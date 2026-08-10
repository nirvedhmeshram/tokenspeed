# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Single-GPU (gfx950) numeric unit test for the MORI EP A16W4 SiTU bridge.

Exercises ``_grouped_a16w4_situ_gemm_3d`` -- the bridge from MORI's padded 3D
``[E,cap,L]`` dispatch buffer to the native gfx950 grouped A16W4 SiTU FFN -- with no
``mori``/``flydsl`` package and no multi-rank collective, so it runs on one MI350X and
skips elsewhere. The bridge drives the native SiTU host fn with a synthetic single-route
(``top_k=1``, unit weight) so its masked top-k reduce is an identity passthrough; this test
is the guard that that reuse yields the correct per-row FFN, that padding rows are left
untouched, and that the path is HIP-graph capturable.
"""

from __future__ import annotations

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip(
        "MORI EP A16W4 SiTU bridge is gfx950 (CDNA4) only", allow_module_level=True
    )

import tokenspeed_kernel  # noqa: E402,F401
from kimi3_reference import a16w4_mxfp4_moe_reference  # noqa: E402
from tokenspeed_kernel.ops.moe.mori.situ import (  # noqa: E402
    _grouped_a16w4_situ_gemm_3d,
)
from utils import make_mxfp4_moe_weights  # noqa: E402

SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()


def _make_module(num_local: int, L: int, I: int, gen: torch.Generator):
    """Raw K-packed linear-checkpoint MXFP4 weights + the attrs the bridge reads."""
    raw = make_mxfp4_moe_weights(num_local, L, I, gen)
    w = torch.nn.Module()
    w.w13_weight = raw["w13_weight"]
    w.w13_weight_scale = raw["w13_scale"]
    w.w2_weight = raw["w2_weight"]
    w.w2_weight_scale = raw["w2_scale"]
    w.num_local_experts = num_local
    w.activation_situ_beta = SITU_BETA
    w.activation_situ_linear_beta = SITU_LINEAR_BETA
    return w, raw


def _expert_major_valid_rows(packed: torch.Tensor, counts_list: list[int]):
    """Gather valid rows in the SAME expert-major/local order the bridge uses, plus their
    local expert ids -- so a single-route (weight=1) reference is row-aligned to the bridge.
    """
    rows, ids = [], []
    for e, n in enumerate(counts_list):
        for r in range(n):
            rows.append(packed[e, r])
            ids.append(e)
    x = torch.stack(rows) if rows else packed.new_zeros((0, packed.shape[-1]))
    lid = torch.tensor(ids, device=packed.device, dtype=torch.int32).reshape(-1, 1)
    return x, lid


def test_grouped_a16w4_situ_gemm_3d_bridge() -> None:
    """The MORI 3-D [E,cap,L] <-> native SiTU host-fn bridge, A16W4, in place.

    ``_grouped_a16w4_situ_gemm_3d`` writes the FFN output back into ``packed_x`` IN PLACE
    (returns None) and leaves padding rows untouched (MORI combine reads only valid rows).
    So we compute the reference from a pre-call clone of the valid rows, then check the valid
    rows match and padding rows are unchanged.
    """
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(11)
    # L (activation width) % 256 == 0 and I % 128 == 0 for the grouped A16W4 host fn.
    E, cap, L, I = 6, 32, 512, 512
    counts_list = [5, 0, 32, 1, 17, 8]
    counts = torch.tensor(counts_list, device=dev, dtype=torch.int32)
    w, raw = _make_module(E, L, I, gen)

    # padding rows deliberately non-zero -- must be left untouched by the gather/scatter
    packed = (
        torch.randn(E, cap, L, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1
    )
    packed_before = packed.clone()

    # reference: pure per-row SiTU FFN over the ORIGINAL valid rows (single local route, w=1)
    x_ref, local_ids = _expert_major_valid_rows(packed_before, counts_list)
    weights = torch.ones_like(local_ids, dtype=torch.float32)
    ref = a16w4_mxfp4_moe_reference(
        x_ref,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        local_ids,
        weights,
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    )

    # exact receive bound == sum(counts): exercises the m < E*cap truncation path and checks
    # no valid row is dropped (a too-small bound would silently lose tokens).
    n_recv_bound = int(counts.sum().item())
    out = _grouped_a16w4_situ_gemm_3d(packed, counts, w, n_recv_bound)  # in place
    assert out is None

    got, _ = _expert_major_valid_rows(packed, counts_list)
    # A16W4 grouped kernel vs the bf16-dequant SiTU reference -> ~0.97 cos is expected-correct
    # (activations bf16; only weights are MXFP4). A real bug (wrong route/activation/truncation)
    # collapses this far below the bar.
    assert _cos(got, ref) > 0.97, f"cos={_cos(got, ref)}"

    valid = torch.arange(cap, device=dev)[None, :] < counts[:, None].to(torch.long)
    assert torch.equal(packed[~valid], packed_before[~valid])


def test_grouped_a16w4_situ_gemm_3d_bridge_is_cuda_graph_capturable() -> None:
    """The bridge must replay under HIP-graph capture (no host sync): the static receive
    bound and device-side argsort/where keep ``m`` shape-stable across replays."""
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(7)
    E, cap, L, I = 4, 16, 512, 512
    counts_list = [3, 0, 16, 9]
    counts = torch.tensor(counts_list, device=dev, dtype=torch.int32)
    w, _ = _make_module(E, L, I, gen)
    # static bound (as under capture): ep_size*max_tokens*top_k, capped by the bridge at E*cap
    n_recv_bound = E * cap

    base = torch.randn(E, cap, L, generator=gen, device=dev, dtype=torch.bfloat16) * 0.1
    packed = base.clone()

    def run() -> torch.Tensor:
        packed.copy_(base)
        _grouped_a16w4_situ_gemm_3d(packed, counts, w, n_recv_bound)
        return packed.clone()

    eager = run()
    warmup = torch.cuda.Stream()
    with torch.cuda.stream(warmup):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed.copy_(base)
        _grouped_a16w4_situ_gemm_3d(packed, counts, w, n_recv_bound)
        captured = packed.clone()
    graph.replay()
    torch.cuda.synchronize()
    valid = torch.arange(cap, device=dev)[None, :] < counts[:, None].to(torch.long)
    assert _cos(captured[valid], eager[valid]) > 0.999
