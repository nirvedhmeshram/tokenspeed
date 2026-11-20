# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

"""Two-node Petit (EP = 2 x local_world) vs one-node Petit (EP = local_world).

Launch with torchrun, world ranks node-major. LOCAL_WORLD (default 8) sets the
node size; on one 8-GPU node, LOCAL_WORLD=4 emulates two 4-GPU "nodes" whose
pairs are load/store peers, which exercises everything except the GIN path.

Both configurations hold the same experts (weights generated per global expert
id), so for identical tokens and routing the outputs must agree, and the sparse
weights give an exact oracle (from the Petit distributed test) for both.

Per token count T (tokens per GPU) it prints, max over ranks:
  ep_node  : Petit EP over this node, 384 experts, T tokens per GPU (baseline)
  ep_2node : the two-level version, 192 experts per node, same T per GPU
Env: TOKENS="32,64,..." ITERS, WARMUP, CHECK=1 (numerics), ROUTING=random|local
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from petit_hier import HierPetitMoE, PetitNode, RcclGinProvider  # noqa: E402
from tokenspeed_kernel.ops.moe.gluon import (  # noqa: E402
    _DSV41_FLASH_PROFILE,
    _pad_gate_up_pair,
    _pad_mxfp4_pair,
)
from tokenspeed_kernel.thirdparty.gluon_petit import load_petit_kernel  # noqa: E402

E, K, H, INTER_LOGICAL, INTER = 384, 6, 5120, 2304, 2560
CLAMP = 10.0


# ── weights: the sparse oracle of test_gluon_petit_distributed.py ─────────────


def logical_weights(experts, device):
    """Sparse MXFP4 weights for the given global expert ids (checkpoint layout)."""
    n = len(experts)
    w13 = torch.zeros((n, 2 * INTER_LOGICAL, H // 2), dtype=torch.uint8, device=device)
    w2 = torch.zeros((n, H, INTER_LOGICAL // 2), dtype=torch.uint8, device=device)
    rows = torch.arange(2 * INTER_LOGICAL, device=device)
    gate = rows < INTER_LOGICAL
    channel = rows % INTER_LOGICAL
    cols = (channel * 7 + torch.where(gate, 0, 3)) % H
    out_rows = torch.arange(H, device=device)
    for local, expert in enumerate(experts):
        codes = torch.where(gate, 2 + expert % 3, 2 + (expert + 1) % 3).to(torch.uint8)
        w13[local, rows, cols // 2] = (codes << ((cols % 2) * 4)).to(torch.uint8)
        out_cols = (out_rows * 11 + expert) % INTER_LOGICAL
        w2[local, out_rows, out_cols // 2] = (
            (2 + expert % 3) << ((out_cols % 2) * 4)
        ).to(torch.uint8)
    s13 = torch.full(
        (n, 2 * INTER_LOGICAL, H // 32), 123, dtype=torch.uint8, device=device
    )
    s2 = torch.full((n, H, INTER_LOGICAL // 32), 123, dtype=torch.uint8, device=device)
    return w13, s13, w2, s2


def petit_weights(pk, experts, device):
    w13, s13, w2, s2 = logical_weights(experts, device)
    w13, s13 = _pad_gate_up_pair(w13, s13, _DSV41_FLASH_PROFILE)
    w2, s2 = _pad_mxfp4_pair(w2, s2, rows=H, columns=INTER)
    layout = pk.MoeKernelLayout.native_mxfp4
    w13, s13 = pk.repack_moe_kernel_layout(w13, s13, layout=layout, petit_format=True)
    w2, s2 = pk.repack_moe_kernel_layout(w2, s2, layout=layout, petit_format=True)
    return w13.contiguous(), w2.contiguous(), s13.contiguous(), s2.contiguous()


def quantized_reference(value, *, input_quantization):
    shape = value.shape
    blocks = value.float().reshape(-1, 32)
    maximum = blocks.abs().amax(dim=1)
    if input_quantization:
        bits = (maximum * (1.0 / 6.0)).contiguous().view(torch.int32)
        exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).int()
    else:
        bits = maximum.contiguous().view(torch.int32)
        exponent = (((bits + 0x00400000) & 0x7F800000) >> 23).clamp(min=2) - 2
    scale = (exponent << 23).view(torch.float32)[:, None]
    scale = torch.where(scale == 0, 1.0, scale)
    normalized = blocks.abs() / scale
    levels = torch.tensor([0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0], device=value.device)
    closest = (normalized[..., None] - levels).abs().argmin(dim=-1)
    return (levels[closest] * blocks.sign() * scale).reshape(shape)


def reference(x, ids, weights):
    xq = quantized_reference(x, input_quantization=True)
    channel = torch.arange(INTER_LOGICAL, device=x.device)
    output_cols = torch.arange(H, device=x.device)
    output = torch.zeros_like(x, dtype=torch.float32)
    levels = (1.0, 1.5, 2.0)
    for expert in ids.unique().tolist():
        token, slot = torch.where(ids == expert)
        gate = xq[token[:, None], (channel * 7) % H] * (levels[expert % 3] / 16)
        up = xq[token[:, None], (channel * 7 + 3) % H] * (levels[(expert + 1) % 3] / 16)
        hidden = torch.nn.functional.silu(gate.clamp(max=CLAMP)) * up.clamp(
            -CLAMP, CLAMP
        )
        hidden = quantized_reference(hidden, input_quantization=False)
        result = hidden[:, (output_cols * 11 + expert) % INTER_LOGICAL] * (
            levels[expert % 3] / 16
        )
        routed = (result * weights[token, slot, None]).to(torch.bfloat16).float()
        output.index_add_(0, token, routed)
    return output.to(torch.bfloat16)


# ── inputs ────────────────────────────────────────────────────────────────────


_BANK = {}


def replay_bank(T, rank, device):
    """Real routing captured from a serving run (TOKENSPEED_PETIT_DUMP): every
    dumped call with at least T rows, truncated to T rows, as [N, T, K]."""
    key = (T, rank)
    if key not in _BANK:
        node_size = int(os.environ.get("REPLAY_RANKS", "4"))
        d = torch.load(f"{os.environ['REPLAY_DIR']}/routing-rank{rank % node_size}.pt")
        sel = [(i, w) for i, w in zip(d["ids"], d["w"]) if i.shape[0] >= T]
        ids = torch.stack([i[:T] for i, _ in sel]).to(device)
        ws = torch.stack([w[:T] for _, w in sel]).to(device)
        _BANK[key] = (ids.contiguous(), ws.contiguous())
    return _BANK[key]


def make_inputs(T, rank, step, device, routing):
    g = torch.Generator(device=device).manual_seed(1000 * rank + 7 * T + step)
    if routing == "replay":
        bank_ids, bank_w = replay_bank(T, rank, device)
        values = torch.randint(-63, 64, (T, H), generator=g, device=device)
        x = (values.float() * 4.0).to(torch.bfloat16)
        n = bank_ids.shape[0]
        return x.contiguous(), bank_ids[step % n].clone(), bank_w[step % n].clone()
    if routing == "random":
        ids = (
            torch.rand((T, E), generator=g, device=device)
            .topk(K, dim=1)
            .indices.to(torch.int32)
        )
    else:  # "local": all experts on this rank's own node (no-traffic control)
        half = E // 2
        node = rank // int(os.environ.get("LOCAL_WORLD", "8"))
        ids = (
            torch.rand((T, half), generator=g, device=device).topk(K, dim=1).indices
            + node * half
        ).to(torch.int32)
    raw = 1.0 + torch.rand((T, K), generator=g, device=device)
    weights = (raw / raw.sum(dim=1, keepdim=True)).float()
    values = torch.randint(-63, 64, (T, H), generator=g, device=device)
    x = (values.float() * 4.0).to(torch.bfloat16)
    return x.contiguous(), ids.contiguous(), weights.contiguous()


# ── timing ────────────────────────────────────────────────────────────────────


def time_it(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000.0 / iters  # us


def max_over_ranks(value):
    t = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def main():
    local_world = int(os.environ.get("LOCAL_WORLD", "8"))
    os.environ.setdefault("LOCAL_WORLD", str(local_world))
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert (
        world == 2 * local_world
    ), f"world {world} must be 2 x LOCAL_WORLD {local_world}"
    gpu = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    node, local = divmod(rank, local_world)
    groups = [
        dist.new_group(
            list(range(n * local_world, (n + 1) * local_world)), backend="gloo"
        )
        for n in range(2)
    ]
    node_group = groups[node]

    tokens = [int(t) for t in os.environ.get("TOKENS", "32,64,128,256,512").split(",")]
    iters, warmup = int(os.environ.get("ITERS", "50")), int(
        os.environ.get("WARMUP", "10")
    )
    check = os.environ.get("CHECK", "1") == "1"
    routing = os.environ.get("ROUTING", "random")
    max_tokens = max(tokens)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    diag = os.environ.get("DIAG") in ("1", "2")

    def mark(msg):
        # Sync between the two MoEs in the checks: back-to-back unsynced
        # ep_node -> ep_2node across nodes has hit a "GPU Hang" (under study).
        torch.cuda.synchronize()
        if diag:
            print(f"[mark r{rank}] {msg}", flush=True)

    pk = load_petit_kernel()
    t0 = time.time()
    # Baseline: one node, every expert, EP = local_world.
    base = PetitNode(pk, node_group, E, K, H, INTER, max_tokens)
    per = E // local_world
    base_w = petit_weights(pk, list(range(local * per, (local + 1) * per)), device)
    # Two-level: this node owns half the experts, EP = local_world inside it.
    backend = os.environ.get("BACKEND", "gin")
    provider = RcclGinProvider() if backend == "gin" else None
    hier = HierPetitMoE(pk, provider, node_group, num_experts=E, topk=K, model_dim=H, inter_dim=INTER,
                        max_tokens=max_tokens, backend=backend)  # fmt: skip
    per2 = E // 2 // local_world
    first = node * (E // 2) + local * per2
    hier_w = petit_weights(pk, list(range(first, first + per2)), device)
    torch.cuda.synchronize()
    dist.barrier()
    log(f"setup {time.time() - t0:.1f}s world={world} local_world={local_world} backend={backend} gin={provider.gin_type_name if provider else '-'} "
        f"peer_direct={hier.direct} row_bytes={hier.row_bytes} routing={routing}")  # fmt: skip

    def run_base(x, ids, w):
        T = x.shape[0]
        views = pk.MegaMoeInputViews(
            tokens=base.views.tokens[:T], scales=base.views.scales[:T],
            expert_ids=base.views.expert_ids[:T], expert_weights=base.views.expert_weights[:T],
        )  # fmt: skip
        base.config.quantize(x, out=views)
        views.expert_ids.copy_(ids)
        views.expert_weights.copy_(w)
        return base.run(base_w, T, inputs=views)

    for T in tokens:
        x, ids, w = make_inputs(T, rank, 0, device, routing)
        if check:
            for step in range(int(os.environ.get("CHECK_STEPS", "2"))):
                x, ids, w = make_inputs(T, rank, step, device, routing)
                mark(f"T={T} step={step} base")
                yb = run_base(x, ids, w).clone()
                mark(f"T={T} step={step} hier")
                yh = hier.forward(x, ids, w, hier_w).clone()
                yh2 = hier.forward(x, ids, w, hier_w).clone()
                mark(f"T={T} step={step} done")
                torch.cuda.synchronize()
                ref = reference(x, ids, w) if T <= 128 else yb
                scale = ref.float().abs().max().item() or 1.0

                def err(y):
                    return (y.float() - ref.float()).abs().max().item() / scale

                # The two-level path adds two BF16-rounded partial sums, so allow
                # about one BF16 ulp of the output scale.
                errs = torch.tensor([err(yb), err(yh), err(yh2)], dtype=torch.float64)
                dist.all_reduce(errs, op=dist.ReduceOp.MAX)
                eb, eh, eh2 = errs.tolist()
                ok = eb <= 2**-7 and eh <= 2**-7 and eh2 <= 2**-7
                log(f"T={T} step={step} check {'PASS' if ok else 'FAIL'}  max rel err vs "
                    f"{'oracle' if T <= 128 else 'ep_node'}: ep_node {eb:.3g}  ep_2node {eh:.3g} (repeat {eh2:.3g})")  # fmt: skip
                if os.environ.get("DIAG") == "1" and err(yh) > 2**-7:
                    bad_rows = (
                        ((yh.float() - ref.float()).abs().amax(dim=1) > 2**-7 * scale)
                        .nonzero()
                        .flatten()
                    )
                    own_ok = (yh.float() - ref.float()).abs()
                    print(f"rank {rank} T={T} step={step}: {bad_rows.numel()} bad rows, first {bad_rows[:8].tolist()}; "
                          f"hier/ref norm ratio {(yh.float().norm() / ref.float().norm()).item():.3f}", flush=True)  # fmt: skip

        if routing == "replay":
            # Rotate through the captured calls; the two small copies are paid by both sides.
            bank_ids, bank_w = replay_bank(T, rank, device)
            ctr = [0]

            def rotate():
                i = ctr[0] % bank_ids.shape[0]
                ctr[0] += 1
                ids.copy_(bank_ids[i])
                w.copy_(bank_w[i])

        else:

            def rotate():
                pass

        tb = max_over_ranks(
            time_it(lambda: (rotate(), run_base(x, ids, w)), iters, warmup)
        )
        th = max_over_ranks(
            time_it(lambda: (rotate(), hier.forward(x, ids, w, hier_w)), iters, warmup)
        )
        graph_note = ""
        if os.environ.get("GRAPH", "1") == "1":
            # Decode runs under CUDA graphs; eager timing above includes Python
            # launch overhead (Petit's argument checks, ~a dozen launches).
            torch.cuda.synchronize()
            dist.barrier()
            gb, gh = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(gb):
                yb_static = run_base(x, ids, w)
            torch.cuda.synchronize()
            dist.barrier()
            with torch.cuda.graph(gh):
                yh_static = hier.forward(x, ids, w, hier_w)
            torch.cuda.synchronize()
            dist.barrier()
            gtb = max_over_ranks(
                time_it(lambda: (rotate(), gb.replay()), iters, warmup)
            )
            gth = max_over_ranks(
                time_it(lambda: (rotate(), gh.replay()), iters, warmup)
            )
            if check:
                yb = run_base(x, ids, w)
                torch.cuda.synchronize()
                yh = hier.forward(x, ids, w, hier_w)
                torch.cuda.synchronize()
                same = torch.tensor(
                    [int(torch.equal(yb, yb_static) and torch.equal(yh, yh_static))]
                )
                dist.all_reduce(same, op=dist.ReduceOp.MIN)
                graph_note = "" if same.item() else "  GRAPH OUTPUT MISMATCH"
            log(
                f"T={T:5d}  graph: ep_node {gtb:8.1f} us   ep_2node {gth:8.1f} us   speedup {gtb / gth:5.2f}x{graph_note}"
            )
            del gb, gh
        # Phase breakdown of the two-level path, from one instrumented run.
        ev = []
        for _ in range(3):
            ev.clear()
            hier.forward(x, ids, w, hier_w, events=ev)
        torch.cuda.synchronize()
        phases = [ev[i].elapsed_time(ev[i + 1]) * 1000 for i in range(len(ev) - 1)]
        phases = [max_over_ranks(p) for p in phases]
        log(f"T={T:5d}  eager: ep_node {tb:8.1f} us   ep_2node {th:8.1f} us   speedup {tb / th:5.2f}x   "
            f"[quant+route {phases[0]:.0f} | dispatch {phases[1]:.0f} | petit {phases[2]:.0f} | "
            f"combine {phases[3]:.0f} | add {phases[4]:.0f}]")  # fmt: skip

    hier.close()
    if provider is not None:
        provider.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
