# Copyright (c) 2026 LightSeek Foundation
#
# SPDX-License-Identifier: MIT
"""Registered A16W4 SiTU MoE apply using MORI all-to-all EP (Kimi-K3).

Native A16W4 SiTU path (NO weight/activation requant): the routed-expert weights are the raw
K-packed MXFP4 "linear checkpoint" (validated at load, not preshuffled); each forward MORI
dispatches bf16 hidden states at the routed-expert (latent) width, the tokens -- already grouped
per local expert in MORI's 3D ``[E_local, cap, H]`` buffer -- are gathered into a contiguous buffer
and run through the gfx950 Gluon grouped A16W4 SiTU FFN, then scattered back and combined.

Sibling of ``mori/mxfp4.py`` (the silu W4A4 kernel). Differences: SiTU (not SwiGLU) activation,
A16W4 (activations stay bf16 -- no ``_quantize_mxfp4_activation``), and the raw linear-checkpoint
weight layout (``validate_linear_mxfp4_moe_weights``, not the preshuffled Triton-tensor layout).

The FFN reuses the existing native host fn ``gluon_a16w4_situ_grouped_ep_gfx950`` wholesale, driven
with a synthetic single-route-per-row topk (``top_k=1``, ``weights=1``, local expert ids,
``expert_start=0``): its built-in masked top-k reduce then collapses to an identity per-row
passthrough, yielding exactly the pure per-row FFN MORI needs (MORI ``combine`` applies the real
gate weights). Selected for Kimi-K3-MXFP4 with --enable-expert-parallel --all2all-backend mori.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

platform = current_platform()


if platform.is_amd:
    from tokenspeed_kernel.ops.communication.mori_ep import get_dispatcher
    from tokenspeed_kernel.ops.moe.gluon.mxfp4 import validate_linear_mxfp4_moe_weights
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.situ_grouped import (
        gluon_a16w4_situ_grouped_ep_gfx950,
    )

    def _situ_betas(w: torch.nn.Module) -> tuple[float, float | None]:
        """SiTU gate params for the grouped reducer. Mirrors the native SiTU apply
        (ops.moe.gluon.mxfp4): ``activation_situ_beta`` defaults to 1.0 and
        ``activation_situ_linear_beta`` to None (no tanh clamp on the linear branch)."""
        return (
            float(getattr(w, "activation_situ_beta", 1.0)),
            getattr(w, "activation_situ_linear_beta", None),
        )

    def _grouped_a16w4_situ_gemm_3d(
        packed_x: torch.Tensor,
        counts: torch.Tensor,
        w: torch.nn.Module,
        n_recv_bound: int,
    ) -> None:
        """Native A16W4 SiTU grouped FFN over MORI's 3D [E,cap,L] padded buffer, IN PLACE.

        Bridges MORI's padded 3D layout to the native SiTU host fn with NO host sync (eager OR
        under HIP-graph capture), so the whole MoE forward captures. ``n_recv_bound`` is a STATIC
        (shape-derived, not ``.item()``) upper bound on the rows this rank received this step; the
        gathered buffer is sized to ``m = min(E*cap, n_recv_bound)`` so the gather/scatter + FFN
        stay proportional to real work rather than the full E*cap capacity.

        Build a fixed-size [E*cap] permutation of slots (valid rows -- local < counts[e] -- first
        in expert-major order, padding rows after), take the first ``m`` as the gather buffer, run
        the SiTU FFN with a synthetic top_k=1 route (local expert id per row, unit weight;
        padding rows get an out-of-range expert id so alignment/reduce drop them), then scatter the
        result back into ``packed_x`` itself. Padding rows are written back unchanged; MORI combine
        reads only valid rows via src_info, so padding is ignored. Pure FFN; combine applies the
        gate weights. A16W4: activations stay bf16 (no requant), only weights dequant from MXFP4.
        """
        E, cap, L = packed_x.shape
        dev = packed_x.device
        n_slots = E * cap
        counts_long = counts.to(torch.long)
        total = counts_long.sum()  # sum(counts), 0-dim tensor (no host sync)

        # Static, shape-derived row bound -- same rationale/pattern as the mxfp4 MORI bridge:
        # avoids a per-layer device->host ``.item()`` sync that would drain the GPU pipeline, and
        # never truncates received rows under DP imbalance. See ops/moe/mori/mxfp4.py.
        m = min(n_slots, n_recv_bound)
        if m == 0:
            return

        # Fixed-size [E*cap] permutation: valid rows (local < counts[e]) first in expert-major
        # order, padding rows after; take the first ``m``. argsort on a per-slot key -- no
        # ``.item()``, no data-dependent shape -- so it stays HIP-graph capturable; the [:m] prefix
        # is a collision-free permutation prefix, so the scatter never double-writes a slot.
        slot = torch.arange(n_slots, device=dev)
        e_of_slot = slot // cap
        l_of_slot = slot - e_of_slot * cap
        is_pad = (l_of_slot >= counts_long[e_of_slot]).to(torch.int64)
        gather_idx = torch.argsort(is_pad * n_slots + slot)[:m]  # [m] unique slots
        row = torch.arange(m, device=dev)

        flat = packed_x.reshape(n_slots, L)
        x_flat = flat[gather_idx]  # [m, L] valid front, pad back

        # Synthetic single-route-per-row topk driving the native SiTU host fn. Valid rows carry
        # their LOCAL expert id (weights are the local expert tensors, expert_start=0); padding
        # rows get id ``E`` (>= num_local) so moe_align/reduce discard them. Unit weights make the
        # host fn's masked top-k reduce an identity passthrough -> pure per-row FFN.
        local_ids = torch.where(
            row < total,
            e_of_slot[gather_idx],
            torch.full((m,), E, device=dev, dtype=torch.long),
        ).to(torch.int32)[:, None]
        weights = torch.ones((m, 1), device=dev, dtype=torch.float32)

        situ_beta, situ_linear_beta = _situ_betas(w)
        y_flat = gluon_a16w4_situ_grouped_ep_gfx950(
            x_flat,
            w.w13_weight,
            w.w13_weight_scale,
            w.w2_weight,
            w.w2_weight_scale,
            weights,
            local_ids,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            expert_start=0,
        ).to(packed_x.dtype)

        # Scatter back IN PLACE. gather_idx is a permutation prefix, so valid rows land in their
        # slots collision-free; padding rows (row >= total) write their own original value back.
        flat[gather_idx] = torch.where((row < total)[:, None], y_flat, x_flat)

    @register_kernel(
        "moe",
        "apply",
        name="mori_ep_a16w4_situ_moe_apply",
        solution="mori",
        weight_preprocessor=validate_linear_mxfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            # Kimi-K3 routed experts use the SiTU gate (activation_situ_beta / _linear_beta).
            "activation": frozenset({"situ"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({True}),
            "ispp_alignment": frozenset({1}),
            # A16W4: activations stay bf16 through both GEMMs (only weights dequant from MXFP4),
            # so the internal activation is the input dtype -- matches the planner's requested
            # trait for K3's native SiTU MoE (internal_activation_dtype_override="input").
            "internal_activation_dtype": frozenset({"input"}),
        },
        priority=Priority.SPECIALIZED + 2,
    )
    def mori_ep_a16w4_situ_moe_apply(
        plan: dict,
        x: torch.Tensor,
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        num_tokens_global: int | None = None,
        max_num_tokens_per_gpu: int | None = None,
        do_finalize: bool = True,
        enable_pdl: bool = False,
        low_latency: bool | None = None,
        overlap_fn: Callable[[], None] | None = None,
    ):
        # ``low_latency`` / ``overlap_fn`` follow the all-to-all apply contract (see the mxfp4
        # MORI kernel): MORI has a single dispatch/combine path, so ``low_latency`` is accepted and
        # ignored; ``overlap_fn`` (if any) runs after the dispatch send.
        if not do_finalize:
            raise ValueError("MORI A16W4 SiTU MoE cannot defer finalization")
        # SiTU uses sigmoid / noaux_tc routing computed upstream (KimiLinearMoE); never recompute a
        # softmax here. The caller (precomputed_topk) always supplies the routing.
        if topk_weights is None or topk_ids is None:
            raise ValueError("MORI A16W4 SiTU MoE requires precomputed top-k")

        ep_size = int(getattr(w, "ep_size", dist.get_world_size()))
        ep_rank = int(getattr(w, "ep_rank", dist.get_rank()))
        num_local = int(getattr(w, "num_local_experts"))
        top_k = int(getattr(w, "top_k"))

        # MORI dispatch capacity (tokens/rank). Symmetric buffers size to cap at op-creation, so
        # cap must upper-bound tokens/rank; derive from the runtime per-GPU capacity (falling back
        # to this step's count). ``MORI_EP_MAX_TOKENS_PER_RANK`` is an explicit override.
        import os as _os

        _cap_env = _os.environ.get("MORI_EP_MAX_TOKENS_PER_RANK")
        cap = (
            int(_cap_env)
            if _cap_env
            else max(int(max_num_tokens_per_gpu or 0), int(x.shape[0]))
        )
        # hidden_dim = x.shape[1] is the routed-expert (latent) width for K3 (e.g. 3584); the
        # dispatcher keys and MORI symmetric buffers size to that automatically.
        dispatcher = get_dispatcher(
            rank=ep_rank,
            world_size=ep_size,
            hidden_dim=x.shape[1],
            num_local_experts=num_local,
            num_experts_per_token=top_k,
            max_num_inp_token_per_rank=cap,
            data_type=torch.bfloat16,
        )

        handle = dispatcher.dispatch(x, topk_weights.float(), topk_ids)
        if overlap_fn is not None:
            overlap_fn()
        packed = handle["packed_x"]  # [E_local, cap, L]
        # Static receive bound: (max tokens any EP rank sends) * ep_size * top_k, capped at E*cap
        # by the bridge. Independent of this rank's local x.shape[0] (see the mxfp4 kernel), so DP
        # idle/imbalanced ranks still process the rows peers routed to their experts.
        per_rank = max(int(max_num_tokens_per_gpu or 0), int(x.shape[0]))
        n_recv_bound = ep_size * per_rank * top_k
        _grouped_a16w4_situ_gemm_3d(
            packed, handle["counts"], w, n_recv_bound
        )  # in place
        # Return the COMPLETE per-token routed result on every rank; do NOT reduce/scale here.
        return dispatcher.combine(handle)
