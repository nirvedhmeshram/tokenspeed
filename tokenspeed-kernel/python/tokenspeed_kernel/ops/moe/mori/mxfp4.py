# Copyright (c) 2026 LightSeek Foundation
#
# SPDX-License-Identifier: MIT
"""Registered MXFP4 MoE apply using MORI all-to-all EP.

Native MXFP4 path (NO dequant): weights are swizzled once at load via the shared
``triton_mxfp4_moe_weights`` preprocessor; each forward MORI dispatches bf16 hidden
states, the tokens (already grouped per local expert in MORI's 3D [E,cap,H] buffer)
are gathered into a contiguous expert-sorted buffer and run through the triton mxfp4
grouped-GEMM (``grouped_mxfp4_ffn``) directly on the packed MXFP4 weights, scattered
back into the 3D buffer, and combined. Selected for Kimi-K2.5-MXFP4 with
--enable-expert-parallel --all2all-backend mori.

Replaces the earlier correctness-first v1 that dequantized weights to bf16 every
forward (~4 s/token + transient memory pressure); the native GEMM eliminates both.
"""
from __future__ import annotations

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
    from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton

    with redirect_triton_to_tokenspeed_triton():
        from triton_kernels.tensor import make_ragged_tensor_metadata

    from tokenspeed_kernel.ops.communication.mori_ep import get_dispatcher
    from tokenspeed_kernel.ops.moe.triton.mxfp4 import (
        grouped_mxfp4_ffn,
        triton_mxfp4_moe_weights,
    )

    def _grouped_mxfp4_gemm_3d(
        packed_x: torch.Tensor, counts: torch.Tensor, w: torch.nn.Module
    ) -> torch.Tensor:
        """Native MXFP4 grouped SwiGLU FFN over MORI's 3D [E,cap,H] padded buffer.

        Bridges MORI's padded 3D layout to the triton ragged (contiguous, expert-sorted)
        layout: gather valid rows (< counts[e]) into a flat buffer, run the mxfp4 grouped
        GEMM on the raw packed weights (no dequant), scatter results back into a fresh 3D
        buffer (padding rows = 0, ignored by MORI combine). Drop-in for the bf16
        masked_grouped_gemm. Pure FFN — MORI combine applies the gate weights.
        """
        E, cap, H = packed_x.shape
        out = torch.zeros_like(packed_x)
        counts_long = counts.to(torch.long)
        total = int(counts_long.sum().item())
        if total == 0:
            return out
        dev = packed_x.device
        offsets = torch.cumsum(counts_long, 0) - counts_long          # [E] start per expert
        expert_of_row = torch.repeat_interleave(
            torch.arange(E, device=dev), counts_long
        )                                                             # [total]
        local_of_row = torch.arange(total, device=dev) - offsets[expert_of_row]
        flat_idx = expert_of_row * cap + local_of_row                # [total] pos in [E*cap]

        x_flat = packed_x.reshape(E * cap, H)[flat_idx]              # gather -> [total, H]
        meta = make_ragged_tensor_metadata(counts.to(torch.int32), total)
        y_flat = grouped_mxfp4_ffn(x_flat, meta, w).to(packed_x.dtype)   # [total, H]
        out.reshape(E * cap, H)[flat_idx] = y_flat                   # scatter back
        return out

    @register_kernel(
        "moe",
        "apply",
        name="mori_ep_mxfp4_moe_apply",
        solution="mori",
        weight_preprocessor=triton_mxfp4_moe_weights,
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({True}),
            "ispp_alignment": frozenset({1}),
            "internal_activation_dtype": frozenset({"input", "fp8"}),
            "supports_bias": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 2,
    )
    def mori_ep_mxfp4_moe_apply(
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
    ):
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        ep_size = int(getattr(w, "ep_size", dist.get_world_size()))
        ep_rank = int(getattr(w, "ep_rank", dist.get_rank()))
        num_local = int(getattr(w, "num_local_experts"))
        # Fixed MORI dispatch capacity (tokens/rank). MORI allocates symmetric buffers
        # ~[num_local, ws*cap, H] at op-creation, so cap must upper-bound tokens/rank but
        # stay memory-bounded. Env override for larger prefill. (Prefill-mode dispatch is
        # a deferred follow-on; this targets decode / short prompts.)
        import os as _os

        cap = int(_os.environ.get("MORI_EP_MAX_TOKENS_PER_RANK", "2048"))
        dispatcher = get_dispatcher(
            rank=ep_rank,
            world_size=ep_size,
            hidden_dim=x.shape[1],
            num_local_experts=num_local,
            num_experts_per_token=int(getattr(w, "top_k")),
            max_num_inp_token_per_rank=cap,
            data_type=torch.bfloat16,
        )

        handle = dispatcher.dispatch(x, topk_weights.float(), topk_ids)
        packed = handle["packed_x"]  # [E_local, cap, H]
        packed.copy_(_grouped_mxfp4_gemm_3d(packed, handle["counts"], w))
        return dispatcher.combine(handle)
