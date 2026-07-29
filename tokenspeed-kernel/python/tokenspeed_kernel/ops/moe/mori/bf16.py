# Copyright (c) 2026 LightSeek Foundation
#
# SPDX-License-Identifier: MIT
"""Registered bf16 MoE apply using MORI all-to-all EP (real dispatch/combine).

Mirrors ops/moe/flashinfer/cutedsl_deepep_nvfp4.py (dispatch -> grouped-GEMM ->
combine) but backed by MORI's v2 op on AMD. Selected for
vendor=amd, gfx950, bf16/unquant, expert-parallel, all-to-all EP. bf16 is the
proving ground (M0); an MXFP4 variant is a follow-on.
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
    from tokenspeed_kernel.ops.communication.mori_ep import (
        get_dispatcher,
        masked_grouped_gemm,
    )

    @register_kernel(
        "moe",
        "apply",
        name="mori_ep_bf16_moe_apply",
        solution="mori",
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({True}),
            "ispp_alignment": frozenset({128}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
    )
    def mori_ep_bf16_moe_apply(
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
        """bf16 MoE FFN with real EP dispatch/combine via MORI.

        ``w`` exposes ``w13_weight`` ``[E_local, 2*I, H]`` (gate [0:I], up [I:2I]),
        ``w2_weight`` ``[E_local, H, I]`` (bf16), ``top_k``, and EP mapping
        ``ep_rank`` / ``ep_size`` / ``num_local_experts``. ``topk_ids`` are GLOBAL
        expert ids (MORI routes by global id); do NOT pre-mask to local space.
        """
        if topk_weights is None or topk_ids is None:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                scores, k=getattr(w, "top_k"), dim=-1, sorted=False
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        ep_size = int(getattr(w, "ep_size", dist.get_world_size()))
        ep_rank = int(getattr(w, "ep_rank", dist.get_rank()))
        num_local = int(getattr(w, "num_local_experts", w.w13_weight.shape[0]))
        # Fixed dispatch capacity (per rank). Must upper-bound tokens/rank across all
        # calls; MORI allocates symmetric buffers for this at op creation. One
        # dispatcher is shared across all same-shape MoE layers (see get_dispatcher).
        cap = int(max_num_tokens_per_gpu or x.shape[0])
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
        packed.copy_(masked_grouped_gemm(packed, handle["counts"], w.w13_weight, w.w2_weight))
        # Return the COMPLETE per-token routed result; the model compensates for the
        # framework all_reduce in dp=1 mode (DeepseekV3MoE.forward) and bypasses it in
        # dp>1 DP-attention (forward_alltoall). See mori/mxfp4.py. Do NOT compensate here.
        return dispatcher.combine(handle)
