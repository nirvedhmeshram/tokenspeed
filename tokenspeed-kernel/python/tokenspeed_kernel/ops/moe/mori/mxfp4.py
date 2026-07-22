# Copyright (c) 2026 LightSeek Foundation
#
# SPDX-License-Identifier: MIT
"""Registered MXFP4 MoE apply using MORI all-to-all EP.

Correctness-first v1: dispatch bf16 hidden states via MORI, dequantize the MXFP4
expert weights to bf16 (cached on the module), run the bf16 masked grouped-GEMM,
and combine. This reuses the validated bf16 path; a native MXFP4 grouped-GEMM (no
dequant) is the perf follow-on. Selected for Kimi-K2.5-MXFP4 with
--enable-expert-parallel --all2all-backend mori.
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
        dequant_mxfp4,
        get_dispatcher,
        masked_grouped_gemm,
    )

    def _dequant_experts_bf16(w: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize the local experts' MXFP4 weights to bf16 (TRANSIENT — NOT cached).

        Caching per layer would accumulate ~8GB x num_layers of bf16 experts on top of the
        MXFP4 weights (OOM on a full model). Transient dequant keeps only the current layer's
        bf16 live (freed after this layer's GEMM). Slow (re-dequants each forward) but memory-
        feasible; the perf fix is a native MXFP4 grouped-GEMM (no dequant).
        w13_weight [E,2I,H//2] uint8 + w13_weight_scale [E,2I,H//32] -> bf16 [E,2I,H]; likewise w2."""
        w13 = dequant_mxfp4(w.w13_weight, w.w13_weight_scale)   # [E, 2I, H]
        w2 = dequant_mxfp4(w.w2_weight, w.w2_weight_scale)      # [E, H, I]
        return w13, w2

    @register_kernel(
        "moe",
        "apply",
        name="mori_ep_mxfp4_moe_apply",
        solution="mori",
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
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
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

        w13_bf16, w2_bf16 = _dequant_experts_bf16(w)

        ep_size = int(getattr(w, "ep_size", dist.get_world_size()))
        ep_rank = int(getattr(w, "ep_rank", dist.get_rank()))
        num_local = int(getattr(w, "num_local_experts", w13_bf16.shape[0]))
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
        packed.copy_(masked_grouped_gemm(packed, handle["counts"], w13_bf16, w2_bf16))
        return dispatcher.combine(handle)
