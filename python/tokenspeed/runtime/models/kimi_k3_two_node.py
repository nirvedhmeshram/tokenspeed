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

"""Prototype: Kimi-K3 MoE over two nodes with native gfx950 EP kernels."""

from __future__ import annotations

import torch
from torch import nn

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    token_all_gather_rows,
    token_reduce_scatter,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.query_shard import scatter_count
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.latent import (
    Kimi3LatentProjection,
    Kimi3MoEExecutionPlan,
)
from tokenspeed.runtime.layers.moe.topk import TopK, TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import (
    RoutingMethodType,
    get_moe_backend,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.models.kimi_k3 import (
    KimiLinearMLP,
    KimiLinearMoEGate,
    _situ_betas,
)
from tokenspeed.runtime.utils import add_prefix

# Rows per TP slice the GIN window holds (decode and 8192-token prefill chunks).
_MAX_SLICE_ROWS = 1024
_SHARED_EXCHANGE = None


class KimiLinearMoETwoNode(nn.Module):
    """Prototype: K3 MoE with attention TP inside a node and DP across two nodes.

    Every GPU of a node holds the node's tokens after attention. Node n owns
    experts [n * E/2, (n + 1) * E/2), one EP16 shard per GPU, run by the native
    gfx950 SiTU EP kernels over the node's own tokens plus the pair node's
    tokens: each rank swaps a 1/TP slice of routed inputs with the rank of the
    same local index on the other node (RCCL GIN), and the node all-gathers the
    received slices. Partial outputs for the other node's tokens are
    reduce-scattered and swapped back; each rank adds the slice it receives
    into its own rows, and one all-reduce over the node completes the routed
    latent. Shared experts and the up projection shard over attention TP.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        layer_index: int,
        model_scope: str,
        moe_block_count: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        alt_stream: torch.cuda.Stream | None,
    ) -> None:
        super().__init__()
        del model_scope, moe_block_count
        if (
            mapping.moe.tp_size != 1
            or mapping.moe.dp_size != 1
            or mapping.attn.dp_size != 2
            or mapping.moe.ep_size != mapping.attn.tp_size * mapping.attn.dp_size
        ):
            raise ValueError(
                "two-node K3 MoE requires attention DP2, MoE TP1, and EP == world"
            )
        self.mapping = mapping
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.routed_scaling_factor = config.routed_scaling_factor
        self.routed_hidden = config.routed_expert_hidden_size or config.hidden_size
        situ_beta, situ_linear_beta = _situ_betas(config)
        self.gate = KimiLinearMoEGate(config.hidden_size, config.num_experts)
        self.experts = MoELayer(
            top_k=self.top_k,
            num_experts=self.num_experts,
            hidden_size=self.routed_hidden,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=mapping.moe.tp_rank,
            tp_size=mapping.moe.tp_size,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            activation="situ",
            activation_situ_beta=situ_beta,
            activation_situ_linear_beta=situ_linear_beta,
            routing_config={
                "n_group": config.num_expert_group,
                "topk_group": config.topk_group,
                "routed_scaling_factor": self.routed_scaling_factor,
                "normalize_topk_weights": config.moe_renormalize,
                "correction_bias": self.gate.e_score_correction_bias,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "activation_situ_beta": situ_beta,
                "activation_situ_linear_beta": situ_linear_beta,
            },
            routing_mode="precomputed_topk",
            internal_activation_dtype_override="input",
        )
        self._exchange = None

        self.topk = TopK(
            top_k=self.top_k,
            renormalize=config.moe_renormalize,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            num_fused_shared_experts=0,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            output_format=TopKOutputFormat.STANDARD,
            topk_weights_dtype=torch.float32,
        )
        self.routed_expert_down_proj = Kimi3LatentProjection(
            config.hidden_size,
            self.routed_hidden,
            params_dtype=None,
            prefix=add_prefix("routed_expert_down_proj", prefix),
            solution="auto",
            shard_group=None,
            column_group=None,
            shard_rank=0,
            shard_size=1,
            multicast_down=None,
        )
        tp = mapping.attn
        self._shard_up_projection = (
            tp.tp_size > 1 and config.hidden_size % tp.tp_size == 0
        )
        self.routed_expert_up_proj = Kimi3LatentProjection(
            self.routed_hidden,
            config.hidden_size,
            params_dtype=None,
            prefix=add_prefix("routed_expert_up_proj", prefix),
            solution="auto",
            shard_group=tp.tp_group if self._shard_up_projection else None,
            column_group=None,
            shard_rank=tp.tp_rank,
            shard_size=tp.tp_size,
            multicast_down=None,
        )
        self.routed_expert_norm = (
            RMSNorm(self.routed_hidden, eps=config.rms_norm_eps)
            if config.latent_moe_use_norm
            else None
        )
        self.shared_experts = KimiLinearMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
            tp_rank=tp.tp_rank,
            tp_size=tp.tp_size,
            tp_group=tp.tp_group,
            shared_parallel=None,
            quant_config=quant_config,
            prefix=add_prefix("shared_experts", prefix),
            reduce_results=False,
            is_shared_expert=True,
            activation_situ_beta=situ_beta,
            activation_situ_linear_beta=situ_linear_beta,
        )

    def pack_input_projection_weights(self) -> None:
        """Keep routed and shared producers separate."""

    def _pair_exchange(self):
        # One exchange (GIN communicator + window) for every MoE layer: layers
        # run one after another and each swap completes before the next.
        # Built at the first forward: the GIN window is collective over the world.
        global _SHARED_EXCHANGE
        if self._exchange is None and _SHARED_EXCHANGE is not None:
            self._exchange = _SHARED_EXCHANGE
        if self._exchange is None:
            # petit_hier imports Iris under the tokenspeed_triton redirect.
            from petit_hier import PairExchange, RcclGinProvider

            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager as pg_manager,
            )

            tp = self.mapping.attn
            self._exchange = PairExchange(
                RcclGinProvider(),
                pg_manager.get_process_group("gloo", tp.tp_group),
                max_rows=_MAX_SLICE_ROWS,
                dispatch_row_bytes=self.routed_hidden * 2 + 2 * self.top_k * 4,
                combine_row_bytes=self.routed_hidden * 2,
            )
            _SHARED_EXCHANGE = self._exchange
        return self._exchange

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext | None,
        *,
        prefix_is_sharded: bool,
        next_mix=None,
        attention_gather=None,
        attention_mix=None,
    ) -> torch.Tensor:
        del num_global_tokens, attention_mix
        if prefix_is_sharded or next_mix is not None or attention_gather is not None:
            raise ValueError("two-node K3 MoE requires a replicated residual")
        if ctx is None:
            raise RuntimeError("two-node K3 MoE requires a ForwardContext")
        tp = self.mapping.attn
        T = hidden_states.shape[0]
        # Largest TP slice over both replicas (CommManager.scattered_num_tokens):
        # both ranks of a pair swap this many rows.
        S = int(max_num_tokens_per_gpu)
        if S > _MAX_SLICE_ROWS:
            raise ValueError(f"slice of {S} rows exceeds {_MAX_SLICE_ROWS}")
        counts = scatter_count(T, tp.tp_size)
        start, mine = sum(counts[: tp.tp_rank]), counts[tp.tp_rank]
        L, K = self.routed_hidden, self.top_k
        logits = (
            self.gate(hidden_states)
            if T
            else hidden_states.new_empty((0, self.num_experts), dtype=torch.float32)
        )
        routing = (
            self.topk(hidden_states, logits, output_format=TopKOutputFormat.STANDARD)
            if T
            else self.topk.empty_topk_output(
                hidden_states.device, hidden_states=hidden_states, router_logits=logits
            )
        )
        ids = routing.topk_ids.to(torch.int32)
        weights = routing.topk_weights.to(torch.float32)
        if T and ctx.forward_mode.is_decode_or_idle():
            live = ctx.attn_backend.decode_window_locations()[:T] > 0
            ids = ids.masked_fill(~live[:, None], -1)
            weights = weights.masked_fill(~live[:, None], 0)
        routed_in = (
            self.routed_expert_down_proj(hidden_states)[0]
            if T
            else hidden_states.new_empty((0, L))
        )
        shared = None

        def compute_shared() -> None:
            nonlocal shared
            shared = self.shared_experts(hidden_states, down_out=None)

        compute_shared()
        ex = self._pair_exchange()
        if S:
            # Pack this rank's slice: [latent bf16 | ids int32 | weights f32].
            send = ex.dsend[:S]
            send[:mine, : L * 2].view(torch.bfloat16).copy_(
                routed_in[start : start + mine]
            )
            send[:, L * 2 : L * 2 + K * 4].view(torch.int32)[:mine].copy_(
                ids[start : start + mine]
            )
            send[:mine, L * 2 + K * 4 :].view(torch.float32).copy_(
                weights[start : start + mine]
            )
            # Padding rows: no expert (-1), zero weight.
            send[mine:, L * 2 : L * 2 + K * 4].view(torch.int32).fill_(-1)
            send[mine:, L * 2 + K * 4 :].view(torch.float32).zero_()
            recv = ex.dispatch(S)
            # Every rank of the node gets the pair node's tokens.
            peer_rows = token_all_gather_rows(recv, tp.tp_group, [S] * tp.tp_size)
            peer_x = peer_rows[:, : L * 2].contiguous().view(torch.bfloat16)
            peer_ids = (
                peer_rows[:, L * 2 : L * 2 + K * 4].contiguous().view(torch.int32)
            )
            peer_w = peer_rows[:, L * 2 + K * 4 :].contiguous().view(torch.float32)
            x_all = torch.cat([routed_in, peer_x])
            ids_all = torch.cat([ids, peer_ids])
            w_all = torch.cat([weights, peer_w])
        else:
            x_all, ids_all, w_all = routed_in, ids, weights
        routing_all = routing._replace(topk_weights=w_all, topk_ids=ids_all)
        out_all = self.experts(
            hidden_states=x_all,
            topk_output=routing_all,
            num_global_tokens=x_all.shape[0],
            max_num_tokens_per_gpu=x_all.shape[0],
            do_finalize=True,
        )
        own = out_all[:T]
        if S:
            # Node-summed partials for the pair node's tokens, one slice per rank.
            back = token_reduce_scatter(
                out_all[T:].contiguous(), tp.tp_group, [S] * tp.tp_size
            )
            ex.csend[:S].view(torch.bfloat16).view(S, L).copy_(back)
            got = ex.combine(S).view(torch.bfloat16).view(S, L)
            own = own.clone()
            own[start : start + mine] += got[:mine]
        if T == 0:
            return prefix_sum
        routed = all_reduce(own.contiguous(), tp.tp_group)
        if self.routed_expert_norm is not None:
            routed = self.routed_expert_norm(routed)
        if self._shard_up_projection:
            projected = self.routed_expert_up_proj.project_shard(routed)
            column, width = self.routed_expert_up_proj.shard_slice
            shared[:, column : column + width].add_(projected)
        else:
            projected, _ = self.routed_expert_up_proj(routed)
            if tp.tp_rank == 0:
                shared.add_(projected)
        shared = all_reduce(shared, tp.tp_group)
        return prefix_sum + shared
