# Copyright (c) 2026 LightSeek Foundation
#
# SPDX-License-Identifier: MIT
"""AMD-native EP dispatch/combine backed by MORI (github.com/ROCm/mori).

MORI-EP provides production dispatch/combine kernels for MoE expert parallelism on
AMD GPUs (IntraNode over XGMI, InterNode over RDMA) — real all-to-all, unlike the
masked-replicate ``all2all_backend=none`` fallback. This module adapts MORI's
``EpDispatchCombineOp`` to tokenspeed's MoE expert path.

Selected when ``--all2all-backend mori`` (``All2AllBackend.MORI``) + expert parallel.

Validated usage (mirrors MORI's own intranode test, zero-copy path):
    d_out, d_w, d_s, d_idx, recv = op.dispatch(x, topk_w, scales, topk_ids)
    cbuf = op.get_registered_combine_input_buffer(dtype)   # expert output written here
    cbuf[:n].copy_(expert_out[:n])                          # (identity example)
    out, _ = op.combine(d_out, d_w, topk_ids, call_reset=False)  # original topk_ids, dispatch weights

Requires: `pip install -e ~/mori` (built ENABLE_STANDARD_MOE_ADAPT=ON) and
`ROCM_PATH` pointing at a hipcc that supports `--genco` for the target arch.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_MORI_AVAILABLE = None
_SHMEM_INITED = False


def mori_available() -> bool:
    """True if the `mori` package (with EP ops) can be imported."""
    global _MORI_AVAILABLE
    if _MORI_AVAILABLE is None:
        try:
            import mori  # noqa: F401
            import mori.ops  # noqa: F401
            import mori.shmem  # noqa: F401

            _MORI_AVAILABLE = True
        except Exception as e:  # pragma: no cover
            logger.warning("MORI not available: %s", e)
            _MORI_AVAILABLE = False
    return _MORI_AVAILABLE


def ensure_mori_shmem(group_name: str = "default", heap_size: str | None = None) -> None:
    """Initialize MORI symmetric memory once, bound to a torch process group.

    torch.distributed must already be initialized. Idempotent per process.
    """
    global _SHMEM_INITED
    if _SHMEM_INITED:
        return
    import mori  # noqa: E402

    if heap_size:
        os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", heap_size)
    assert dist.is_initialized(), "torch.distributed must be initialized before MORI shmem init"
    # Register the (world) process group under `group_name` so MORI can bootstrap
    # its symmetric heap over it.
    torch._C._distributed_c10d._register_process_group(group_name, dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init(group_name)
    _SHMEM_INITED = True


class MoriEpDispatcher:
    """Thin wrapper over ``mori.ops.EpDispatchCombineOp`` for one MoE layer group.

    One instance owns a MORI op configured for a fixed (world_size, hidden, experts,
    capacity) shape. dispatch() routes tokens to expert owners; combine() gathers and
    weight-sums expert outputs back to origin tokens.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        hidden_dim: int,
        num_local_experts: int,
        num_experts_per_token: int,
        max_num_inp_token_per_rank: int,
        data_type: torch.dtype = torch.bfloat16,
        gpu_per_node: int | None = None,
        scale_dim: int = 0,
        kernel_type: str = "intranode",  # "intranode" | "intranode_ll" | "internode" | ...
    ) -> None:
        assert mori_available(), "MORI is not importable; cannot build MoriEpDispatcher"
        import mori

        ensure_mori_shmem()

        kt = {
            "intranode": mori.ops.EpDispatchCombineKernelType.IntraNode,
            "intranode_ll": mori.ops.EpDispatchCombineKernelType.IntraNodeLL,
            "internode": mori.ops.EpDispatchCombineKernelType.InterNode,
            "internode_v1": mori.ops.EpDispatchCombineKernelType.InterNodeV1,
            "internode_v1_ll": mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
            "asyncll": mori.ops.EpDispatchCombineKernelType.AsyncLL,
        }[kernel_type]

        # gpu_per_node must be a power of 2 and divide world_size (MORI assertion).
        # Single-node EP<8: gpu_per_node = world_size.
        if gpu_per_node is None:
            gpu_per_node = world_size if world_size <= 8 else 8

        self.data_type = data_type
        self.num_experts_per_token = num_experts_per_token
        self._cfg = mori.ops.EpDispatchCombineConfig(
            data_type=data_type,
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_dim,
            scale_dim=scale_dim,
            scale_type_size=torch.tensor([], dtype=torch.float8_e4m3fnuz).element_size(),
            max_token_type_size=torch.tensor([], dtype=torch.float32).element_size(),
            max_num_inp_token_per_rank=max_num_inp_token_per_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=num_experts_per_token,
            kernel_type=kt,
            gpu_per_node=gpu_per_node,
            use_external_inp_buf=False,  # zero-copy: expert writes into registered buffer
        )
        self._op = mori.ops.EpDispatchCombineOp(self._cfg)

    def dispatch(
        self,
        hidden_states: torch.Tensor,   # [num_tokens, hidden]
        topk_weights: torch.Tensor,    # [num_tokens, topk]  fp32
        topk_ids: torch.Tensor,        # [num_tokens, topk]  int32 (global expert ids)
        scales: torch.Tensor | None = None,
        block_num: int = 80,
        warp_per_block: int = 16,
    ) -> dict[str, Any]:
        """Route tokens to their experts' owner ranks.

        Returns a handle dict with the received tokens and metadata. The caller runs
        local expert compute writing results into ``combine_input_buffer`` (zero-copy),
        then calls :meth:`combine` with the SAME ``topk_ids``.
        """
        if scales is None:
            scales = torch.empty(
                hidden_states.size(0), 0, dtype=torch.float8_e4m3fnuz, device=hidden_states.device
            )
        d_out, d_w, d_s, d_idx, recv = self._op.dispatch(
            hidden_states, topk_weights, scales, topk_ids.to(torch.int32),
            block_num=block_num, warp_per_block=warp_per_block,
        )
        return {
            "recv_hidden": d_out,        # [recv_capacity, hidden]; first recv_num valid
            "recv_weights": d_w,
            "recv_scales": d_s,
            "recv_indices": d_idx,
            "recv_num": int(recv[0].item()),
            "block_num": block_num,
            "warp_per_block": warp_per_block,
        }

    def combine_input_buffer(self) -> torch.Tensor:
        """The registered buffer the caller writes local expert outputs into (zero-copy)."""
        return self._op.get_registered_combine_input_buffer(self.data_type)

    def combine(
        self,
        recv_hidden: torch.Tensor,     # the dispatch handle's recv_hidden (buffer handle)
        recv_weights: torch.Tensor,    # dispatch-returned weights
        topk_ids: torch.Tensor,        # ORIGINAL topk_ids (same as dispatch input)
        block_num: int = 80,
        warp_per_block: int = 16,
    ) -> torch.Tensor:
        """Gather expert outputs back to origin tokens, weight-summed. Returns [num_tokens, hidden]."""
        out, _ = self._op.combine(
            recv_hidden, recv_weights, topk_ids.to(torch.int32),
            block_num=block_num, warp_per_block=warp_per_block, call_reset=False,
        )
        return out

    def reset(self) -> None:
        self._op.reset()


# TODO(M3/M4): the registered MoE `apply` kernel (mirroring
# ops/moe/flashinfer/cutedsl_deepep_nvfp4.py) wires: dispatch() -> local masked
# grouped-GEMM over recv tokens grouped by local expert (reuse gluon_bf16_moe /
# mxfp4 per the M0 decision; MORI's convert_dispatch_output gives the 3D
# [experts, capacity, hidden] + per-expert counts layout that grouped-GEMM wants)
# -> combine(). Register gated vendors={"amd"}, supports_all_to_all_ep={True},
# priority above triton_mxfp4_ep_precomputed_moe_apply.
