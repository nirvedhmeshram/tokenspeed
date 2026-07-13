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

"""Hybrid linear attention backend for Qwen3.5 GDN models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention import GdnCheckpointLayout, gdn_chunk_prefill
from tokenspeed_kernel.ops.attention.triton.gdn_qkv_split import (
    fused_qkv_split_gdn_prefill,
)
from tokenspeed_kernel.ops.attention.triton.linear.chunk_delta_h import (
    CHUNK_SIZE as FLA_CHUNK_SIZE,
)
from tokenspeed_kernel.ops.attention.triton.linear.index import (
    set_total_chunks_hint,
    set_total_chunks_hint_uniform,
)

from tokenspeed.runtime.configs.paged_cache_spec import LINEAR_ATTENTION
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_forward_ctx,
    scrub_padding_tail,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import (
    AttentionBackend,
    init_backend_cuda_graph_state,
)
from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from tokenspeed.runtime.layers.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating
from tokenspeed.runtime.layers.attention.linear.mamba_state_scatter_triton import (
    fused_mamba_state_copy,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

# Flat KV-cache group id carrying GDN/mamba2 state pages.
_STATE_GROUP_ID = LINEAR_ATTENTION
# M18c state binning publishes k sharded groups ("linear_attention_shard{i}"),
# each with its own block table; page ids are per-shard.
_STATE_SHARD_PREFIX = _STATE_GROUP_ID + "_shard"


def compute_state_page_indices(
    rows: torch.Tensor,
    page_size: int,
    seq_lens_before: torch.Tensor,
    seq_lens_after: torch.Tensor,
    *,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dual-index state pages: in = page of position n-1 (0/null when no
    history), out = page of the step's last position. rows: [bs, max_pages]
    int32 page ids (-1 pad, 0 hole). Within a page in == out (in-place
    evolution); crossing a boundary reads the old page and writes the new
    one; resuming from a prefix hit reads the claimed snapshot page and
    writes the fresh working page.

    Thin single-table wrapper over ``compute_state_page_indices_batched``
    (k == 1). Returns ([bs], [bs]) int32 (state_in, state_out).
    """
    state_in, state_out = compute_state_page_indices_batched(
        rows.unsqueeze(0),
        page_size,
        seq_lens_before,
        seq_lens_after,
        validate=validate,
    )
    return state_in.squeeze(0), state_out.squeeze(0)


def compute_state_page_indices_batched(
    rows: torch.Tensor,
    page_size: int,
    seq_lens_before: torch.Tensor,
    seq_lens_after: torch.Tensor,
    *,
    validate: bool = True,
    out_in: torch.Tensor | None = None,
    out_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched dual-index over k stacked shard tables (the semantics of
    ``compute_state_page_indices`` applied per table). Every shard pages
    the same request positions, so the slot math (div/clamp over
    before/after) is shared and computed once, and both gathers run
    batched over dim 2.

    Args:
        rows: [k, bs, max_pages] page ids (-1 pad, 0 hole); row i is the
            "linear_attention_shard{i}" block table. int32 (required when
            ``out_in``/``out_out`` are passed).
        page_size: State page size P (tokens).
        seq_lens_before / seq_lens_after: [bs] token counts before/after
            this forward, shared across shards.
        validate: Host-syncing guard checks (pad/hole in-page reads, table
            overrun, out-page uniqueness -- enforced per shard).
        out_in / out_out: Optional persistent [k, >= bs] int32 buffers
            (pass both or neither). Page ids are gathered directly into
            their ``[:, :bs]`` slices -- the CUDA-graph replay path's
            zero-intermediate write -- and those slices are returned.

    Returns:
        (state_in, state_out): [k, bs] int32 page ids (the buffer slices
        when ``out_in``/``out_out`` were passed).
    """
    if (out_in is None) != (out_out is None):
        raise ValueError("out_in and out_out must be passed together")
    bs = seq_lens_before.shape[0]
    k = rows.shape[0]
    rows = rows[:, :bs]
    before = seq_lens_before.to(torch.int64)
    after = seq_lens_after.to(torch.int64)
    max_slots = rows.shape[2]

    in_slots = torch.div(before - 1, page_size, rounding_mode="floor").clamp_(min=0)
    out_slots = torch.div(after - 1, page_size, rounding_mode="floor")
    out_slots_safe = out_slots.clamp(min=0, max=max_slots - 1)
    in_index = in_slots.view(1, bs, 1).expand(k, bs, 1)
    out_index = out_slots_safe.view(1, bs, 1).expand(k, bs, 1)

    if out_in is not None:
        state_in = out_in[:, :bs]
        state_out = out_out[:, :bs]
        torch.gather(rows, 2, in_index, out=state_in.unsqueeze(2))
        torch.gather(rows, 2, out_index, out=state_out.unsqueeze(2))
        # before == 0 -> no history -> in page 0 (the null page).
        state_in.masked_fill_((before <= 0).view(1, bs), 0)
    else:
        state_in = rows.gather(2, in_index).squeeze(2)
        state_in = torch.where(before > 0, state_in, torch.zeros_like(state_in))
        state_out = rows.gather(2, out_index).squeeze(2)

    if validate:
        if bool((after <= 0).any()):
            raise ValueError(
                "state paging: seq_lens_after must be >= 1 for every request"
            )
        if bool((out_slots >= max_slots).any()):
            raise ValueError(
                "state paging: out page slot exceeds flat table width "
                f"{max_slots} (page_size={page_size})"
            )
        if bool((state_in[:, before > 0] <= 0).any()):
            raise ValueError(
                "state paging: in page is a pad (-1) or hole (0) for a "
                "request with history; reading it would silently resume "
                f"from the zero state (flat {_STATE_GROUP_ID!r} table)"
            )
        if bool((state_out <= 0).any()):
            raise ValueError(
                "state paging: out page is a pad (-1) or hole (0); the "
                "request's working state page must be present in the flat "
                f"{_STATE_GROUP_ID!r} table"
            )
        # The <= 0 raise above guarantees every state_out entry is positive.
        # Uniqueness holds per shard: each shard table hands out its own
        # page-id space, so cross-shard collisions are meaningless.
        for shard_out in state_out:
            if torch.unique(shard_out).numel() != shard_out.numel():
                raise ValueError("state out pages must be unique per batch")
    if out_in is not None:
        return state_in, state_out
    return state_in.to(torch.int32), state_out.to(torch.int32)


@dataclass
class MambaForwardMetadata:
    query_start_loc: torch.Tensor | None
    mamba_cache_indices: torch.Tensor
    mamba_output_indices: torch.Tensor | None = None
    mamba_req_pool_indices: torch.Tensor | None = None
    extend_prefix_lens: torch.Tensor | None = None
    extend_seq_lens_cpu: torch.Tensor | None = None
    # Pre-computed src/dst indices for extracting Mamba prefix-cache snapshots.
    track_ssm_h_src: torch.Tensor | None = None
    track_ssm_h_src_fla: torch.Tensor | None = None
    track_ssm_h_dst: torch.Tensor | None = None
    track_conv_indices: torch.Tensor | None = None
    track_ssm_final_src: torch.Tensor | None = None
    track_ssm_final_dst: torch.Tensor | None = None
    # Flat path (state shard views): dual in/out page indices per request,
    # one row per shard group: [k, bs] (row i indexes the
    # "linear_attention_shard{i}" block table).
    state_in_pages: torch.Tensor | None = None
    state_out_pages: torch.Tensor | None = None
    # seq_lens_before for the flat path: rows with before == 0 have no
    # history and must be zero-seeded rather than read from the aliased
    # (dirty) null page row 0 (R1, see state_shard_view WARNING).
    state_seq_lens_before: torch.Tensor | None = None
    # Extend-only precomputes (hoisted out of the per-state-layer forward):
    # int64 copies of the page tables for advanced indexing, and the R1
    # fresh mask (before == 0). None on decode/replay metadata.
    state_in_pages_i64: torch.Tensor | None = None
    state_out_pages_i64: torch.Tensor | None = None
    state_fresh_mask: torch.Tensor | None = None
    # Flat prefill boundary harvest (flat counterpart of the radix track_*
    # fields): each extend chunk's LAST interior whole-page boundary state is
    # scattered into that boundary page. Compact over the m tracked requests;
    # per-shard page/sel pairs are pre-filtered of holes/pads and the
    # forbidden null row 0.
    flat_track_mask: torch.Tensor | None = None  # [bs] bool
    flat_track_lens: torch.Tensor | None = None  # [bs] chunk-relative bounds
    flat_track_ssm_src: torch.Tensor | None = None  # [m] FLASHINFER grid
    flat_track_ssm_src_fla: torch.Tensor | None = None  # [m] FLA 64 grid
    flat_track_pages: tuple[torch.Tensor, ...] | None = None  # k x [m_s] i64
    flat_track_sel: tuple[torch.Tensor, ...] | None = None  # k x [m_s] into m
    # Lazily filled on the first state layer's forward (needs conv_state_len
    # from the bound views); shared by the rest of the step's layers.
    flat_track_conv_indices: torch.Tensor | None = None  # [m, conv_len]


class LayerMappedKVPool:
    """Wraps a KV pool to map global layer IDs to internal pool indices.

    For hybrid models, only full attention layers have KV cache. This wrapper
    translates global layer IDs (e.g., 3, 7, 11) to pool indices (0, 1, 2).
    """

    def __init__(
        self, inner_pool: BaseTokenToKVPool, full_attention_layer_ids: list[int]
    ):
        self.inner = inner_pool
        self.layer_ids = list(full_attention_layer_ids)
        self.layer_map = {
            global_id: pool_idx
            for pool_idx, global_id in enumerate(full_attention_layer_ids)
        }
        # Expose page_size from inner pool for the scheduler
        self.page_size = getattr(inner_pool, "page_size", 1)

    def _map(self, layer_id: int) -> int:
        return self.layer_map.get(layer_id, layer_id)

    def set_kv_buffer(
        self,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor | None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ):
        orig = layer.layer_id
        layer.layer_id = self._map(orig)
        self.inner.set_kv_buffer(layer, out_cache_loc, k, v, k_scale, v_scale)
        layer.layer_id = orig

    def get_kv_buffer(self, layer_id: int):
        return self.inner.get_kv_buffer(self._map(layer_id))

    def get_key_buffer(self, layer_id: int):
        return self.inner.get_key_buffer(self._map(layer_id))

    def get_value_buffer(self, layer_id: int):
        return self.inner.get_value_buffer(self._map(layer_id))

    def __getattr__(self, name):
        return getattr(self.inner, name)


class SimpleMambaPool:
    """Mamba state pool indexed by scheduler-assigned cache slots."""

    def __init__(
        self,
        size: int,
        num_mamba_layers: int,
        conv_state_shape: tuple,
        temporal_state_shape: tuple,
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        mamba_layer_ids: list[int],
        device: str,
        page_size: int = 1,
        speculative_num_draft_tokens: int = 0,
        max_req_pool_size: int = 0,
    ):
        self.size = size
        self.device = device
        self.mamba_layer_ids = list(mamba_layer_ids)
        self.page_size = page_size
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}
        self.is_kda_cache = False
        self.max_req_pool_size = max_req_pool_size

        # Base slots (working + checkpoint) are allocated by C++ scheduler.
        # Python-only draft rows live after the scheduler-owned range and are
        # addressed by normal row indices in the same tensors.
        self.base_size = size
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.current_input_size = (
            max_req_pool_size + 1 if max_req_pool_size > 0 else size
        )
        self.draft_slots_per_req = max(0, speculative_num_draft_tokens - 1)
        self.draft_base = size
        self.draft_total_slots = self.current_input_size * self.draft_slots_per_req
        total_size = size + self.draft_total_slots
        self.total_size = total_size

        # Allocate conv state: (num_mamba_layers, total_size, conv_dim, state_len)
        self.conv_state = torch.zeros(
            num_mamba_layers,
            total_size,
            *conv_state_shape,
            dtype=conv_dtype,
            device=device,
        )
        # Allocate temporal/SSM state: (num_mamba_layers, total_size, heads, key_dim, val_dim)
        self.ssm_state = torch.zeros(
            num_mamba_layers,
            total_size,
            *temporal_state_shape,
            dtype=ssm_dtype,
            device=device,
        )

        self.mamba_cache = (self.conv_state, self.ssm_state)
        self.layer_transfer_counter = None

        self.current_input_indices = torch.full(
            (self.current_input_size,), -1, dtype=torch.int32, device=device
        )

    def get_mamba_indices(self, mamba_pool_indices: torch.Tensor) -> torch.Tensor:
        """Return mamba cache indices directly (allocated by C++ scheduler)."""
        return mamba_pool_indices.to(torch.int32)

    @staticmethod
    @torch.compile(dynamic=True)
    def _build_mtp_output_indices_kernel(
        output_indices: torch.Tensor,
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        draft_base: int,
        draft_slots_per_req: int,
        draft_token_num: int,
    ) -> None:
        """Fused fill of MTP target-verify output index table.

        Inductor fuses the working-column write and the draft-grid write into
        as few elementwise kernels as possible.  The host-side early returns
        (draft_token_num<=0, ``out is None``) are kept in the wrapper.
        """
        bs = working_indices.shape[0]
        working = working_indices.to(torch.int32)
        valid = working >= 0
        output_indices[:, 0] = torch.where(valid, working, -1)

        if draft_token_num > 1 and draft_slots_per_req > 0:
            req = req_pool_indices[:bs].to(torch.int32)
            steps = torch.arange(
                draft_token_num - 1, dtype=torch.int32, device=working.device
            )
            draft = draft_base + req[:, None] * draft_slots_per_req + steps[None, :]
            output_indices[:, 1:] = torch.where(
                valid[:, None] & (req >= 0)[:, None],
                draft,
                -1,
            )

    def get_mtp_output_indices(
        self,
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        draft_token_num: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build per-request target-verify outputs: [working, draft0, ...]."""
        bs = working_indices.shape[0]
        if out is not None:
            output_indices = out
            output_indices.fill_(-1)
        else:
            output_indices = torch.full(
                (bs, draft_token_num),
                -1,
                dtype=torch.int32,
                device=working_indices.device,
            )
        if draft_token_num <= 0:
            return output_indices

        self._build_mtp_output_indices_kernel(
            output_indices,
            req_pool_indices,
            working_indices,
            self.draft_base,
            self.draft_slots_per_req,
            draft_token_num,
        )
        return output_indices

    @staticmethod
    @torch.compile(dynamic=True)
    def _get_current_input_indices_kernel(
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        current_input_indices_buf: torch.Tensor,
        current_input_size: int,
    ) -> torch.Tensor:
        """Fused gather + masked-where for the no-COW path."""
        n = working_indices.shape[0]
        req = req_pool_indices[:n].to(torch.int32)
        working = working_indices.to(torch.int32)
        valid = (working >= 0) & (req >= 0)
        safe = req.clamp(0, current_input_size - 1).to(torch.int64)
        stored = current_input_indices_buf[safe]
        current = torch.where(valid & (stored >= 0), stored, working)
        current = torch.where(valid, current, torch.full_like(current, -1))
        return current

    @staticmethod
    @torch.compile(dynamic=True)
    def _get_current_input_indices_with_cow_kernel(
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        cow_src_indices: torch.Tensor,
        current_input_indices_buf: torch.Tensor,
        current_input_size: int,
    ) -> torch.Tensor:
        """Fused gather + masked-where for the COW path."""
        n = working_indices.shape[0]
        req = req_pool_indices[:n].to(torch.int32)
        working = working_indices.to(torch.int32)
        cow = cow_src_indices[:n].to(torch.int32)
        valid = (working >= 0) & (req >= 0)
        safe = req.clamp(0, current_input_size - 1).to(torch.int64)
        stored = current_input_indices_buf[safe]
        current = torch.where(valid & (stored >= 0), stored, working)
        current = torch.where(valid, current, torch.full_like(current, -1))
        current = torch.where(
            (cow >= 0) & valid & (current == working),
            cow,
            current,
        )
        return current

    def get_current_input_indices(
        self,
        req_pool_indices: torch.Tensor,
        working_indices: torch.Tensor,
        cow_src_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the row each request should read at the start of target verify."""
        if cow_src_indices is None:
            return self._get_current_input_indices_kernel(
                req_pool_indices,
                working_indices,
                self.current_input_indices,
                self.current_input_size,
            )
        return self._get_current_input_indices_with_cow_kernel(
            req_pool_indices,
            working_indices,
            cow_src_indices,
            self.current_input_indices,
            self.current_input_size,
        )

    def reset_current_inputs(
        self, req_pool_indices: torch.Tensor, working_indices: torch.Tensor
    ) -> None:
        """Mark freshly allocated/reused scheduler slots as canonical."""
        req_pool_indices = req_pool_indices[: working_indices.shape[0]].to(torch.int32)
        working_indices = working_indices.to(torch.int32)
        self.current_input_indices[req_pool_indices.long()] = working_indices

    @staticmethod
    @torch.compile(dynamic=True)
    def _update_current_inputs_after_verify_kernel(
        req_pool_indices: torch.Tensor,
        output_indices: torch.Tensor,
        accepted_lengths: torch.Tensor,
        current_input_indices: torch.Tensor,
        max_col: int,
    ) -> None:
        """Fused gather-scatter for the after-verify input pointer update.

        Inductor fuses clamp/arange/sub/dtype-convert into a single elementwise
        kernel; the gather and the in-place scatter on ``current_input_indices``
        each remain a single kernel.  All tensors stay on GPU; no host sync.
        """
        n = accepted_lengths.shape[0]
        req = req_pool_indices[:n].to(torch.int64)
        idx = (accepted_lengths.clamp(min=1, max=max_col) - 1).to(torch.int64)
        rows = torch.arange(n, device=accepted_lengths.device, dtype=torch.int64)
        selected = output_indices[rows, idx].to(torch.int32)
        current_input_indices[req] = selected

    def update_current_inputs_after_verify(
        self,
        req_pool_indices: torch.Tensor,
        output_indices: torch.Tensor,
        accepted_lengths: torch.Tensor,
    ) -> None:
        if output_indices is None or output_indices.numel() == 0:
            return
        self._update_current_inputs_after_verify_kernel(
            req_pool_indices,
            output_indices,
            accepted_lengths,
            self.current_input_indices,
            output_indices.shape[1],
        )

    def register_layer_transfer_counter(self, layer_transfer_counter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_mamba_params(self, layer_id: int):
        """Return per-layer cache slices."""
        internal_idx = self.mamba_map[layer_id]
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(internal_idx)
        return [self.mamba_cache[i][internal_idx] for i in range(len(self.mamba_cache))]

    def get_mamba_params_all_layers(self):
        """Return all layers for all cache components."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(self.conv_state.shape[0] - 1)
        return [self.mamba_cache[i] for i in range(len(self.mamba_cache))]

    def get_contiguous_buf_infos(self):
        """Return per-layer mamba cache buffers for disaggregated transfer."""
        data_ptrs = []
        data_lens = []
        item_lens = []
        for cache in self.mamba_cache:
            for layer_id in range(cache.shape[0]):
                layer_cache = cache[layer_id]
                data_ptrs.append(layer_cache.data_ptr())
                data_lens.append(layer_cache.nbytes)
                item_lens.append(layer_cache[0].nbytes)
        return data_ptrs, data_lens, item_lens

    def get_contiguous_buf_unit_lens(self):
        unit_lens = []
        for cache in self.mamba_cache:
            for layer_id in range(cache.shape[0]):
                layer_cache = cache[layer_id]
                unit_lens.append(layer_cache[0, 0].nbytes)
        return unit_lens

    def get_contiguous_buf_layer_ids(self):
        """Return global layer ids aligned with get_contiguous_buf_infos()."""
        return self.mamba_layer_ids * len(self.mamba_cache)


class MambaAttnBackend(AttentionBackend):
    """Attention backend for Mamba/GDN linear attention layers."""

    def __init__(self, config: BaseAttnConfig):
        super().__init__(config)
        self.pad_slot_id = -1
        self.forward_metadata: MambaForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.output_indices_list = []
        self.speculative_num_draft_tokens = getattr(
            config, "speculative_num_draft_tokens", 0
        )
        self.pool: SimpleMambaPool = None
        # Flat path (state shard views over the KV slabs, dual in/out page
        # indexing per shard group).
        self.kv_pool = None
        self.flat_state_active = False
        self._num_state_shards = 0
        self._shard_group_ids: tuple[str, ...] = ()
        self._flat_state_page_size = 1
        self.flat_state_in_list: list[torch.Tensor] = []
        self.flat_state_out_list: list[torch.Tensor] = []
        # Layers whose head groups already passed the one-time GQA ratio
        # alignment check (lazy: the q/v head counts only arrive with the
        # first forward's kwargs).
        self._gqa_checked_layers: set[int] = set()

    def set_pool(self, pool: SimpleMambaPool):
        self.pool = pool

    def set_kv_pool(self, kv_pool) -> None:
        """Bind the (layer-mapped) KV pool. Flat state paging turns on iff the
        pool's StateShardView is active AND it publishes the k sharded state
        cache groups — publication (paged_cache_spec.
        publish_paged_cache_groups) is the upstream signal that flat block
        tables will actually be delivered (radix ext / spec decode never
        publish)."""
        self.kv_pool = kv_pool
        specs = getattr(kv_pool, "paged_cache_group_specs", ())
        num_shards = sum(
            1 for spec in specs if str(spec.group_id).startswith(_STATE_SHARD_PREFIX)
        )
        view = getattr(kv_pool, "state_shard_view", None)
        self.flat_state_active = (
            bool(view is not None and view.is_active) and num_shards > 0
        )
        self._num_state_shards = num_shards if self.flat_state_active else 0
        # Prebuilt once per pool binding so the per-step metadata paths
        # never re-run the k f-strings.
        self._shard_group_ids = tuple(
            f"{_STATE_SHARD_PREFIX}{i}" for i in range(self._num_state_shards)
        )
        self._flat_state_page_size = int(getattr(kv_pool, "page_size", 1))

    def _stack_shard_rows(self, bs: int, kwargs: dict) -> torch.Tensor:
        """[k, bs, W] stack of the k shard block tables (a view when k == 1);
        fails loud on a missing shard table."""
        flat_tables = kwargs.get("flat_block_tables")
        shard_ids = self._shard_group_ids
        missing = [
            gid for gid in shard_ids if not flat_tables or gid not in flat_tables
        ]
        if missing:
            raise RuntimeError(
                "MambaAttnBackend: flat state paging is active (pool "
                f"publishes {self._num_state_shards} "
                f"{_STATE_SHARD_PREFIX}* groups) but flat_block_tables is "
                f"missing {missing!r} "
                f"(got {sorted(flat_tables) if flat_tables else flat_tables!r})"
            )
        if len(shard_ids) == 1:
            return flat_tables[shard_ids[0]][:bs].unsqueeze(0)
        return torch.stack([flat_tables[gid][:bs] for gid in shard_ids], dim=0)

    def _flat_state_pages(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        kwargs: dict,
        *,
        validate: bool | None = None,
        out_in: torch.Tensor | None = None,
        out_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(state_in, state_out, before) for this forward: [k, bs] page ids
        (row i from the "linear_attention_shard{i}" block table; every shard
        pages the same request positions, so before/after are shared) plus
        the [bs] seq_lens_before used for R1 zero-seeding. seq_lens counts
        the tokens computed AFTER this forward (decode: q_len 1; extend:
        prefix + chunk).

        validate: explicit True/False wins; None (the hot-path default)
        validates only under TOKENSPEED_FLAT_DEBUG=1 (the checks host-sync).
        out_in/out_out: optional persistent [k, >= bs] buffers gathered
        into directly (the CUDA-graph replay path), see
        ``compute_state_page_indices_batched``.
        """
        if validate is None:
            validate = os.environ.get("TOKENSPEED_FLAT_DEBUG") == "1"
        after = seq_lens[:bs]
        if forward_mode.is_decode_or_idle():
            before = after - 1
        else:
            extend_prefix_lens = kwargs.get("extend_prefix_lens")
            if extend_prefix_lens is not None:
                before = extend_prefix_lens[:bs].to(
                    device=after.device, dtype=after.dtype
                )
            else:
                before = torch.zeros_like(after)
        # One [k, bs, W] stack (a view when k == 1), then a single batched
        # dual-index over all shards -- the per-request slot math runs once
        # instead of k times.
        rows = self._stack_shard_rows(bs, kwargs)
        state_in, state_out = compute_state_page_indices_batched(
            rows,
            self._flat_state_page_size,
            before,
            after,
            validate=validate,
            out_in=out_in,
            out_out=out_out,
        )
        return state_in, state_out, before

    def reset_current_inputs(
        self, req_pool_indices: torch.Tensor, working_indices: torch.Tensor
    ):
        if self.pool is not None:
            self.pool.reset_current_inputs(req_pool_indices, working_indices)

    def init_forward_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = ForwardMode.DECODE,
        **kwargs,
    ):
        mamba_pool_indices = kwargs.get("mamba_pool_indices")
        if self.pool is None:
            # Poolless flat mode: states are addressed via state_in/out_pages;
            # every consumer of mamba_cache_indices is pool/radix-gated.
            mamba_cache_indices = None
        elif mamba_pool_indices is not None:
            mamba_cache_indices = self.pool.get_mamba_indices(mamba_pool_indices[:bs])
        else:
            mamba_cache_indices = self.pool.get_mamba_indices(req_pool_indices[:bs])

        is_target_verify = (
            forward_mode.is_decode_or_idle()
            and not self.is_draft
            and self.spec_num_tokens > 1
        )
        is_draft_extend = (
            forward_mode.is_decode_or_idle()
            and self.is_draft
            and self.spec_num_tokens > 1
        )

        mamba_output_indices = None
        extend_seq_lens_cpu = None
        if is_target_verify:
            draft_token_num = int(
                kwargs.get("tokens_per_req", self.speculative_num_draft_tokens)
            )
            cow_src_indices = kwargs.get("mamba_cow_src_indices")
            mamba_input_indices = self.pool.get_current_input_indices(
                req_pool_indices[:bs], mamba_cache_indices, cow_src_indices
            )
            mamba_output_indices = self.pool.get_mtp_output_indices(
                req_pool_indices[:bs],
                mamba_cache_indices,
                draft_token_num,
            )
            mamba_cache_indices = mamba_input_indices

        if forward_mode.is_decode_or_idle() and self.spec_num_tokens == 1:
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
        elif forward_mode.is_extend_or_mixed() or is_target_verify or is_draft_extend:
            if is_target_verify or is_draft_extend:
                tokens_per_req = kwargs.get(
                    "tokens_per_req", self.speculative_num_draft_tokens
                )
                query_start_loc = torch.arange(
                    0,
                    bs * tokens_per_req + 1,
                    step=tokens_per_req,
                    dtype=torch.int32,
                    device=self.device,
                )
                set_total_chunks_hint_uniform(bs, tokens_per_req, query_start_loc)
            else:
                extend_start_loc = kwargs.get("extend_start_loc")
                extend_seq_lens = kwargs.get("extend_seq_lens")
                if extend_start_loc is not None and extend_seq_lens is not None:
                    query_start_loc = torch.empty(
                        (bs + 1,), dtype=torch.int32, device=self.device
                    )
                    query_start_loc[:bs] = extend_start_loc
                    query_start_loc[bs] = extend_start_loc[-1] + extend_seq_lens[-1]
                    extend_seq_lens_cpu = extend_seq_lens[:bs].to(
                        device="cpu", dtype=torch.int32
                    )
                else:
                    extend_prefix_lens = kwargs.get("extend_prefix_lens")
                    if extend_prefix_lens is not None:
                        extend_lens = (seq_lens[:bs] - extend_prefix_lens[:bs]).to(
                            torch.int32
                        )
                    else:
                        # No prefix: all tokens are new
                        extend_lens = seq_lens[:bs].to(torch.int32)
                    query_start_loc = torch.zeros(
                        bs + 1, dtype=torch.int32, device=self.device
                    )
                    torch.cumsum(extend_lens, dim=0, out=query_start_loc[1:])
                    extend_seq_lens_cpu = extend_lens.to(device="cpu")
                set_total_chunks_hint(extend_seq_lens_cpu, query_start_loc)
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        state_in_pages = None
        state_out_pages = None
        state_seq_lens_before = None
        state_in_pages_i64 = None
        state_out_pages_i64 = None
        state_fresh_mask = None
        # Idle/bs==0 forwards carry no requests and never reach the mamba
        # forward (router returns early), so no tables are required.
        if self.flat_state_active and bs > 0 and not forward_mode.is_idle():
            if is_target_verify or is_draft_extend:
                raise RuntimeError(
                    "flat GDN state paging does not support speculative "
                    "decoding; the pool must not publish flat groups under "
                    "spec (TODO(flat+spec))"
                )
            state_in_pages, state_out_pages, state_seq_lens_before = (
                self._flat_state_pages(bs, seq_lens, forward_mode, kwargs)
            )
            if not forward_mode.is_decode_or_idle():
                # Extend consumes int64 advanced indices and the R1 fresh
                # mask once per state layer; precompute them here instead
                # of in every layer's forward.
                state_in_pages_i64 = state_in_pages.to(torch.int64)
                state_out_pages_i64 = state_out_pages.to(torch.int64)
                state_fresh_mask = state_seq_lens_before == 0

        track_ssm_h_src = None
        track_ssm_h_src_fla = None
        track_ssm_h_dst = None
        track_conv_indices = None
        track_ssm_final_src = None
        track_ssm_final_dst = None
        flat_track_mask = None
        flat_track_lens = None
        flat_track_ssm_src = None
        flat_track_ssm_src_fla = None
        flat_track_pages = None
        flat_track_sel = None
        if (
            forward_mode.is_extend_or_mixed() or is_draft_extend
        ) and not is_target_verify:
            if self.flat_state_active:
                # Flat prefix snapshots: harvest the chunk's last interior
                # page-boundary state into that boundary page (radix instead
                # tracks into the SimpleMambaPool row space below).
                flat_track = self._flat_boundary_track_metadata(
                    bs, seq_lens, state_seq_lens_before, kwargs
                )
                if flat_track is not None:
                    (
                        flat_track_mask,
                        flat_track_lens,
                        flat_track_ssm_src,
                        flat_track_ssm_src_fla,
                        flat_track_pages,
                        flat_track_sel,
                    ) = flat_track
            else:
                extend_prefix_lens_kw = kwargs.get("extend_prefix_lens")
                mamba_track_pool_indices = kwargs.get("mamba_track_pool_indices")
                if (
                    extend_prefix_lens_kw is not None
                    and mamba_track_pool_indices is not None
                ):
                    prefix = extend_prefix_lens_kw[:bs].to(
                        dtype=torch.int32, device=self.device
                    )
                    track_indices = mamba_track_pool_indices[:bs].to(
                        dtype=torch.int32, device=self.device
                    )
                    extend_lens = (seq_lens[:bs] - prefix).to(torch.int32)
                    checkpoint_mask = (track_indices >= 0) & (mamba_cache_indices >= 0)

                    page_size = getattr(self.pool, "page_size", 1)
                    final_lens = prefix + extend_lens
                    last_inserted_lens = (final_lens // page_size) * page_size
                    track_lens = last_inserted_lens - prefix
                    track_inside = (
                        checkpoint_mask & (track_lens > 0) & (track_lens < extend_lens)
                    )
                    track_mask = track_inside & ((track_lens % FLA_CHUNK_SIZE) == 0)
                    # C++ attaches the checkpoint slot to the last KV page
                    # inserted for this chunk. When a chunk has an intermediate
                    # branch and ends exactly on a page boundary, the final
                    # state must win.
                    final_mask = (
                        checkpoint_mask
                        & (final_lens >= page_size)
                        & ((final_lens % page_size) == 0)
                    )
                    if final_mask.any():
                        track_ssm_final_src = mamba_cache_indices[final_mask]
                        track_ssm_final_dst = track_indices[final_mask]

                    if track_mask.any():
                        (
                            track_ssm_h_src,
                            track_ssm_h_src_fla,
                            track_ssm_h_dst,
                        ) = self._compute_track_ssm_indices(
                            track_lens,
                            track_mask,
                            track_indices,
                            seq_lens[:bs] - prefix,  # extend_seq_lens
                        )
                        track_conv_indices = self._compute_track_conv_indices(
                            query_start_loc,
                            track_lens,
                            track_mask,
                        )

        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            mamba_output_indices=mamba_output_indices,
            mamba_req_pool_indices=req_pool_indices[:bs],
            extend_prefix_lens=kwargs.get("extend_prefix_lens"),
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_src_fla=track_ssm_h_src_fla,
            track_ssm_h_dst=track_ssm_h_dst,
            track_conv_indices=track_conv_indices,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
            state_in_pages=state_in_pages,
            state_out_pages=state_out_pages,
            state_seq_lens_before=state_seq_lens_before,
            state_in_pages_i64=state_in_pages_i64,
            state_out_pages_i64=state_out_pages_i64,
            state_fresh_mask=state_fresh_mask,
            flat_track_mask=flat_track_mask,
            flat_track_lens=flat_track_lens,
            flat_track_ssm_src=flat_track_ssm_src,
            flat_track_ssm_src_fla=flat_track_ssm_src_fla,
            flat_track_pages=flat_track_pages,
            flat_track_sel=flat_track_sel,
        )

    def _compute_track_conv_indices(
        self,
        query_start_loc: torch.Tensor,
        track_lens: torch.Tensor,
        track_mask: torch.Tensor,
        conv_state_len: int | None = None,
    ):
        """Compute packed input indices for conv windows at tracked boundaries."""
        if conv_state_len is None:
            conv_state_len = self.pool.conv_state.shape[-1]
        lens_m = track_lens[track_mask]
        start = query_start_loc[:-1][track_mask] + lens_m - conv_state_len
        indices = start.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start.dtype,
        )
        return indices.clamp(0, query_start_loc[-1] - 1)

    @staticmethod
    def _compute_boundary_ssm_src(
        track_lens: torch.Tensor,
        track_mask: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        fi_interval: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(src_fi, src_fla) checkpoint indices of the masked boundary states.

        ``fi_interval`` is the FLASHINFER checkpoint grid (the
        ``checkpoint_interval`` the kernel will run with); the FLA ``h``
        workspace is always on the fixed FLA_CHUNK_SIZE grid.
        """
        # flashinfer ckpts[k] = state after interval k = FLA h[k+1]
        num_fi_ckpts = extend_seq_lens // fi_interval
        offset = torch.zeros_like(num_fi_ckpts)
        offset[1:] = torch.cumsum(num_fi_ckpts[:-1], dim=0)
        num_fla_states = (extend_seq_lens - 1) // FLA_CHUNK_SIZE + 1
        fla_offset = torch.zeros_like(num_fla_states)
        fla_offset[1:] = torch.cumsum(num_fla_states[:-1], dim=0)

        lens_m = track_lens[track_mask]
        # FLA h[lens//C] = state after the first `lens` tokens (h[0] = h0);
        # track_mask guarantees lens_m is on-grid and >= one interval.
        src_fi = offset[track_mask] + (lens_m // fi_interval - 1)
        src_fla = fla_offset[track_mask] + (lens_m // FLA_CHUNK_SIZE)
        return src_fi, src_fla

    def _compute_track_ssm_indices(
        self,
        track_lens: torch.Tensor,
        track_mask: torch.Tensor,
        mamba_track_indices: torch.Tensor,
        extend_seq_lens: torch.Tensor,
    ):
        """Compute src/dst indices for extracting intermediate SSM states.

        Matching conv windows are gathered separately from packed pre-conv inputs.
        """
        src_fi, src_fla = self._compute_boundary_ssm_src(
            track_lens, track_mask, extend_seq_lens, FLA_CHUNK_SIZE
        )
        return src_fi, src_fla, mamba_track_indices[track_mask]

    def _flat_boundary_track_metadata(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        before: torch.Tensor | None,
        kwargs: dict,
    ):
        """Harvest metadata for the chunk's LAST interior whole-page boundary.

        Returns ``(mask, track_lens, src_fi, src_fla, pages, sel)`` or
        ``None`` when no request harvests. The kernel checkpoints on the
        P-token grid (FLASHINFER) / the fixed 64-token grid (FLA), so the
        boundary must be grid-aligned: P % FLA_CHUNK_SIZE (real pools satisfy
        it by registry construction) and track_lens % P (<=> the chunk starts
        page-aligned).
        """
        P = self._flat_state_page_size
        if before is None or P % FLA_CHUNK_SIZE != 0:
            return None
        final_lens = seq_lens[:bs].to(torch.int64)
        prefix = before.to(torch.int64)
        extend_lens = final_lens - prefix
        last_inserted = (final_lens // P) * P
        track_lens = last_inserted - prefix
        # Strictly interior boundaries only: a chunk ending ON a page
        # boundary already lands that state via the kernel's final-state
        # write (state_out IS the boundary page there).
        mask = (track_lens > 0) & (track_lens < extend_lens) & (track_lens % P == 0)
        # Host sync is fine: extend never runs under CUDA-graph capture/replay
        # (the radix branch below relies on the same property).
        if not bool(mask.any()):
            return None
        src_fi, src_fla = self._compute_boundary_ssm_src(
            track_lens, mask, extend_lens, P
        )
        rows = self._stack_shard_rows(bs, kwargs)
        slot = (last_inserted // P - 1).clamp(min=0)
        pages_all = rows.gather(
            2, slot.view(1, bs, 1).expand(rows.shape[0], bs, 1)
        ).squeeze(2)
        pages_m = pages_all[:, mask]
        pages, sel = [], []
        for shard_pages in pages_m:
            # Boundary page is a hole (0: never write the aliased null row)
            # or a pad (-1): skip that request's harvest on this shard.
            keep = (shard_pages > 0).nonzero(as_tuple=False).squeeze(1)
            sel.append(keep)
            pages.append(shard_pages[keep].to(torch.int64))
        if all(s.numel() == 0 for s in sel):
            # Every boundary page is a hole: don't pay the kernel's
            # checkpoint write for a no-op scatter.
            return None
        return mask, track_lens, src_fi, src_fla, tuple(pages), tuple(sel)

    # ---- CUDA graph state ----

    def init_cuda_graph_state(
        self, max_num_tokens: int, seq_lens_buf: torch.Tensor = None
    ):
        del seq_lens_buf  # mamba doesn't use seq_lens_buf.
        for i in range(max_num_tokens):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
            if self.flat_state_active:
                # [k, bs] persistent tables: one row per state shard group.
                self.flat_state_in_list.append(
                    torch.full(
                        (self._num_state_shards, i + 1),
                        self.pad_slot_id,
                        dtype=torch.int32,
                        device=self.device,
                    )
                )
                self.flat_state_out_list.append(
                    torch.full(
                        (self._num_state_shards, i + 1),
                        self.pad_slot_id,
                        dtype=torch.int32,
                        device=self.device,
                    )
                )
            if self.speculative_num_draft_tokens > 0:
                self.output_indices_list.append(
                    torch.full(
                        (i + 1, self.speculative_num_draft_tokens),
                        self.pad_slot_id,
                        dtype=torch.int32,
                        device=self.device,
                    )
                )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_num_tokens + 1, dtype=torch.int32, device=self.device
        )
        if self.speculative_num_draft_tokens > 0:
            # Need max_num_tokens+1 entries (one per request + sentinel).
            # Each entry is request_index * spec_num_draft_tokens.
            self.cached_cuda_graph_verify_query_start_loc = torch.arange(
                0,
                (max_num_tokens + 1) * self.speculative_num_draft_tokens,
                step=self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        self._qsl_dirty = [False] * max_num_tokens
        self._qsl_last_mode = [None] * max_num_tokens

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        is_target_verify = (
            forward_mode.is_decode_or_idle()
            and not self.is_draft
            and self.spec_num_tokens > 1
        )
        is_draft_extend = (
            forward_mode.is_decode_or_idle()
            and self.is_draft
            and self.spec_num_tokens > 1
        )

        if forward_mode.is_decode_or_idle() and self.spec_num_tokens == 1:
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif is_target_verify or is_draft_extend:
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        mamba_pool_indices = kwargs.get("mamba_pool_indices")
        # Reuse the pre-allocated [bs]-length buffer as mamba_indices so the
        # capture path matches the replay path: zero allocation, single write.
        padded_mamba_indices = self.state_indices_list[bs - 1]
        if self.pool is None:
            # Poolless flat mode: the buffer stays all pad_slot_id (consumers
            # are pool/radix-gated; states travel via state_in/out_pages).
            padded_mamba_indices.fill_(self.pad_slot_id)
        elif mamba_pool_indices is not None:
            padded_mamba_indices[:bs].copy_(
                self.pool.get_mamba_indices(mamba_pool_indices[:bs])
            )
        else:
            padded_mamba_indices[:bs].copy_(
                self.pool.get_mamba_indices(req_pool_indices[:bs])
            )
        mamba_output_indices = None
        if is_target_verify:
            cow_src_indices = kwargs.get("mamba_cow_src_indices")
            mamba_input_indices = self.pool.get_current_input_indices(
                req_pool_indices[:bs], padded_mamba_indices, cow_src_indices
            )
            mamba_output_indices = self.output_indices_list[bs - 1]
            self.pool.get_mtp_output_indices(
                req_pool_indices[:bs],
                padded_mamba_indices,
                self.speculative_num_draft_tokens,
                out=mamba_output_indices,
            )
            padded_mamba_indices.copy_(mamba_input_indices)
        state_in_pages = None
        state_out_pages = None
        if self.flat_state_active:
            # Real tables only arrive at replay; capture binds the persistent
            # buffers (all pad_slot_id: kernels skip reads/writes at capture,
            # so state slab rows are never dirtied by the capture pass).
            if is_target_verify or is_draft_extend:
                raise RuntimeError(
                    "flat GDN state paging: CUDA-graph capture supports "
                    "plain decode only (flat+spec unsupported)"
                )
            flat_ids = kwargs.get("flat_cache_group_ids", ())
            missing = [gid for gid in self._shard_group_ids if gid not in flat_ids]
            if missing:
                raise RuntimeError(
                    "flat GDN state paging: capture is missing the "
                    f"{missing!r} flat cache group ids "
                    f"(got {tuple(flat_ids)!r})"
                )
            state_in_pages = self.flat_state_in_list[bs - 1]
            state_out_pages = self.flat_state_out_list[bs - 1]
            state_in_pages.fill_(self.pad_slot_id)
            state_out_pages.fill_(self.pad_slot_id)
        self._qsl_dirty[bs - 1] = False
        self._qsl_last_mode[bs - 1] = (forward_mode, self.spec_num_tokens > 1)
        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
            mamba_output_indices=mamba_output_indices,
            mamba_req_pool_indices=req_pool_indices[:bs],
            state_in_pages=state_in_pages,
            state_out_pages=state_out_pages,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        num_padding = kwargs.get("num_padding", 0)
        mamba_pool_indices = kwargs.get("mamba_pool_indices")

        real_bs = bs - num_padding
        req_pool_indices = req_pool_indices[:bs]

        # Reuse the pre-allocated [bs]-length buffer as the padded mamba_indices
        # so downstream ops (get_mtp_output_indices, get_current_input_indices)
        # see the full-batch shape with padding rows already set to -1.
        # Zero extra allocations on this hot path.
        padded_mamba_indices = self.state_indices_list[bs - 1]
        if self.pool is None:
            # Poolless flat mode: the buffer stays all pad_slot_id (consumers
            # are pool/radix-gated; states travel via state_in/out_pages).
            padded_mamba_indices.fill_(self.pad_slot_id)
        else:
            if mamba_pool_indices is not None:
                padded_mamba_indices[:real_bs].copy_(
                    self.pool.get_mamba_indices(mamba_pool_indices[:real_bs])
                )
            else:
                padded_mamba_indices[:real_bs].copy_(
                    self.pool.get_mamba_indices(req_pool_indices[:real_bs])
                )
            if num_padding > 0:
                padded_mamba_indices[real_bs:].fill_(-1)

        is_target_verify = (
            forward_mode is not None
            and forward_mode.is_decode_or_idle()
            and not self.is_draft
            and self.spec_num_tokens > 1
        )
        is_draft_extend = (
            forward_mode is not None
            and forward_mode.is_decode_or_idle()
            and self.is_draft
            and self.spec_num_tokens > 1
        )

        mamba_output_indices = None
        if is_target_verify:
            cow_src_indices = kwargs.get("mamba_cow_src_indices")
            mamba_input_indices = self.pool.get_current_input_indices(
                req_pool_indices, padded_mamba_indices, cow_src_indices
            )
            mamba_output_indices = self.output_indices_list[bs - 1]
            self.pool.get_mtp_output_indices(
                req_pool_indices,
                padded_mamba_indices,
                self.speculative_num_draft_tokens,
                out=mamba_output_indices,
            )
            # mamba_input_indices already encodes padding via padded_mamba_indices.
            padded_mamba_indices.copy_(mamba_input_indices)

        if num_padding == 0:
            need_copy = self._qsl_dirty[bs - 1] or self._qsl_last_mode[bs - 1] != (
                forward_mode,
                self.spec_num_tokens > 1,
            )
            if need_copy:
                if forward_mode.is_decode_or_idle() and self.spec_num_tokens == 1:
                    self.query_start_loc_list[bs - 1].copy_(
                        self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                    )
                elif is_target_verify or is_draft_extend:
                    self.query_start_loc_list[bs - 1].copy_(
                        self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                    )
                self._qsl_dirty[bs - 1] = False
                self._qsl_last_mode[bs - 1] = (forward_mode, self.spec_num_tokens > 1)
        else:
            if forward_mode.is_decode_or_idle() and self.spec_num_tokens == 1:
                self.query_start_loc_list[bs - 1][:real_bs].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[:real_bs]
                )
                self.query_start_loc_list[bs - 1][real_bs:].fill_(real_bs)
            elif is_target_verify or is_draft_extend:
                self.query_start_loc_list[bs - 1][:real_bs].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[:real_bs]
                )
                self.query_start_loc_list[bs - 1][real_bs:].fill_(
                    real_bs * self.speculative_num_draft_tokens
                )
            else:
                raise ValueError(f"Invalid forward mode: {forward_mode=}")
            self._qsl_dirty[bs - 1] = True
            self._qsl_last_mode[bs - 1] = (forward_mode, self.spec_num_tokens > 1)

        state_in_pages = None
        state_out_pages = None
        if self.flat_state_active:
            # Decode-only (q_len == 1): before = seq_lens - 1. Padded rows
            # (zero table rows, seq_lens 1) are overwritten with pad_slot_id
            # below, so validation is skipped to avoid a host sync.
            state_in_pages = self.flat_state_in_list[bs - 1]
            state_out_pages = self.flat_state_out_list[bs - 1]
            # Gather straight into the persistent [k, bs] tables: no
            # intermediate stack + copy on this per-step path.
            self._flat_state_pages(
                bs,
                seq_lens,
                forward_mode,
                kwargs,
                validate=False,
                out_in=state_in_pages,
                out_out=state_out_pages,
            )
            if num_padding > 0:
                state_in_pages[:, real_bs:].fill_(self.pad_slot_id)
                state_out_pages[:, real_bs:].fill_(self.pad_slot_id)

        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
            mamba_output_indices=mamba_output_indices,
            mamba_req_pool_indices=req_pool_indices,
            state_in_pages=state_in_pages,
            state_out_pages=state_out_pages,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ---- Forward ----

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # Multi-token decode (target verify or drafter compound) reuses
        # the multi-token kernel path in forward_extend. `q` is None for
        # hybrid linear-attn layers; the token count comes from mixed_qkv.
        q_len_per_req = kwargs["mixed_qkv"].shape[0] // bs if bs > 0 else 1
        if q_len_per_req > 1:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                forward_mode=ForwardMode.DECODE,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        state_in_pages = self.forward_metadata.state_in_pages
        state_out_pages = self.forward_metadata.state_out_pages
        use_flat = state_in_pages is not None

        if use_flat:
            # Dual-index: read the page holding position n-1, write the page
            # holding position n (in == out within a page; a boundary crossing
            # reads the old page and writes the new one). pad_slot_id rows
            # (graph padding) are skipped by both kernels. state_in/out_pages
            # are [k, bs]: row g.shard pages the group's ssm rows, row
            # g.conv_shard the layer's conv rows.
            #
            # in == 0 (the aliased, dirty null row) is unreachable here: a
            # decode is always preceded by an extend (after >= 2 so
            # before >= 1), and graph-padding rows carry pad_slot_id, which
            # both kernels skip. R1 zero-seeding lives on the extend side.
            head_groups = self.kv_pool.get_state_buffers(layer_id)
            # The conv view is one whole-layer tensor repeated across the
            # layer's head groups; run the conv update once off the
            # head_begin == 0 group (get_state_buffers sorts by head_begin).
            conv_group = head_groups[0]
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_group.conv,
                conv_weights,
                bias,
                activation,
                conv_state_indices=state_in_pages[conv_group.conv_shard],
                output_state_indices=state_out_pages[conv_group.conv_shard].view(-1, 1),
            )
        else:
            conv_states, ssm_states, *rest = self.pool.get_mamba_params(layer_id)
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_states,
                conv_weights,
                bias,
                activation,
                conv_state_indices=cache_indices,
                output_state_indices=None,
            )

        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attn_tp_size,
                key_dim // attn_tp_size,
                value_dim // attn_tp_size,
            ],
            dim=-1,
        )
        seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        query = query.view(1, seq_len, num_heads, head_k_dim)
        key = key.view(1, seq_len, num_heads, head_k_dim)
        value = value.view(1, seq_len, value.shape[1] // head_v_dim, head_v_dim)

        if use_flat:
            # One kernel launch per ssm head group: q/k/v/a/b are sliced to
            # the group's value heads (q/k by the GQA ratio), the group's
            # strided ssm view is the h0 source (row addressing is
            # h0_row_stride-parameterized in the kernel), and pages come from
            # the group's own shard table.
            num_v_heads = value.shape[2]
            ratio = num_v_heads // num_heads
            if layer_id not in self._gqa_checked_layers:
                # One-time lazy gate per layer: the q/v head counts only
                # arrive with the first forward's kwargs (set_kv_pool sees
                # neither the per-layer head dims nor the bound head
                # groups), so the ratio alignment can't be hoisted there.
                # The head-group layout itself (ascending, contiguous,
                # head_begin 0 first) is asserted once in
                # StateShardView.bind.
                for g in head_groups:
                    hb, he = g.head_begin, g.head_begin + g.num_heads
                    if hb % ratio or g.num_heads % ratio:
                        raise RuntimeError(
                            "flat GDN state paging: ssm head group "
                            f"[{hb}, {he}) is not aligned to the GQA ratio "
                            f"{ratio} (HV={num_v_heads}, H={num_heads})"
                        )
                self._gqa_checked_layers.add(layer_id)
            outs = []
            for g in head_groups:
                hb, he = g.head_begin, g.head_begin + g.num_heads
                outs.append(
                    fused_sigmoid_gating_delta_rule_update(
                        A_log=A_log[hb:he],
                        dt_bias=dt_bias[hb:he],
                        q=query[:, :, hb // ratio : he // ratio],
                        k=key[:, :, hb // ratio : he // ratio],
                        v=value[:, :, hb:he],
                        a=a[..., hb:he],
                        b=b[..., hb:he],
                        initial_state_source=g.ssm,
                        initial_state_indices=state_in_pages[g.shard],
                        cu_seqlens=query_start_loc,
                        use_qk_l2norm_in_kernel=True,
                        softplus_beta=1.0,
                        softplus_threshold=20.0,
                        # Flat: don't write back to the (possibly shared) in
                        # page; the post-step state lands on the out page.
                        disable_state_update=True,
                        output_state_indices=state_out_pages[g.shard],
                    )
                )
            core_attn_out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        else:
            core_attn_out = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                dt_bias=dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                cu_seqlens=query_start_loc,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                disable_state_update=False,
                output_state_indices=None,
            )
        return core_attn_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        forward_mode: ForwardMode,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]
        seq_len = kwargs["seq_len"]

        # `q` is None for hybrid linear-attn layers; the token count comes
        # from seq_len carried in kwargs.
        q_len_per_req = seq_len // bs if bs > 0 else 1
        is_target_verify = (
            forward_mode is not None
            and forward_mode.is_decode_or_idle()
            and not self.is_draft
            and q_len_per_req > 1
        )

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        if is_target_verify:
            draft_token_num = kwargs.get(
                "draft_token_num", self.speculative_num_draft_tokens
            )
            conv_states, ssm_states = self.pool.get_mamba_params(layer_id)
            output_indices = self.forward_metadata.mamba_output_indices

            batch_size = seq_len // draft_token_num
            # shouldn't use contiguous here, because causal_conv1d_update
            # support input non-contiguous
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                conv_weights,
                bias,
                activation,
                conv_state_indices=cache_indices[:batch_size],
                output_state_indices=output_indices[:batch_size],
            )
            # needn't contiguous here.
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            state_in_pages = self.forward_metadata.state_in_pages
            state_out_pages = self.forward_metadata.state_out_pages
            use_flat = state_in_pages is not None
            if use_flat:
                # state_in/out_pages are [k, bs]: row g.shard pages a group's
                # ssm rows, row g.conv_shard the layer's conv rows. The int64
                # index copies and the R1 fresh mask are precomputed once in
                # init_forward_metadata (shared by every state layer).
                head_groups = self.kv_pool.get_state_buffers(layer_id)
                state_in_long = self.forward_metadata.state_in_pages_i64
                state_out_long = self.forward_metadata.state_out_pages_i64
                # R1: requests with no history read row 0, which aliases the
                # KV dummy page and is NOT zero (padded tokens write it, see
                # state_shard_view WARNING) — zero-seed instead of copying.
                fresh = self.forward_metadata.state_fresh_mask
                if state_in_long is None or fresh is None:
                    raise RuntimeError(
                        "flat GDN state paging: extend forward requires "
                        "state_seq_lens_before metadata for R1 zero-seeding"
                    )
                conv_group = head_groups[0]
                conv_states = conv_group.conv
                # Seed the out page's conv window from the in page (identity
                # within a page; boundary crossing carries the previous
                # window over), then run the conv read+write entirely on the
                # out page so a shared snapshot in-page is never written.
                conv_seed = conv_states[state_in_long[conv_group.conv_shard]]
                conv_seed[fresh] = 0
                conv_states[state_out_long[conv_group.conv_shard]] = conv_seed
                conv_cache_indices = state_out_pages[conv_group.conv_shard]
            else:
                conv_states, ssm_states = self.pool.get_mamba_params(layer_id)
                conv_cache_indices = cache_indices
            extend_prefix_lens = kwargs.get("extend_prefix_lens")
            if extend_prefix_lens is None:
                extend_prefix_lens = self.forward_metadata.extend_prefix_lens
            extend_seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")
            if extend_seq_lens_cpu is None:
                extend_seq_lens_cpu = self.forward_metadata.extend_seq_lens_cpu
            has_initial_states = (
                extend_prefix_lens > 0 if extend_prefix_lens is not None else None
            )
            need_h_track = (
                self.forward_metadata.track_ssm_h_src is not None
                and self.forward_metadata.track_ssm_h_src.numel() > 0
            )
            need_flat_track = (
                use_flat and self.forward_metadata.flat_track_ssm_src is not None
            )

            # Zero padded rows so garbage can't reach recurrent state (see scrub_padding_tail).
            if extend_seq_lens_cpu is not None:
                ntok = int(sum(int(x) for x in extend_seq_lens_cpu))
                scrub_padding_tail(ntok, mixed_qkv, a, b)

            mixed_qkv_t = mixed_qkv.transpose(0, 1)
            if need_h_track:
                if self.forward_metadata.track_conv_indices is None:
                    raise RuntimeError(
                        "Missing conv indices for intermediate mamba track"
                    )
                conv_states[self.forward_metadata.track_ssm_h_dst] = mixed_qkv_t[
                    :, self.forward_metadata.track_conv_indices
                ].transpose(0, 1)
            if need_flat_track:
                # Boundary conv window = the raw (pre-conv) inputs of the last
                # kernel-1 tokens before the boundary, written into the
                # boundary page of the layer's conv view (once per layer).
                md = self.forward_metadata
                if md.flat_track_conv_indices is None:
                    # Lazy: conv_state_len only arrives with the bound views.
                    md.flat_track_conv_indices = self._compute_track_conv_indices(
                        query_start_loc,
                        md.flat_track_lens,
                        md.flat_track_mask,
                        conv_state_len=conv_states.shape[-1],
                    )
                sel = md.flat_track_sel[conv_group.conv_shard]
                conv_states[md.flat_track_pages[conv_group.conv_shard]] = (
                    mixed_qkv_t[:, md.flat_track_conv_indices[sel]]
                    .transpose(0, 1)
                    .to(conv_states.dtype)
                )

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv_t,
                conv_weights,
                bias,
                activation=activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=conv_cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        key_split_dim = key_dim // attn_tp_size
        value_split_dim = value_dim // attn_tp_size
        num_heads = key_split_dim // head_k_dim
        num_value_heads = value_split_dim // head_v_dim

        query, key, value = fused_qkv_split_gdn_prefill(
            mixed_qkv,
            num_q_heads=num_heads,
            num_k_heads=num_heads,
            num_v_heads=num_value_heads,
            head_q=head_k_dim,
            head_k=head_k_dim,
            head_v=head_v_dim,
        )

        if is_target_verify:
            draft_token_num = kwargs.get(
                "draft_token_num", self.speculative_num_draft_tokens
            )
            core_attn_out = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                dt_bias=dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                cu_seqlens=query_start_loc,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                # target_verify specific parameters
                disable_state_update=True,
                output_state_indices=self.forward_metadata.mamba_output_indices,
            )
        else:
            beta = b.sigmoid()
            g = fused_gdn_gating(A_log, a, dt_bias)
            g = g.unsqueeze(0)
            beta = beta.unsqueeze(0)

            if use_flat:
                # Gather the full-head initial state across the shard views
                # (advanced-indexing COPY: reading a copy is safe, writes
                # below go back through the views). R1: zero no-history rows
                # instead of trusting the aliased dirty null row 0.
                recurrent_state = torch.cat(
                    [g.ssm[state_in_long[g.shard]] for g in head_groups], dim=1
                )
                recurrent_state[fresh] = 0
            else:
                recurrent_state = ssm_states[cache_indices]
            need_final_track = (
                self.forward_metadata.track_ssm_final_src is not None
                and self.forward_metadata.track_ssm_final_src.numel() > 0
            )

            fi_h_checkpoints = None
            h_src = None
            need_track = need_h_track or need_flat_track
            gdn_result = gdn_chunk_prefill(
                query,
                key,
                value,
                g,
                beta,
                scale=head_k_dim**-0.5,
                initial_state=recurrent_state,
                cu_seqlens=query_start_loc,
                qk_l2norm=True,
                output_final_state=True,
                output_h=need_track,
                # Flat boundaries live on the state-page grid; None keeps the
                # radix path on the historical 64-token grid, byte-identical.
                checkpoint_interval=(
                    self._flat_state_page_size if need_flat_track else None
                ),
            )
            core_attn_out = gdn_result.out
            last_recurrent_state = gdn_result.final_state
            if need_track:
                if gdn_result.h is None:
                    raise RuntimeError(
                        "gdn_chunk_prefill(output_h=True) must return checkpoints"
                    )
                if gdn_result.h_layout is GdnCheckpointLayout.FLASHINFER:
                    fi_h_checkpoints = gdn_result.h
                    h_src = (
                        self.forward_metadata.flat_track_ssm_src
                        if need_flat_track
                        else self.forward_metadata.track_ssm_h_src
                    )
                elif gdn_result.h_layout is GdnCheckpointLayout.FLA:
                    fi_h_checkpoints = gdn_result.h.squeeze(0)
                    h_src = (
                        self.forward_metadata.flat_track_ssm_src_fla
                        if need_flat_track
                        else self.forward_metadata.track_ssm_h_src_fla
                    )
                else:
                    raise RuntimeError(
                        "gdn_chunk_prefill(output_h=True) returned unsupported "
                        f"checkpoint layout {gdn_result.h_layout}"
                    )
            if use_flat:
                # Scatter each group's head slice back through its strided
                # view (advanced-indexing WRITE on the view itself; never
                # .contiguous()/.reshape() — those copies drop the write).
                last_recurrent_state = last_recurrent_state.to(
                    head_groups[0].ssm.dtype, copy=False
                )
                for g in head_groups:
                    g.ssm[state_out_long[g.shard]] = last_recurrent_state[
                        :, g.head_begin : g.head_begin + g.num_heads
                    ]
            else:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if need_h_track:
                ssm_states[self.forward_metadata.track_ssm_h_dst] = fi_h_checkpoints[
                    h_src
                ].to(ssm_states.dtype, copy=False)

            if need_flat_track:
                # Boundary ssm state into the boundary page's head-group views
                # (advanced-indexing WRITE on the strided views; fp32 ckpts ->
                # fp32 ssm, no rounding). The page then satisfies the C++
                # final-page registration predicate and becomes hittable.
                md = self.forward_metadata
                for grp in head_groups:
                    sel = md.flat_track_sel[grp.shard]
                    grp.ssm[md.flat_track_pages[grp.shard]] = fi_h_checkpoints[
                        h_src[sel]
                    ][:, grp.head_begin : grp.head_begin + grp.num_heads].to(
                        grp.ssm.dtype, copy=False
                    )

            if need_final_track:
                fused_mamba_state_copy(
                    conv_states,
                    self.forward_metadata.track_ssm_final_src,
                    self.forward_metadata.track_ssm_final_dst,
                    single_layer=True,
                )
                fused_mamba_state_copy(
                    ssm_states,
                    self.forward_metadata.track_ssm_final_src,
                    self.forward_metadata.track_ssm_final_dst,
                    single_layer=True,
                )

        return core_attn_out


class HybridLinearAttnBackend(AttentionBackend):
    """Hybrid backend that routes between full attention and linear attention by layer ID."""

    # Both sub-backends consume flat per-group tables (MHA: KV pages; mamba:
    # dual-index state pages). Safety comes from the publication rule
    # (paged_cache_spec.publish_paged_cache_groups): a radix ext or spec
    # decode never publishes groups, so no tables (and no flat capture
    # buffers) exist on those paths.
    uses_flat_cache_groups: bool = True

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackend,
        full_attn_layers: list[int],
    ):
        self.device = full_attn_backend.device
        self.full_attn_layers = set(full_attn_layers)
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend

    def _backends(self):
        return [self.full_attn_backend, self.linear_attn_backend]

    def _backend_for_layer(self, layer_id: int) -> AttentionBackend:
        if self.linear_attn_backend is None or layer_id in self.full_attn_layers:
            return self.full_attn_backend
        return self.linear_attn_backend

    _MAMBA_KWARGS = frozenset(
        {
            "mamba_pool_indices",
            "mamba_cow_src_indices",
            "mamba_branching_seqlens",
            "mamba_track_pool_indices",
        }
    )

    @staticmethod
    def _split_mamba_kwargs(kwargs: dict) -> tuple[dict, dict]:
        mamba_kw = {}
        common_kw = {}
        for k, v in kwargs.items():
            if k in HybridLinearAttnBackend._MAMBA_KWARGS:
                mamba_kw[k] = v
            else:
                common_kw[k] = v
        return common_kw, mamba_kw

    # ---- Metadata delegation ----

    def init_forward_metadata(self, *args, **kwargs):
        common_kw, mamba_kw = self._split_mamba_kwargs(kwargs)
        self.full_attn_backend.init_forward_metadata(*args, **common_kw)
        self.linear_attn_backend.init_forward_metadata(*args, **common_kw, **mamba_kw)

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor, **kwargs):
        # kwargs (e.g. paged_cache_group_specs, so the full backend sheds
        # state-family groups) are forwarded through the shared signature
        # filter: the full backend is user-selectable and may have a narrow
        # signature (e.g. TRTLLM MHA takes only (max_bs, seq_lens_buf)), and
        # the mamba backend keeps its narrow signature today.
        init_backend_cuda_graph_state(
            self.full_attn_backend, max_bs, seq_lens_buf, **kwargs
        )
        init_backend_cuda_graph_state(
            self.linear_attn_backend, max_bs, seq_lens_buf, **kwargs
        )

    def register_step_counter(self, step_counter):
        # Hybrid layerwise transfer needs one global step per model layer,
        # including both full-attention and mamba layers. Record steps in this
        # wrapper instead of in child backends to avoid double counting.
        self.step_counter = step_counter

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        common_kw, mamba_kw = self._split_mamba_kwargs(kwargs)
        self.full_attn_backend.init_forward_metadata_capture_cuda_graph(
            *args, **common_kw
        )
        self.linear_attn_backend.init_forward_metadata_capture_cuda_graph(
            *args, **common_kw, **mamba_kw
        )

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        common_kw, mamba_kw = self._split_mamba_kwargs(kwargs)
        self.full_attn_backend.init_forward_metadata_replay_cuda_graph(
            *args, **common_kw
        )
        self.linear_attn_backend.init_forward_metadata_replay_cuda_graph(
            *args, **common_kw, **mamba_kw
        )

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return self.full_attn_backend.support_kv_cache_prewrite(forward_mode)

    # ---- Forward dispatch ----

    @break_point
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc,
        token_to_kv_pool,
        forward_mode: ForwardMode,
        bs: int,
        save_kv_cache: bool = True,
        record_kv_cache: bool | None = None,
        **kwargs,
    ):
        """Dispatch one layer to its full-attention or GDN backend (the break point).

        Overrides the base forward, so it carries its own ``@break_point``;
        the frozen capture-time scalars (forward_mode/bs) are re-read from the
        ambient ctx (semantics: see breakable_cuda_graph). The GDN scan's
        batched [1, T, Hv, D] output is collapsed to z-shaped [T, Hv, D].
        """
        if forward_mode is None:
            return super().forward(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                forward_mode,
                bs,
                save_kv_cache,
                record_kv_cache=record_kv_cache,
                **kwargs,
            )

        # Frozen capture-time scalars, re-read live (see docstring); no-op in eager.
        amb = current_forward_ctx()
        if amb is not None:
            forward_mode = amb.forward_mode
            bs = amb.bs

        if forward_mode.is_idle():
            if layer is None:
                return torch.empty_like(kwargs["z"])
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        backend = self._backend_for_layer(layer_id)

        # See AttentionBackend.forward for the record_kv_cache contract; the step
        # is recorded in this wrapper (not the child backends) to keep one step
        # per model layer across full-attn + mamba. Idle already returned above.
        with self.record_pd_cache_step(forward_mode, save_kv_cache, record_kv_cache):
            if forward_mode.is_decode():
                ret = backend.forward_decode(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    bs,
                    save_kv_cache=save_kv_cache,
                    **kwargs,
                )
            else:
                ret = backend.forward_extend(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    bs,
                    save_kv_cache=save_kv_cache,
                    forward_mode=forward_mode,
                    **kwargs,
                )
        # Collapse the GDN scan's batched [1, T, Hv, D] to z-shaped (see docstring).
        if ret is not None and ret.dim() == 4:
            # Strictly [1, T, Hv, D]: a genuine B>1 must fail loud, not corrupt the handoff.
            assert (
                ret.shape[0] == 1
            ), f"GDN scan batched rank expected leading 1, got {ret.shape}"
            ret = ret.flatten(0, 1)
        return ret

    def forward_decode(
        self, q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        return self._backend_for_layer(layer_id).forward_decode(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
        )

    def forward_extend(
        self, q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        return self._backend_for_layer(layer_id).forward_extend(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
        )

    def reset_current_inputs(self, *args, **kwargs):
        if self.linear_attn_backend is None:
            return
        if hasattr(self.linear_attn_backend, "reset_current_inputs"):
            self.linear_attn_backend.reset_current_inputs(*args, **kwargs)

    def update_mamba_state_after_mtp_verify(self, accepted_length, model):
        # mamba_cache_indices are input rows during target-verify. The first
        # output row is always the scheduler-owned working slot, so use the
        # output index table to update the next-round input pointer.
        output_indices = self.linear_attn_backend.forward_metadata.mamba_output_indices
        if output_indices is None:
            return
        req_pool_indices = (
            self.linear_attn_backend.forward_metadata.mamba_req_pool_indices
        )
        if req_pool_indices is None:
            return
        request_number = accepted_length.shape[0]
        self.linear_attn_backend.pool.update_current_inputs_after_verify(
            req_pool_indices[:request_number],
            output_indices[:request_number],
            accepted_length,
        )
