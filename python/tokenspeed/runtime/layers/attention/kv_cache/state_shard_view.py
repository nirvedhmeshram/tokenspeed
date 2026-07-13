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

"""GDN/mamba2 state views over the flat pool's K/V slabs (M18c binning).

Replaces ``FlatStateSlabs``: the recurrent state no longer owns device
memory. The state shard bin table (``flat_memory_plan.shard_bin_table``)
packs every state layer's conv row and ssm head groups into segments of
the full layers' K/V page rows, and this class reinterprets those byte
ranges as typed tensors -- the fp32 ssm views live INSIDE the bf16 KV
slabs (dtype reinterpret, not a cast). Row index == page id over the
single shared page-id space.

WARNING: row 0 aliases the KV dummy page (page 0), which padded tokens
DO write (see ``mha.py`` ``_create_buffers``) -- unlike the retired
standalone ``FlatStateSlabs``, row 0 must NOT be assumed zero. A
consumer seeding recurrent state from "no history" (``state_in == 0``)
must zero-fill rather than copy row 0 (handled in the backend, M18c T4).

Flat-GDN gate semantics carry over verbatim from ``FlatStateSlabs``: one
boolean gates both skipping per-layer KV on state layers and binding the
state views, with the added requirement that a bin table is present
(``bin_table is None`` -> inactive, the planned transitional off-switch).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from tokenspeed.runtime.configs import paged_cache_spec
from tokenspeed.runtime.configs.flat_memory_plan import StateShardBinTable
from tokenspeed.runtime.configs.paged_cache_spec import STATE_LAYER_TYPES
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class StateHeadGroup(NamedTuple):
    """One ssm head group of a state layer, plus the layer's conv view.

    ``conv`` is the WHOLE layer's conv window view (repeated across the
    layer's head groups); ``ssm`` covers only heads
    ``[head_begin, head_begin + num_heads)``. ``shard`` selects which
    ``linear_attention_shard{i}`` block table pages this group's ssm rows,
    ``conv_shard`` which one pages the conv rows (the bin packing may land
    them on different shards).

    Both ``ssm`` and ``conv`` are NON-contiguous strided views: dim 0
    strides a whole page row (the K/V slab row), not the group's own
    element count, so a tail group is necessarily discontiguous. Address
    them only by ``.stride()`` or advanced indexing. Calling
    ``.contiguous()`` / ``.reshape()`` on them yields a COPY; writing into
    that copy (e.g. a harvest scatter) silently drops the update. A
    flashinfer-class kernel must check the layout explicitly before
    consuming these views."""

    conv: torch.Tensor  # (N, *conv_state_shape) view, whole layer
    ssm: torch.Tensor  # (N, num_heads, *state_dims) view, this group
    shard: int  # ssm pages come from shard table `shard`
    conv_shard: int  # conv pages come from shard table `conv_shard`
    head_begin: int
    num_heads: int


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major strides of a contiguous tensor of ``shape``."""
    strides = []
    acc = 1
    for dim in reversed(shape):
        strides.append(acc)
        acc *= dim
    return tuple(reversed(strides))


class StateShardView:
    """Reinterpret views of the GDN/mamba2 state over the K/V slabs.

    Args:
        bin_table: ``StateShardBinTable`` (flat_memory_plan.shard_bin_table)
            placing every conv row / ssm head group into a (shard, slot,
            kv_side, byte_offset) segment of the full layers' page rows.
            ``None`` -> inactive (legacy / radix / non-binned profiles).
        layer_types: Per-layer type labels; state layers carry the
            ``STATE_LAYER_TYPES`` labels. Drives the layer -> pair mapping.
        conv_state_shape / temporal_state_shape: Per-state-layer mamba2
            state tensor shapes (configs' mamba2_cache_params); ``None`` on
            pure-attention models.
        conv_dtype / ssm_dtype: State dtypes; default to ``default_dtype``.
        default_dtype: Pool store dtype used when a state dtype is ``None``.
        page_size: Page size P (tokens); must equal the bin table's.
        size: Total token slots (the flat pool size; must be whole pages
            when the views are active).

    No device memory is owned here: ``bind`` builds the views over the
    pool's existing K/V slab tensors once and caches them, so
    ``get_state_buffers`` returns identical tensor objects every call
    (stable addresses -- the CUDA-graph capture prerequisite).
    """

    def __init__(
        self,
        *,
        bin_table: StateShardBinTable | None,
        layer_types: tuple[str, ...],
        conv_state_shape: tuple[int, ...] | None,
        temporal_state_shape: tuple[int, ...] | None,
        conv_dtype: torch.dtype | None,
        ssm_dtype: torch.dtype | None,
        default_dtype: torch.dtype,
        page_size: int,
        size: int,
    ):
        self._bin_table = bin_table
        self._layer_types = tuple(layer_types or ())
        # Per-state-layer mamba2 shapes (configs' mamba2_cache_params);
        # None on pure-attention models.
        self._conv_state_shape = (
            tuple(conv_state_shape) if conv_state_shape is not None else None
        )
        self._temporal_state_shape = (
            tuple(temporal_state_shape) if temporal_state_shape is not None else None
        )
        self._conv_dtype = conv_dtype if conv_dtype is not None else default_dtype
        self._ssm_dtype = ssm_dtype if ssm_dtype is not None else default_dtype
        self.page_size = page_size
        self.size = size

        # layer_id -> state pair index (the n-th state layer binds pair n).
        # Derives purely from layer_types; shared by the KV skip set on the
        # pool and the view binding below.
        self._layer_state_pair: dict[int, int] = {
            layer_id: pair
            for pair, layer_id in enumerate(
                layer_id
                for layer_id, label in enumerate(self._layer_types)
                if label in STATE_LAYER_TYPES
            )
        }

        # Flat GDN predicate: ONE boolean gates both skipping per-layer KV on
        # state layers and binding the state views -- the plan sizing
        # (registry) charges exactly full-layer KV rows (state is aliased
        # over them), so the two decisions must never diverge. A missing bin
        # table switches the whole feature off (transitional legacy state).
        self._flat_gdn = (
            bool(self._layer_state_pair)
            and self._conv_state_shape is not None
            and self._temporal_state_shape is not None
            and self._bin_table is not None
            and paged_cache_spec.scheduler_ext_flat_kvcache()
        )
        if self._flat_gdn and self._bin_table.block_size != self.page_size:
            raise ValueError(
                "state bin table was solved for block_size="
                f"{self._bin_table.block_size} but the pool pages "
                f"{self.page_size} tokens; the two must match"
            )

        self.num_pages_with_null: int | None = None
        # layer_id -> cached head groups; built once in bind so repeated
        # get_state_buffers calls return the SAME tensor objects (address
        # stability is the CUDA-graph prerequisite).
        self._state_buffers: dict[int, list[StateHeadGroup]] = {}

    @property
    def is_active(self) -> bool:
        """True iff the flat-GDN gate is on (state views will be bound).

        When False the pool must NOT skip per-layer KV on state layers and
        must not bind views (pure attention, radix ext, spec decode,
        missing state shapes, or no bin table)."""
        return self._flat_gdn

    @property
    def state_layer_ids(self) -> frozenset[int]:
        """Layer ids that carry NO per-layer KV (state layers under flat
        GDN). Empty unless active, so non-flat profiles keep full KV."""
        return frozenset(self._layer_state_pair) if self._flat_gdn else frozenset()

    def is_state_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id`` is a KV-less state layer under flat GDN."""
        return self._flat_gdn and layer_id in self._layer_state_pair

    def bind(
        self,
        k_buffers: list[torch.Tensor],
        v_buffers: list[torch.Tensor],
    ) -> None:
        """Build (and cache) the state views over the pool's K/V slabs.

        Args:
            k_buffers / v_buffers: The full layers' slab tensors in SLOT
                order (layer_id ascending over non-state layers -- exactly
                the bin table's slot enumeration). Each has
                ``size + page_size`` leading rows (the +P dummy page).

        Only call when ``is_active``; raises otherwise. Views are pure
        reinterprets (``as_strided`` over a possibly dtype-reinterpreted
        row matrix): no allocation, addresses stable for the pool's
        lifetime.
        """
        if not self._flat_gdn:
            raise ValueError("StateShardView.bind called while inactive")
        if self._state_buffers:
            raise RuntimeError("StateShardView already bound")
        if len(k_buffers) != len(v_buffers):
            raise ValueError(
                f"bind got {len(k_buffers)} K slabs but {len(v_buffers)} V "
                "slabs; the full layers' slot enumeration must be symmetric"
            )
        seen: dict[int, None] = {}
        for buf in (*k_buffers, *v_buffers):
            if id(buf) in seen:
                raise ValueError(
                    "bind got a duplicate slab tensor across K/V; two slot "
                    "segments would alias the same storage"
                )
            seen[id(buf)] = None
        if self.size % self.page_size != 0:
            raise ValueError("flat pool size must be whole pages")
        n = self.size // self.page_size + 1
        self.num_pages_with_null = n
        table = self._bin_table
        sides = (k_buffers, v_buffers)

        row_matrices: dict[tuple[int, int], torch.Tensor] = {}

        def _row_matrix(kv_side: int, slot: int) -> torch.Tensor:
            # (n, P * head_num * head_dim) elements: one page row per line.
            key = (kv_side, slot)
            pm = row_matrices.get(key)
            if pm is None:
                slab = sides[kv_side][slot]
                assert slab is not None, f"slot {slot} kv_side {kv_side} has no slab"
                pm = slab.reshape(n, -1)
                row_matrices[key] = pm
            return pm

        def _reinterpret(entry, dtype: torch.dtype, shape: tuple[int, ...]):
            """View ``entry``'s byte range as a (n, *shape) tensor of dtype."""
            pm = _row_matrix(entry.kv_side, entry.slot)
            typed = pm if dtype == pm.dtype else pm.view(dtype)
            itemsize = typed.element_size()
            row_bytes = typed.shape[1] * itemsize
            if entry.byte_offset % itemsize != 0:
                raise ValueError(
                    f"bin entry offset {entry.byte_offset} is not aligned to "
                    f"{dtype} itemsize {itemsize}"
                )
            if entry.byte_offset + entry.nbytes > row_bytes:
                raise ValueError(
                    f"bin entry [{entry.byte_offset}, "
                    f"{entry.byte_offset + entry.nbytes}) overruns the "
                    f"{row_bytes}-byte page row (slot {entry.slot}, "
                    f"kv_side {entry.kv_side})"
                )
            return typed.as_strided(
                (n, *shape),
                (typed.stride(0), *_contiguous_strides(shape)),
                storage_offset=typed.storage_offset() + entry.byte_offset // itemsize,
            )

        # Conv: one whole-layer view per state layer (repeated across the
        # layer's head groups below).
        conv_views: dict[int, tuple[torch.Tensor, int]] = {}
        for entry in table.conv_entries:
            if entry.nbytes != (
                math.prod(self._conv_state_shape) * self._conv_dtype.itemsize
            ):
                raise ValueError("conv bin entry size disagrees with conv_state_shape")
            conv_views[entry.state_layer] = (
                _reinterpret(entry, self._conv_dtype, self._conv_state_shape),
                entry.shard,
            )

        # Ssm: one view per head group. Shapes derive from
        # temporal_state_shape = (heads, *state_dims); nothing hardcodes the
        # 128x128 cell.
        state_dims = self._temporal_state_shape[1:]
        head_elems = math.prod(state_dims)
        ssm_views: dict[int, list[tuple]] = {}
        for entry in table.ssm_entries:
            if entry.nbytes != (
                entry.num_heads * head_elems * self._ssm_dtype.itemsize
            ):
                raise ValueError(
                    "ssm bin entry size disagrees with temporal_state_shape"
                )
            view = _reinterpret(entry, self._ssm_dtype, (entry.num_heads, *state_dims))
            ssm_views.setdefault(entry.state_layer, []).append((entry, view))

        self._state_buffers = {
            layer_id: [
                StateHeadGroup(
                    conv=conv_views[pair][0],
                    ssm=view,
                    shard=entry.shard,
                    conv_shard=conv_views[pair][1],
                    head_begin=entry.head_begin,
                    num_heads=entry.num_heads,
                )
                for entry, view in sorted(
                    ssm_views[pair], key=lambda ev: ev[0].head_begin
                )
            ]
            for layer_id, pair in self._layer_state_pair.items()
        }
        logger.info(
            "State shard views: %d state layers x %d head groups over %d "
            "K/V slab page rows (row 0 = null page), k=%d shards",
            len(self._layer_state_pair),
            len(table.ssm_entries) // max(len(self._layer_state_pair), 1),
            n,
            table.num_shards,
        )

    def get_state_buffers(self, layer_id: int) -> list[StateHeadGroup]:
        """Head groups of a state layer; the n-th state layer
        (within-state-label occurrence order, the pairing order) binds the
        n-th bin-table layer. Raises ValueError for non-state layers."""
        pair = self._layer_state_pair.get(layer_id)
        if pair is None:
            raise ValueError(
                f"layer {layer_id} is not a state layer "
                f"(layer_types={self._layer_types!r})"
            )
        if not self._state_buffers:
            raise ValueError(
                f"layer {layer_id} is a state layer but no state "
                "views were bound (state shapes/bin table missing or "
                "radix ext)"
            )
        return self._state_buffers[layer_id]
