"""Flat KV-cache memory plan: pure sizing/binding decisions, no torch.

Components declare per-block bytes as a function of P (block_size):
linear components scale (bytes_per_slot > 0), constant components do not
(const_bytes > 0, mamba state snapshots). Same-(group, layer) components
pack into one page row ([conv|ssm|pad], the vLLM hybrid layout). One
equalizer move: constant rows inflate P until the widest linear row
covers them (vLLM align). plan_tensors then pairs physical slot j with
the j-th layer of every group over a single page-id space and sizes each
slab by its own packed row from the budget.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace

# Labels whose group is state-family (recurrent state rows, not KV history).
# Deliberate one-line duplicate of paged_cache_spec.STATE_LAYER_TYPES: both
# modules are direct-loaded standalone by their tests (importlib, no package
# context), so a cross-module import would break either loader. Keep in sync.
STATE_LAYER_TYPES = frozenset({"linear_attention"})


@dataclass(frozen=True)
class ComponentSpec:
    group_id: str
    layer: int
    component: str
    bytes_per_slot: int  # linear in P; 0 for constant components
    const_bytes: int  # constant in P; 0 for linear components


@dataclass(frozen=True)
class BlockGeometry:
    block_size: int
    block_bytes: int
    num_blocks: int = 0  # filled by the planners from the memory budget


def occurrence_index(labels):
    """Within-label occurrence index per position.

    Args:
        labels: Iterable of hashable labels (e.g. per-layer type strings).

    Returns:
        list[int]: out[i] == number of earlier positions carrying the same
        label as position i — the slab pairing order shared by
        components_from_layers and the KV pool's slab layout.
    """
    counts: dict = {}
    out: list[int] = []
    for label in labels:
        idx = counts.get(label, 0)
        counts[label] = idx + 1
        out.append(idx)
    return out


def state_const_bytes(conv_shape, conv_dtype, ssm_shape, ssm_dtype):
    """Constant per-page state row bytes of one GDN/mamba2 state layer.

    Args:
        conv_shape / ssm_shape: Per-layer state tensor shapes (the configs'
            mamba2_cache_params conv and temporal shapes).
        conv_dtype / ssm_dtype: Matching dtypes (anything with ``itemsize``).

    Returns:
        dict[str, int]: {"conv": bytes, "ssm": bytes} — the exact
        ``state_const_bytes`` mapping components_from_layers consumes
        (insertion order = row_offset order).
    """
    return {
        "conv": math.prod(conv_shape) * conv_dtype.itemsize,
        "ssm": math.prod(ssm_shape) * ssm_dtype.itemsize,
    }


def components_from_layers(*, layer_types, kv_bytes_per_slot, state_const_bytes):
    """Per-layer ComponentSpecs: history layers carry one linear kv component;
    state layers one constant component per state tensor. Layer index is the
    within-group occurrence count (the slab pairing order). State component
    order (hence row_offset order downstream) follows state_const_bytes
    insertion order."""
    comps: list[ComponentSpec] = []
    for label, idx in zip(layer_types, occurrence_index(layer_types)):
        if label in STATE_LAYER_TYPES:
            for name, nbytes in state_const_bytes.items():
                comps.append(ComponentSpec(label, idx, name, 0, nbytes))
        else:
            comps.append(ComponentSpec(label, idx, "kv", kv_bytes_per_slot, 0))
    return comps


def _row_demands(components):
    """Per-(group, layer) row: (linear bytes-per-slot sum, constant bytes sum)."""
    rows = defaultdict(lambda: [0, 0])
    for c in components:
        row = rows[(c.group_id, c.layer)]
        row[0] += c.bytes_per_slot
        row[1] += c.const_bytes
    return rows


def solve_page_geometry(components, *, block_size, alignment):
    """Smallest P >= block_size (multiple of `alignment` when inflated)
    such that the widest linear row covers the widest constant row."""
    rows = _row_demands(components).values()
    # NOTE: a row mixing linear and constant components is not needed by any
    # known model; reject it so the math stays honest.
    for lin, const in rows:
        if lin > 0 and const > 0:
            raise ValueError("a row must be all-linear or all-constant")
    max_linear = max((lin for lin, _ in rows), default=0)
    max_const = max((const for _, const in rows), default=0)
    if max_const > 0:
        if max_linear == 0:
            raise ValueError("constant components need a linear row to size P against")
        needed = -(-max_const // max_linear)  # exact integer ceil
        if needed > block_size:
            block_size = alignment * math.ceil(needed / alignment)
    block_bytes = max(max_linear * block_size, max_const)
    return BlockGeometry(block_size=block_size, block_bytes=block_bytes)


@dataclass(frozen=True)
class LayerBinding:
    slot: int
    group_id: str
    layer: int
    component: str
    nbytes_per_block: int
    row_offset: int  # byte offset of this component within its (group, layer) page row


@dataclass(frozen=True)
class TensorPlan:
    name: str
    nbytes: int
    bindings: tuple[LayerBinding, ...]


@dataclass(frozen=True)
class FlatMemoryPlan:
    geometry: BlockGeometry
    tensors: tuple[TensorPlan, ...]


def plan_component_tensors(
    components, *, block_size, budget_bytes, reserved_bytes_per_block=0
):
    """One tensor per ComponentSpec, honestly sized: row bytes = that
    component's per-block bytes, num_blocks = budget // (sum of all rows +
    reserved_bytes_per_block). No cross-component packing, no padding —
    every tensor keeps today's standalone-slab shape, so kernels, CUDA
    graphs and the host mirror stay untouched. reserved_bytes_per_block
    carries co-resident rows outside these components (the MTP draft
    pool's KV rows ride the same block-id space). Under this planner each
    component is its own slot, in input order."""
    row_bytes = [c.bytes_per_slot * block_size + c.const_bytes for c in components]
    per_block = sum(row_bytes) + reserved_bytes_per_block
    num_blocks = budget_bytes // per_block
    if num_blocks <= 1:
        raise ValueError("budget too small for one usable block")
    geo = BlockGeometry(
        block_size=block_size, block_bytes=per_block, num_blocks=num_blocks
    )
    tensors = tuple(
        TensorPlan(
            name=f"flat_{c.group_id}_{c.layer}_{c.component}",
            nbytes=num_blocks * nbytes,
            bindings=(LayerBinding(i, c.group_id, c.layer, c.component, nbytes, 0),),
        )
        for i, (c, nbytes) in enumerate(zip(components, row_bytes))
    )
    return FlatMemoryPlan(geometry=geo, tensors=tensors)


def plan_tensors(components, *, block_size, alignment, budget_bytes):
    """Pair slot j with the j-th layer of every group over one page-id space.
    Each slot tensor is sized by its own packed row (the sum of its bindings'
    per-block bytes); geometry.block_bytes accounts one block's total across
    all slots."""
    geo = solve_page_geometry(components, block_size=block_size, alignment=alignment)
    layers_by_group: dict[str, list[int]] = {}
    for c in components:
        layers = layers_by_group.setdefault(c.group_id, [])
        if c.layer not in layers:
            layers.append(c.layer)
    num_slots = max(len(v) for v in layers_by_group.values())

    slot_bindings: list[tuple[LayerBinding, ...]] = []
    for slot in range(num_slots):
        bindings = []
        for gid, layers in layers_by_group.items():
            if slot >= len(layers):
                continue
            layer = layers[slot]
            row_offset = 0
            for c in components:
                if c.group_id != gid or c.layer != layer:
                    continue
                nbytes = c.bytes_per_slot * geo.block_size + c.const_bytes
                bindings.append(
                    LayerBinding(slot, gid, layer, c.component, nbytes, row_offset)
                )
                row_offset += nbytes
        slot_bindings.append(tuple(bindings))
    slot_rows = [sum(b.nbytes_per_block for b in bs) for bs in slot_bindings]

    num_blocks = budget_bytes // sum(slot_rows)
    if num_blocks <= 1:
        raise ValueError("budget too small for one usable block per slot")
    geo = replace(geo, block_bytes=sum(slot_rows), num_blocks=num_blocks)

    tensors = tuple(
        TensorPlan(
            name=f"flat_slab_{slot}",
            nbytes=num_blocks * slot_rows[slot],
            bindings=bindings,
        )
        for slot, bindings in enumerate(slot_bindings)
    )
    return FlatMemoryPlan(geometry=geo, tensors=tensors)


@dataclass(frozen=True)
class ShardBinEntry:
    state_layer: int  # within-state occurrence index (slab pairing order)
    head_begin: int  # first ssm head of this group; 0 for conv entries
    num_heads: int  # ssm heads in this group; 0 for conv entries
    nbytes: int
    shard: int  # 0..k-1 (which shard-group block carries this piece)
    slot: int  # 0..num_full_layers-1 (which full layer's slab)
    kv_side: int  # 0 = K slab, 1 = V slab
    byte_offset: int  # byte offset inside the page row


@dataclass(frozen=True)
class StateShardBinTable:
    num_shards: int
    block_size: int
    segment_bytes: int
    heads_per_group: int
    ssm_entries: tuple[ShardBinEntry, ...]
    conv_entries: tuple[ShardBinEntry, ...]


def shard_bin_table(
    *,
    num_full_layers,
    num_state_layers,
    ssm_heads_per_layer,
    ssm_head_bytes,
    conv_bytes_per_layer,
    kv_cell_bytes_per_tok,
    block_size,
):
    """Pack GDN state (conv+ssm rows of every state layer) into segments of
    the full layers' K/V page rows; num_shards (k) falls out of the packing.
    Segment = one full layer's K (or V) page row = (cell // 2) * P bytes.
    Pure function: no torch, deterministic first-fit.

    Args:
        num_full_layers: Full-attention layers per group; each contributes
            two segments (K slab + V slab) to a shard-group block.
        num_state_layers: GDN/mamba2 state layers whose conv+ssm rows are
            packed into the full layers' spare K/V segments.
        ssm_heads_per_layer: SSM heads in one state layer; split into groups
            of ``heads_per_group`` so each group fits one segment.
        ssm_head_bytes: Bytes of one head's state cell (fp32 128x128 = 65536).
        conv_bytes_per_layer: Conv state row bytes of one state layer.
        kv_cell_bytes_per_tok: K+V combined bytes per token; ``// 2`` gives the
            single-side (K or V) segment width.
        block_size: Page block size P (tokens); must be a multiple of 64.

    Returns:
        StateShardBinTable: num_shards (k), segment/head-group geometry, and
        the packed ssm/conv entries.
    """
    if block_size % 64 != 0:
        raise ValueError(f"block_size must be a multiple of 64, got {block_size}")
    if num_full_layers < 1 or num_state_layers < 1:
        raise ValueError(
            "need at least one full layer and one state layer, got "
            f"{num_full_layers} full / {num_state_layers} state"
        )
    if kv_cell_bytes_per_tok % 2 != 0:
        raise ValueError(
            f"kv_cell_bytes_per_tok must be even (K+V split in half), got "
            f"{kv_cell_bytes_per_tok}"
        )
    segment_bytes = (kv_cell_bytes_per_tok // 2) * block_size
    heads_per_group = segment_bytes // ssm_head_bytes
    if heads_per_group < 1:
        raise ValueError(
            f"segment {segment_bytes}B cannot hold one ssm head cell "
            f"({ssm_head_bytes}B); raise block_size"
        )
    if conv_bytes_per_layer > segment_bytes:
        raise ValueError(
            f"conv row exceeds one segment: {conv_bytes_per_layer}B > "
            f"{segment_bytes}B; raise block_size"
        )
    segs_per_shard = num_full_layers * 2

    def seg_pos(s):
        return (s // segs_per_shard, (s % segs_per_shard) // 2, s % 2)

    ssm, seg = [], 0
    for layer in range(num_state_layers):
        for begin in range(0, ssm_heads_per_layer, heads_per_group):
            nh = min(heads_per_group, ssm_heads_per_layer - begin)
            sh, sl, side = seg_pos(seg)
            ssm.append(
                ShardBinEntry(layer, begin, nh, nh * ssm_head_bytes, sh, sl, side, 0)
            )
            seg += 1
    conv, off = [], 0
    for layer in range(num_state_layers):
        if off + conv_bytes_per_layer > segment_bytes:
            seg, off = seg + 1, 0
        sh, sl, side = seg_pos(seg)
        conv.append(ShardBinEntry(layer, 0, 0, conv_bytes_per_layer, sh, sl, side, off))
        off += conv_bytes_per_layer
    total_segs = seg + 1
    return StateShardBinTable(
        num_shards=-(-total_segs // segs_per_shard),
        block_size=block_size,
        segment_bytes=segment_bytes,
        heads_per_group=heads_per_group,
        ssm_entries=tuple(ssm),
        conv_entries=tuple(conv),
    )


@dataclass(frozen=True)
class StateLayerHeadMap:
    # Each field's length == that state layer's ssm head count (heads_per_layer).
    #
    # WARNING: (head_shard, head_elem_offset) is NOT an absolute address. The K
    # and V segments of one state layer are packed into distinct slab rows that
    # both start at byte_offset 0, so a K-side head and a V-side head can share
    # the very same (shard, elem_offset) yet point at different memory. Resolve
    # the base with the per-head ssm view's data_ptr() (which already encodes
    # kv_side / slot / in-row byte offset); head_shard here only selects which
    # runtime page-table row of state_pages to read, and head_elem_offset is for
    # validation/asserts, not addressing.
    head_shard: tuple[
        int, ...
    ]  # head h's page row = this row of state_pages (= entry.shard)
    head_elem_offset: tuple[
        int, ...
    ]  # head h's element offset inside its shard's ssm view
    #                                    (ssm dtype units, not bytes)


def head_addressing_maps(bin_table, *, ssm_head_elems):
    """Per-state-layer per-head (shard, in-view element offset), expanded from
    the bin table's ssm entries. Each ssm entry covers a head group; head h in
    a group at byte_offset b (ssm dtype itemsize s) sits at element offset
    (b // s) + (h - head_begin) * ssm_head_elems within that shard's slab row.
    Returns list indexed by state-layer occurrence order (len = num_state_layers).

    WARNING: the returned (head_shard, head_elem_offset) pair is NOT an absolute
    address. Because each state layer's K and V segments occupy separate slab
    rows that both begin at byte_offset 0, a head on the K side and a head on the
    V side can carry identical (shard, elem_offset) while aliasing distinct
    memory. The consumer MUST take the base from each head's ssm view data_ptr()
    (it already encodes kv_side / slot / segment offset); head_shard is only for
    picking the state_pages runtime page-table row, and head_elem_offset is only
    a validation/assert aid.

    Args:
        bin_table: StateShardBinTable.
        ssm_head_elems: elements per ssm head cell (= prod(temporal_state_shape[1:]),
            e.g. 128*128 = 16384). Element unit, matching the kernel's per-head
            addressing (dtype-agnostic; itemsize handled by caller's byte_offset).
    Returns:
        list[StateLayerHeadMap]
    """
    by_layer: dict[int, list] = {}
    order: list[int] = []
    for e in bin_table.ssm_entries:
        if e.state_layer not in by_layer:
            by_layer[e.state_layer] = []
            order.append(e.state_layer)
        by_layer[e.state_layer].append(e)

    # (layer, sorted entries, expanded lists, tail head count) per state layer.
    expanded: list[tuple[int, int, list[int], list[int]]] = []
    for layer in order:
        head_shard: list[int] = []
        head_elem_offset: list[int] = []
        next_head = 0
        for e in sorted(by_layer[layer], key=lambda e: e.head_begin):
            if e.head_begin != next_head:
                raise ValueError(
                    f"state layer {layer}: ssm head groups do not tile [0, N) "
                    f"contiguously; expected head_begin {next_head}, got "
                    f"{e.head_begin}"
                )
            if e.num_heads < 1:
                raise ValueError(
                    f"state layer {layer}: ssm entry has num_heads {e.num_heads}"
                )
            # byte_offset is bytes; recover ssm dtype itemsize from this entry so
            # the pure function need not know the dtype: one head is nbytes//num_heads
            # bytes = itemsize * ssm_head_elems elements.
            if e.nbytes % e.num_heads != 0:
                raise ValueError(
                    f"state layer {layer}: entry nbytes {e.nbytes} is not divisible "
                    f"by num_heads {e.num_heads}"
                )
            head_bytes = e.nbytes // e.num_heads
            itemsize = head_bytes // ssm_head_elems
            if itemsize < 1 or itemsize * ssm_head_elems != head_bytes:
                raise ValueError(
                    f"state layer {layer}: head bytes {head_bytes} is not a whole "
                    f"multiple of ssm_head_elems {ssm_head_elems}"
                )
            elem_base = e.byte_offset // itemsize
            for h in range(e.num_heads):
                head_shard.append(e.shard)
                head_elem_offset.append(elem_base + h * ssm_head_elems)
            next_head += e.num_heads
        # next_head == last group's (head_begin + num_heads): head order tiles
        # [0, next_head). The prefix check above already forbids interior gaps,
        # so the only remaining failure is a dropped *tail* group -> a short map.
        expanded.append((layer, next_head, head_shard, head_elem_offset))

    # Tail-coverage: every state layer of a model carries the same ssm head
    # count, so the expected upper bound is the largest per-layer head total.
    # A layer whose last group was dropped tiles [0, short) cleanly yet expands
    # to fewer heads than its peers -> caught here (never silently truncated).
    expected_heads = max(total for _, total, _, _ in expanded)
    maps: list[StateLayerHeadMap] = []
    for layer, total, head_shard, head_elem_offset in expanded:
        if total != expected_heads:
            raise ValueError(
                f"state layer {layer}: ssm head groups cover only heads "
                f"[0, {total}) but layers reach [0, {expected_heads}); a tail "
                f"head group is missing"
            )
        maps.append(
            StateLayerHeadMap(
                head_shard=tuple(head_shard),
                head_elem_offset=tuple(head_elem_offset),
            )
        )
    return maps
