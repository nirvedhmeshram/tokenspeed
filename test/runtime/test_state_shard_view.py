"""StateShardView (M18c T3): GDN state as reinterpret views over K/V slabs.

The state shard bin table places every state layer's conv row and ssm head
groups into segments of the full layers' K/V page rows; StateShardView
turns those byte ranges into typed tensors (fp32 ssm views INSIDE the bf16
KV slabs). Covers: head-group shapes/dtypes, byte-exact roundtrips through
the views in both directions, address stability (CUDA-graph prerequisite),
the flat-GDN gate (bin_table=None -> inactive), and the per-shard
no-overlap invariant of the packing as consumed by the views.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_FLAT_GATE = "tokenspeed.runtime.configs.paged_cache_spec.scheduler_ext_flat_kvcache"


class StateShardViewTest(unittest.TestCase):
    # Qwen3.5-ish tp8 numbers shrunk to 2 full + 2 state layers:
    # cell = 1024 B/tok (K side 512 B), P = 256 -> segment = 131072 B;
    # ssm head cell = 128*128 fp32 = 65536 B -> 2 heads per group, so the
    # 4 ssm heads of a layer split into 2 groups; conv row = 1024*3 bf16
    # = 6144 B. size = 8 pages -> n = 9 rows (row 0 = null page).
    LAYER_TYPES = ("linear_attention", "full_attention") * 2
    CONV_SHAPE = (1024, 3)
    TEMPORAL_SHAPE = (4, 128, 128)
    PAGE_SIZE = 256
    SIZE = 256 * 8
    HEAD_NUM = 1
    HEAD_DIM = 256  # K row/tok = 1*256*2 B = 512 B -> cell 1024 B

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.configs.flat_memory_plan import (
                shard_bin_table,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.state_shard_view import (  # noqa: E501
                StateShardView,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed package: {exc}")
        self.torch = torch
        self.StateShardView = StateShardView
        self.shard_bin_table = shard_bin_table

    def _bin_table(self):
        return self.shard_bin_table(
            num_full_layers=2,
            num_state_layers=2,
            ssm_heads_per_layer=self.TEMPORAL_SHAPE[0],
            ssm_head_bytes=self.TEMPORAL_SHAPE[1] * self.TEMPORAL_SHAPE[2] * 4,
            conv_bytes_per_layer=self.CONV_SHAPE[0] * self.CONV_SHAPE[1] * 2,
            kv_cell_bytes_per_tok=2 * self.HEAD_NUM * self.HEAD_DIM * 2,
            block_size=self.PAGE_SIZE,
        )

    def _view(self, *, bin_table="default", flat_gate=True):
        torch = self.torch
        with mock.patch(_FLAT_GATE, return_value=flat_gate):
            return self.StateShardView(
                bin_table=self._bin_table() if bin_table == "default" else bin_table,
                layer_types=self.LAYER_TYPES,
                conv_state_shape=self.CONV_SHAPE,
                temporal_state_shape=self.TEMPORAL_SHAPE,
                conv_dtype=torch.bfloat16,
                ssm_dtype=torch.float32,
                default_dtype=torch.bfloat16,
                page_size=self.PAGE_SIZE,
                size=self.SIZE,
            )

    def _slabs(self):
        torch = self.torch
        rows = self.SIZE + self.PAGE_SIZE
        k = [
            torch.zeros((rows, self.HEAD_NUM, self.HEAD_DIM), dtype=torch.bfloat16)
            for _ in range(2)
        ]
        v = [
            torch.zeros((rows, self.HEAD_NUM, self.HEAD_DIM), dtype=torch.bfloat16)
            for _ in range(2)
        ]
        return k, v

    def _bound_view(self):
        view = self._view()
        k, v = self._slabs()
        view.bind(k, v)
        return view, k, v

    def _slab_u8(self, slab):
        # (n, segment_bytes) uint8 rows of one slab: row index == page id.
        n = self.SIZE // self.PAGE_SIZE + 1
        return slab.reshape(n, -1).view(self.torch.uint8)

    # 1. Head-group shapes/dtypes -------------------------------------------

    def test_head_group_shapes_and_dtypes(self):
        torch = self.torch
        view, _k, _v = self._bound_view()
        n = self.SIZE // self.PAGE_SIZE + 1  # 9
        for layer_id in (0, 2):
            groups = view.get_state_buffers(layer_id)
            self.assertEqual(len(groups), 2)
            self.assertEqual([g.head_begin for g in groups], [0, 2])
            self.assertEqual([g.num_heads for g in groups], [2, 2])
            for g in groups:
                self.assertEqual(tuple(g.ssm.shape), (n, 2, 128, 128))
                self.assertEqual(g.ssm.dtype, torch.float32)
                self.assertEqual(tuple(g.conv.shape), (n, *self.CONV_SHAPE))
                self.assertEqual(g.conv.dtype, torch.bfloat16)
            # The conv view is the WHOLE layer's window, repeated per group.
            self.assertEqual(groups[0].conv.data_ptr(), groups[1].conv.data_ptr())
            # This packing lands every ssm group on shard 0 and both conv
            # rows on shard 1 (conv_shard may differ from the ssm shard).
            self.assertEqual([g.shard for g in groups], [0, 0])
            self.assertEqual([g.conv_shard for g in groups], [1, 1])

    # 2. Byte roundtrip: view write -> raw slab bytes -----------------------

    def test_view_write_reads_back_from_raw_slab_bytes(self):
        torch = self.torch
        view, k, v = self._bound_view()
        table = view._bin_table
        # ssm (shard 0) and conv (shard 1) footprints overlap byte-wise in
        # the slab rows; runtime disambiguation is the page id (each shard
        # table hands out disjoint pages), so write them on distinct pages.
        ssm_page, conv_page = 3, 4

        groups0 = view.get_state_buffers(0)
        ssm_pattern = (
            torch.arange(2 * 128 * 128, dtype=torch.float32).reshape(2, 128, 128) / 7.0
        )
        conv_pattern = (
            torch.arange(self.CONV_SHAPE[0] * self.CONV_SHAPE[1], dtype=torch.float32)
            .reshape(self.CONV_SHAPE)
            .to(torch.bfloat16)
        )
        groups0[0].ssm[ssm_page] = ssm_pattern
        groups0[0].conv[conv_page] = conv_pattern

        # layer 0, head group 0 -> first ssm entry: shard 0, slot 0, K side,
        # byte offset 0. Its raw bytes in the K slab must match the pattern.
        entry = table.ssm_entries[0]
        self.assertEqual(
            (entry.state_layer, entry.head_begin, entry.kv_side, entry.slot),
            (0, 0, 0, 0),
        )
        raw = self._slab_u8(k[entry.slot])[
            ssm_page, entry.byte_offset : entry.byte_offset + entry.nbytes
        ]
        self.assertTrue(torch.equal(raw, ssm_pattern.reshape(-1).view(torch.uint8)))
        # Conv entry of layer 0: shard 1, slot 0, K side, byte offset 0.
        centry = table.conv_entries[0]
        self.assertEqual((centry.state_layer, centry.slot, centry.kv_side), (0, 0, 0))
        rawc = self._slab_u8(k[centry.slot])[
            conv_page, centry.byte_offset : centry.byte_offset + centry.nbytes
        ]
        self.assertTrue(torch.equal(rawc, conv_pattern.reshape(-1).view(torch.uint8)))
        # The null page (row 0) was never touched.
        self.assertEqual(int(self._slab_u8(k[0])[0].sum()), 0)

    # 3. Reverse roundtrip: raw slab bytes -> view read ---------------------

    def test_raw_slab_write_reads_back_through_view(self):
        torch = self.torch
        view, k, v = self._bound_view()
        table = view._bin_table
        page = 5

        # Second head group of layer 2 -> ssm entry 3: shard 0, slot 1, V.
        entry = table.ssm_entries[3]
        self.assertEqual(
            (entry.state_layer, entry.head_begin, entry.kv_side, entry.slot),
            (1, 2, 1, 1),
        )
        pattern = torch.randn(2, 128, 128, dtype=torch.float32)
        self._slab_u8(v[entry.slot])[
            page, entry.byte_offset : entry.byte_offset + entry.nbytes
        ] = pattern.reshape(-1).view(torch.uint8)
        got = view.get_state_buffers(2)[1].ssm[page]
        # Bit-exact: the fp32 pattern lives inside the bf16 slab as raw
        # bytes (reinterpret view, no cast), so nothing is lost.
        self.assertTrue(
            torch.equal(
                got.reshape(-1).view(torch.uint8).cpu(),
                pattern.reshape(-1).view(torch.uint8),
            )
        )

    # 4. Address stability ---------------------------------------------------

    def test_repeated_get_state_buffers_is_address_stable(self):
        view, _k, _v = self._bound_view()
        for layer_id in (0, 2):
            first = view.get_state_buffers(layer_id)
            second = view.get_state_buffers(layer_id)
            for g1, g2 in zip(first, second):
                self.assertIs(g1.ssm, g2.ssm)
                self.assertIs(g1.conv, g2.conv)
                self.assertEqual(g1.ssm.data_ptr(), g2.ssm.data_ptr())
                self.assertEqual(g1.conv.data_ptr(), g2.conv.data_ptr())

    def test_views_alias_slab_storage(self):
        view, k, v = self._bound_view()
        slab_ptrs = {t.data_ptr(): t for t in k + v}
        for layer_id in (0, 2):
            for g in view.get_state_buffers(layer_id):
                for t in (g.ssm, g.conv):
                    inside = any(
                        s.data_ptr() <= t.data_ptr() < s.data_ptr() + s.nbytes
                        for s in slab_ptrs.values()
                    )
                    self.assertTrue(inside, "view does not alias any K/V slab")

    # 5. Gate ----------------------------------------------------------------

    def test_gate_inactive_without_bin_table(self):
        view = self._view(bin_table=None)
        self.assertFalse(view.is_active)
        self.assertEqual(view.state_layer_ids, frozenset())
        self.assertFalse(view.is_state_layer(0))
        with self.assertRaisesRegex(ValueError, r"no state\s+views were bound"):
            view.get_state_buffers(0)
        with self.assertRaisesRegex(ValueError, r"inactive"):
            view.bind(*self._slabs())

    def test_gate_inactive_on_radix_ext(self):
        view = self._view(flat_gate=False)
        self.assertFalse(view.is_active)
        self.assertEqual(view.state_layer_ids, frozenset())

    def test_gate_active_and_non_state_layer_raises(self):
        view, _k, _v = self._bound_view()
        self.assertTrue(view.is_active)
        self.assertEqual(view.state_layer_ids, frozenset({0, 2}))
        self.assertTrue(view.is_state_layer(0))
        self.assertFalse(view.is_state_layer(1))
        with self.assertRaisesRegex(ValueError, r"not a state layer"):
            view.get_state_buffers(1)
        with self.assertRaisesRegex(ValueError, r"not a state layer"):
            view.get_state_buffers(3)

    def test_bin_table_page_size_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, r"must match"):
            torch = self.torch
            with mock.patch(_FLAT_GATE, return_value=True):
                self.StateShardView(
                    bin_table=self._bin_table(),  # solved for P=256
                    layer_types=self.LAYER_TYPES,
                    conv_state_shape=self.CONV_SHAPE,
                    temporal_state_shape=self.TEMPORAL_SHAPE,
                    conv_dtype=torch.bfloat16,
                    ssm_dtype=torch.float32,
                    default_dtype=torch.bfloat16,
                    page_size=128,
                    size=self.SIZE,
                )

    # 6. No-overlap within a shard -------------------------------------------

    def test_entries_of_one_shard_never_overlap(self):
        # Entries sharing a shard live on the SAME runtime pages, so their
        # (slot, kv_side, byte range) footprints must be disjoint. Write a
        # byte-uniform marker through every view, each shard on its own
        # page row (cross-shard aliasing at equal offsets is fine: shard
        # tables hand out disjoint page ids). Marked bytes across both
        # slabs must equal the sum of entry sizes -- any overlap within a
        # shard would collapse marks and undershoot.
        torch = self.torch
        view, k, v = self._bound_view()
        table = view._bin_table
        marker_f32 = torch.tensor([0x3F3F3F3F], dtype=torch.int32).view(torch.float32)[
            0
        ]
        marker_bf16 = torch.tensor([0x3F3F], dtype=torch.int16).view(torch.bfloat16)[0]

        for layer_id in (0, 2):
            for g in view.get_state_buffers(layer_id):
                g.ssm[1 + g.shard].fill_(marker_f32)
                g.conv[1 + g.conv_shard].fill_(marker_bf16)

        marked = sum(int((self._slab_u8(slab) == 0x3F).sum()) for slab in k + v)
        expected = sum(e.nbytes for e in table.ssm_entries + table.conv_entries)
        self.assertEqual(marked, expected)

    # 7. Bind guards ---------------------------------------------------------

    def test_repeated_bind_raises(self):
        view, k, v = self._bound_view()
        with self.assertRaisesRegex(RuntimeError, r"already bound"):
            view.bind(k, v)

    def test_bind_asymmetric_kv_lengths_raises(self):
        view = self._view()
        k, v = self._slabs()
        with self.assertRaisesRegex(ValueError, r"K slabs but"):
            view.bind(k, v[:-1])

    def test_bind_duplicate_slab_tensor_raises(self):
        view = self._view()
        k, v = self._slabs()
        # Alias a K slab into the V list: two slot segments would share
        # storage, silently colliding.
        v[0] = k[0]
        with self.assertRaisesRegex(ValueError, r"duplicate slab tensor"):
            view.bind(k, v)

    # 8. Head-group layout invariants (backend decode-loop contract) ---------

    def _mutated_bin_table(self, mutate_ssm_entries):
        import dataclasses

        table = self._bin_table()
        return dataclasses.replace(
            table, ssm_entries=tuple(mutate_ssm_entries(table.ssm_entries))
        )

    def test_bind_rejects_gapped_head_groups(self):
        # Shift layer 0's first ssm group off head 0: the groups no longer
        # tile [0, heads) from head_begin 0, the invariant the backend's
        # decode loop (conv update off groups[0]) relies on.
        import dataclasses

        def mutate(entries):
            return [
                (
                    dataclasses.replace(e, head_begin=1)
                    if e.state_layer == 0 and e.head_begin == 0
                    else e
                )
                for e in entries
            ]

        view = self._view(bin_table=self._mutated_bin_table(mutate))
        with self.assertRaisesRegex(ValueError, r"contiguously"):
            view.bind(*self._slabs())

    def test_bind_rejects_missing_head_group(self):
        # Drop layer 0's tail group: coverage falls short of the layer's
        # ssm head count.
        def mutate(entries):
            return [
                e for e in entries if not (e.state_layer == 0 and e.head_begin == 2)
            ]

        view = self._view(bin_table=self._mutated_bin_table(mutate))
        with self.assertRaisesRegex(ValueError, r"cover"):
            view.bind(*self._slabs())


if __name__ == "__main__":
    unittest.main()
