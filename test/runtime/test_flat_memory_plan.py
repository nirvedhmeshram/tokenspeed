from __future__ import annotations

import dataclasses
import importlib.util
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_CONFIGS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "configs"
)


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _CONFIGS_DIR / file_name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: on py3.9 @dataclass + `from __future__ import
    # annotations` resolves field types via sys.modules[cls.__module__].
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_fmp = _load("flat_memory_plan_under_test", "flat_memory_plan.py")
ComponentSpec = _fmp.ComponentSpec
BlockGeometry = _fmp.BlockGeometry
solve_page_geometry = _fmp.solve_page_geometry
plan_tensors = _fmp.plan_tensors
plan_component_tensors = _fmp.plan_component_tensors
components_from_layers = _fmp.components_from_layers
shard_bin_table = _fmp.shard_bin_table
ShardBinEntry = _fmp.ShardBinEntry
StateShardBinTable = _fmp.StateShardBinTable
head_addressing_maps = _fmp.head_addressing_maps
StateLayerHeadMap = _fmp.StateLayerHeadMap
solve_base_block_geometry = _fmp.solve_base_block_geometry
BaseBlockGeometry = _fmp.BaseBlockGeometry


class FlatStateCapacityTest(unittest.TestCase):
    def _plan(self, *, num_blocks=101, block_bytes=10_000):
        return _fmp.FlatMemoryPlan(
            geometry=BlockGeometry(
                block_size=TP8_KW["block_size"],
                block_bytes=block_bytes,
                num_blocks=num_blocks,
            ),
            tensors=(),
        )

    def test_capacity_uses_physical_plan_and_state_bin_geometry(self):
        table = shard_bin_table(**TP8_KW)
        capacity = _fmp.flat_state_capacity_from_plan(
            self._plan(),
            state_bin_table=table,
            verify_width=4,
            num_page_ids=81,
        )

        self.assertEqual(capacity.physical_page_bytes, 10_000)
        self.assertEqual(
            capacity.logical_state_payload_bytes_per_checkpoint, 30 * (4 * 65536 + 6144)
        )
        self.assertEqual(capacity.usable_page_ids, 80)
        self.assertEqual(capacity.state_shard_count, 4)
        self.assertEqual(capacity.canonical_pages_per_request, 8)
        self.assertEqual(capacity.speculative_pages_per_request, 16)
        self.assertEqual(capacity.total_state_pages_per_request, 24)
        self.assertEqual(capacity.state_limited_request_concurrency, 3)

    def test_width_one_has_no_speculative_page_charge(self):
        capacity = _fmp.flat_state_capacity_from_plan(
            self._plan(),
            state_bin_table=shard_bin_table(**TP8_KW),
            verify_width=1,
            num_page_ids=81,
        )

        self.assertEqual(capacity.speculative_pages_per_request, 0)
        self.assertEqual(capacity.total_state_pages_per_request, 8)
        self.assertEqual(capacity.state_limited_request_concurrency, 10)

    def test_graph_gate_caps_only_effective_capture_max(self):
        capacity = _fmp.flat_state_capacity_from_plan(
            self._plan(),
            state_bin_table=shard_bin_table(**TP8_KW),
            verify_width=4,
            num_page_ids=81,
        )
        config = SimpleNamespace(
            max_cudagraph_capture_size=64,
            max_num_seqs=256,
            cudagraph_capture_sizes=[1, 2, 4, 8],
        )

        reduction = _fmp.apply_flat_state_capacity_graph_gate(config, capacity)

        self.assertEqual(reduction, (64, 3))
        self.assertEqual(config.max_cudagraph_capture_size, 3)
        self.assertEqual(config.max_num_seqs, 256)
        self.assertEqual(config.cudagraph_capture_sizes, [1, 2, 4, 8])

    def test_graph_gate_keeps_one_reachable_explicit_capture_bucket(self):
        capacity = _fmp.flat_state_capacity_from_plan(
            self._plan(),
            state_bin_table=shard_bin_table(**TP8_KW),
            verify_width=4,
            num_page_ids=81,
        )
        config = SimpleNamespace(
            max_cudagraph_capture_size=64,
            cudagraph_capture_sizes=[4, 8],
        )

        _fmp.apply_flat_state_capacity_graph_gate(config, capacity)

        self.assertEqual(config.cudagraph_capture_sizes, [4, 8, 3])
        reachable = [
            size
            for size in config.cudagraph_capture_sizes
            if 0 < size <= config.max_cudagraph_capture_size
        ]
        self.assertEqual(reachable, [3])

    def test_graph_gate_does_not_repair_preexisting_empty_capture_set(self):
        capacity = _fmp.flat_state_capacity_from_plan(
            self._plan(),
            state_bin_table=shard_bin_table(**TP8_KW),
            verify_width=4,
            num_page_ids=241,
        )
        config = SimpleNamespace(
            max_cudagraph_capture_size=3,
            cudagraph_capture_sizes=[4, 8],
        )

        reduction = _fmp.apply_flat_state_capacity_graph_gate(config, capacity)

        self.assertEqual(reduction, (3, 3))
        self.assertEqual(config.cudagraph_capture_sizes, [4, 8])

    def test_graph_gate_fails_when_one_request_cannot_fit(self):
        capacity = _fmp.flat_state_capacity_from_plan(
            self._plan(),
            state_bin_table=shard_bin_table(**TP8_KW),
            verify_width=4,
            num_page_ids=24,
        )
        config = SimpleNamespace(max_cudagraph_capture_size=64)

        with self.assertRaisesRegex(RuntimeError, "cannot represent one request"):
            _fmp.apply_flat_state_capacity_graph_gate(config, capacity)

    def test_graph_gate_is_noop_without_mtp(self):
        config = SimpleNamespace(max_cudagraph_capture_size=64)
        width_one = _fmp.flat_state_capacity_from_plan(
            self._plan(),
            state_bin_table=shard_bin_table(**TP8_KW),
            verify_width=1,
            num_page_ids=81,
        )

        self.assertIsNone(_fmp.apply_flat_state_capacity_graph_gate(config, None))
        self.assertIsNone(_fmp.apply_flat_state_capacity_graph_gate(config, width_one))
        self.assertEqual(config.max_cudagraph_capture_size, 64)


class BaseBlockGeometryTest(unittest.TestCase):
    """Shared base-block sizing: one 64KB homogeneous block divides both the
    per-layer ssm bytes and the per-full-layer KV bytes/token across Qwen3.5
    sizes/TPs (the dual-quantum coincidence)."""

    HEAD_CELL = 128 * 128 * 4  # ssm head cell, fp32 = 64KB, same for 35B/397B

    def test_397b_tp8_dual_quantum_coincides(self):
        # 397B-A17B tp8: 8 ssm heads/layer -> 512KB; full-layer KV 1024 B/token.
        geo = solve_base_block_geometry(
            ssm_bytes_per_layer=8 * self.HEAD_CELL,
            kv_bytes_per_token=1024,
            base_block_bytes=self.HEAD_CELL,
        )
        self.assertEqual(geo.base_block_bytes, 65536)
        self.assertEqual(geo.blocks_per_state_layer, 8)
        self.assertEqual(
            geo.kv_tokens_per_block, 64
        )  # KV P=64 (was 512 under equalizer)

    def test_35b_tp8_smaller_n(self):
        # 35B-A3B tp8: 4 ssm heads/layer -> 256KB; KV 1024 B/token.
        geo = solve_base_block_geometry(
            ssm_bytes_per_layer=4 * self.HEAD_CELL,
            kv_bytes_per_token=1024,
            base_block_bytes=self.HEAD_CELL,
        )
        self.assertEqual(geo.blocks_per_state_layer, 4)
        self.assertEqual(geo.kv_tokens_per_block, 64)

    def test_35b_tp1_larger_n_smaller_kv_grain(self):
        # 35B-A3B tp1: 32 ssm heads/layer -> 2MB; KV 2048 B/token (tp1).
        geo = solve_base_block_geometry(
            ssm_bytes_per_layer=32 * self.HEAD_CELL,
            kv_bytes_per_token=2048,
            base_block_bytes=self.HEAD_CELL,
        )
        self.assertEqual(geo.blocks_per_state_layer, 32)
        self.assertEqual(geo.kv_tokens_per_block, 32)  # 64KB / 2048

    def test_397b_tp4_dual_quantum_coincides(self):
        # 397B tp4: 16 ssm heads/layer -> 1MB; KV 1024 B/token.
        geo = solve_base_block_geometry(
            ssm_bytes_per_layer=16 * self.HEAD_CELL,
            kv_bytes_per_token=1024,
            base_block_bytes=self.HEAD_CELL,
        )
        self.assertEqual(geo.blocks_per_state_layer, 16)
        self.assertEqual(geo.kv_tokens_per_block, 64)

    def test_ssm_not_divisible_raises(self):
        with self.assertRaisesRegex(ValueError, "state layer would need padding"):
            solve_base_block_geometry(
                ssm_bytes_per_layer=100_000,  # not a multiple of 64KB
                kv_bytes_per_token=1024,
                base_block_bytes=self.HEAD_CELL,
            )

    def test_kv_not_whole_tokens_raises(self):
        with self.assertRaisesRegex(ValueError, "whole number of KV"):
            solve_base_block_geometry(
                ssm_bytes_per_layer=8 * self.HEAD_CELL,
                kv_bytes_per_token=1500,  # 64KB not a whole number of these
                base_block_bytes=self.HEAD_CELL,
            )

    def test_nonpositive_base_raises(self):
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            solve_base_block_geometry(
                ssm_bytes_per_layer=8 * self.HEAD_CELL,
                kv_bytes_per_token=1024,
                base_block_bytes=0,
            )


class EqualizerTest(unittest.TestCase):
    def test_gpt_oss_degenerate_keeps_page_size(self):
        comps = [
            ComponentSpec(
                group_id="full_attention",
                layer=0,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            ),
            ComponentSpec(
                group_id="sliding_attention",
                layer=1,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            ),
        ]
        geo = solve_page_geometry(comps, block_size=16, alignment=256)
        self.assertEqual(geo.block_size, 16)
        self.assertEqual(geo.block_bytes, 16 * 1024)

    def test_qwen35_constant_state_inflates_page_size(self):
        comps = [
            ComponentSpec(
                group_id="full_attention",
                layer=0,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            ),
            ComponentSpec(
                group_id="linear_attention",
                layer=1,
                component="conv",
                bytes_per_slot=0,
                const_bytes=40 * 1024,
            ),
            ComponentSpec(
                group_id="linear_attention",
                layer=1,
                component="ssm",
                bytes_per_slot=0,
                const_bytes=60 * 1024,
            ),
        ]
        geo = solve_page_geometry(comps, block_size=16, alignment=4)
        # A state layer's components pack into ONE page row ([conv|ssm|pad]),
        # so the constant demand is their SUM: ceil((40+60)KiB / 1KiB) = 100.
        self.assertEqual(geo.block_size, 100)
        self.assertEqual(geo.block_bytes, 100 * 1024)

    def test_inflation_rounds_up_to_alignment(self):
        comps = [
            ComponentSpec(
                "full_attention", 0, "kv", bytes_per_slot=1024, const_bytes=0
            ),
            ComponentSpec(
                "linear_attention",
                1,
                "state",
                bytes_per_slot=0,
                const_bytes=101 * 1024,
            ),
        ]
        geo = solve_page_geometry(comps, block_size=16, alignment=16)
        # ceil(101K / 1K) = 101 -> rounded up to the next multiple of 16.
        self.assertEqual(geo.block_size, 112)
        self.assertEqual(geo.block_bytes, 112 * 1024)

    def test_dsv4_linear_rows_pad_not_inflate(self):
        comps = [
            ComponentSpec("full_mla", 0, "latent", bytes_per_slot=1152, const_bytes=0),
            ComponentSpec(
                "full_mla", 0, "indexer_k", bytes_per_slot=132, const_bytes=0
            ),
        ]
        geo = solve_page_geometry(comps, block_size=64, alignment=256)
        self.assertEqual(geo.block_size, 64)
        # Same-layer components pack into one row.
        self.assertEqual(geo.block_bytes, 64 * (1152 + 132))

    def test_constant_components_require_a_linear_row(self):
        comps = [
            ComponentSpec(
                "linear_attention", 0, "state", bytes_per_slot=0, const_bytes=1024
            )
        ]
        with self.assertRaises(ValueError):
            solve_page_geometry(comps, block_size=16, alignment=4)


class PlanTensorsTest(unittest.TestCase):
    def _comps_qwen35(self):
        return [
            ComponentSpec(
                "full_attention",
                layer=0,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            ),
            ComponentSpec(
                "full_attention",
                layer=1,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            ),
            ComponentSpec(
                "linear_attention",
                layer=0,
                component="conv",
                bytes_per_slot=0,
                const_bytes=40 * 1024,
            ),
            ComponentSpec(
                "linear_attention",
                layer=0,
                component="ssm",
                bytes_per_slot=0,
                const_bytes=60 * 1024,
            ),
        ]

    def test_slot_pairing_one_layer_per_group_per_slot(self):
        plan = plan_tensors(
            self._comps_qwen35(),
            block_size=16,
            alignment=4,
            budget_bytes=100 * 1024 * 1024,
        )
        self.assertEqual(len(plan.tensors), 2)
        slot0 = plan.tensors[0]
        self.assertEqual(
            {(b.group_id, b.layer) for b in slot0.bindings},
            {("full_attention", 0), ("linear_attention", 0)},
        )
        slot1 = plan.tensors[1]
        self.assertEqual(
            {(b.group_id, b.layer) for b in slot1.bindings},
            {("full_attention", 1)},
        )
        for t in plan.tensors:
            seen = {}
            for b in t.bindings:
                key = (b.slot, b.group_id)
                self.assertEqual(seen.setdefault(key, b.layer), b.layer)

    def test_row_offsets_accumulate_within_a_row(self):
        plan = plan_tensors(
            self._comps_qwen35(),
            block_size=16,
            alignment=4,
            budget_bytes=100 * 1024 * 1024,
        )
        state = [
            b for b in plan.tensors[0].bindings if b.group_id == "linear_attention"
        ]
        by_comp = {b.component: b for b in state}
        self.assertEqual(by_comp["conv"].row_offset, 0)
        self.assertEqual(by_comp["ssm"].row_offset, 40 * 1024)
        full = [b for b in plan.tensors[0].bindings if b.group_id == "full_attention"]
        self.assertEqual(full[0].row_offset, 0)

    def test_num_blocks_from_budget_shared_across_slots(self):
        plan = plan_tensors(
            self._comps_qwen35(),
            block_size=16,
            alignment=4,
            budget_bytes=100 * 1024 * 1024,
        )
        geo = plan.geometry
        self.assertEqual(geo.block_size, 100)
        self.assertEqual(geo.block_bytes, 300 * 1024)
        self.assertEqual(geo.num_blocks, 100 * 1024 * 1024 // (300 * 1024))  # 341
        slot0, slot1 = plan.tensors
        self.assertEqual(slot0.nbytes, geo.num_blocks * 200 * 1024)
        self.assertEqual(slot1.nbytes, geo.num_blocks * 100 * 1024)

    def test_gpt_oss_pairing_matches_hybrid_slab(self):
        comps = [
            ComponentSpec(
                "full_attention",
                layer=i,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            )
            for i in range(2)
        ]
        comps += [
            ComponentSpec(
                "sliding_attention",
                layer=i,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            )
            for i in range(2)
        ]
        plan = plan_tensors(
            comps, block_size=16, alignment=4, budget_bytes=64 * 1024 * 1024
        )
        self.assertEqual(len(plan.tensors), 2)
        for t in plan.tensors:
            self.assertEqual(
                {b.group_id for b in t.bindings},
                {"full_attention", "sliding_attention"},
            )

    def test_budget_too_small_raises(self):
        with self.assertRaises(ValueError):
            plan_tensors(
                self._comps_qwen35(),
                block_size=16,
                alignment=4,
                budget_bytes=100 * 1024,
            )

    def test_cross_group_rows_sized_by_own_bindings(self):
        comps = [
            ComponentSpec("full", 0, "kv", 100, 0),
            ComponentSpec("state", 0, "conv", 0, 300),
            ComponentSpec("state", 1, "conv", 0, 300),
        ]
        plan = plan_tensors(comps, block_size=4, alignment=1, budget_bytes=100_000)
        # slot0 packs 100*4 + 300 = 700, slot1 packs 300.
        self.assertEqual(plan.geometry.num_blocks, 100_000 // 1000)
        slot0, slot1 = plan.tensors
        self.assertEqual(slot0.nbytes, plan.geometry.num_blocks * 700)
        self.assertEqual(slot1.nbytes, plan.geometry.num_blocks * 300)


class PlanComponentTensorsTest(unittest.TestCase):
    def test_qwen_shape(self):
        kv_per_slot = 2048
        state = {"conv": 848_256, "ssm": 1_298_048}
        layers = (["linear_attention"] * 3 + ["full_attention"]) * 12
        comps = components_from_layers(
            layer_types=layers,
            kv_bytes_per_slot=kv_per_slot,
            state_const_bytes=state,
        )
        plan = plan_component_tensors(comps, block_size=1088, budget_bytes=10 * 1024**3)
        row_sum = 12 * 1088 * kv_per_slot + 36 * sum(state.values())
        self.assertEqual(plan.geometry.num_blocks, (10 * 1024**3) // row_sum)
        self.assertGreaterEqual(plan.geometry.num_blocks, 100)
        self.assertEqual(len(plan.tensors), 12 + 72)
        for t in plan.tensors:
            (b,) = t.bindings
            self.assertEqual(b.row_offset, 0)
            self.assertEqual(t.nbytes, plan.geometry.num_blocks * b.nbytes_per_block)

    def test_reserved_bytes_shrink_blocks(self):
        comps = components_from_layers(
            layer_types=["full_attention"] * 2,
            kv_bytes_per_slot=100,
            state_const_bytes={},
        )
        base = plan_component_tensors(comps, block_size=4, budget_bytes=10_000)
        tighter = plan_component_tensors(
            comps, block_size=4, budget_bytes=10_000, reserved_bytes_per_block=800
        )
        self.assertEqual(base.geometry.num_blocks, 10_000 // 800)
        self.assertEqual(tighter.geometry.num_blocks, 10_000 // 1600)

    def test_budget_too_small_raises(self):
        comps = components_from_layers(
            layer_types=["full_attention"],
            kv_bytes_per_slot=100,
            state_const_bytes={},
        )
        with self.assertRaises(ValueError):
            plan_component_tensors(comps, block_size=4, budget_bytes=500)


class GptOssCapacityTest(unittest.TestCase):
    def test_plan_counts_every_layer_row(self):
        comps = [
            ComponentSpec(
                "full_attention",
                layer=i,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            )
            for i in range(24)
        ]
        comps += [
            ComponentSpec(
                "sliding_attention",
                layer=i,
                component="kv",
                bytes_per_slot=1024,
                const_bytes=0,
            )
            for i in range(24)
        ]
        budget = 10 * 1024**3
        plan = plan_tensors(comps, block_size=16, alignment=4, budget_bytes=budget)
        self.assertEqual(plan.geometry.num_blocks, budget // (48 * 16 * 1024))


class ComponentsFromLayersTest(unittest.TestCase):
    def test_qwen35_shape(self):
        comps = components_from_layers(
            layer_types=["linear_attention", "full_attention", "linear_attention"],
            kv_bytes_per_slot=1024,
            state_const_bytes={"conv": 40 * 1024, "ssm": 60 * 1024},
        )
        by_key = {(c.group_id, c.layer, c.component): c for c in comps}
        self.assertIn(("full_attention", 0, "kv"), by_key)
        self.assertIn(("linear_attention", 0, "conv"), by_key)
        self.assertIn(("linear_attention", 1, "ssm"), by_key)
        self.assertEqual(by_key[("full_attention", 0, "kv")].bytes_per_slot, 1024)
        self.assertEqual(by_key[("linear_attention", 1, "conv")].const_bytes, 40 * 1024)

    def test_pure_attention_model_has_no_state_components(self):
        comps = components_from_layers(
            layer_types=["full_attention", "sliding_attention"],
            kv_bytes_per_slot=512,
            state_const_bytes={},
        )
        self.assertTrue(all(c.const_bytes == 0 for c in comps))
        self.assertEqual(len(comps), 2)

    def test_plan_end_to_end_from_layers(self):
        comps = components_from_layers(
            layer_types=["linear_attention", "full_attention"],
            kv_bytes_per_slot=1024,
            state_const_bytes={"conv": 40 * 1024, "ssm": 60 * 1024},
        )
        plan = plan_tensors(
            comps, block_size=16, alignment=4, budget_bytes=100 * 1024 * 1024
        )
        self.assertEqual(plan.geometry.block_size, 100)  # inflated by the state row


TP8_KW = dict(  # Qwen3.5-35B-A3B @ tp8 真值
    num_full_layers=10,
    num_state_layers=30,
    ssm_heads_per_layer=4,
    ssm_head_bytes=65536,  # 128*128*fp32
    conv_bytes_per_layer=6144,  # 49152/8
    kv_cell_bytes_per_tok=1024,  # K+V 一层一 tok
    block_size=256,
)


class ShardBinTableTest(unittest.TestCase):
    def test_bin_table_tp8_truth(self):
        bt = shard_bin_table(**TP8_KW)
        self.assertEqual(bt.num_shards, 4)
        self.assertEqual(bt.segment_bytes, 512 * 256)
        self.assertEqual(bt.heads_per_group, 2)
        ssm_by_layer = {}
        for e in bt.ssm_entries:
            ssm_by_layer.setdefault(e.state_layer, []).append(e)
        self.assertEqual(len(ssm_by_layer), 30)
        self.assertTrue(all(len(v) == 2 for v in ssm_by_layer.values()))
        self.assertEqual(len(bt.conv_entries), 30)
        for e in bt.ssm_entries + bt.conv_entries:
            self.assertTrue(0 <= e.shard < 4)
            self.assertTrue(0 <= e.slot < 10)
            self.assertIn(e.kv_side, (0, 1))
            self.assertLessEqual(e.byte_offset + e.nbytes, bt.segment_bytes)
        # Golden enumeration order: side varies fastest, then slot, then shard.
        self.assertEqual(
            [(e.shard, e.slot, e.kv_side) for e in bt.ssm_entries[:3]],
            [(0, 0, 0), (0, 0, 1), (0, 1, 0)],
        )
        e0 = bt.conv_entries[0]
        self.assertEqual((e0.shard, e0.slot, e0.kv_side, e0.byte_offset), (3, 0, 0, 0))

    def test_bin_table_no_overlap_tp8(self):
        bt = shard_bin_table(**TP8_KW)
        spans = []
        for e in bt.ssm_entries + bt.conv_entries:
            key = (e.shard, e.slot, e.kv_side)
            for k2, a, b in spans:
                if k2 == key:
                    self.assertTrue(e.byte_offset >= b or e.byte_offset + e.nbytes <= a)
            spans.append((key, e.byte_offset, e.byte_offset + e.nbytes))

    def test_bin_table_k_parametrized(self):
        cases = [
            (8, 1024, 30, 4, 6144, 256, 4),
            (8, 1024, 30, 4, 6144, 128, 7),
            (4, 1024, 30, 8, 12288, 256, 7),
            (1, 2048, 30, 32, 49152, 256, 13),
            (16, 1024, 30, 2, 3072, 256, 2),
        ]
        for tp, cell, s_layer, heads, conv, p, expect_k in cases:
            with self.subTest(tp=tp, cell=cell, block_size=p):
                bt = shard_bin_table(
                    num_full_layers=10,
                    num_state_layers=s_layer,
                    ssm_heads_per_layer=heads,
                    ssm_head_bytes=65536,
                    conv_bytes_per_layer=conv,
                    kv_cell_bytes_per_tok=cell,
                    block_size=p,
                )
                self.assertEqual(bt.num_shards, expect_k)

    def test_bin_table_rejects_non64_block(self):
        with self.assertRaisesRegex(ValueError, "multiple of 64"):
            shard_bin_table(**{**TP8_KW, "block_size": 200})

    def test_bin_table_rejects_segment_below_one_cell(self):
        with self.assertRaisesRegex(ValueError, "one ssm head cell"):
            shard_bin_table(**{**TP8_KW, "block_size": 64})

    def test_bin_table_rejects_conv_row_over_segment(self):
        with self.assertRaisesRegex(ValueError, "conv row exceeds one segment"):
            shard_bin_table(**{**TP8_KW, "conv_bytes_per_layer": 512 * 256 + 1})

    def test_bin_table_rejects_zero_layers(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            shard_bin_table(**{**TP8_KW, "num_state_layers": 0})
        with self.assertRaisesRegex(ValueError, "at least one"):
            shard_bin_table(**{**TP8_KW, "num_full_layers": 0})

    def test_bin_table_rejects_odd_kv_cell(self):
        with self.assertRaisesRegex(ValueError, "even"):
            shard_bin_table(**{**TP8_KW, "kv_cell_bytes_per_tok": 1023})


SSM_HEAD_ELEMS = 128 * 128  # temporal_state_shape[1:] = (128, 128); fp32 -> itemsize 4


class HeadAddressingTest(unittest.TestCase):
    def test_tp8_per_head_maps_match_ssm_entries(self):
        bt = shard_bin_table(**TP8_KW)
        maps = head_addressing_maps(bt, ssm_head_elems=SSM_HEAD_ELEMS)
        # 30 state layers -> 30 maps, each covering heads_per_layer == 4 heads.
        self.assertEqual(len(maps), 30)
        self.assertEqual(len(maps[0].head_shard), 4)
        self.assertEqual(len(maps[0].head_elem_offset), 4)
        # Hand calc from the real bin table (printed ssm_entries[:2] of layer 0):
        #   entry0: head_begin=0 num_heads=2 shard=0 nbytes=131072 byte_offset=0
        #   entry1: head_begin=2 num_heads=2 shard=0 nbytes=131072 byte_offset=0
        # itemsize = (131072//2)//16384 = 65536//16384 = 4 bytes (fp32).
        # entry0 elem_base = 0//4 = 0 -> heads 0,1 at 0, 0+16384.
        # entry1 elem_base = 0//4 = 0 -> heads 2,3 at 0, 0+16384.
        # Both groups sit at byte_offset 0 of their (distinct kv_side) segments,
        # so heads 0 and 2 both land at element 0 within their shard's ssm view.
        self.assertEqual(maps[0].head_shard, (0, 0, 0, 0))
        self.assertEqual(maps[0].head_elem_offset, (0, 16384, 0, 16384))
        # Two heads within one group differ by exactly one ssm head cell.
        self.assertEqual(
            maps[0].head_elem_offset[1] - maps[0].head_elem_offset[0], SSM_HEAD_ELEMS
        )
        self.assertEqual(
            maps[0].head_elem_offset[3] - maps[0].head_elem_offset[2], SSM_HEAD_ELEMS
        )

    def test_tp1_32_heads_span_multiple_shards(self):
        # tp1: 32 ssm heads/layer, H_g=4 -> 8 head groups per state layer.
        # num_full_layers=1 gives segs_per_shard=2*1=2, so consecutive groups
        # spread across shards fastest; the layer's 8 groups cover 4 distinct
        # shards (the K and V side of each shard-slab pair share a shard, so 8
        # groups can never reach 8 distinct shards here).
        kw = dict(
            num_full_layers=1,
            num_state_layers=30,
            ssm_heads_per_layer=32,
            ssm_head_bytes=65536,
            conv_bytes_per_layer=49152,
            kv_cell_bytes_per_tok=2048,
            block_size=256,
        )
        bt = shard_bin_table(**kw)
        self.assertEqual(bt.heads_per_group, 4)
        maps = head_addressing_maps(bt, ssm_head_elems=SSM_HEAD_ELEMS)
        self.assertEqual(len(maps[0].head_shard), 32)
        self.assertEqual(sorted(set(maps[0].head_shard)), [0, 1, 2, 3])
        # Within each 4-head group the offsets tile one cell apart from base 0.
        for begin in range(0, 32, 4):
            grp = maps[0].head_elem_offset[begin : begin + 4]
            self.assertEqual(grp, tuple(h * SSM_HEAD_ELEMS for h in range(4)))

    def test_non_contiguous_head_groups_raise(self):
        bt = shard_bin_table(**TP8_KW)
        # Drop layer 0's first ssm group (head_begin=0) so the surviving group
        # starts at head_begin=2 -> head order no longer tiles from 0.
        pruned = tuple(
            e for e in bt.ssm_entries if not (e.state_layer == 0 and e.head_begin == 0)
        )
        holed = dataclasses.replace(bt, ssm_entries=pruned)
        with self.assertRaisesRegex(ValueError, "contiguously"):
            head_addressing_maps(holed, ssm_head_elems=SSM_HEAD_ELEMS)

    def test_dropped_tail_head_group_raises(self):
        # Use tp1's 8-groups-per-layer table so a state layer has a genuine tail
        # group (head_begin=28) to drop; the survivors still tile [0, 28)
        # contiguously, so only the tail-coverage check can catch the loss.
        kw = dict(
            num_full_layers=1,
            num_state_layers=30,
            ssm_heads_per_layer=32,
            ssm_head_bytes=65536,
            conv_bytes_per_layer=49152,
            kv_cell_bytes_per_tok=2048,
            block_size=256,
        )
        bt = shard_bin_table(**kw)
        last_begin = max(e.head_begin for e in bt.ssm_entries if e.state_layer == 0)
        pruned = tuple(
            e
            for e in bt.ssm_entries
            if not (e.state_layer == 0 and e.head_begin == last_begin)
        )
        docked = dataclasses.replace(bt, ssm_entries=pruned)
        with self.assertRaisesRegex(ValueError, "tail head group is missing"):
            head_addressing_maps(docked, ssm_head_elems=SSM_HEAD_ELEMS)

    def test_single_shard_degenerate_flat_offsets(self):
        # k=1: one segment holds all heads of a layer (heads_per_group >= heads),
        # so every head maps to shard 0 with a plain per-cell stride.
        kw = dict(
            num_full_layers=16,
            num_state_layers=30,
            ssm_heads_per_layer=4,
            ssm_head_bytes=65536,
            conv_bytes_per_layer=6144,
            kv_cell_bytes_per_tok=4096,
            block_size=256,
        )
        bt = shard_bin_table(**kw)
        self.assertEqual(bt.num_shards, 1)
        maps = head_addressing_maps(bt, ssm_head_elems=SSM_HEAD_ELEMS)
        self.assertEqual(maps[0].head_shard, (0, 0, 0, 0))
        self.assertEqual(
            maps[0].head_elem_offset,
            tuple(h * SSM_HEAD_ELEMS for h in range(4)),
        )


if __name__ == "__main__":
    unittest.main()
