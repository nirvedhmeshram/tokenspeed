"""Flat assembly line for GDN hybrids (M17 C4, M18c state binning).

Two contracts: the Qwen3.5 config exposes ``layer_types`` in the
paged-cache label vocabulary; and an MHAConfig carrying state shapes plus
a state shard bin table builds a full-coverage pool whose GDN state is
head-group reinterpret views over the full layers' K/V slabs, with the
full group and the k ``linear_attention_shard{i}`` groups published.
Flat GDN sizing itself is plan-driven (plan_component_tensors,
test_flat_memory_plan); the bin packing is shard_bin_table's
(test_flat_memory_plan too).
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")

_CONFIGS_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "configs"
)

_PKG_FLAT_PROBE = (
    "tokenspeed.runtime.configs.paged_cache_spec.scheduler_ext_flat_kvcache"
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


_plan = _load("flat_memory_plan_gdn_assembly_under_test", "flat_memory_plan.py")
shard_bin_table = _plan.shard_bin_table


class Qwen3_5LayerTypesTest(unittest.TestCase):
    """The config's layer_types property (interleaving + label vocabulary)."""

    def setUp(self):
        try:
            from tokenspeed.runtime.configs.qwen3_5_text_base_config import (
                Qwen3_5BaseTextConfig,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + transformers: {exc}")
        self.config_cls = Qwen3_5BaseTextConfig

    def test_layer_types_interleaving(self):
        cfg = self.config_cls(num_hidden_layers=8, full_attention_interval=4)
        self.assertEqual(
            cfg.layer_types,
            (["linear_attention"] * 3 + ["full_attention"]) * 2,
        )

    def test_layers_block_type_keeps_checkpoint_label(self):
        # models/qwen3_5.py keys layer construction on the checkpoint's
        # "attention" label; layer_types must not change it.
        cfg = self.config_cls(num_hidden_layers=4, full_attention_interval=4)
        self.assertEqual(
            cfg.layers_block_type, ["linear_attention"] * 3 + ["attention"]
        )

    def test_tracks_nextn_interval_override(self):
        # models/qwen3_5_nextn.py rewrites full_attention_interval AFTER
        # construction; layer_types must follow (property, not __init__).
        cfg = self.config_cls(num_hidden_layers=2, full_attention_interval=4)
        cfg.full_attention_interval = 1
        self.assertEqual(cfg.layer_types, ["full_attention"] * 2)


class GdnFlatPoolAssemblyTest(unittest.TestCase):
    """MHAConfig with state shapes + state shard bin table -> create_pool:
    full-coverage pool whose state layers carry no KV and expose head-group
    reinterpret views over the full layers' K/V slabs, with the full group
    and the k shard groups published."""

    # 3 linear + 1 full; kv cell = 2 (K+V) * 1 head * 8 dim * 2 B = 32 B/tok
    # -> segment = 16 * 64 = 1024 B at P=64. ssm head cell = 4*4 fp32 =
    # 64 B (16 heads/group, so each layer's 2 heads make ONE group); conv
    # row = 4*3 bf16 = 24 B. 3 ssm segments + 1 conv segment over
    # 1 full layer * 2 sides -> k = 2 shards.
    LAYER_TYPES = ("linear_attention",) * 3 + ("full_attention",)
    CONV_SHAPE = (4, 3)
    TEMPORAL_SHAPE = (2, 4, 4)
    PAGE_SIZE = 64
    SIZE = 256  # 4 pages -> 5 page rows with the null row

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.configs.mha import (
                MHAConfig,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.MHAConfig = MHAConfig

    def _bin_table(self):
        return shard_bin_table(
            num_full_layers=1,
            num_state_layers=3,
            ssm_heads_per_layer=self.TEMPORAL_SHAPE[0],
            ssm_head_bytes=self.TEMPORAL_SHAPE[1] * self.TEMPORAL_SHAPE[2] * 4,
            conv_bytes_per_layer=self.CONV_SHAPE[0] * self.CONV_SHAPE[1] * 2,
            kv_cell_bytes_per_tok=32,
            block_size=self.PAGE_SIZE,
        )

    def _config(self, *, with_bin_table: bool = True, **overrides):
        torch = self.torch
        config = self.MHAConfig(
            device="cpu",
            backend_name=None,
            num_attention_heads=2,
            num_kv_heads=1,
            head_dim=8,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            page_size=self.PAGE_SIZE,
            context_len=64,
            max_bs=2,
            max_graph_bs=2,
            kv_cache_quant_method=None,
            layer_types=self.LAYER_TYPES,
            max_scheduled_tokens=16,
            conv_state_shape=self.CONV_SHAPE,
            temporal_state_shape=self.TEMPORAL_SHAPE,
            conv_dtype=torch.bfloat16,
            ssm_dtype=torch.float32,
            **overrides,
        )
        if with_bin_table:
            # The registry's flat-GDN branch sets this after solving the
            # packing; single source of truth for the fan-out k.
            config.state_bin_table = self._bin_table()
        return config

    def _pool(self, **kwargs):
        config = self._config(**kwargs)
        with mock.patch(_PKG_FLAT_PROBE, return_value=True):
            return config.create_pool(len(self.LAYER_TYPES), self.SIZE, 0, False)

    def test_assembly_with_bin_table(self):
        torch = self.torch
        pool = self._pool()
        # k = 2 shard groups published next to the full-history group
        # (upstream signal for flat state paging).
        self.assertEqual(
            sorted(spec.group_id for spec in pool.paged_cache_group_specs),
            [
                "full_attention",
                "linear_attention_shard0",
                "linear_attention_shard1",
            ],
        )
        # The k/v lists stay layer-indexed, but state layers carry no KV
        # tensors (None slots) -- only the full-attention layer allocates.
        self.assertEqual(len(pool.k_buffer), len(self.LAYER_TYPES))
        for layer_id, label in enumerate(self.LAYER_TYPES):
            if label == "linear_attention":
                self.assertIsNone(pool.k_buffer[layer_id])
                self.assertIsNone(pool.v_buffer[layer_id])
            else:
                self.assertIsNotNone(pool.k_buffer[layer_id])
                self.assertIsNotNone(pool.v_buffer[layer_id])
        # Each state layer exposes ONE head group (2 heads fit a segment):
        # views over the page-id space (size // P + 1 null row), the fp32
        # ssm reinterpret living inside the bf16 KV slab.
        n = self.SIZE // self.PAGE_SIZE + 1
        for layer_id in (0, 1, 2):
            (group,) = pool.get_state_buffers(layer_id)
            self.assertEqual(tuple(group.conv.shape), (n, *self.CONV_SHAPE))
            self.assertEqual(group.conv.dtype, torch.bfloat16)
            self.assertEqual(tuple(group.ssm.shape), (n, *self.TEMPORAL_SHAPE))
            self.assertEqual(group.ssm.dtype, torch.float32)
            self.assertEqual((group.head_begin, group.num_heads), (0, 2))
            # No state memory of its own: the views alias the full layer's
            # K or V slab storage.
            slabs = [pool.k_buffer[3], pool.v_buffer[3]]
            for t in (group.ssm, group.conv):
                self.assertTrue(
                    any(
                        s.data_ptr() <= t.data_ptr() < s.data_ptr() + s.nbytes
                        for s in slabs
                    )
                )
        self.assertTrue(pool.state_shard_view.is_active)

    def test_without_bin_table_state_stays_legacy(self):
        # Transitional off-switch: no bin table -> the flat-GDN gate stays
        # off, state layers keep per-layer KV, and no shard groups publish.
        pool = self._pool(with_bin_table=False)
        self.assertFalse(pool.state_shard_view.is_active)
        self.assertTrue(all(t is not None for t in pool.k_buffer))
        self.assertEqual(
            sorted(spec.group_id for spec in pool.paged_cache_group_specs),
            ["full_attention", "linear_attention"],
        )
        with self.assertRaisesRegex(ValueError, r"views were bound"):
            pool.get_state_buffers(0)

    def test_pd_disaggregation_rejected_when_active(self):
        with self.assertRaisesRegex(
            RuntimeError, r"state binning is incompatible with PD"
        ):
            self._pool(pd_disaggregation_enabled=True)


if __name__ == "__main__":
    unittest.main()
