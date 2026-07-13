"""hybrid_slab_group_size: the single activation predicate for the unified
KV slab pool (M12), and its two consumers (registry sizing divisor and
MHATokenToKVPool buffer layout).

The predicate returns the common layers-per-group count exactly when the
slab layout may activate (flat ext, >= 2 equal-size known groups) and None
otherwise (legacy per-layer layout). The installed ext's
real build flavor must not decide these tests, so the
scheduler_ext_flat_kvcache probe is patched per case.
"""

from __future__ import annotations

import contextlib
import importlib.util
import itertools
import os
import pathlib
import sys
import unittest
from unittest import mock

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


_pcs = _load("paged_cache_spec_slab_under_test", "paged_cache_spec.py")
hybrid_slab_group_size = _pcs.hybrid_slab_group_size

GPT_OSS_LAYER_TYPES = ("sliding_attention", "full_attention") * 12


class HybridSlabGroupSizeTest(unittest.TestCase):
    """Each case pins exactly ONE reason the predicate returns None (or the
    single shape where it activates)."""

    @contextlib.contextmanager
    def _flat_ext(self, value: bool):
        # The predicate resolves the probe from its own module globals at
        # call time, so the patch must target the path-loaded module.
        with mock.patch.object(_pcs, "scheduler_ext_flat_kvcache", return_value=value):
            yield

    def test_gpt_oss_shape_returns_group_size(self):
        # gpt-oss: 12 sliding + 12 full, alternating -> 12 layers per group.
        with self._flat_ext(True):
            self.assertEqual(
                hybrid_slab_group_size(GPT_OSS_LAYER_TYPES),
                12,
            )

    def test_none_when_radix_ext(self):
        with self._flat_ext(False):
            self.assertIsNone(hybrid_slab_group_size(GPT_OSS_LAYER_TYPES))

    def test_none_when_single_group(self):
        with self._flat_ext(True):
            self.assertIsNone(hybrid_slab_group_size(("full_attention",) * 24))

    def test_none_when_unequal_groups(self):
        lt = ("sliding_attention",) * 8 + ("full_attention",) * 16
        with self._flat_ext(True):
            self.assertIsNone(hybrid_slab_group_size(lt))

    def test_none_when_unknown_label(self):
        # Unknown input degrades to None (safe legacy layout), never raises;
        # loud rejection is group_specs_from_layer_types' job.
        lt = GPT_OSS_LAYER_TYPES + ("banana_attention",)
        with self._flat_ext(True):
            self.assertIsNone(hybrid_slab_group_size(lt))

    def test_none_when_empty(self):
        # Plain models pass empty or None layer_types.
        with self._flat_ext(True):
            self.assertIsNone(hybrid_slab_group_size(()))
            self.assertIsNone(hybrid_slab_group_size(None))

    def test_none_when_multi_window_sequence(self):
        with self._flat_ext(True):
            it = itertools.cycle((4, 512))
            windows = [
                next(it) if t == "sliding_attention" else None
                for t in GPT_OSS_LAYER_TYPES
            ]
            self.assertIsNone(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES,
                    sliding_window_tokens=windows,
                )
            )

    def test_uniform_window_sequence_stays_active(self):
        with self._flat_ext(True):
            windows = [
                None if t == "full_attention" else 128 for t in GPT_OSS_LAYER_TYPES
            ]
            self.assertEqual(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES,
                    sliding_window_tokens=windows,
                ),
                12,
            )

    def test_scalar_window_stays_active(self):
        with self._flat_ext(True):
            self.assertEqual(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES,
                    sliding_window_tokens=128,
                ),
                12,
            )

    def test_none_when_window_sequence_length_mismatch(self):
        with self._flat_ext(True):
            self.assertIsNone(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES,
                    sliding_window_tokens=[128],
                )
            )

    def test_garbage_elements_ignored_not_raised(self):
        with self._flat_ext(True):
            self.assertEqual(
                hybrid_slab_group_size(
                    GPT_OSS_LAYER_TYPES,
                    sliding_window_tokens=["a"] * len(GPT_OSS_LAYER_TYPES),
                ),
                12,
            )


class KvProfileLayerDivisorTest(unittest.TestCase):
    """Registry sizing consumer: _kv_profile_layer_divisor charges
    layers-per-group exactly when the predicate activates, all layers
    otherwise. Imports the real registry, so skips on a bare interpreter.
    Patch target is the PACKAGE paged_cache_spec probe -- the path-loaded
    _pcs copy above is a distinct module object the registry never sees.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import tokenspeed.runtime.configs.paged_cache_spec as pkg_pcs
            from tokenspeed.runtime.layers.attention import registry
        except ImportError as exc:
            raise unittest.SkipTest(f"real attention registry unimportable here: {exc}")
        cls._registry = registry
        cls._pkg_pcs = pkg_pcs

    @contextlib.contextmanager
    def _pkg_flat_ext(self, value: bool):
        with mock.patch.object(
            self._pkg_pcs, "scheduler_ext_flat_kvcache", return_value=value
        ):
            yield

    def test_gpt_oss_flat_ext_charges_group_size(self):
        # 24 layers, 12+12 alternating -> charge 12 (per-token bytes halve).
        with self._pkg_flat_ext(True):
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(24, GPT_OSS_LAYER_TYPES),
                12,
            )

    def test_all_layers_when_radix_ext(self):
        with self._pkg_flat_ext(False):
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(24, GPT_OSS_LAYER_TYPES),
                24,
            )

    def test_all_layers_when_no_layer_types(self):
        # () from MHAConfig's default, None from MLA configs via getattr.
        with self._pkg_flat_ext(True):
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(24, ()),
                24,
            )
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(24, None),
                24,
            )

    def test_all_layers_when_multi_window_sequence(self):
        # M14: the registry must forward sliding_window_tokens so sizing
        # matches the pool's layout decision (divergence is the hazard).
        with self._pkg_flat_ext(True):
            it = itertools.cycle((4, 512))
            windows = [
                next(it) if t == "sliding_attention" else None
                for t in GPT_OSS_LAYER_TYPES
            ]
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(
                    24,
                    GPT_OSS_LAYER_TYPES,
                    sliding_window_tokens=windows,
                ),
                24,
            )

    def test_group_size_when_uniform_window_sequence(self):
        with self._pkg_flat_ext(True):
            windows = [
                128 if t == "sliding_attention" else None for t in GPT_OSS_LAYER_TYPES
            ]
            self.assertEqual(
                self._registry._kv_profile_layer_divisor(
                    24,
                    GPT_OSS_LAYER_TYPES,
                    sliding_window_tokens=windows,
                ),
                12,
            )


_PKG_FLAT_PROBE = (
    "tokenspeed.runtime.configs.paged_cache_spec.scheduler_ext_flat_kvcache"
)


class MHAPoolSlabLayoutTest(unittest.TestCase):
    """Layout consumer (kv_cache/mha.py _create_buffers): when the predicate
    activates, paired layers bind to the SAME slab tensors; otherwise the
    legacy per-layer layout holds and the PD guard never fires.
    Constructs a real (tiny, CPU) MHATokenToKVPool; skips without deps.
    Patch target is the PACKAGE paged_cache_spec probe (see above).
    """

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.MHATokenToKVPool = MHATokenToKVPool

    def _pool(self, *, flat_ext: bool = True, **overrides):
        kwargs = dict(
            size=32,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=8,
            layer_num=24,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=64,
            page_size=16,
            rank=0,
            layer_types=GPT_OSS_LAYER_TYPES,
            sliding_window_tokens=128,
            enable_alt_stream=False,
        )
        kwargs.update(overrides)
        with mock.patch(_PKG_FLAT_PROBE, return_value=flat_ext):
            return self.MHATokenToKVPool(**kwargs)

    def test_slab_pairing_binds_same_tensor(self):
        pool = self._pool()
        # 24 layer entries alias 12 slabs: accessors stay layer-indexed.
        self.assertEqual(len(pool.k_buffer), 24)
        self.assertEqual(len({id(t) for t in pool.k_buffer}), 12)
        self.assertEqual(len({id(t) for t in pool.v_buffer}), 12)
        # The i-th sliding layer (2i) pairs the i-th full layer (2i+1) on
        # the SAME tensor object -- shared storage, not a copy or a view.
        for i in range(12):
            self.assertIs(pool.k_buffer[2 * i], pool.k_buffer[2 * i + 1])
            self.assertIs(pool.v_buffer[2 * i], pool.v_buffer[2 * i + 1])
            self.assertEqual(
                pool.k_buffer[2 * i].data_ptr(),
                pool.k_buffer[2 * i + 1].data_ptr(),
            )
        # Every slab is referenced by exactly one layer of EACH group.
        for buffers in (pool.k_buffer, pool.v_buffer):
            slab_to_layers: dict[int, list[int]] = {}
            for layer_id, tensor in enumerate(buffers):
                slab_to_layers.setdefault(id(tensor), []).append(layer_id)
            self.assertEqual(len(slab_to_layers), 12)
            for layer_ids in slab_to_layers.values():
                sliding = [lid for lid in layer_ids if lid % 2 == 0]
                full = [lid for lid in layer_ids if lid % 2 == 1]
                self.assertEqual(len(sliding), 1)
                self.assertEqual(len(full), 1)
        # Distinct slabs own distinct storage.
        self.assertEqual(len({t.data_ptr() for t in pool.k_buffer}), 12)
        self.assertEqual(len({t.data_ptr() for t in pool.v_buffer}), 12)
        # Per-layer host (L2) copies would alias shared slabs, so the slab
        # pool opts out of the hierarchical cache surface.
        self.assertFalse(pool.supports_hierarchical_kv_cache)

    def test_fallback_matrix_keeps_24_buffers(self):
        cases = dict(
            radix_ext=dict(flat_ext=False),
            single_group=dict(
                layer_types=("full_attention",) * 24,
                sliding_window_tokens=None,
            ),
            unequal_groups=dict(
                layer_types=("sliding_attention",) * 8 + ("full_attention",) * 16
            ),
        )
        for name, overrides in cases.items():
            with self.subTest(name):
                pool = self._pool(**overrides)
                self.assertEqual(len({id(t) for t in pool.k_buffer}), 24)
                self.assertEqual(len({id(t) for t in pool.v_buffer}), 24)
                self.assertTrue(pool.supports_hierarchical_kv_cache)

    def test_guard_raises_on_pd_with_slab(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"hybrid slab KV layout is incompatible with PD disaggregation"
            r".*radix-built",
        ):
            self._pool(pd_disaggregation_enabled=True)

    def test_no_guard_when_fallback(self):
        # The flag only conflicts with the slab layout, not the legacy one.
        pool = self._pool(
            flat_ext=False,
            pd_disaggregation_enabled=True,
        )
        self.assertEqual(len({id(t) for t in pool.k_buffer}), 24)


GDN_LAYER_TYPES = ("linear_attention", "full_attention") * 2


class StatePagedCacheGroupPageCountTest(unittest.TestCase):
    """compute_paged_cache_group_page_counts: the family="state" branch is
    positive and bounded by the full-history formula for the same inputs
    (state rows keep <= 2 live pages per request -- the W=2 write window --
    and snapshots are bounded by the shared page-id space).
    The direct-loaded module still imports ceil_div from the real package
    at call time, so this skips on a bare interpreter.
    """

    def setUp(self):
        try:
            from tokenspeed.runtime.utils.common import ceil_div  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"page-count math needs the real package: {exc}")

    def _counts(self, **overrides):
        specs = _pcs.group_specs_from_layer_types(
            layer_types=("linear_attention", "full_attention"),
            sliding_window_tokens=None,
            page_size=16,
        )
        params = dict(
            max_live_requests=2,
            max_scheduled_tokens=64,
            max_total_tokens=1024,
            max_context_len=4096,
        )
        params.update(overrides)
        return _pcs.compute_paged_cache_group_page_counts(specs, **params)

    def test_state_count_positive_and_bounded_by_full_history(self):
        counts = self._counts()
        self.assertGreater(counts["linear_attention"], 0)
        self.assertLessEqual(counts["linear_attention"], counts["full_attention"])

    def test_state_branch_departs_from_full_history_formula(self):
        # B=0 with a non-page-multiple T distinguishes the state branch
        # (floor(T/P) + 0 live) from the full-history one (ceil(T/P) + B):
        # 1000/16 -> state 62+1=63 < full 63+1=64.
        counts = self._counts(max_live_requests=0, max_total_tokens=1000)
        self.assertLess(counts["linear_attention"], counts["full_attention"])


class MHAPoolStateShardViewTest(unittest.TestCase):
    """State-view consumer (kv_cache/mha.py): a GDN hybrid keeps a per-layer
    KV layout on attention layers (state layers carry None slots, M18a),
    and flat ext + mamba2 shapes + a state shard bin table expose per-layer
    head groups whose conv/ssm tensors are reinterpret views over the full
    layers' K/V slabs (M18c) -- row-indexed by page id (row 0 = null page,
    never written), no state memory of their own.
    Constructs a real (tiny, CPU) MHATokenToKVPool; skips without deps.
    Patch target is the PACKAGE paged_cache_spec probe (see above).
    """

    # cell = 2 * 1 head * 8 dim * 2 B = 32 B/tok -> segment 1024 B at P=64;
    # ssm head cell = 4*4 fp32 = 64 B (both heads of a layer make ONE
    # group); conv row = 4*8 fp32 = 128 B. 2 ssm + 1 conv segment over
    # 2 full layers * 2 sides -> k = 1 shard.
    CONV_SHAPE = (4, 8)
    SSM_SHAPE = (2, 4, 4)
    PAGE_SIZE = 64
    SIZE = 128

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.configs.flat_memory_plan import (
                shard_bin_table,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.MHATokenToKVPool = MHATokenToKVPool
        self.shard_bin_table = shard_bin_table

    def _bin_table(self):
        return self.shard_bin_table(
            num_full_layers=2,
            num_state_layers=2,
            ssm_heads_per_layer=self.SSM_SHAPE[0],
            ssm_head_bytes=self.SSM_SHAPE[1] * self.SSM_SHAPE[2] * 4,
            conv_bytes_per_layer=self.CONV_SHAPE[0] * self.CONV_SHAPE[1] * 4,
            kv_cell_bytes_per_tok=32,
            block_size=self.PAGE_SIZE,
        )

    def _pool(self, *, flat_ext: bool = True, **overrides):
        kwargs = dict(
            size=self.SIZE,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=8,
            layer_num=4,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=64,
            page_size=self.PAGE_SIZE,
            rank=0,
            layer_types=GDN_LAYER_TYPES,
            sliding_window_tokens=None,
            enable_alt_stream=False,
            conv_state_shape=self.CONV_SHAPE,
            temporal_state_shape=self.SSM_SHAPE,
            conv_dtype=self.torch.float32,
            ssm_dtype=self.torch.float32,
            state_bin_table=self._bin_table(),
        )
        kwargs.update(overrides)
        with mock.patch(_PKG_FLAT_PROBE, return_value=flat_ext):
            return self.MHATokenToKVPool(**kwargs)

    def test_state_views_one_group_per_state_layer(self):
        pool = self._pool()
        # KV side stays per-layer (no aliasing), but flat GDN state layers
        # (0/2) carry no KV tensors -- None slots (M18a T4).
        self.assertEqual(len(pool.k_buffer), 4)
        self.assertIsNone(pool.k_buffer[0])
        self.assertIsNone(pool.k_buffer[2])
        self.assertEqual(len({id(t) for t in pool.k_buffer if t is not None}), 2)
        self.assertTrue(pool.state_shard_view.is_active)
        num_pages_with_null = self.SIZE // self.PAGE_SIZE + 1  # row 0 = null
        slabs = [pool.k_buffer[1], pool.k_buffer[3], pool.v_buffer[1], pool.v_buffer[3]]
        for layer_id in (0, 2):
            (group,) = pool.get_state_buffers(layer_id)
            self.assertEqual(group.conv.shape, (num_pages_with_null, *self.CONV_SHAPE))
            self.assertEqual(group.ssm.shape, (num_pages_with_null, *self.SSM_SHAPE))
            self.assertEqual(group.conv.dtype, self.torch.float32)
            self.assertEqual(group.ssm.dtype, self.torch.float32)
            self.assertEqual((group.head_begin, group.num_heads), (0, 2))
            # No state memory of its own: views alias the K/V slab storage.
            for t in (group.ssm, group.conv):
                self.assertTrue(
                    any(
                        s.data_ptr() <= t.data_ptr() < s.data_ptr() + s.nbytes
                        for s in slabs
                    )
                )

    def test_get_state_buffers_occurrence_indexed(self):
        pool = self._pool()
        # Layers 0 and 2 are the 0th/1st linear layers -> bin-table layers
        # 0/1: ssm segments land on slot 0's K then V slab (byte offset 0),
        # and both conv rows pack into slot 1's K slab at offsets 0 / 128.
        (g0,) = pool.get_state_buffers(0)
        (g2,) = pool.get_state_buffers(2)
        self.assertEqual(g0.ssm.data_ptr(), pool.k_buffer[1].data_ptr())
        self.assertEqual(g2.ssm.data_ptr(), pool.v_buffer[1].data_ptr())
        self.assertEqual(g0.conv.data_ptr(), pool.k_buffer[3].data_ptr())
        self.assertEqual(g2.conv.data_ptr(), pool.k_buffer[3].data_ptr() + 128)
        with self.assertRaisesRegex(ValueError, r"not a state layer"):
            pool.get_state_buffers(1)

    def test_no_state_shapes_no_views(self):
        # Missing shapes switch the gate off even with a bin table present.
        pool = self._pool(
            conv_state_shape=None,
            temporal_state_shape=None,
            conv_dtype=None,
            ssm_dtype=None,
        )
        self.assertFalse(pool.state_shard_view.is_active)
        with self.assertRaisesRegex(ValueError, r"views were bound"):
            pool.get_state_buffers(0)

    def test_no_bin_table_stays_legacy(self):
        # Transitional off-switch: no bin table -> full per-layer KV, no
        # state views.
        pool = self._pool(state_bin_table=None)
        self.assertFalse(pool.state_shard_view.is_active)
        self.assertTrue(all(t is not None for t in pool.k_buffer))
        with self.assertRaisesRegex(ValueError, r"views were bound"):
            pool.get_state_buffers(0)

    def test_radix_ext_no_views(self):
        pool = self._pool(flat_ext=False)
        self.assertFalse(pool.state_shard_view.is_active)
        with self.assertRaisesRegex(ValueError, r"views were bound"):
            pool.get_state_buffers(0)

    def test_published_shard_group_counts_positive_and_bounded(self):
        # k = 1 for this packing: the state group publishes as
        # linear_attention_shard0 (never the bare name).
        pool = self._pool()
        counts = pool.paged_cache_group_page_counts
        self.assertNotIn("linear_attention", counts)
        self.assertGreater(counts["linear_attention_shard0"], 0)
        self.assertLessEqual(
            counts["linear_attention_shard0"], counts["full_attention"]
        )


# Qwen3.5-ish interleaving: 3 linear layers then 1 full, times 12 (48 layers).
QWEN_LIKE_LAYER_TYPES = (("linear_attention",) * 3 + ("full_attention",)) * 12


class FlatGDNStateLayerNoKVTest(unittest.TestCase):
    """M18a T4: under the flat GDN predicate the pool allocates NO per-layer
    KV tensors on state layers (`k_buffer[lid] is None`); the component plan
    (registry sizing) already charges only full-layer KV + state rows, so
    keeping the tensors would oversubscribe the layout. Every None-exposed
    surface must either skip (pointer tables, size accounting, clear) or
    reject loudly (per-layer accessors, PD transfer). Radix pools keep the
    full per-layer KV coverage (zero impact pin).
    Constructs a real (tiny, CPU) MHATokenToKVPool; skips without deps.
    Patch target is the PACKAGE paged_cache_spec probe (see above).
    """

    CONV_SHAPE = (4, 8)
    SSM_SHAPE = (2, 4, 4)
    NUM_LAYERS = 48
    NUM_FULL = 12
    SIZE = 128
    PAGE_SIZE = 64

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.configs.flat_memory_plan import (
                shard_bin_table,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.MHATokenToKVPool = MHATokenToKVPool
        self.shard_bin_table = shard_bin_table

    def _bin_table(self):
        return self.shard_bin_table(
            num_full_layers=self.NUM_FULL,
            num_state_layers=self.NUM_LAYERS - self.NUM_FULL,
            ssm_heads_per_layer=self.SSM_SHAPE[0],
            ssm_head_bytes=self.SSM_SHAPE[1] * self.SSM_SHAPE[2] * 4,
            conv_bytes_per_layer=self.CONV_SHAPE[0] * self.CONV_SHAPE[1] * 4,
            kv_cell_bytes_per_tok=32,
            block_size=self.PAGE_SIZE,
        )

    def _pool(self, *, flat_ext: bool = True, **overrides):
        kwargs = dict(
            size=self.SIZE,
            dtype=self.torch.bfloat16,
            head_num=1,
            head_dim=8,
            layer_num=self.NUM_LAYERS,
            device="cpu",
            enable_memory_saver=False,
            max_batch_size=2,
            max_context_len=64,
            page_size=self.PAGE_SIZE,
            rank=0,
            layer_types=QWEN_LIKE_LAYER_TYPES,
            sliding_window_tokens=None,
            enable_alt_stream=False,
            conv_state_shape=self.CONV_SHAPE,
            temporal_state_shape=self.SSM_SHAPE,
            conv_dtype=self.torch.float32,
            ssm_dtype=self.torch.float32,
            state_bin_table=self._bin_table(),
        )
        kwargs.update(overrides)
        with mock.patch(_PKG_FLAT_PROBE, return_value=flat_ext):
            return self.MHATokenToKVPool(**kwargs)

    def test_flat_gdn_state_layers_have_no_kv(self):
        pool = self._pool()
        rows = self.SIZE + self.PAGE_SIZE  # +page_size dummy page
        self.assertEqual(len(pool.k_buffer), self.NUM_LAYERS)
        self.assertEqual(len(pool.v_buffer), self.NUM_LAYERS)
        for layer_id, label in enumerate(QWEN_LIKE_LAYER_TYPES):
            if label == "linear_attention":
                self.assertIsNone(pool.k_buffer[layer_id])
                self.assertIsNone(pool.v_buffer[layer_id])
            else:
                self.assertEqual(pool.k_buffer[layer_id].shape[0], rows)
                self.assertEqual(pool.v_buffer[layer_id].shape[0], rows)
        # State views still cover every state layer (reinterpret views over
        # the full layers' slabs; no state memory of their own).
        for layer_id, label in enumerate(QWEN_LIKE_LAYER_TYPES):
            if label == "linear_attention":
                (group,) = pool.get_state_buffers(layer_id)
                self.assertEqual(group.ssm.shape[0], self.SIZE // self.PAGE_SIZE + 1)

    def test_flat_gdn_data_ptrs_exclude_none(self):
        # _kv_copy launches one block per data_ptrs entry (grid = numel), so
        # a placeholder entry for a skipped layer would be dereferenced --
        # the pointer/stride tables must carry the REAL tensors only.
        pool = self._pool()
        self.assertEqual(pool.k_data_ptrs.numel(), self.NUM_FULL)
        self.assertEqual(pool.v_data_ptrs.numel(), self.NUM_FULL)
        self.assertEqual(pool.data_ptrs.numel(), 2 * self.NUM_FULL)
        self.assertEqual(pool.data_strides.numel(), 2 * self.NUM_FULL)
        self.assertTrue(bool((pool.data_strides > 0).all()))

    def test_flat_gdn_kv_size_counts_only_real_tensors(self):
        pool = self._pool()
        # (rows, head_num=1, head_dim=8) bf16 per full-attention layer.
        per_tensor = (self.SIZE + self.PAGE_SIZE) * 1 * 8 * 2
        k_size, v_size = pool.get_kv_size_bytes()
        self.assertEqual(k_size, self.NUM_FULL * per_tensor)
        self.assertEqual(v_size, self.NUM_FULL * per_tensor)

    def test_flat_gdn_get_key_buffer_raises_on_state_layer(self):
        pool = self._pool()
        with self.assertRaisesRegex(ValueError, r"state layer"):
            pool.get_key_buffer(0)
        with self.assertRaisesRegex(ValueError, r"state layer"):
            pool.get_value_buffer(0)
        # Full-attention layers stay served (layer 3 is the first full one).
        self.assertIsNotNone(pool.get_key_buffer(3))
        self.assertIsNotNone(pool.get_value_buffer(3))

    def test_flat_gdn_pd_transfer_rejected(self):
        # PD disaggregation registers per-layer buffer pointers; with state
        # layers carrying no KV that surface must reject loudly.
        pool = self._pool()
        with self.assertRaisesRegex(ValueError, r"PD disaggregation"):
            pool.get_contiguous_buf_infos()

    def test_flat_gdn_disables_hierarchical_offload(self):
        # Structural pin: event_loop builds the radix MemoryExecutor (even
        # with the kvstore off, for retraction offload) only when the pool
        # advertises hierarchical support. State layers carry no per-layer
        # KV, so the flat GDN pool must opt out rather than rely on
        # cache-op-kind rejection to keep layer-indexed derefs unreachable.
        pool = self._pool()
        self.assertFalse(pool.supports_hierarchical_kv_cache)

    def test_flat_gdn_clear_buffers_survives_none(self):
        pool = self._pool()
        self.assertIsNone(pool.k_buffer[0])  # None entries really present
        pool.k_buffer[3].fill_(1.0)
        pool.clear_kv_buffers()  # sleep/wake repair path; must not raise
        self.assertEqual(pool.k_buffer[3].abs().sum().item(), 0.0)

    def test_radix_pool_unaffected(self):
        # Zero-impact pin: without the flat ext the same layer_types (and
        # even a present bin table) keep full per-layer KV coverage and no
        # state views.
        pool = self._pool(flat_ext=False)
        self.assertTrue(all(t is not None for t in pool.k_buffer))
        self.assertTrue(all(t is not None for t in pool.v_buffer))
        self.assertEqual(pool.data_ptrs.numel(), 2 * self.NUM_LAYERS)
        self.assertFalse(pool.state_shard_view.is_active)
        pool.get_contiguous_buf_infos()  # PD surface stays available
        self.assertTrue(pool.supports_hierarchical_kv_cache)


if __name__ == "__main__":
    unittest.main()
