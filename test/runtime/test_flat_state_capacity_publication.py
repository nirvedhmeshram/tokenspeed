from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _load_base_pool_class():
    stubs = {
        "tokenspeed.runtime.configs": types.ModuleType("tokenspeed.runtime.configs"),
        "tokenspeed.runtime.configs.paged_cache_spec": types.ModuleType(
            "tokenspeed.runtime.configs.paged_cache_spec"
        ),
        "tokenspeed.runtime.layers.paged_attention": types.ModuleType(
            "tokenspeed.runtime.layers.paged_attention"
        ),
        "tokenspeed.runtime.utils": types.ModuleType("tokenspeed.runtime.utils"),
    }
    stubs["tokenspeed.runtime.configs.paged_cache_spec"].PagedCacheGroupSpec = object
    stubs["tokenspeed.runtime.layers.paged_attention"].PagedAttention = object
    stubs["tokenspeed.runtime.utils"].get_colorful_logger = (
        lambda _name: types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    )
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "python/tokenspeed/runtime/layers/attention/kv_cache/base.py"
        )
        spec = importlib.util.spec_from_file_location(
            "flat_state_capacity_base_under_test", path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.BaseTokenToKVPool
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


BaseTokenToKVPool = _load_base_pool_class()


class FlatStateCapacityPublicationTest(unittest.TestCase):
    def test_capacity_is_none_by_default_and_published_once(self):
        pool = BaseTokenToKVPool(
            size=16,
            dtype=torch.bfloat16,
            device="cpu",
            max_batch_size=4,
            max_context_len=16,
            page_size=4,
            rank=0,
        )
        capacity = object()

        self.assertIsNone(pool.flat_state_capacity)
        pool.publish_flat_state_capacity(capacity)
        self.assertIs(pool.flat_state_capacity, capacity)
        with self.assertRaisesRegex(RuntimeError, "already published"):
            pool.publish_flat_state_capacity(object())

    def test_registry_publishes_actual_effective_flat_capacity(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        source = (
            root / "python/tokenspeed/runtime/layers/attention/registry.py"
        ).read_text()

        self.assertIn("flat_state_capacity_from_plan(", source)
        self.assertIn("num_page_ids=max_num_tokens // server_args.block_size", source)
        self.assertIn("pool.publish_flat_state_capacity(flat_state_capacity)", source)

    def test_event_loop_gates_graph_config_before_executor_creation(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        source = (root / "python/tokenspeed/runtime/engine/event_loop.py").read_text()

        config_at = source.index("model_executor_config = ModelExecutorConfig")
        gate_at = source.index("apply_flat_state_capacity_graph_gate(")
        executor_at = source.index("self.model_executor = create_model_executor(")
        self.assertLess(config_at, gate_at)
        self.assertLess(gate_at, executor_at)


if __name__ == "__main__":
    unittest.main()
