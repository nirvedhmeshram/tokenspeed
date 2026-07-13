"""Accepted-checkpoint commit tests for Flat Qwen MTP state pages."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class FlatMtpCheckpointSelectionTest(unittest.TestCase):
    def setUp(self):
        from tokenspeed.runtime.layers.attention.linear.flat_mtp_state import (
            selected_checkpoint_steps,
        )

        self.select = selected_checkpoint_steps

    def test_same_page_collapses_to_last_accepted_step(self):
        out_pages = [10, 10, 10, 10]
        self.assertEqual(self.select(out_pages, 1), (0,))
        self.assertEqual(self.select(out_pages, 2), (1,))
        self.assertEqual(self.select(out_pages, 3), (2,))
        self.assertEqual(self.select(out_pages, 4), (3,))

    def test_one_and_multiple_page_crossings(self):
        out_pages = [10, 10, 11, 11]
        self.assertEqual(self.select(out_pages, 1), (0,))
        self.assertEqual(self.select(out_pages, 2), (1,))
        self.assertEqual(self.select(out_pages, 3), (1, 2))
        self.assertEqual(self.select(out_pages, 4), (1, 3))

        self.assertEqual(self.select([10, 11, 12, 12], 4), (0, 1, 3))

    def test_rejects_invalid_accept_length(self):
        with self.assertRaisesRegex(ValueError, "accepted_length"):
            self.select([10, 10, 11, 11], 0)
        with self.assertRaisesRegex(ValueError, "accepted_length"):
            self.select([10, 10, 11, 11], 5)


class FlatMtpStateOrchestrationTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.linear import flat_mtp_state
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs CPU PyTorch: {exc}")
        self.torch = torch
        self.module = flat_mtp_state

    def test_conv_once_and_every_ssm_group(self):
        torch = self.torch
        conv = torch.empty(20, 3, 2)
        ssm0 = torch.empty(20, 1, 2, 2)
        ssm1 = torch.empty(20, 2, 2, 2)
        groups = [
            SimpleNamespace(conv=conv, ssm=ssm0, conv_shard=1, shard=0),
            SimpleNamespace(conv=conv, ssm=ssm1, conv_shard=1, shard=1),
        ]
        pool = SimpleNamespace(
            state_shard_view=SimpleNamespace(state_layer_ids=frozenset({7})),
            get_state_buffers=lambda layer_id: groups if layer_id == 7 else None,
        )
        spec = torch.tensor(
            [
                [[1, 2, 3, 4], [-1, -1, -1, -1]],
                [[5, 6, 7, 8], [-1, -1, -1, -1]],
            ],
            dtype=torch.int32,
        )
        out = torch.tensor(
            [
                [[10, 10, 11, 11], [-1, -1, -1, -1]],
                [[12, 12, 13, 13], [-1, -1, -1, -1]],
            ],
            dtype=torch.int32,
        )
        accepted = torch.tensor([3], dtype=torch.int32)
        calls = []

        def record(state, src, dst, lengths, *, validate_accepted_lengths=True):
            calls.append(
                (
                    state,
                    src.clone(),
                    dst.clone(),
                    lengths.clone(),
                    validate_accepted_lengths,
                )
            )

        with mock.patch.object(
            self.module, "flat_mtp_state_commit", side_effect=record
        ):
            self.module.commit_flat_mtp_state_pages(pool, spec, out, accepted)

        self.assertEqual([call[0] for call in calls], [conv, ssm0, ssm1])
        self.assertEqual(calls[0][1].tolist(), [[5, 6, 7, 8]])
        self.assertEqual(calls[0][2].tolist(), [[12, 12, 13, 13]])
        self.assertEqual(calls[1][1].tolist(), [[1, 2, 3, 4]])
        self.assertEqual(calls[2][1].tolist(), [[5, 6, 7, 8]])
        self.assertTrue(all(call[3].tolist() == [3] for call in calls))
        self.assertTrue(all(call[4] is False for call in calls))


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


_CPU_BACKEND_MODULE = None


def _load_backend_with_cpu_stubs():
    """Load the real backend while stubbing only unavailable CUDA modules."""

    global _CPU_BACKEND_MODULE
    if _CPU_BACKEND_MODULE is not None:
        return _CPU_BACKEND_MODULE

    class AttentionBackend:
        pass

    class ForwardMode:
        DECODE = object()

    noop = lambda *_args, **_kwargs: None
    stub_modules = {
        "tokenspeed_kernel": _module("tokenspeed_kernel", __path__=[]),
        "tokenspeed_kernel.ops": _module("tokenspeed_kernel.ops", __path__=[]),
        "tokenspeed_kernel.ops.attention": _module(
            "tokenspeed_kernel.ops.attention",
            GdnCheckpointLayout=object,
            gdn_chunk_prefill=noop,
        ),
        "tokenspeed_kernel.ops.attention.triton.gdn_qkv_split": _module(
            "tokenspeed_kernel.ops.attention.triton.gdn_qkv_split",
            fused_qkv_split_gdn_prefill=noop,
        ),
        "tokenspeed_kernel.ops.attention.triton.linear.chunk_delta_h": _module(
            "tokenspeed_kernel.ops.attention.triton.linear.chunk_delta_h",
            CHUNK_SIZE=64,
        ),
        "tokenspeed_kernel.ops.attention.triton.linear.index": _module(
            "tokenspeed_kernel.ops.attention.triton.linear.index",
            set_total_chunks_hint=noop,
            set_total_chunks_hint_uniform=noop,
        ),
        "tokenspeed.runtime.configs.paged_cache_spec": _module(
            "tokenspeed.runtime.configs.paged_cache_spec",
            LINEAR_ATTENTION="linear_attention",
        ),
        "tokenspeed.runtime.execution.breakable_cuda_graph": _module(
            "tokenspeed.runtime.execution.breakable_cuda_graph",
            break_point=lambda fn: fn,
            current_forward_ctx=noop,
            scrub_padding_tail=noop,
        ),
        "tokenspeed.runtime.execution.forward_batch_info": _module(
            "tokenspeed.runtime.execution.forward_batch_info", ForwardMode=ForwardMode
        ),
        "tokenspeed.runtime.layers.attention.backends.base": _module(
            "tokenspeed.runtime.layers.attention.backends.base",
            AttentionBackend=AttentionBackend,
            init_backend_cuda_graph_state=noop,
        ),
        "tokenspeed.runtime.layers.attention.linear.causal_conv1d": _module(
            "tokenspeed.runtime.layers.attention.linear.causal_conv1d",
            causal_conv1d_fn=noop,
            causal_conv1d_update=noop,
        ),
        "tokenspeed.runtime.layers.attention.linear.fused_sigmoid_gating_recurrent": _module(
            "tokenspeed.runtime.layers.attention.linear.fused_sigmoid_gating_recurrent",
            fused_sigmoid_gating_delta_rule_update=noop,
        ),
        "tokenspeed.runtime.layers.attention.linear.gdn": _module(
            "tokenspeed.runtime.layers.attention.linear.gdn", fused_gdn_gating=noop
        ),
        "tokenspeed.runtime.layers.attention.linear.mamba_state_scatter_triton": _module(
            "tokenspeed.runtime.layers.attention.linear.mamba_state_scatter_triton",
            fused_mamba_state_copy=noop,
        ),
    }
    module_name = "_task5_hybrid_linear_attn"
    path = (
        Path(__file__).resolve().parents[2]
        / "python/tokenspeed/runtime/layers/attention/backends/hybrid_linear_attn.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    import torch

    passthrough_compile = lambda fn=None, **_kwargs: (
        fn if fn is not None else lambda f: f
    )
    with mock.patch.dict(sys.modules, stub_modules), mock.patch.object(
        torch, "compile", passthrough_compile
    ):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    _CPU_BACKEND_MODULE = module
    return module


class FlatVerifyPageIndicesTest(unittest.TestCase):
    """CPU contract for runtime-resolved target-verify State pages."""

    def setUp(self):
        try:
            import torch
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs CPU PyTorch: {exc}")
        self.torch = torch
        self.fn = (
            _load_backend_with_cpu_stubs().compute_state_verify_page_indices_batched
        )

    def test_three_verify_positions_cross_page(self):
        torch = self.torch
        rows = torch.tensor([[[10, 11, 12, 13, 14]]], dtype=torch.int32)
        state_in, state_out = self.fn(rows, 2, torch.tensor([8], dtype=torch.int32), 3)
        self.assertEqual(state_in.tolist(), [[12]])
        self.assertEqual(state_out.tolist(), [[[12, 13, 13]]])

    def test_before_zero_clamps_then_masks_input(self):
        torch = self.torch
        rows = torch.tensor([[[10, 11, 12]]], dtype=torch.int32)
        state_in, state_out = self.fn(rows, 2, torch.tensor([3], dtype=torch.int32), 3)
        self.assertEqual(state_in.tolist(), [[0]])
        self.assertEqual(state_out.tolist(), [[[10, 10, 11]]])

    def test_verify_width_is_output_dimension_for_every_shard(self):
        torch = self.torch
        rows = torch.tensor([[[10, 11, 12, 13]], [[20, 21, 22, 23]]], dtype=torch.int32)
        state_in, state_out = self.fn(rows, 2, torch.tensor([6], dtype=torch.int32), 3)
        self.assertEqual(tuple(state_in.shape), (2, 1))
        self.assertEqual(tuple(state_out.shape), (2, 1, 3))
        self.assertEqual(state_out.tolist(), [[[11, 12, 12]], [[21, 22, 22]]])

    def test_rejects_invalid_width_negative_before_and_short_table(self):
        torch = self.torch
        rows = torch.tensor([[[10, 11]]], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "verify_width"):
            self.fn(rows, 2, torch.tensor([2], dtype=torch.int32), 0)
        with self.assertRaisesRegex(ValueError, "before"):
            self.fn(rows, 2, torch.tensor([2], dtype=torch.int32), 3)
        with self.assertRaisesRegex(ValueError, "table width"):
            self.fn(rows, 2, torch.tensor([6], dtype=torch.int32), 1)

    def test_loadback_extension_keeps_absolute_slot_meaning(self):
        torch = self.torch
        # The first two entries are the device prefix. Host loadback appends
        # slots 2 and 3; it must not renumber the existing absolute slots.
        rows = torch.tensor([[[10, 11, 50, 60]]], dtype=torch.int32)
        state_in, state_out = self.fn(rows, 2, torch.tensor([8], dtype=torch.int32), 3)
        self.assertEqual(state_in.tolist(), [[50]])
        self.assertEqual(state_out.tolist(), [[[50, 60, 60]]])


class FlatExecutionPageResolutionTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs CPU PyTorch: {exc}")
        self.torch = torch
        self.module = _load_backend_with_cpu_stubs()
        self.backend = object.__new__(self.module.MambaAttnBackend)
        self.backend._num_state_shards = 1
        self.backend._shard_group_ids = ("linear_attention_shard0",)
        self.backend._flat_state_page_size = 2
        self.backend.pad_slot_id = -1
        self.backend.speculative_num_draft_tokens = 3

    def test_mixed_batch_keeps_extend_pages_and_gathers_decode_suffix(self):
        torch = self.torch

        class MixedMode:
            @staticmethod
            def is_decode_or_idle():
                return False

        state_in, state_out, before = self.backend._flat_state_pages(
            2,
            torch.tensor([6, 8], dtype=torch.int32),
            MixedMode(),
            {
                "extend_prefix_lens": torch.tensor([0], dtype=torch.int32),
                "flat_state_pages": torch.tensor(
                    [[[0, -1]], [[12, -1]]], dtype=torch.int32
                ),
                "flat_block_tables": {
                    "linear_attention_shard0": torch.tensor(
                        [[10, 11, 12, 13], [20, 21, 22, 23]],
                        dtype=torch.int32,
                    )
                },
            },
            validate=True,
        )

        self.assertEqual(before.tolist(), [0, 7])
        self.assertEqual(state_in.tolist(), [[0, 23]])
        self.assertEqual(state_out.tolist(), [[12, 23]])

    def test_verify_uses_gpu_lengths_and_only_scheduler_spec_pages(self):
        torch = self.torch
        state_in, state_spec, state_out = self.backend._flat_state_verify_pages(
            1,
            torch.tensor([8], dtype=torch.int32),
            {
                "flat_state_spec_pages": torch.tensor(
                    [[[101, 102, 103]]], dtype=torch.int32
                ),
                "flat_block_tables": {
                    "linear_attention_shard0": torch.tensor(
                        [[10, 11, 50, 60]], dtype=torch.int32
                    )
                },
            },
            validate=True,
        )

        self.assertEqual(state_in.tolist(), [[50]])
        self.assertEqual(state_spec.tolist(), [[[101, 102, 103]]])
        self.assertEqual(state_out.tolist(), [[[50, 60, 60]]])


class FlatMtpBackendPostSamplingTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs CPU PyTorch: {exc}")
        self.torch = torch
        self.backend_module = _load_backend_with_cpu_stubs()

    def test_flat_commit_uses_real_rows_and_never_simple_pool(self):
        torch = self.torch
        backend = object.__new__(self.backend_module.HybridLinearAttnBackend)
        spec = torch.tensor(
            [[[101, 102, 103, 104], [-1, -1, -1, -1]]], dtype=torch.int32
        )
        out = torch.tensor([[[9, 9, 10, 10], [-1, -1, -1, -1]]], dtype=torch.int32)
        forbidden_pool = SimpleNamespace(
            update_current_inputs_after_verify=mock.Mock(
                side_effect=AssertionError("Flat must not touch SimpleMambaPool")
            )
        )
        flat_kv_pool = object()
        backend.linear_attn_backend = SimpleNamespace(
            flat_state_active=True,
            kv_pool=flat_kv_pool,
            pool=forbidden_pool,
            forward_metadata=SimpleNamespace(
                state_verify_spec_pages=spec,
                state_verify_out_pages=out,
            ),
        )
        accepted = torch.tensor([3], dtype=torch.int32)

        with mock.patch.object(
            self.backend_module,
            "commit_flat_mtp_state_pages",
            create=True,
        ) as commit:
            backend.update_mamba_state_after_mtp_verify(accepted, None)

        commit.assert_called_once()
        args = commit.call_args.args
        self.assertIs(args[0], flat_kv_pool)
        self.assertEqual(args[1].tolist(), [[[101, 102, 103, 104]]])
        self.assertEqual(args[2].tolist(), [[[9, 9, 10, 10]]])
        self.assertIs(args[3], accepted)
        forbidden_pool.update_current_inputs_after_verify.assert_not_called()

    def test_radix_still_updates_simple_pool_pointer(self):
        torch = self.torch
        backend = object.__new__(self.backend_module.HybridLinearAttnBackend)
        pool = SimpleNamespace(update_current_inputs_after_verify=mock.Mock())
        output_indices = torch.tensor([[31, 32, 33, 34]], dtype=torch.int32)
        req_indices = torch.tensor([7], dtype=torch.int32)
        backend.linear_attn_backend = SimpleNamespace(
            flat_state_active=False,
            pool=pool,
            forward_metadata=SimpleNamespace(
                mamba_output_indices=output_indices,
                mamba_req_pool_indices=req_indices,
            ),
        )
        accepted = torch.tensor([2], dtype=torch.int32)

        backend.update_mamba_state_after_mtp_verify(accepted, None)

        pool.update_current_inputs_after_verify.assert_called_once()
        args = pool.update_current_inputs_after_verify.call_args.args
        self.assertEqual(args[0].tolist(), req_indices.tolist())
        self.assertEqual(args[1].tolist(), output_indices.tolist())
        self.assertIs(args[2], accepted)

    def test_wrapper_hook_stays_after_sampling_without_stream_sync(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "python/tokenspeed/runtime/execution/cuda_graph_wrapper.py"
        ).read_text()
        hook = source.index("# Update mamba/GDN state after speculative verify")
        eager_sampling = source.rindex("result = self._forward_func", 0, hook)
        graph_sampling_result = source.rindex("result = (", 0, hook)

        self.assertGreater(hook, eager_sampling)
        self.assertGreater(hook, graph_sampling_result)
        post_sampling = source[min(eager_sampling, graph_sampling_result) : hook]
        self.assertNotIn(".synchronize(", post_sampling)
        self.assertNotIn("torch.cuda.stream(", post_sampling)


class FlatMtpStateCudaTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.linear.flat_mtp_state import (
                commit_flat_mtp_state_pages,
                commit_flat_mtp_state_rows,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs PyTorch + Triton: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        self.torch = torch
        self.commit_pages = commit_flat_mtp_state_pages
        self.commit_rows = commit_flat_mtp_state_rows

    def _strided_state(self, rows, shape, dtype):
        torch = self.torch
        row_elements = 1
        for dim in shape:
            row_elements *= dim
        row_stride = row_elements + 17
        inner_strides = []
        stride = 1
        for dim in reversed(shape):
            inner_strides.append(stride)
            stride *= dim
        state = torch.empty_strided(
            (rows, *shape),
            (row_stride, *reversed(inner_strides)),
            dtype=dtype,
            device="cuda",
        )
        for row in range(rows):
            values = torch.arange(
                row_elements, dtype=torch.float32, device="cuda"
            ).reshape(shape)
            state[row].copy_((values + row * 100).to(dtype))
        return state

    def test_strided_conv_and_ssm_rows_copy_selected_pages_exactly(self):
        torch = self.torch
        spec = torch.tensor(
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 11],
                    [-1, -1, -1, -1],
                ]
            ],
            dtype=torch.int32,
            device="cuda",
        )
        out = torch.tensor(
            [
                [
                    [8, 8, 9, 9],
                    [10, 10, 10, 10],
                    [-1, -1, -1, -1],
                ]
            ],
            dtype=torch.int32,
            device="cuda",
        )
        accepted = torch.tensor([3, 2, 4], dtype=torch.int32, device="cuda")

        for dtype, shape in (
            (torch.bfloat16, (5, 3)),
            (torch.float32, (2, 3, 4)),
        ):
            with self.subTest(dtype=dtype, shape=shape):
                state = self._strided_state(16, shape, dtype)
                before = state.clone()
                self.commit_rows(state, spec[0], out[0], accepted)
                torch.cuda.synchronize()

                self.assertTrue(torch.equal(state[8], before[2]))
                self.assertTrue(torch.equal(state[9], before[3]))
                self.assertTrue(torch.equal(state[10], before[6]))
                # Rejected checkpoint 4 and unrelated canonical page 12 are
                # byte-identical; the padded request is a -1 no-op.
                self.assertTrue(torch.equal(state[4], before[4]))
                self.assertTrue(torch.equal(state[12], before[12]))

    def test_multiple_shards_commit_conv_once_and_every_ssm_group(self):
        torch = self.torch
        conv = self._strided_state(20, (4, 3), torch.bfloat16)
        ssm0 = self._strided_state(20, (1, 2, 3), torch.float32)
        ssm1 = self._strided_state(20, (2, 2, 3), torch.float32)
        before = [tensor.clone() for tensor in (conv, ssm0, ssm1)]
        groups = [
            SimpleNamespace(conv=conv, ssm=ssm0, conv_shard=1, shard=0),
            SimpleNamespace(conv=conv, ssm=ssm1, conv_shard=1, shard=1),
        ]
        pool = SimpleNamespace(
            state_shard_view=SimpleNamespace(state_layer_ids=frozenset({7})),
            get_state_buffers=lambda _layer_id: groups,
        )
        spec = torch.tensor(
            [
                [[1, 2, 3, 4], [-1, -1, -1, -1]],
                [[5, 6, 7, 8], [-1, -1, -1, -1]],
            ],
            dtype=torch.int32,
            device="cuda",
        )
        out = torch.tensor(
            [
                [[10, 10, 11, 11], [-1, -1, -1, -1]],
                [[12, 12, 13, 13], [-1, -1, -1, -1]],
            ],
            dtype=torch.int32,
            device="cuda",
        )
        accepted = torch.tensor([3, 4], dtype=torch.int32, device="cuda")

        self.commit_pages(pool, spec, out, accepted)
        torch.cuda.synchronize()

        # conv uses conv_shard=1; both SSM groups use their own shard.
        self.assertTrue(torch.equal(conv[12], before[0][6]))
        self.assertTrue(torch.equal(conv[13], before[0][7]))
        self.assertTrue(torch.equal(ssm0[10], before[1][2]))
        self.assertTrue(torch.equal(ssm0[11], before[1][3]))
        self.assertTrue(torch.equal(ssm1[12], before[2][6]))
        self.assertTrue(torch.equal(ssm1[13], before[2][7]))


if __name__ == "__main__":
    unittest.main()
