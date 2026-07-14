"""GDN dual-index state paging on the flat path (M17).

compute_state_page_indices maps per-request (seq_len_before, seq_len_after)
to (in, out) state page ids over the flat "linear_attention" block table;
the GPU test drives MambaAttnBackend in flat mode (prefill + decodes over
paged state slabs) against the FLA chunk_gated_delta_rule oracle run once
over the full contiguous sequence.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-1gpu")


class ComputeStatePageIndicesTest(unittest.TestCase):
    """CPU-only contract tests for the pure dual-index helper."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
                compute_state_page_indices,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.fn = compute_state_page_indices

    def _run(self, rows, before, after, page_size=4):
        torch = self.torch
        return self.fn(
            torch.tensor(rows, dtype=torch.int32),
            page_size,
            torch.tensor(before, dtype=torch.int32),
            torch.tensor(after, dtype=torch.int32),
        )

    def test_across_boundary(self):
        state_in, state_out = self._run([[7, 9, 12]], [4], [5])
        self.assertEqual(state_in.tolist(), [7])
        self.assertEqual(state_out.tolist(), [9])

    def test_within_page(self):
        state_in, state_out = self._run([[7, 9, 12]], [5], [6])
        self.assertEqual(state_in.tolist(), [9])
        self.assertEqual(state_out.tolist(), [9])

    def test_first_step_null_in_page(self):
        state_in, state_out = self._run([[7, 9, 12]], [0], [3])
        self.assertEqual(state_in.tolist(), [0])
        self.assertEqual(state_out.tolist(), [7])

    def test_resume_from_prefix_hit(self):
        state_in, state_out = self._run([[3, 5, 8]], [8], [9])
        self.assertEqual(state_in.tolist(), [5])
        self.assertEqual(state_out.tolist(), [8])

    def test_batch_mixed(self):
        # Distinct rows per request: out pages are exclusive per batch (the scheduler
        # invariant the validate path enforces).
        rows = [
            [7, 9, 12],
            [21, 22, 23],
            [31, 33, 35],
            [3, 5, 8],
        ]
        state_in, state_out = self._run(rows, [4, 5, 0, 8], [5, 6, 3, 9])
        self.assertEqual(state_in.tolist(), [7, 22, 0, 5])
        self.assertEqual(state_out.tolist(), [9, 22, 31, 8])

    def test_out_slot_hole_raises(self):
        with self.assertRaises(ValueError):
            self._run([[7, 0, 12]], [4], [5])

    def test_out_slot_pad_raises(self):
        with self.assertRaises(ValueError):
            self._run([[7, -1, 12]], [4], [5])

    def test_out_slot_past_table_raises(self):
        with self.assertRaises(ValueError):
            self._run([[7, 9]], [8], [9])

    def test_in_slot_hole_raises(self):
        # before=5 -> in slot 1 is a hole (0): a silent zero-state resume
        # must fail loud like the out-page case.
        with self.assertRaises(ValueError):
            self._run([[7, 0, 12]], [5], [6])

    def test_in_slot_pad_raises(self):
        with self.assertRaises(ValueError):
            self._run([[7, -1, 12]], [5], [6])

    def test_duplicate_out_pages_raise(self):
        # req0: before=4 after=5 -> out slot 1 -> page 9; req1: before=0
        # after=1 -> out slot 0 -> page 9. All other guards pass (pages
        # positive, in-page valid/no history), so only the batch-uniqueness
        # invariant fires: two requests writing the same working state page
        # would silently clobber each other.
        with self.assertRaisesRegex(ValueError, "unique"):
            self._run([[7, 9, 12], [9, 22, 23]], [4, 0], [5, 1])

    def test_no_history_null_in_page_passes(self):
        # before=0 legitimately reads the null page 0 (see
        # test_first_step_null_in_page); the in-page guard must not fire.
        state_in, state_out = self._run([[7, 9, 12]], [0], [1])
        self.assertEqual(state_in.tolist(), [0])
        self.assertEqual(state_out.tolist(), [7])

    def test_validate_off_masks_guards(self):
        torch = self.torch
        state_in, state_out = self.fn(
            torch.tensor([[0, 0, 0]], dtype=torch.int32),
            4,
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            validate=False,
        )
        self.assertEqual(state_in.tolist(), [0])
        self.assertEqual(state_out.tolist(), [0])


class PoollessFlatMetadataTest(unittest.TestCase):
    """Flat mode runs without a SimpleMambaPool (the runner no longer creates
    one), so every metadata entry point must tolerate ``pool is None``.
    CPU-only: pure index math, no kernels."""

    P = 4  # state page size (tokens)

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.execution.forward_batch_info import (
                ForwardMode,
            )
            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
                MambaAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.ForwardMode = ForwardMode
        config = SimpleNamespace(
            device="cpu",
            num_attention_heads=16,
            num_kv_heads=16,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            head_dim=128,
            is_draft=False,
            speculative_num_draft_tokens=1,
        )
        backend = MambaAttnBackend(config)
        stub_pool = SimpleNamespace(
            state_shard_view=SimpleNamespace(is_active=True),
            paged_cache_group_specs=(
                SimpleNamespace(group_id="linear_attention_shard0"),
            ),
            page_size=self.P,
        )
        # set_pool is intentionally never called: flat mode has no
        # SimpleMambaPool.
        backend.set_kv_pool(stub_pool)
        self.assertTrue(backend.flat_state_active)
        self.assertIsNone(backend.pool)
        self.backend = backend

    def test_decode_metadata_without_pool(self):
        torch = self.torch
        backend = self.backend
        backend.init_forward_metadata(
            bs=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([9], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            flat_block_tables={
                "linear_attention_shard0": torch.tensor([[1, 2, 3]], dtype=torch.int32)
            },
        )
        md = backend.forward_metadata
        # before = 8 -> page slot 1 (row 2); after = 9 -> page slot 2 (row 3).
        # [k, bs] with k = 1.
        self.assertEqual(md.state_in_pages.tolist(), [[2]])
        self.assertEqual(md.state_out_pages.tolist(), [[3]])
        self.assertEqual(md.state_seq_lens_before.tolist(), [8])

    def test_extend_metadata_without_pool(self):
        torch = self.torch
        backend = self.backend
        backend.init_forward_metadata(
            bs=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
            forward_mode=self.ForwardMode.EXTEND,
            extend_prefix_lens=torch.zeros(1, dtype=torch.int32),
            flat_block_tables={
                "linear_attention_shard0": torch.tensor([[1, 2]], dtype=torch.int32)
            },
        )
        md = backend.forward_metadata
        self.assertEqual(md.state_in_pages.tolist(), [[0]])
        self.assertEqual(md.state_out_pages.tolist(), [[2]])
        self.assertEqual(md.state_seq_lens_before.tolist(), [0])

    def test_capture_replay_metadata_without_pool(self):
        torch = self.torch
        backend = self.backend
        backend.init_cuda_graph_state(max_num_tokens=2)
        backend.init_forward_metadata_capture_cuda_graph(
            bs=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([1], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            flat_cache_group_ids=("linear_attention_shard0",),
        )
        md = backend.forward_metadata
        # Capture binds the persistent pad-filled buffers.
        self.assertEqual(md.state_in_pages.tolist(), [[-1]])
        self.assertEqual(md.state_out_pages.tolist(), [[-1]])

        backend.init_forward_metadata_replay_cuda_graph(
            bs=1,
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([9], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            flat_block_tables={
                "linear_attention_shard0": torch.tensor([[1, 2, 3]], dtype=torch.int32)
            },
        )
        md = backend.forward_metadata
        self.assertEqual(md.state_in_pages.tolist(), [[2]])
        self.assertEqual(md.state_out_pages.tolist(), [[3]])


class ShardedFlatMetadataTest(unittest.TestCase):
    """k = 2 state shards (M18c): per-shard block tables stack into [k, bs]
    dual-index page tables; missing shard tables fail loud. CPU-only."""

    P = 4

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.execution.forward_batch_info import (
                ForwardMode,
            )
            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
                MambaAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.ForwardMode = ForwardMode
        self.MambaAttnBackend = MambaAttnBackend

    def _config(self):
        torch = self.torch
        return SimpleNamespace(
            device="cpu",
            num_attention_heads=16,
            num_kv_heads=16,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            head_dim=128,
            is_draft=False,
            speculative_num_draft_tokens=1,
        )

    def _backend(self, group_ids, view_active=True):
        backend = self.MambaAttnBackend(self._config())
        backend.set_kv_pool(
            SimpleNamespace(
                state_shard_view=SimpleNamespace(is_active=view_active),
                paged_cache_group_specs=tuple(
                    SimpleNamespace(group_id=gid) for gid in group_ids
                ),
                page_size=self.P,
            )
        )
        return backend

    def test_two_shard_tables_stack_per_group(self):
        torch = self.torch
        backend = self._backend(["linear_attention_shard0", "linear_attention_shard1"])
        self.assertTrue(backend.flat_state_active)
        self.assertEqual(backend._num_state_shards, 2)
        backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([9, 5], dtype=torch.int32),
            forward_mode=self.ForwardMode.DECODE,
            flat_block_tables={
                "linear_attention_shard0": torch.tensor(
                    [[1, 2, 3], [4, 5, -1]], dtype=torch.int32
                ),
                "linear_attention_shard1": torch.tensor(
                    [[11, 12, 13], [14, 15, -1]], dtype=torch.int32
                ),
            },
        )
        md = backend.forward_metadata
        # req0: before 8 -> slot 1, after 9 -> slot 2; req1: before 4 ->
        # slot 0, after 5 -> slot 1. Each shard pages the same slots from
        # its OWN table.
        self.assertEqual(tuple(md.state_in_pages.shape), (2, 2))
        self.assertEqual(md.state_in_pages.tolist(), [[2, 4], [12, 14]])
        self.assertEqual(md.state_out_pages.tolist(), [[3, 5], [13, 15]])
        self.assertEqual(md.state_seq_lens_before.tolist(), [8, 4])

    def test_missing_shard_table_raises(self):
        torch = self.torch
        backend = self._backend(["linear_attention_shard0", "linear_attention_shard1"])
        with self.assertRaisesRegex(RuntimeError, "linear_attention_shard1"):
            backend.init_forward_metadata(
                bs=1,
                req_pool_indices=torch.tensor([0], dtype=torch.int32),
                seq_lens=torch.tensor([9], dtype=torch.int32),
                forward_mode=self.ForwardMode.DECODE,
                flat_block_tables={
                    "linear_attention_shard0": torch.tensor(
                        [[1, 2, 3]], dtype=torch.int32
                    )
                },
            )

    def test_gate_requires_active_view_and_shard_groups(self):
        # Inactive view -> off, even with shard groups published.
        backend = self._backend(["linear_attention_shard0"], view_active=False)
        self.assertFalse(backend.flat_state_active)
        self.assertEqual(backend._num_state_shards, 0)
        # Legacy bare-name publication (no shard groups) -> off.
        backend = self._backend(["linear_attention"])
        self.assertFalse(backend.flat_state_active)
        # No state_shard_view attribute at all (legacy pools) -> off.
        backend = self.MambaAttnBackend(self._config())
        backend.set_kv_pool(
            SimpleNamespace(
                paged_cache_group_specs=(
                    SimpleNamespace(group_id="linear_attention_shard0"),
                ),
                page_size=self.P,
            )
        )
        self.assertFalse(backend.flat_state_active)


class GDNFlatStatePagingGPUTest(unittest.TestCase):
    """MambaAttnBackend in flat mode (paged state slabs, dual-index) vs the
    FLA chunk_gated_delta_rule oracle over the full contiguous sequence."""

    # Smallest fastpath parametrization: Hk = Hv = 16, D = 128 (sm100 GDN).
    H = 16
    D = 128
    P = 4  # state page size (tokens)
    PREFILL = 8
    DECODES = 3
    WIDTH = 4  # conv kernel width; state_len = WIDTH - 1

    def setUp(self):
        try:
            import torch
            from tokenspeed_kernel.ops.attention.flashinfer import (
                gated_delta_rule as gdn,
            )

            from tokenspeed.runtime.execution.forward_batch_info import (
                ForwardMode,
            )
            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
                MambaAttnBackend,
            )
            from tokenspeed.runtime.layers.attention.kv_cache.state_shard_view import (  # noqa: E501
                StateHeadGroup,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        if not gdn.is_available():
            self.skipTest("sm100 GDN kernel unavailable")
        self.torch = torch
        self.ForwardMode = ForwardMode
        self.MambaAttnBackend = MambaAttnBackend
        self.StateHeadGroup = StateHeadGroup
        torch.manual_seed(0)

    def _make_backend(self, head_groups, num_shards):
        torch = self.torch
        config = SimpleNamespace(
            device="cuda",
            num_attention_heads=self.H,
            num_kv_heads=self.H,
            attn_tp_size=1,
            dtype=torch.bfloat16,
            head_dim=self.D,
            is_draft=False,
            speculative_num_draft_tokens=1,
        )
        backend = self.MambaAttnBackend(config)
        # Flat mode is poolless: states live in the shard views only.
        stub_pool = SimpleNamespace(
            state_shard_view=SimpleNamespace(is_active=True),
            paged_cache_group_specs=tuple(
                SimpleNamespace(group_id=f"linear_attention_shard{i}")
                for i in range(num_shards)
            ),
            page_size=self.P,
            get_state_buffers=lambda layer_id: list(head_groups),
        )
        backend.set_kv_pool(stub_pool)
        self.assertTrue(backend.flat_state_active)
        return backend

    def _make_inputs(self):
        torch = self.torch
        H, D = self.H, self.D
        total = self.PREFILL + self.DECODES  # 11 tokens
        key_dim = H * D
        value_dim = H * D
        conv_dim = 2 * key_dim + value_dim
        return SimpleNamespace(
            total=total,
            key_dim=key_dim,
            value_dim=value_dim,
            conv_dim=conv_dim,
            mixed_full=torch.randn(
                total, conv_dim, device="cuda", dtype=torch.bfloat16
            ),
            conv_weights=torch.randn(
                conv_dim, self.WIDTH, device="cuda", dtype=torch.bfloat16
            )
            * 0.1,
            bias=torch.randn(conv_dim, device="cuda", dtype=torch.bfloat16) * 0.1,
            A_log=torch.randn(H, device="cuda", dtype=torch.float32) * 0.1,
            dt_bias=torch.randn(H, device="cuda", dtype=torch.float32) * 0.1,
            a_full=torch.randn(total, H, device="cuda", dtype=torch.float32),
            b_full=torch.randn(total, H, device="cuda", dtype=torch.float32),
        )

    def _oracle(self, inp):
        """One contiguous pass over all tokens: (o_ref, st_ref)."""
        torch = self.torch
        from tokenspeed_kernel.ops.attention.triton.linear.chunk import (
            chunk_gated_delta_rule,
        )

        from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
            causal_conv1d_fn,
        )
        from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating

        H, D = self.H, self.D
        total = inp.total
        ref_conv_state = torch.zeros(
            1, inp.conv_dim, self.WIDTH - 1, device="cuda", dtype=torch.bfloat16
        )
        conv_out = causal_conv1d_fn(
            inp.mixed_full.transpose(0, 1),
            inp.conv_weights,
            inp.bias,
            activation="silu",
            conv_states=ref_conv_state,
            has_initial_state=torch.zeros(1, dtype=torch.bool, device="cuda"),
            cache_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
            query_start_loc=torch.tensor([0, total], dtype=torch.int32, device="cuda"),
            seq_lens_cpu=torch.tensor([total], dtype=torch.int32),
        ).transpose(0, 1)[:total]
        q_ref, k_ref, v_ref = torch.split(
            conv_out, [inp.key_dim, inp.key_dim, inp.value_dim], dim=-1
        )
        q_ref = q_ref.view(1, total, H, D)
        k_ref = k_ref.view(1, total, H, D)
        v_ref = v_ref.view(1, total, H, D)
        g_ref = fused_gdn_gating(inp.A_log, inp.a_full, inp.dt_bias).view(1, total, H)
        beta_ref = inp.b_full.sigmoid().to(torch.bfloat16).view(1, total, H)
        return chunk_gated_delta_rule(
            q=q_ref,
            k=k_ref,
            v=v_ref,
            g=g_ref,
            beta=beta_ref,
            initial_state=torch.zeros(1, H, D, D, device="cuda", dtype=torch.float32),
            output_final_state=True,
            cu_seqlens=torch.tensor([0, total], device="cuda").long(),
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )

    def _drive_flat(self, backend, inp, num_shards, snapshot):
        """Prefill + 3 decodes over the flat path; returns (o_flat,
        after_prefill_snapshots) where the snapshots come from calling
        ``snapshot()`` right after the prefill step."""
        torch = self.torch
        ForwardMode = self.ForwardMode
        req_pool_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
        common = dict(
            conv_weights=inp.conv_weights,
            bias=inp.bias,
            activation="silu",
            key_dim=inp.key_dim,
            value_dim=inp.value_dim,
            attention_tp_size=1,
            head_k_dim=self.D,
            head_v_dim=self.D,
            A_log=inp.A_log,
            dt_bias=inp.dt_bias,
            layer_id=0,
        )
        stub = backend.kv_pool

        def tables(rows):
            t = torch.tensor([rows], dtype=torch.int32, device="cuda")
            return {f"linear_attention_shard{i}": t for i in range(num_shards)}

        # Prefill 8 tokens: in = null page 0, out = page 2 (slot 1); the
        # [k, bs] tables carry the same slots from every shard's table.
        backend.init_forward_metadata(
            bs=1,
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor([self.PREFILL], dtype=torch.int32, device="cuda"),
            forward_mode=ForwardMode.EXTEND,
            extend_prefix_lens=torch.zeros(1, dtype=torch.int32, device="cuda"),
            flat_block_tables=tables([1, 2]),
        )
        self.assertEqual(
            backend.forward_metadata.state_in_pages.tolist(), [[0]] * num_shards
        )
        self.assertEqual(
            backend.forward_metadata.state_out_pages.tolist(), [[2]] * num_shards
        )
        outputs = [
            backend.forward_extend(
                None,
                None,
                None,
                layer=None,
                out_cache_loc=None,
                token_to_kv_pool=stub,
                bs=1,
                forward_mode=ForwardMode.EXTEND,
                mixed_qkv=inp.mixed_full[: self.PREFILL],
                a=inp.a_full[: self.PREFILL],
                b=inp.b_full[: self.PREFILL],
                seq_len=self.PREFILL,
                **common,
            )
        ]
        after_prefill = snapshot()

        # 3 decode steps: page ids (in, out) = (2, 3), (3, 3), (3, 3).
        expected_pages = [(2, 3), (3, 3), (3, 3)]
        for i in range(self.DECODES):
            pos = self.PREFILL + i
            backend.init_forward_metadata(
                bs=1,
                req_pool_indices=req_pool_indices,
                seq_lens=torch.tensor([pos + 1], dtype=torch.int32, device="cuda"),
                forward_mode=ForwardMode.DECODE,
                flat_block_tables=tables([1, 2, 3]),
            )
            self.assertEqual(
                backend.forward_metadata.state_in_pages.tolist(),
                [[expected_pages[i][0]]] * num_shards,
            )
            self.assertEqual(
                backend.forward_metadata.state_out_pages.tolist(),
                [[expected_pages[i][1]]] * num_shards,
            )
            outputs.append(
                backend.forward_decode(
                    None,
                    None,
                    None,
                    layer=None,
                    out_cache_loc=None,
                    token_to_kv_pool=stub,
                    bs=1,
                    mixed_qkv=inp.mixed_full[pos : pos + 1],
                    a=inp.a_full[pos : pos + 1],
                    b=inp.b_full[pos : pos + 1],
                    **common,
                )
            )
        return torch.cat(outputs, dim=1), after_prefill

    def _check_oracle_match(self, o_flat, o_ref, ssm_slab, st_ref):
        torch = self.torch
        self.assertEqual(tuple(o_flat.shape), tuple(o_ref.shape))
        # Fastpath-test tolerances: mean diff is the real bar, loose max.
        out_diff = (o_flat.float() - o_ref.float()).abs()
        self.assertLess(out_diff.mean().item(), 1e-3)
        self.assertTrue(
            torch.allclose(o_flat.float(), o_ref.float(), atol=1e-1, rtol=1e-2)
        )
        st_diff = (ssm_slab[3] - st_ref[0].float()).abs()
        self.assertLess(st_diff.mean().item(), 1e-3)

    def _slabs(self):
        torch = self.torch
        num_pages = (self.PREFILL + self.DECODES) // self.P + 2  # null + 1..3
        conv_slab = torch.zeros(
            num_pages,
            2 * self.H * self.D + self.H * self.D,
            self.WIDTH - 1,
            device="cuda",
            dtype=torch.bfloat16,
        )
        ssm_slab = torch.zeros(
            num_pages, self.H, self.D, self.D, device="cuda", dtype=torch.float32
        )
        return conv_slab, ssm_slab

    def test_flat_paged_states_match_fla_oracle(self):
        torch = self.torch
        inp = self._make_inputs()
        o_ref, st_ref = self._oracle(inp)

        # ---- Flat path: page 0 = null, pages fill as the sequence grows ----
        conv_slab, ssm_slab = self._slabs()
        group = self.StateHeadGroup(
            conv=conv_slab,
            ssm=ssm_slab,
            shard=0,
            conv_shard=0,
            head_begin=0,
            num_heads=self.H,
        )
        backend = self._make_backend([group], num_shards=1)

        o_flat, (conv_page2, ssm_page2) = self._drive_flat(
            backend,
            inp,
            num_shards=1,
            snapshot=lambda: (conv_slab[2].clone(), ssm_slab[2].clone()),
        )
        self._check_oracle_match(o_flat, o_ref, ssm_slab, st_ref)

        # Null page 0 must never be written; page 2 (prefill's out page)
        # keeps the shared snapshot untouched by the boundary-crossing decode.
        self.assertEqual(conv_slab[0].abs().max().item(), 0.0)
        self.assertEqual(ssm_slab[0].abs().max().item(), 0.0)
        self.assertTrue(torch.equal(conv_slab[2], conv_page2))
        self.assertTrue(torch.equal(ssm_slab[2], ssm_page2))
        self.assertGreater(ssm_slab[2].abs().max().item(), 0.0)
        self.assertGreater(ssm_slab[3].abs().max().item(), 0.0)

    def test_r1_zero_seed_survives_dirty_null_row(self):
        """Row 0 aliases the KV dummy page and is NOT zero (M18c): dirty it
        with garbage and verify the R1 zero-seeding still reproduces the
        oracle, without ever writing row 0."""
        torch = self.torch
        inp = self._make_inputs()
        o_ref, st_ref = self._oracle(inp)

        conv_slab, ssm_slab = self._slabs()
        conv_slab[0].fill_(7.0)
        ssm_slab[0].fill_(11.0)
        group = self.StateHeadGroup(
            conv=conv_slab,
            ssm=ssm_slab,
            shard=0,
            conv_shard=0,
            head_begin=0,
            num_heads=self.H,
        )
        backend = self._make_backend([group], num_shards=1)

        o_flat, _ = self._drive_flat(backend, inp, num_shards=1, snapshot=lambda: None)
        self._check_oracle_match(o_flat, o_ref, ssm_slab, st_ref)
        # The dirty null row was read around (never through) and never written.
        self.assertTrue(torch.all(conv_slab[0] == 7.0))
        self.assertTrue(torch.all(ssm_slab[0] == 11.0))

    def test_two_head_group_shards_match_fla_oracle(self):
        """k = 2: the layer's ssm heads split across two shard groups whose
        views are NON-contiguous head slices of the slab (row stride = the
        whole slab row) — exercising the decode single launch's per-head
        absolute addressing across two shards and the extend gather/scatter."""
        torch = self.torch
        inp = self._make_inputs()
        o_ref, st_ref = self._oracle(inp)

        conv_slab, ssm_slab = self._slabs()
        half = self.H // 2
        groups = [
            self.StateHeadGroup(
                conv=conv_slab,
                ssm=ssm_slab[:, :half],
                shard=0,
                conv_shard=0,
                head_begin=0,
                num_heads=half,
            ),
            self.StateHeadGroup(
                conv=conv_slab,
                ssm=ssm_slab[:, half:],
                shard=1,
                conv_shard=0,
                head_begin=half,
                num_heads=half,
            ),
        ]
        for g in groups:
            self.assertFalse(g.ssm.is_contiguous())
        backend = self._make_backend(groups, num_shards=2)

        o_flat, _ = self._drive_flat(backend, inp, num_shards=2, snapshot=lambda: None)
        self._check_oracle_match(o_flat, o_ref, ssm_slab, st_ref)
        self.assertEqual(conv_slab[0].abs().max().item(), 0.0)
        self.assertEqual(ssm_slab[0].abs().max().item(), 0.0)

    def test_decode_single_issue_per_head_maps(self):
        """The flat decode collapses the k head groups into ONE kernel launch
        driven by per-head absolute-addressing maps: one map per layer, HV
        entries, head h's byte base == its ssm view's page-0 address, head h's
        shard == the group that owns it. The maps are constant tensors reused
        across steps (same object) -- the CUDA-graph capture prerequisite."""
        torch = self.torch
        inp = self._make_inputs()

        conv_slab, ssm_slab = self._slabs()
        half = self.H // 2
        groups = [
            self.StateHeadGroup(
                conv=conv_slab,
                ssm=ssm_slab[:, :half],
                shard=0,
                conv_shard=0,
                head_begin=0,
                num_heads=half,
            ),
            self.StateHeadGroup(
                conv=conv_slab,
                ssm=ssm_slab[:, half:],
                shard=1,
                conv_shard=0,
                head_begin=half,
                num_heads=half,
            ),
        ]
        backend = self._make_backend(groups, num_shards=2)
        self._drive_flat(backend, inp, num_shards=2, snapshot=lambda: None)

        self.assertEqual(list(backend._per_head_maps), [0])
        head_base, head_shard, page_row_stride = backend._per_head_maps[0]
        self.assertEqual(tuple(head_base.shape), (self.H,))
        self.assertEqual(tuple(head_shard.shape), (self.H,))
        self.assertEqual(head_base.dtype, torch.int64)
        self.assertEqual(head_shard.dtype, torch.int32)
        self.assertEqual(page_row_stride, ssm_slab[:, :half].stride(0))
        # Head h's base is its own ssm view's page-0 byte address, shard the
        # group that owns it (heads [0, half) -> shard 0, [half, H) -> 1).
        expected_base = [ssm_slab[:, h].data_ptr() for h in range(self.H)]
        self.assertEqual(head_base.tolist(), expected_base)
        self.assertEqual(head_shard.tolist(), [0] * half + [1] * (self.H - half))
        # A second decode reuses the SAME tensor objects (stable addresses).
        again = backend._flat_head_addressing(0, groups, self.H, self.H)
        self.assertIs(again[0], head_base)
        self.assertIs(again[1], head_shard)


if __name__ == "__main__":
    unittest.main()
