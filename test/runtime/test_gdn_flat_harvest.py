"""Flat GDN prefill boundary-state harvest (M18c T5).

Each extend chunk's LAST interior whole-page boundary state (conv + ssm) is
harvested from the prefill kernel's checkpoints and scattered into that
boundary page's head-group views, making the page prefix-hittable. CPU tests
cover the harvest metadata (mask / checkpoint src / boundary page gather);
the GPU test checks the harvested page against a truncation oracle: rerunning
the same prompt cut at the boundary and reading its final state.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="runtime-1gpu")


def _make_flat_backend(
    torch, MambaAttnBackend, *, device, H, D, P, head_groups=None, num_shards=1
):
    config = SimpleNamespace(
        device=device,
        num_attention_heads=H,
        num_kv_heads=H,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        head_dim=D,
        is_draft=False,
        speculative_num_draft_tokens=1,
    )
    backend = MambaAttnBackend(config)
    stub_pool = SimpleNamespace(
        state_shard_view=SimpleNamespace(is_active=True),
        paged_cache_group_specs=tuple(
            SimpleNamespace(group_id=f"linear_attention_shard{i}")
            for i in range(num_shards)
        ),
        page_size=P,
        get_state_buffers=(
            (lambda layer_id: list(head_groups)) if head_groups is not None else None
        ),
    )
    backend.set_kv_pool(stub_pool)
    return backend


class FlatHarvestMetadataTest(unittest.TestCase):
    """CPU-only contract tests for the flat boundary-harvest metadata."""

    P = 64

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

    def _metadata(self, seq_lens, prefix_lens, rows, page_size=None):
        torch = self.torch
        backend = _make_flat_backend(
            torch,
            self.MambaAttnBackend,
            device="cpu",
            H=16,
            D=128,
            P=page_size or self.P,
        )
        self.assertTrue(backend.flat_state_active)
        bs = len(seq_lens)
        backend.init_forward_metadata(
            bs=bs,
            req_pool_indices=torch.arange(bs, dtype=torch.int32),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            forward_mode=self.ForwardMode.EXTEND,
            extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32),
            flat_block_tables={
                "linear_attention_shard0": torch.tensor(rows, dtype=torch.int32)
            },
        )
        return backend.forward_metadata

    def test_interior_boundary_tracks(self):
        # 100 tokens cross the slot-0 boundary at 64: harvest ckpt 0 into
        # the slot-0 page.
        md = self._metadata([100], [0], [[7, 9]])
        self.assertEqual(md.flat_track_mask.tolist(), [True])
        self.assertEqual(md.flat_track_lens.tolist(), [64])
        self.assertEqual(md.flat_track_ssm_src.tolist(), [0])
        self.assertEqual(md.flat_track_ssm_src_fla.tolist(), [1])
        self.assertEqual(md.flat_track_pages[0].tolist(), [7])
        self.assertEqual(md.flat_track_sel[0].tolist(), [0])

    def test_chunk_ending_on_boundary_skips(self):
        # Boundary == chunk end: the kernel's final-state write to state_out
        # already lands the boundary state on that page.
        md = self._metadata([128], [0], [[7, 9]])
        self.assertIsNone(md.flat_track_ssm_src)

    def test_short_prompt_skips(self):
        md = self._metadata([63], [0], [[7]])
        self.assertIsNone(md.flat_track_ssm_src)

    def test_prefix_chunk_tracks_on_page_grid(self):
        # Chunk [64, 228): last boundary 192 -> slot 2, chunk-relative len
        # 128 -> flashinfer ckpt 1 (interval = P), FLA h[2].
        md = self._metadata([228], [64], [[3, 5, 8, 9]])
        self.assertEqual(md.flat_track_mask.tolist(), [True])
        self.assertEqual(md.flat_track_lens.tolist(), [128])
        self.assertEqual(md.flat_track_ssm_src.tolist(), [1])
        self.assertEqual(md.flat_track_ssm_src_fla.tolist(), [2])
        self.assertEqual(md.flat_track_pages[0].tolist(), [8])

    def test_unaligned_prefix_skips(self):
        # prefix 32: the boundary is off the kernel checkpoint grid.
        md = self._metadata([161], [32], [[3, 5, 8]])
        self.assertIsNone(md.flat_track_ssm_src)

    def test_hole_boundary_page_skips(self):
        # Boundary page is a hole (0 = the aliased null row, never written).
        md = self._metadata([100], [0], [[0, 9]])
        self.assertIsNone(md.flat_track_ssm_src)

    def test_mixed_batch_masks_per_request(self):
        md = self._metadata(
            [100, 64, 200],
            [0, 0, 0],
            [[7, 9, -1, -1], [21, -1, -1, -1], [31, 33, 35, 36]],
        )
        self.assertEqual(md.flat_track_mask.tolist(), [True, False, True])
        # Checkpoint offsets accumulate per request (L // P = [1, 1, 3]):
        # req0 harvests its ckpt 0; req2 its ckpt 2 (boundary 192, slot 2).
        self.assertEqual(md.flat_track_ssm_src.tolist(), [0, 4])
        self.assertEqual(md.flat_track_pages[0].tolist(), [7, 35])
        self.assertEqual(md.flat_track_sel[0].tolist(), [0, 1])

    def test_non_multiple_page_size_disables_harvest(self):
        # P off the 64-token kernel grid (stub/test pools only): no harvest.
        md = self._metadata([10], [0], [[7, 9, 12]], page_size=4)
        self.assertIsNone(md.flat_track_ssm_src)
        self.assertIsNone(md.flat_track_mask)


class GDNFlatHarvestGPUTest(unittest.TestCase):
    """Harvested boundary page vs the truncation oracle (same prompt cut at
    the boundary, rerun on fresh slabs, final state read from its out page)."""

    H = 16
    D = 128
    P = 64
    PROMPT = 897  # single chunk; last boundary at 896 = slot 13
    WIDTH = 4

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
        self.torch = torch
        self.gdn = gdn
        self.ForwardMode = ForwardMode
        self.MambaAttnBackend = MambaAttnBackend
        self.StateHeadGroup = StateHeadGroup
        torch.manual_seed(0)

    # ---- fixtures ----

    def _make_inputs(self, total):
        torch = self.torch
        H, D = self.H, self.D
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
            A_log=torch.randn(self.H, device="cuda", dtype=torch.float32) * 0.1,
            dt_bias=torch.randn(self.H, device="cuda", dtype=torch.float32) * 0.1,
            a_full=torch.randn(total, self.H, device="cuda", dtype=torch.float32),
            b_full=torch.randn(total, self.H, device="cuda", dtype=torch.float32),
        )

    def _slabs(self, num_pages):
        torch = self.torch
        conv_slab = torch.zeros(
            num_pages,
            3 * self.H * self.D,
            self.WIDTH - 1,
            device="cuda",
            dtype=torch.bfloat16,
        )
        ssm_slab = torch.zeros(
            num_pages, self.H, self.D, self.D, device="cuda", dtype=torch.float32
        )
        return conv_slab, ssm_slab

    def _whole_layer_group(self, conv_slab, ssm_slab):
        return [
            self.StateHeadGroup(
                conv=conv_slab,
                ssm=ssm_slab,
                shard=0,
                conv_shard=0,
                head_begin=0,
                num_heads=self.H,
            )
        ]

    def _split_groups(self, conv_slab, ssm_slab):
        half = self.H // 2
        return [
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

    def _common_kwargs(self, inp):
        return dict(
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

    def _prefill_chunks(self, backend, inp, chunks, rows_per_shard):
        """Run extend chunks [(prefix, end), ...] over the flat backend."""
        torch = self.torch
        tables = {
            f"linear_attention_shard{i}": torch.tensor(
                [rows], dtype=torch.int32, device="cuda"
            )
            for i, rows in enumerate(rows_per_shard)
        }
        common = self._common_kwargs(inp)
        for prefix, end in chunks:
            backend.init_forward_metadata(
                bs=1,
                req_pool_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
                seq_lens=torch.tensor([end], dtype=torch.int32, device="cuda"),
                forward_mode=self.ForwardMode.EXTEND,
                extend_prefix_lens=torch.tensor(
                    [prefix], dtype=torch.int32, device="cuda"
                ),
                flat_block_tables=tables,
            )
            backend.forward_extend(
                None,
                None,
                None,
                layer=None,
                out_cache_loc=None,
                token_to_kv_pool=backend.kv_pool,
                bs=1,
                forward_mode=self.ForwardMode.EXTEND,
                mixed_qkv=inp.mixed_full[prefix:end],
                a=inp.a_full[prefix:end],
                b=inp.b_full[prefix:end],
                seq_len=end - prefix,
                **common,
            )

    def _oracle_boundary_page(self, inp, boundary):
        """Final conv/ssm state of the prompt truncated at the boundary,
        read from a fresh flat run's out page (no harvest fires there)."""
        conv_slab, ssm_slab = self._slabs(num_pages=boundary // self.P + 2)
        backend = _make_flat_backend(
            self.torch,
            self.MambaAttnBackend,
            device="cuda",
            H=self.H,
            D=self.D,
            P=self.P,
            head_groups=self._whole_layer_group(conv_slab, ssm_slab),
        )
        rows = list(range(1, boundary // self.P + 1))
        self._prefill_chunks(backend, inp, [(0, boundary)], [rows])
        self.assertIsNone(backend.forward_metadata.flat_track_ssm_src)
        out_page = rows[-1]
        return conv_slab[out_page].clone(), ssm_slab[out_page].clone()

    # ---- oracle equality ----

    def _run_and_check_oracle(
        self, chunks, head_group_builder, num_shards, ckpt_dtype=None
    ):
        torch = self.torch
        inp = self._make_inputs(self.PROMPT)
        boundary = (self.PROMPT // self.P) * self.P  # 896, slot 13
        oracle_conv, oracle_ssm = self._oracle_boundary_page(inp, boundary)
        if ckpt_dtype is not None:
            # The FLA h workspace is stored in k.dtype (bf16): the harvest is
            # a bf16 snapshot of the same fp32 accumulator the final state
            # comes from, so equality is exact after quantizing the oracle.
            oracle_ssm = oracle_ssm.to(ckpt_dtype).to(oracle_ssm.dtype)

        num_slots = (self.PROMPT - 1) // self.P + 1  # 15
        conv_slab, ssm_slab = self._slabs(num_pages=num_slots + 1)
        backend = _make_flat_backend(
            torch,
            self.MambaAttnBackend,
            device="cuda",
            H=self.H,
            D=self.D,
            P=self.P,
            head_groups=head_group_builder(conv_slab, ssm_slab),
            num_shards=num_shards,
        )
        rows = list(range(1, num_slots + 1))
        self._prefill_chunks(backend, inp, chunks, [rows] * num_shards)

        md = backend.forward_metadata
        self.assertIsNotNone(md.flat_track_ssm_src)
        boundary_page = rows[boundary // self.P - 1]  # 14
        for s in range(num_shards):
            self.assertEqual(md.flat_track_pages[s].tolist(), [boundary_page])

        # The boundary state is a snapshot of the exact same chunk-sequential
        # scan the truncated rerun performs (identical bf16 inputs, fp32
        # state math, identical 64-token chunking), so fp32 equality is
        # bitwise: tolerance 0.
        self.assertTrue(torch.equal(ssm_slab[boundary_page], oracle_ssm))
        # Conv windows are raw bf16 input copies: bitwise too.
        self.assertTrue(torch.equal(conv_slab[boundary_page], oracle_conv))
        self.assertGreater(ssm_slab[boundary_page].abs().max().item(), 0.0)
        # The null row is never harvested into.
        self.assertEqual(ssm_slab[0].abs().max().item(), 0.0)
        self.assertEqual(conv_slab[0].abs().max().item(), 0.0)

    def test_single_chunk_harvest_matches_truncated_oracle(self):
        if not self.gdn.is_available():
            self.skipTest("sm100 GDN kernel unavailable")
        self._run_and_check_oracle(
            [(0, self.PROMPT)], self._whole_layer_group, num_shards=1
        )

    def test_single_chunk_harvest_matches_oracle_fla_layout(self):
        # Force the portable Triton kernel (FLA checkpoint layout).
        from tokenspeed_kernel.registry import KernelRegistry
        from tokenspeed_kernel.selection import kernel_override

        names = {
            spec.name
            for spec in KernelRegistry.get().get_for_operator(
                "attention", "gdn_chunk_prefill"
            )
        }
        if "triton_gdn_chunk_prefill" not in names:
            self.skipTest("triton gdn_chunk_prefill is not registered")
        with kernel_override(
            "attention", "gdn_chunk_prefill", "triton_gdn_chunk_prefill"
        ):
            self._run_and_check_oracle(
                [(0, self.PROMPT)],
                self._whole_layer_group,
                num_shards=1,
                ckpt_dtype=self.torch.bfloat16,
            )

    def test_chunked_prefill_harvest_with_split_head_groups(self):
        # Page-aligned first chunk (no harvest: it ends ON the boundary),
        # unaligned tail chunk harvests slot 13; k = 2 strided head groups.
        if not self.gdn.is_available():
            self.skipTest("sm100 GDN kernel unavailable")
        self._run_and_check_oracle(
            [(0, 512), (512, self.PROMPT)], self._split_groups, num_shards=2
        )

    # ---- robustness ----

    def test_hole_boundary_page_is_skipped(self):
        if not self.gdn.is_available():
            self.skipTest("sm100 GDN kernel unavailable")
        torch = self.torch
        inp = self._make_inputs(self.PROMPT)
        num_slots = (self.PROMPT - 1) // self.P + 1
        conv_slab, ssm_slab = self._slabs(num_pages=num_slots + 1)
        backend = _make_flat_backend(
            torch,
            self.MambaAttnBackend,
            device="cuda",
            H=self.H,
            D=self.D,
            P=self.P,
            head_groups=self._whole_layer_group(conv_slab, ssm_slab),
        )
        rows = list(range(1, num_slots + 1))
        rows[self.PROMPT // self.P - 1] = 0  # boundary slot is a hole
        self._prefill_chunks(backend, inp, [(0, self.PROMPT)], [rows])
        self.assertIsNone(backend.forward_metadata.flat_track_ssm_src)
        self.assertEqual(ssm_slab[0].abs().max().item(), 0.0)
        self.assertEqual(conv_slab[0].abs().max().item(), 0.0)

    def test_mixed_batch_harvests_only_boundary_crossers(self):
        if not self.gdn.is_available():
            self.skipTest("sm100 GDN kernel unavailable")
        torch = self.torch
        len0, len1 = 100, 64  # req0 crosses slot 0; req1 ends ON it
        inp = self._make_inputs(len0 + len1)
        conv_slab, ssm_slab = self._slabs(num_pages=8)
        backend = _make_flat_backend(
            torch,
            self.MambaAttnBackend,
            device="cuda",
            H=self.H,
            D=self.D,
            P=self.P,
            head_groups=self._whole_layer_group(conv_slab, ssm_slab),
        )
        tables = {
            "linear_attention_shard0": torch.tensor(
                [[1, 2], [3, -1]], dtype=torch.int32, device="cuda"
            )
        }
        backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor([len0, len1], dtype=torch.int32, device="cuda"),
            forward_mode=self.ForwardMode.EXTEND,
            extend_prefix_lens=torch.zeros(2, dtype=torch.int32, device="cuda"),
            flat_block_tables=tables,
        )
        md = backend.forward_metadata
        self.assertEqual(md.flat_track_mask.tolist(), [True, False])
        self.assertEqual(md.flat_track_pages[0].tolist(), [1])
        backend.forward_extend(
            None,
            None,
            None,
            layer=None,
            out_cache_loc=None,
            token_to_kv_pool=backend.kv_pool,
            bs=2,
            forward_mode=self.ForwardMode.EXTEND,
            mixed_qkv=inp.mixed_full,
            a=inp.a_full,
            b=inp.b_full,
            seq_len=len0 + len1,
            **self._common_kwargs(inp),
        )
        # req0's boundary page harvested; req1's only page holds its final
        # state via the normal out-page write.
        self.assertGreater(ssm_slab[1].abs().max().item(), 0.0)
        self.assertGreater(ssm_slab[3].abs().max().item(), 0.0)
        self.assertEqual(ssm_slab[0].abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
