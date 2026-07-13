"""Task1: multi-point interior state-boundary enumeration (per-P harvest).

The GDN flat prefill harvest registers every P-aligned interior boundary of a
chunk whose prefix is also P-aligned. The kernel emits a state checkpoint every
P chunk-relative tokens; ``enumerate_interior_state_boundaries`` supplies the
absolute candidate boundaries and the metadata builder drops candidates that
do not lie on the kernel grid.

CPU-only: pure integer/tensor index math, no kernels.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")


class EnumerateInteriorStateBoundariesTest(unittest.TestCase):
    """Contract for the pure multi-point boundary enumerator."""

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
                enumerate_interior_state_boundaries,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
        self.fn = enumerate_interior_state_boundaries

    def _run(self, prefix, final, page_size):
        torch = self.torch
        return self.fn(
            torch.tensor(prefix, dtype=torch.int64),
            torch.tensor(final, dtype=torch.int64),
            page_size,
        )

    def test_single_request_two_interior_boundaries(self):
        # prefix=0, final=700, P=256: interior P-aligned boundaries strictly
        # inside (0, 700) are 256 and 512 (768 > 700). Their checkpoint slots
        # are b/P - 1 = 0 and 1. The tail remainder 700 % 256 = 188 does not
        # produce a boundary (no full interval past 512).
        boundaries, slots, req_idx = self._run([0], [700], 256)
        self.assertEqual(boundaries.tolist(), [256, 512])
        self.assertEqual(slots.tolist(), [0, 1])
        self.assertEqual(req_idx.tolist(), [0, 0])

    def test_boundary_on_final_is_excluded(self):
        # final=512 lands ON a boundary: the kernel's final-state write already
        # lands slot 1, so only the strictly-interior 256 (slot 0) harvests.
        boundaries, slots, req_idx = self._run([0], [512], 256)
        self.assertEqual(boundaries.tolist(), [256])
        self.assertEqual(slots.tolist(), [0])
        self.assertEqual(req_idx.tolist(), [0])

    def test_boundaries_below_prefix_excluded(self):
        # prefix=300 (already computed): only boundaries strictly inside
        # (300, 700) harvest -> 512 (slot 1). 256 is <= prefix, already cached.
        boundaries, slots, req_idx = self._run([300], [700], 256)
        self.assertEqual(boundaries.tolist(), [512])
        self.assertEqual(slots.tolist(), [1])
        self.assertEqual(req_idx.tolist(), [0])

    def test_no_interior_boundary(self):
        # Short unaligned prompt: final=200 < P=256, no interior boundary.
        boundaries, slots, req_idx = self._run([0], [200], 256)
        self.assertEqual(boundaries.tolist(), [])
        self.assertEqual(slots.tolist(), [])
        self.assertEqual(req_idx.tolist(), [])

    def test_batch_flattened_compact(self):
        # req0: (0, 700) -> 256, 512 (slots 0, 1)
        # req1: (0, 300) -> 256 (slot 0)
        # req2: (600, 900) -> 768 (slot 2)   [512 <= prefix 600, excluded]
        boundaries, slots, req_idx = self._run([0, 0, 600], [700, 300, 900], 256)
        self.assertEqual(boundaries.tolist(), [256, 512, 256, 768])
        self.assertEqual(slots.tolist(), [0, 1, 0, 2])
        self.assertEqual(req_idx.tolist(), [0, 0, 1, 2])

    def test_slot_matches_decode_dual_page_semantics(self):
        # Correctness anchor: boundary b's checkpoint = state after the first b
        # tokens = the last position b-1's state. Decode reads it via
        # slot(after-1) = (b-1)//P. For b a multiple of P, (b-1)//P = b/P - 1,
        # which MUST equal the slot we assign. Verify across several P/b.
        for P in (64, 128, 256):
            for mult in (1, 2, 3):
                b = P * mult
                # decode-side slot for reading state at position b (before=b):
                decode_in_slot = (b - 1) // P
                boundaries, slots, _ = self._run([0], [b + P], P)
                # b is interior of (0, b+P); find its assigned slot.
                idx = boundaries.tolist().index(b)
                self.assertEqual(
                    slots.tolist()[idx],
                    decode_in_slot,
                    f"P={P} b={b}: harvest slot must match decode in-slot",
                )


class MultiBoundaryHarvestMetadataTest(unittest.TestCase):
    """_flat_boundary_track_metadata must harvest EVERY interior P boundary of
    a chunk (not just the last), flattened over (request, boundary) pairs with
    each checkpoint landing in its own page slot. CPU-only: pure index math on
    a stub backend, no kernels."""

    P = 64  # state page size (tokens); must be a multiple of FLA_CHUNK_SIZE=64

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (  # noqa: E501
                MambaAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        self.torch = torch
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
        backend.set_kv_pool(
            SimpleNamespace(
                state_shard_view=SimpleNamespace(is_active=True),
                paged_cache_group_specs=(
                    SimpleNamespace(group_id="linear_attention_shard0"),
                ),
                page_size=self.P,
            )
        )
        self.assertTrue(backend.flat_state_active)
        self.backend = backend

    def _harvest(self, rows, before, final):
        torch = self.torch
        return self.backend._flat_boundary_track_metadata(
            bs=len(final),
            seq_lens=torch.tensor(final, dtype=torch.int32),
            before=torch.tensor(before, dtype=torch.int32),
            kwargs={
                "flat_block_tables": {
                    "linear_attention_shard0": torch.tensor(rows, dtype=torch.int32)
                }
            },
        )

    def test_single_chunk_two_interior_boundaries(self):
        # prefix=0, final=240, P=64: interior boundaries 64,128,192 (all < 240),
        # checkpoint slots 0,1,2 -> pages at those slots. Slot 3 (256>240) is the
        # tail working page, not harvested.
        ft = self._harvest([[5, 6, 7, 8, 9]], before=[0], final=[240])
        self.assertIsNotNone(ft)
        self.assertEqual(ft.req_idx.tolist(), [0, 0, 0])
        self.assertEqual(ft.track_lens.tolist(), [64, 128, 192])
        self.assertEqual(ft.src_fi.tolist(), [0, 1, 2])
        self.assertEqual(ft.src_fla.tolist(), [1, 2, 3])
        self.assertEqual(ft.pages[0].tolist(), [5, 6, 7])
        self.assertEqual(ft.sel[0].tolist(), [0, 1, 2])

    def test_unaligned_prefix_does_not_harvest_off_grid_boundaries(self):
        # The kernel checkpoints at chunk-relative 64,128,..., so an unaligned
        # prefix cannot produce the states for absolute boundaries 128 and 192.
        self.assertIsNone(self._harvest([[5, 6, 7, 8, 9]], before=[96], final=[240]))

    def test_unaligned_request_does_not_suppress_aligned_batch_peer(self):
        ft = self._harvest(
            [[5, 6, 7, 8], [21, 22, 23, 24]],
            before=[0, 96],
            final=[160, 240],
        )
        self.assertIsNotNone(ft)
        self.assertEqual(ft.req_idx.tolist(), [0, 0])
        self.assertEqual(ft.track_lens.tolist(), [64, 128])
        self.assertEqual(ft.src_fi.tolist(), [0, 1])
        self.assertEqual(ft.src_fla.tolist(), [1, 2])
        self.assertEqual(ft.pages[0].tolist(), [5, 6])

    def test_hole_page_filtered_per_boundary(self):
        # Slot 1 is a hole (0): boundary 128 skipped, 64 and 192 kept.
        ft = self._harvest([[5, 0, 7, 8, 9]], before=[0], final=[240])
        self.assertIsNotNone(ft)
        self.assertEqual(ft.pages[0].tolist(), [5, 7])
        self.assertEqual(ft.sel[0].tolist(), [0, 2])

    def test_no_interior_boundary_returns_none(self):
        # final=40 < P=64: no interior boundary -> None.
        self.assertIsNone(self._harvest([[5, 6]], before=[0], final=[40]))

    def test_batch_multi_boundary_flattened(self):
        # req0 (0,240): boundaries 64,128,192 slots 0,1,2. req1 (0,100):
        # boundary 64 slot 0. Flattened req_idx = [0,0,0,1].
        ft = self._harvest(
            [[5, 6, 7, 8, 9], [21, 22, 23, 24, 25]],
            before=[0, 0],
            final=[240, 100],
        )
        self.assertIsNotNone(ft)
        self.assertEqual(ft.req_idx.tolist(), [0, 0, 0, 1])
        self.assertEqual(ft.track_lens.tolist(), [64, 128, 192, 64])
        self.assertEqual(ft.pages[0].tolist(), [5, 6, 7, 21])


if __name__ == "__main__":
    unittest.main()
