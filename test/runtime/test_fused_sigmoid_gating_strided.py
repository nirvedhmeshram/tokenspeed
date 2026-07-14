"""Dense vs strided-view h0 equivalence for the fused sigmoid gating kernel.

The flat GDN path (M18c state binning) feeds
``fused_sigmoid_gating_delta_rule_update`` NON-contiguous h0 sources — state
shard views whose dim 0 strides a whole K/V page row — and head-sliced
q/k/v/a/b per head group. The kernel addresses h0 rows via the runtime
``h0_row_stride`` and ``a`` tokens via ``stride_a``; for dense tensors both
collapse to the previous hardcoded layout, so a dense call and a
strided/sliced call over identical values must produce bitwise-identical
outputs and state writes.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")


class FusedSigmoidGatingStridedTest(unittest.TestCase):
    B = 4  # requests (decode: one token each)
    H = 4  # q/k heads
    HV = 8  # v heads (GQA ratio 2)
    K = 32
    V = 32
    N = 8  # state rows (pages)

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.linear.fused_sigmoid_gating_recurrent import (  # noqa: E501
                fused_sigmoid_gating_delta_rule_update,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        self.torch = torch
        self.fn = fused_sigmoid_gating_delta_rule_update
        torch.manual_seed(0)

    def _inputs(self):
        torch = self.torch
        B, H, HV, K, V, N = self.B, self.H, self.HV, self.K, self.V, self.N
        return SimpleNamespace(
            q=torch.randn(1, B, H, K, device="cuda", dtype=torch.bfloat16),
            k=torch.randn(1, B, H, K, device="cuda", dtype=torch.bfloat16),
            v=torch.randn(1, B, HV, V, device="cuda", dtype=torch.bfloat16),
            a=torch.randn(B, HV, device="cuda", dtype=torch.float32),
            b=torch.randn(B, HV, device="cuda", dtype=torch.float32),
            A_log=torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1,
            dt_bias=torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1,
            h0=torch.randn(N, HV, K, V, device="cuda", dtype=torch.float32),
            # One pad row (-1): the kernel must skip its read AND write.
            idx=torch.tensor([1, 3, -1, 5], dtype=torch.int32, device="cuda"),
            cu=torch.arange(B + 1, dtype=torch.int32, device="cuda"),
        )

    def _call(self, inp, source, indices, **overrides):
        kwargs = dict(
            A_log=inp.A_log,
            a=inp.a,
            dt_bias=inp.dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=inp.q,
            k=inp.k,
            v=inp.v,
            b=inp.b,
            initial_state_source=source,
            initial_state_indices=indices,
            cu_seqlens=inp.cu,
            use_qk_l2norm_in_kernel=True,
        )
        kwargs.update(overrides)
        return self.fn(**kwargs)

    def _strided_copy_of(self, dense):
        """A view with the same values but a padded row stride (2x), the
        state-shard-view layout: dim 0 non-contiguous, rows inner-contiguous."""
        torch = self.torch
        big = torch.zeros(
            dense.shape[0], 2, *dense.shape[1:], device="cuda", dtype=dense.dtype
        )
        strided = big[:, 0]
        strided.copy_(dense)
        self.assertFalse(strided.is_contiguous())
        return big, strided

    def test_strided_h0_write_back_matches_dense(self):
        torch = self.torch
        inp = self._inputs()
        dense = inp.h0.clone()
        big, strided = self._strided_copy_of(inp.h0)

        o_dense = self._call(inp, dense, inp.idx)
        o_strided = self._call(inp, strided, inp.idx)

        self.assertTrue(torch.equal(o_dense, o_strided))
        # Same reads -> same math -> bitwise-identical write-backs; rows not
        # in idx were equal by construction, so the whole sources match.
        self.assertTrue(torch.equal(dense, strided))
        # The padding lanes between strided rows must never be touched.
        self.assertEqual(big[:, 1].abs().max().item(), 0.0)

    def test_strided_h0_output_indices_matches_dense(self):
        torch = self.torch
        inp = self._inputs()
        dense = inp.h0.clone()
        big, strided = self._strided_copy_of(inp.h0)
        out_idx = torch.tensor([2, 4, 6, 7], dtype=torch.int32, device="cuda")

        # Flat decode mode: no in-place write-back, states land on out rows.
        o_dense = self._call(
            inp,
            dense,
            inp.idx,
            disable_state_update=True,
            output_state_indices=out_idx,
        )
        o_strided = self._call(
            inp,
            strided,
            inp.idx,
            disable_state_update=True,
            output_state_indices=out_idx,
        )

        self.assertTrue(torch.equal(o_dense, o_strided))
        self.assertTrue(torch.equal(dense, strided))
        self.assertEqual(big[:, 1].abs().max().item(), 0.0)
        # Out rows really were written (differ from the pristine source).
        self.assertFalse(torch.equal(dense, inp.h0))

    def test_head_sliced_calls_match_full_call(self):
        """Two half-head-group calls over view slices == one full call: the
        flat decode loop's slicing (q/k by the GQA ratio, v/a/b/A_log/dt_bias
        by v heads, h0 by a non-contiguous head slice)."""
        torch = self.torch
        inp = self._inputs()
        half = self.HV // 2
        qhalf = half // (self.HV // self.H)  # GQA ratio 2 -> 2 q heads
        full_src = inp.h0.clone()
        sliced_src = inp.h0.clone()

        o_full = self._call(inp, full_src, inp.idx)

        outs = []
        for lo, hi, qlo, qhi in (
            (0, half, 0, qhalf),
            (half, self.HV, qhalf, self.H),
        ):
            part = SimpleNamespace(
                q=inp.q[:, :, qlo:qhi],
                k=inp.k[:, :, qlo:qhi],
                v=inp.v[:, :, lo:hi],
                a=inp.a[..., lo:hi],
                b=inp.b[..., lo:hi],
                A_log=inp.A_log[lo:hi],
                dt_bias=inp.dt_bias[lo:hi],
                cu=inp.cu,
            )
            source = sliced_src[:, lo:hi]
            self.assertFalse(source.is_contiguous())
            outs.append(self._call(part, source, inp.idx))

        self.assertTrue(torch.equal(o_full, torch.cat(outs, dim=2)))
        self.assertTrue(torch.equal(full_src, sliced_src))

    def test_inner_discontiguous_source_raises(self):
        inp = self._inputs()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self._call(inp, inp.h0.transpose(2, 3), inp.idx)

    def test_head_count_mismatch_raises(self):
        inp = self._inputs()
        with self.assertRaisesRegex(ValueError, "heads"):
            self._call(inp, inp.h0[:, : self.HV // 2].contiguous(), inp.idx)


class FusedSigmoidGatingPerHeadTest(unittest.TestCase):
    """Per-head absolute addressing (M18c opt ①): one launch covers a whole
    layer whose HV ssm heads scatter across k separate K/V slab tensors. The
    single launch must be bitwise-identical to the retired per-shard loop
    (one strided-h0 call per shard, outputs cat'd).
    """

    B = 4  # requests (decode: one token each)
    H = 4  # q/k heads
    HV = 8  # v heads (GQA ratio 2)
    K = 32
    V = 32
    KSHARDS = 2  # heads split HV/2 per shard, each its own slab tensor
    NPAGES = 8  # state rows per slab (decoupled from the B requests)

    def setUp(self):
        try:
            import torch

            from tokenspeed.runtime.layers.attention.linear.fused_sigmoid_gating_recurrent import (  # noqa: E501
                fused_sigmoid_gating_delta_rule_update,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs torch + tokenspeed_kernel: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        self.torch = torch
        self.fn = fused_sigmoid_gating_delta_rule_update
        torch.manual_seed(0)

    def _inputs(self):
        torch = self.torch
        B, H, HV, K, V = self.B, self.H, self.HV, self.K, self.V
        return SimpleNamespace(
            q=torch.randn(1, B, H, K, device="cuda", dtype=torch.bfloat16),
            k=torch.randn(1, B, H, K, device="cuda", dtype=torch.bfloat16),
            v=torch.randn(1, B, HV, V, device="cuda", dtype=torch.bfloat16),
            a=torch.randn(B, HV, device="cuda", dtype=torch.float32),
            b=torch.randn(B, HV, device="cuda", dtype=torch.float32),
            A_log=torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1,
            dt_bias=torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1,
            h0=torch.randn(self.NPAGES, HV, K, V, device="cuda", dtype=torch.float32),
            cu=torch.arange(B + 1, dtype=torch.int32, device="cuda"),
        )

    def _make_shards(self, h0):
        """k separate slab tensors, each a padded (non-contiguous) view holding
        half the heads -- the real aliasing where a shard's ssm view strides a
        whole K/V page row. Returns (slabs, views): slabs are the backing
        storages (for write-back checks), views the strided (N, half, K, V)
        sources the retired per-shard path consumes."""
        torch = self.torch
        half = self.HV // self.KSHARDS
        N, K, V = h0.shape[0], self.K, self.V
        slabs, views = [], []
        for s in range(self.KSHARDS):
            # dim 1 == 2 pads the row so the view's dim-0 stride is a whole
            # (doubled) page row -> non-contiguous like the slab views.
            slab = torch.zeros(N, 2, half, K, V, device="cuda", dtype=torch.float32)
            view = slab[:, 0]
            view.copy_(h0[:, s * half : (s + 1) * half])
            self.assertFalse(view.is_contiguous())
            slabs.append(slab)
            views.append(view)
        return slabs, views

    def _head_maps(self, slabs):
        """head_base (byte addr per head), head_shard, page_row_stride from the
        slab views. All slabs share one page_row_stride (same shape)."""
        torch = self.torch
        half = self.HV // self.KSHARDS
        itemsize = slabs[0].element_size()
        page_row_stride = slabs[0].stride(0)  # elements per full (padded) row
        for slab in slabs:
            self.assertEqual(slab.stride(0), page_row_stride)
        base, shard = [], []
        for i in range(self.HV):
            s, l = i // half, i % half
            # page-0, head-l cell start of shard s's slab (byte address).
            base.append(slabs[s][:, 0][:, l].data_ptr())
            shard.append(s)
        return (
            torch.tensor(base, device="cuda", dtype=torch.int64),
            torch.tensor(shard, device="cuda", dtype=torch.int32),
            page_row_stride,
        )

    def _per_shard_ref(self, inp, views, in_idx, **overrides):
        """Retired path: one strided-h0 call per shard, outputs cat'd."""
        torch = self.torch
        half = self.HV // self.KSHARDS
        ratio = self.HV // self.H
        outs = []
        for s in range(self.KSHARDS):
            lo, hi = s * half, (s + 1) * half
            outs.append(
                self.fn(
                    A_log=inp.A_log[lo:hi],
                    dt_bias=inp.dt_bias[lo:hi],
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=inp.q[:, :, lo // ratio : hi // ratio],
                    k=inp.k[:, :, lo // ratio : hi // ratio],
                    v=inp.v[:, :, lo:hi],
                    b=inp.b[..., lo:hi],
                    a=inp.a[..., lo:hi],
                    initial_state_source=views[s],
                    initial_state_indices=in_idx[s],
                    cu_seqlens=inp.cu,
                    use_qk_l2norm_in_kernel=True,
                    **overrides,
                )
            )
        return torch.cat(outs, dim=2)

    def _per_head(self, inp, carrier, base, shard, prs, **overrides):
        """Single launch over the whole layer via per-head absolute addresses."""
        return self.fn(
            A_log=inp.A_log,
            dt_bias=inp.dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=inp.q,
            k=inp.k,
            v=inp.v,
            b=inp.b,
            a=inp.a,
            initial_state_source=carrier,
            initial_state_indices=None,
            cu_seqlens=inp.cu,
            use_qk_l2norm_in_kernel=True,
            head_base=base,
            head_shard=shard,
            page_row_stride=prs,
            **overrides,
        )

    def test_per_head_writeback_matches_per_shard(self):
        """In-place write-back: single launch == per-shard loop, bitwise, both
        for outputs and for the slab state written back."""
        torch = self.torch
        # One pad page (-1): the shard's read AND write-back must skip it.
        in_pages = torch.tensor(
            [[1, 3, -1, 2], [0, 3, -1, 1]], dtype=torch.int32, device="cuda"
        )

        slabs_ref, views_ref = self._make_shards(self._inputs().h0)
        slabs_new, views_new = self._make_shards(self._inputs().h0)
        # Same seed -> identical inputs for both arms.
        inp = self._inputs()

        o_ref = self._per_shard_ref(inp, views_ref, in_pages)
        o_new = self._per_head(
            inp,
            slabs_new[0],
            *self._head_maps(slabs_new),
            state_in_pages=in_pages,
        )

        self.assertTrue(torch.equal(o_ref, o_new))
        for sr, sn in zip(slabs_ref, slabs_new):
            self.assertTrue(torch.equal(sr, sn))
            # Padding lanes between rows must never be touched.
            self.assertEqual(sn[:, 1].abs().max().item(), 0.0)

    def test_per_head_output_pages_matches_per_shard(self):
        """Flat decode mode: disable_state_update + out pages, state lands on
        the out rows; single launch == per-shard loop, bitwise."""
        torch = self.torch
        in_pages = torch.tensor(
            [[1, 3, -1, 2], [0, 3, -1, 1]], dtype=torch.int32, device="cuda"
        )
        out_pages = torch.tensor(
            [[4, 5, 6, -1], [4, 5, 6, -1]], dtype=torch.int32, device="cuda"
        )

        slabs_ref, views_ref = self._make_shards(self._inputs().h0)
        slabs_new, views_new = self._make_shards(self._inputs().h0)
        inp = self._inputs()

        # Per-shard reference: each shard consumes its own in/out page row.
        half = self.HV // self.KSHARDS
        ratio = self.HV // self.H
        outs = []
        for s in range(self.KSHARDS):
            lo, hi = s * half, (s + 1) * half
            outs.append(
                self.fn(
                    A_log=inp.A_log[lo:hi],
                    dt_bias=inp.dt_bias[lo:hi],
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=inp.q[:, :, lo // ratio : hi // ratio],
                    k=inp.k[:, :, lo // ratio : hi // ratio],
                    v=inp.v[:, :, lo:hi],
                    b=inp.b[..., lo:hi],
                    a=inp.a[..., lo:hi],
                    initial_state_source=views_ref[s],
                    initial_state_indices=in_pages[s],
                    cu_seqlens=inp.cu,
                    use_qk_l2norm_in_kernel=True,
                    disable_state_update=True,
                    output_state_indices=out_pages[s],
                )
            )
        o_ref = torch.cat(outs, dim=2)

        o_new = self._per_head(
            inp,
            slabs_new[0],
            *self._head_maps(slabs_new),
            state_in_pages=in_pages,
            state_out_pages=out_pages,
            disable_state_update=True,
        )

        self.assertTrue(torch.equal(o_ref, o_new))
        for sr, sn in zip(slabs_ref, slabs_new):
            self.assertTrue(torch.equal(sr, sn))
            self.assertEqual(sn[:, 1].abs().max().item(), 0.0)

    def test_null_out_page_not_written(self):
        """A request whose out page is -1 leaves every candidate row of every
        shard untouched (no dirty write through a stale base)."""
        torch = self.torch
        in_pages = torch.tensor(
            [[1, 1, 1, 1], [2, 2, 2, 2]], dtype=torch.int32, device="cuda"
        )
        # Every request maps to out page -1: nothing must be written at all.
        out_pages = torch.full((2, self.B), -1, dtype=torch.int32, device="cuda")
        slabs, _ = self._make_shards(self._inputs().h0)
        before = [s.clone() for s in slabs]
        inp = self._inputs()

        self._per_head(
            inp,
            slabs[0],
            *self._head_maps(slabs),
            state_in_pages=in_pages,
            state_out_pages=out_pages,
            disable_state_update=True,
        )
        for b, s in zip(before, slabs):
            self.assertTrue(torch.equal(b, s))

    def test_missing_head_base_maps_raise(self):
        inp = self._inputs()
        slabs, _ = self._make_shards(inp.h0)
        base, shard, prs = self._head_maps(slabs)
        in_pages = torch.zeros(2, self.B, dtype=self.torch.int32, device="cuda")
        with self.assertRaisesRegex(ValueError, "head_shard"):
            self._per_head(inp, slabs[0], base, None, prs, state_in_pages=in_pages)
        with self.assertRaisesRegex(ValueError, "length HV"):
            self._per_head(
                inp, slabs[0], base[:-1], shard[:-1], prs, state_in_pages=in_pages
            )


if __name__ == "__main__":
    unittest.main()
