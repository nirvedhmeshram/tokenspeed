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


if __name__ == "__main__":
    unittest.main()
