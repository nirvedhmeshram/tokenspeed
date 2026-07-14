# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    # Parameters for target_verify support (unused for decode)
    intermediate_states_buffer,
    cache_steps,
    output_state_indices,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    # ================================================
    # Per-head absolute addressing (flat GDN: one launch covers a whole
    # layer whose HV ssm heads live in k distinct K/V slab tensors, so a
    # single h0_source base can't reach them). Unused when addressing by
    # h0_source + h0_row_stride.
    head_base_ptr,  # int64[HV] byte address of each head's ssm view
    head_shard_ptr,  # int32[HV] which state_pages row pages the head
    state_in_pages_ptr,  # int32[k, bs] in-page table
    state_out_pages_ptr,  # int32[k, bs] out-page table
    state_pages_row_stride,  # elements per page-table row (= bs)
    page_row_stride,  # elements in one full page row (all slabs same shape)
    scale,
    T,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    stride_a,
    h0_row_stride,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    # Optional flags for target_verify support (default False for decode)
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    HAS_OUTPUT_STATE_INDICES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
    # Flat GDN: address each head by its own slab base + page offset.
    USE_PER_HEAD_ADDRESSING: tl.constexpr = False,
    HAS_STATE_IN_PAGES: tl.constexpr = False,
    HAS_STATE_OUT_PAGES: tl.constexpr = False,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    p_a = a + bos * stride_a + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_PER_HEAD_ADDRESSING:
        # Head i_hv's ssm view lives in its own slab tensor; head_base is
        # that view's byte address (h0_source carries only the state dtype),
        # in_page picks a row of that slab. Hoisted for the output write too.
        state_ty = h0_source.dtype.element_ty
        head_base = tl.load(head_base_ptr + i_hv).to(tl.pointer_type(state_ty))
        sh = tl.load(head_shard_ptr + i_hv)
        if HAS_STATE_IN_PAGES:
            in_page = tl.load(state_in_pages_ptr + sh * state_pages_row_stride + i_n)
            if in_page >= 0:
                p_h0 = (
                    head_base
                    + in_page.to(tl.int64) * page_row_stride
                    + o_k[:, None] * V
                    + o_v[None, :]
                )
                b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    elif USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            # h0_source rows may be non-contiguous strided views (flat GDN
            # state shards over the K/V slabs); rows stay inner-contiguous.
            p_h0 = (
                h0_source
                + idx.to(tl.int64) * h0_row_stride
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Prepare intermediate state cache index if enabled
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = i_n

    step_idx = 0
    for _ in range(0, T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Cache intermediate states if enabled
        if USE_PER_HEAD_ADDRESSING:
            # Flat decode writes the post-step state to the head's out page
            # (T == 1, so one out page per request). in_page < 0 heads have
            # nothing seeded and nothing to persist.
            if HAS_STATE_OUT_PAGES:
                out_page = tl.load(
                    state_out_pages_ptr + sh * state_pages_row_stride + i_n
                )
                if out_page >= 0:
                    output_ptr = (
                        head_base
                        + out_page.to(tl.int64) * page_row_stride
                        + o_k[:, None] * V
                        + o_v[None, :]
                    )
                    tl.store(
                        output_ptr, b_h.to(output_ptr.dtype.element_ty), mask=mask_h
                    )
        elif HAS_OUTPUT_STATE_INDICES:
            out_idx = tl.load(output_state_indices + i_n * T + step_idx).to(tl.int64)
            if out_idx >= 0:
                output_ptr = (
                    h0_source
                    + out_idx * h0_row_stride
                    + i_hv * K * V
                    + o_k[:, None] * V
                    + o_v[None, :]
                )
                tl.store(output_ptr, b_h.to(output_ptr.dtype.element_ty), mask=mask_h)
        elif CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_k[:, None] * V
                    + o_v[None, :]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        step_idx += 1

        # Update pointers for next timestep
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        p_a += stride_a

    # Store final state back to h0_source with bounds checking
    if not DISABLE_STATE_UPDATE:
        if USE_PER_HEAD_ADDRESSING:
            # In-place write-back to the head's in page (mirrors the legacy
            # h0 write-back); flat decode uses DISABLE_STATE_UPDATE + out
            # pages instead, so this only fires for an in-place caller.
            if HAS_STATE_IN_PAGES:
                in_page = tl.load(
                    state_in_pages_ptr + sh * state_pages_row_stride + i_n
                )
                if in_page >= 0:
                    p_h0 = (
                        head_base
                        + in_page.to(tl.int64) * page_row_stride
                        + o_k[:, None] * V
                        + o_v[None, :]
                    )
                    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)
        elif USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n)
            if idx >= 0:
                p_h0 = (
                    h0_source
                    + idx.to(tl.int64) * h0_row_stride
                    + i_hv * K * V
                    + o_k[:, None] * V
                    + o_v[None, :]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    # Optional parameters for target_verify support
    disable_state_update: bool = False,
    intermediate_states_buffer: torch.Tensor | None = None,
    cache_steps: int | None = None,
    output_state_indices: torch.Tensor | None = None,
    retrieve_parent_token: torch.Tensor | None = None,
    # Flat GDN per-head absolute addressing: one launch covers a whole
    # layer whose HV ssm heads scatter across k K/V slab tensors, so no
    # single initial_state_source base reaches them. When head_base is
    # given the kernel addresses each head by its own byte base + page,
    # and initial_state_source serves only as the state-dtype carrier.
    head_base: torch.Tensor | None = None,
    head_shard: torch.Tensor | None = None,
    state_in_pages: torch.Tensor | None = None,
    state_out_pages: torch.Tensor | None = None,
    page_row_stride: int | None = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.

    Supports both decode and target_verify modes:
    - decode: standard single-step update with state write-back
    - target_verify: multi-step with intermediate state caching, optional tree attention,
                     and optional state update disable
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    stride_q = q.stride()[1]
    stride_k = k.stride()[1]
    stride_v = v.stride()[1]
    stride_b = b.stride()[-2]
    stride_a = a.stride()[-2]
    HV = v.shape[2]
    use_per_head = head_base is not None
    state_pages_row_stride = 0
    if use_per_head:
        # Per-head mode: initial_state_source is only the state-dtype
        # carrier (each head reaches its own slab via head_base), so the
        # dense (HV, K, V) layout check below doesn't apply. Validate the
        # per-head maps instead.
        if initial_state_source is None:
            raise ValueError(
                "per-head addressing needs initial_state_source as the "
                "state-dtype carrier"
            )
        for name, t in (("head_base", head_base), ("head_shard", head_shard)):
            if t is None:
                raise ValueError(f"per-head addressing requires {name}")
            if t.ndim != 1 or t.shape[0] != HV:
                raise ValueError(
                    f"{name} must be 1-D of length HV={HV}; got " f"{tuple(t.shape)}"
                )
        if head_base.dtype != torch.int64:
            raise ValueError(f"head_base must be int64; got {head_base.dtype}")
        if head_shard.dtype != torch.int32:
            raise ValueError(f"head_shard must be int32; got {head_shard.dtype}")
        for name, t in (
            ("state_in_pages", state_in_pages),
            ("state_out_pages", state_out_pages),
        ):
            if t is not None and t.ndim != 2:
                raise ValueError(f"{name} must be 2-D [k, bs]; got {tuple(t.shape)}")
        # The kernel indexes both page tables with one state_pages_row_stride,
        # so a mismatched row stride would silently corrupt the out table.
        if (
            state_in_pages is not None
            and state_out_pages is not None
            and state_in_pages.stride(0) != state_out_pages.stride(0)
        ):
            raise ValueError(
                "state_in_pages and state_out_pages must share row stride; got "
                f"{state_in_pages.stride(0)} vs {state_out_pages.stride(0)}"
            )
        if page_row_stride is None:
            raise ValueError("per-head addressing requires page_row_stride")
        state_pages = state_in_pages if state_in_pages is not None else state_out_pages
        if state_pages is None:
            raise ValueError(
                "per-head addressing requires state_in_pages or state_out_pages"
            )
        # sh (a head_shard value) indexes page-table rows; i_n indexes columns.
        n_req = B if cu_seqlens is None else len(cu_seqlens) - 1
        if state_pages.shape[1] < n_req:
            raise ValueError(
                f"page table has {state_pages.shape[1]} columns; need >= "
                f"{n_req} requests"
            )
        # head_shard.max() reads device memory, illegal under graph capture;
        # the k bound is static per launch, so skip it while capturing.
        if not torch.cuda.is_current_stream_capturing():
            k_rows = int(head_shard.max().item()) + 1
            if state_pages.shape[0] < k_rows:
                raise ValueError(
                    f"page table has {state_pages.shape[0]} rows; head_shard "
                    f"selects up to row {k_rows - 1}"
                )
        state_pages_row_stride = state_pages.stride(0)
        h0_row_stride = 0
    # h0 rows may be non-contiguous strided views (flat GDN state shards
    # over the K/V slabs): dim 0 strides a whole page row. The kernel
    # addresses rows by h0_row_stride but keeps the dense in-row layout
    # (i_hv*K*V + o_k*V + o_v), so each row must stay inner-contiguous.
    elif initial_state_source is not None:
        src_hv, src_k, src_v = initial_state_source.shape[-3:]
        if initial_state_source.stride()[-3:] != (src_k * src_v, src_v, 1):
            raise ValueError(
                "initial_state_source rows must be contiguous (HV, K, V) "
                f"blocks; got shape {tuple(initial_state_source.shape)} with "
                f"strides {tuple(initial_state_source.stride())}"
            )
        if (src_hv, src_k, src_v) != (HV, K, V):
            raise ValueError(
                "initial_state_source rows must hold exactly this call's "
                f"(HV={HV}, K={K}, V={V}) heads; got "
                f"{tuple(initial_state_source.shape[-3:])}"
            )
        h0_row_stride = initial_state_source.stride(0)
    else:
        h0_row_stride = 0
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)

    # Prepare retrieve_parent_token strides
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq = retrieve_parent_token.stride(0)
        stride_retrieve_parent_token_token = retrieve_parent_token.stride(1)
    else:
        stride_retrieve_parent_token_seq = 0
        stride_retrieve_parent_token_token = 0

    NP2_T = triton.next_power_of_2(T)

    grid = (NK, NV, N * HV)

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate_states_buffer,
        cache_steps=0 if cache_steps is None else cache_steps,
        output_state_indices=output_state_indices,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        head_base_ptr=head_base,
        head_shard_ptr=head_shard,
        state_in_pages_ptr=state_in_pages,
        state_out_pages_ptr=state_out_pages,
        state_pages_row_stride=state_pages_row_stride if use_per_head else 0,
        page_row_stride=page_row_stride if use_per_head else 0,
        scale=scale,
        T=T,
        stride_q=stride_q,
        stride_k=stride_k,
        stride_v=stride_v,
        stride_b=stride_b,
        stride_a=stride_a,
        h0_row_stride=h0_row_stride,
        NP2_T=NP2_T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_OUTPUT_STATE_INDICES=output_state_indices is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        USE_PER_HEAD_ADDRESSING=use_per_head,
        HAS_STATE_IN_PAGES=use_per_head and state_in_pages is not None,
        HAS_STATE_OUT_PAGES=use_per_head and state_out_pages is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o
