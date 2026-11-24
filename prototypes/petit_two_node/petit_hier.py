# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

"""Two-level expert parallelism for Gluon Petit MegaMoE across nodes.

Petit's MegaMoE is one fused dispatch/compute/combine kernel over a node-local
VMM symmetric heap, so it cannot address GPUs on another node. This module
stacks a second, inter-node level on top of it:

* Experts are split into ``num_nodes`` contiguous groups; node ``n`` owns
  ``[n * E_node, (n + 1) * E_node)`` and runs Petit EP over its ``local_world``
  GPUs with ``E_node`` experts (``E_node / local_world`` per GPU).
* Rank ``(n, l)`` pairs with rank ``(1 - n, l)`` (two nodes). Each step it sends
  its quantized tokens, its top-k ids remapped to the peer node's local ids
  (``-1`` for experts that are not there) and its top-k weights to the peer, in
  three RCCL GIN puts through Iris's ``RcclGinProvider``.
* Each rank then calls Petit once on ``[own tokens | peer tokens]``; ids that
  are ``-1`` (or >= E_node) are skipped by Petit's dispatch.
* Petit writes its output straight into the GIN window. The peer rows go back
  in one put; the origin adds them to its own rows.

With top-6 routing over two nodes about 97% of tokens have an expert on the
other node, so every token is sent; there is no compaction and no host sync.

Assumes every rank calls ``forward`` with the same token count (the decode
batch case). The peer's rows land at ``[T, 2T)`` and Petit's token count is the
host value ``2T``; a ragged version would mask unused rows instead.

Buffer reuse needs no barrier: a rank overwrites its peer's receive regions for
step ``k + 1`` only after it got the peer's combine for step ``k``, which the
peer sends after Petit consumed step ``k``'s inputs.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton, tl, triton

with redirect_triton_to_tokenspeed_triton():
    import iris
    from iris.experimental import rccl_gin
    from iris.experimental.rccl_gin import RcclGinProvider


# ── node-local Petit heap ────────────────────────────────────────────────────


@contextmanager
def node_local_dist(group: dist.ProcessGroup):
    """Make Petit's heap exchange see ``group`` as the whole world.

    Petit's VMM heap calls ``torch.distributed.get_rank``, ``get_world_size``, ``all_gather_object``
    and ``barrier`` on the default group from C++; redirect them for the
    duration of heap creation and allocation.
    """
    saved = (dist.get_rank, dist.all_gather_object, dist.barrier, dist.get_world_size)
    local_rank, local_world = dist.get_rank(group), dist.get_world_size(group)
    dist.get_rank = lambda group_=None: local_rank
    dist.get_world_size = lambda group_=None: local_world
    dist.all_gather_object = lambda out, obj, group_=None: saved[1](
        out, obj, group=group
    )
    dist.barrier = lambda *a, **k: saved[2](group=group)
    try:
        yield
    finally:
        dist.get_rank, dist.all_gather_object, dist.barrier, dist.get_world_size = saved


class PetitNode:
    """Petit MegaMoE over one node-local group: config, heap and input views."""

    def __init__(
        self,
        petit_kernel,
        group,
        num_experts,
        topk,
        model_dim,
        inter_dim,
        max_tokens,
        activation_function=None,
    ):
        self.pk = petit_kernel
        self.world = dist.get_world_size(group)
        self.config = petit_kernel.MegaMoeConfig(
            world_size=self.world,
            num_experts=num_experts,
            topk=topk,
            model_dim=model_dim,
            activation=petit_kernel.MegaMoeActivation.mxfp4,
            activation_function=activation_function
            or petit_kernel.MegaMoeActivationFunction.silu_clamped,
            stages=petit_kernel.MegaMoeStages.two_stage,
            inter_dim=inter_dim,
            has_bias=False,
        )
        with node_local_dist(group):
            self.heap = petit_kernel.create_vmm_symmetric_heap(self.world)
            self.views = self.config.input_views(self.heap, max_tokens)
        # Petit's combine sums all top-k route-output rows of a token, valid or
        # not; a skipped route (-1 id) leaves its row holding a stale result
        # from an earlier call. Until the combine masks them, zero the rows.
        from lib.moe.rocm.mega_moe import _MegaMoESolutions

        workspace = _MegaMoESolutions()[self.config._solution_id].Workspace
        start = workspace.RouteOutputBufferOffset(self.heap.rank)
        self._route_row_bytes = topk * model_dim * 2
        self._route_rows = self.heap.local_tensor().view(torch.uint8)[
            start : start + max_tokens * self._route_row_bytes
        ]

    def clear_route_outputs(self, num_tokens):
        self._route_rows[: num_tokens * self._route_row_bytes].zero_()

    def run(self, weights, num_tokens, out=None, inputs=None):
        w13, w2, s13, s2 = weights
        return self.config.run(
            self.heap, w13, w2, s13, s2, num_tokens, out=out, inputs=inputs
        )


# ── transport kernels ────────────────────────────────────────────────────────


@triton.jit(
    do_not_specialize=[
        "s0_src",
        "s0_dst",
        "s0_n",
        "s1_src",
        "s1_dst",
        "s1_n",
        "s2_src",
        "s2_dst",
        "s2_n",
    ]
)
def _send_kernel(win_ptr, heap_bases, comm, win, rank, peer, s0_src, s0_dst, s0_n, s1_src, s1_dst, s1_n, s2_src, s2_dst, s2_n, signal, DIRECT: tl.constexpr, NSEG: tl.constexpr, CHUNKS: tl.constexpr, BLOCK: tl.constexpr,):  # fmt: skip
    """Program (segment, chunk). A direct peer gets plain stores from every
    chunk; a GIN peer gets one put_signal per segment, from chunk 0."""
    seg = tl.program_id(0)
    chunk = tl.program_id(1)
    if seg == 0:
        src_off, dst_off, nbytes = s0_src, s0_dst, s0_n
    elif seg == 1:
        src_off, dst_off, nbytes = s1_src, s1_dst, s1_n
    else:
        src_off, dst_off, nbytes = s2_src, s2_dst, s2_n
    if DIRECT:
        src = (win_ptr + tl.cast(src_off, tl.int64)).to(tl.pointer_type(tl.int64))
        to = (win_ptr + tl.cast(dst_off, tl.int64)).to(tl.pointer_type(tl.int64))
        nwords = tl.cast(nbytes, tl.int64) // 8
        for start in range(tl.cast(chunk, tl.int64) * BLOCK, nwords, CHUNKS * BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < nwords
            iris.store(
                to + offs,
                tl.load(src + offs, mask=mask),
                rank,
                peer,
                heap_bases,
                mask=mask,
            )
    elif chunk == 0:
        if nbytes > 0:
            rccl_gin.put_signal(
                comm, 0, peer, win, dst_off, win, src_off, nbytes, signal, 1
            )


@triton.jit
def _notify_kernel(win_ptr, heap_bases, flag_off, rank, peer, state):
    """Raise the peer's flag to this exchange's next round, ``*state + 1``."""
    flag = (win_ptr + flag_off).to(tl.pointer_type(tl.int64))
    iris.atomic_xchg(
        flag, tl.load(state) + 1, rank, peer, heap_bases, sem="release", scope="sys"
    )


@triton.jit
def _wait_kernel(win_ptr, comm, flag_off, signal, state, inc, DIRECT: tl.constexpr):
    """Wait until the flag (direct) or GIN signal reaches ``*state + inc``, then
    advance ``*state``. The round lives on the device so that a CUDA graph can
    replay the exchange; for GIN, ``*state`` starts at the signal's base."""
    target = tl.load(state) + inc
    if DIRECT:
        flag = (win_ptr + flag_off).to(tl.pointer_type(tl.int64))
        while tl.atomic_add(flag, 0, sem="acquire", scope="sys") < target:
            pass
    else:
        rccl_gin.wait_signal(comm, 0, signal, target)
    tl.store(state, target)


# MORI-CCO GDA transport (GPU-initiated RDMA, no CPU proxy). Imported lazily
# so the GIN path does not need MORI installed.
_cco = None


def _mori():
    global _cco
    if _cco is None:
        with redirect_triton_to_tokenspeed_triton():
            from mori.ir.triton import cco

        _cco = cco
    return _cco


def _mori_kernels():
    cco = _mori()

    @triton.jit
    def send(dc, win, peer, s0_src, s0_dst, s0_n, s1_src, s1_dst, s1_n, s2_src, s2_dst, s2_n, sig,FLUSH: tl.constexpr):  # fmt: skip
        """Program ``seg``: one warp-cooperative put with a fused signal increment."""
        seg = tl.program_id(0)
        if seg == 0:
            src_off, dst_off, nbytes = s0_src, s0_dst, s0_n
        elif seg == 1:
            src_off, dst_off, nbytes = s1_src, s1_dst, s1_n
        else:
            src_off, dst_off, nbytes = s2_src, s2_dst, s2_n
        if nbytes > 0:
            cco.Gda.put(dc, peer, win, dst_off, win, src_off, nbytes, ctx=seg,
                        signal_op=cco.SignalOp.INC, signal_id=sig, coop=cco.CoopScope.WARP)  # fmt: skip
            if FLUSH:
                cco.Gda.flush_peer(dc, peer, ctx=seg, coop=cco.CoopScope.WARP)

    @triton.jit
    def wait(dc, sig, count):
        # MORI's wait consumes `count` increments past its per-signal shadow, so
        # it needs no round bookkeeping (and replays in a CUDA graph as is).
        cco.Gda.wait_signal(dc, sig, count, coop=cco.CoopScope.WARP)

    return send, wait


@triton.jit
def _route_kernel(ids, weights, own_ids, peer_ids, own_w, n, n_total, my_node, peer_node,E_NODE: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    """Split global top-k ids into this node's and the peer node's local ids.
    Routes [n, n_total) are padding rows: id -1 on both sides, weight 0."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n
    mask = offs < n_total
    e = tl.load(ids + offs, mask=valid, other=-1)
    node = tl.where(valid, e // E_NODE, -1)
    local = e - node * E_NODE
    tl.store(own_ids + offs, tl.where(node == my_node, local, -1), mask=mask)
    tl.store(peer_ids + offs, tl.where(node == peer_node, local, -1), mask=mask)
    tl.store(own_w + offs, tl.load(weights + offs, mask=valid, other=0.0), mask=mask)


# ── the two-level MoE ────────────────────────────────────────────────────────


def t_device(window):
    return window.tensor.device


def _align(n, a=256):
    return (n + a - 1) // a * a


class HierPetitMoE:
    """EP over ``num_nodes`` x ``local_world`` GPUs: Petit inside a node, GIN across.

    Collective to construct. ``node_group`` is this rank's node-local group;
    world ranks must be node-major (rank = node * local_world + local).
    """

    def __init__(self, petit_kernel, provider: RcclGinProvider | None, node_group, *, num_experts, topk,model_dim, inter_dim, max_tokens, num_nodes=2, backend="gin", activation_function=None):  # fmt: skip
        if num_nodes != 2:
            raise NotImplementedError("pairs ranks across exactly two nodes")
        self.rank = dist.get_rank()
        self.local_world = dist.get_world_size(node_group)
        self.node = self.rank // self.local_world
        self.peer_node = 1 - self.node
        self.peer = self.peer_node * self.local_world + self.rank % self.local_world
        self.e_node = num_experts // num_nodes
        self.topk, self.model_dim, self.max_tokens = topk, model_dim, max_tokens
        self.petit = PetitNode(
            petit_kernel,
            node_group,
            self.e_node,
            topk,
            model_dim,
            inter_dim,
            2 * max_tokens,
            activation_function,
        )
        self.row_bytes = self.petit.views.tokens.stride(0)
        cap = max_tokens

        # Window layout. IN/IDS/W hold [own | peer] rows, Petit's external inputs.
        off = 0

        def region(nbytes):
            nonlocal off
            start = off
            off = _align(off + nbytes)
            return start

        self.o_flags = region(256)  # dispatch flag at +0, combine flag at +64
        self.o_in = region(2 * cap * self.row_bytes)
        self.o_ids = region(2 * cap * topk * 4)
        self.o_w = region(2 * cap * topk * 4)
        self.o_send_ids = region(cap * topk * 4)
        self.o_out = region(2 * cap * model_dim * 2)
        self.o_crecv = region(cap * model_dim * 2)
        self.backend = backend
        self.provider = provider
        self.return_puts = max(1, min(3, int(os.environ.get("PETIT_RETURN_PUTS", "1"))))
        # PETIT_SKIP_EXCHANGE=1: timing ablation with no cross-node traffic (wrong output).
        self.skip_exchange = os.environ.get("PETIT_SKIP_EXCHANGE") == "1"
        self._epoch = 0  # host-side count, for logging only
        if backend == "mori":
            t = self._init_mori(off)
        else:
            self.window = provider.allocate_window(off)
            self.direct = bool(self.window.address_map.direct[self.peer])
            if not self.direct and not provider.gin_available:
                raise RuntimeError(
                    f"peer {self.peer} is not load/store reachable and RCCL has no GIN backend"
                )
            self.comm = provider.device_comm(signal_count=2)
            self._signal_base = (
                provider.read_signals(self.comm, count=2).tolist()
                if provider.gin_available
                else [0, 0]
            )
            self.window.tensor[: self.o_in].zero_()
            self._lib = rccl_gin.extern_libs()
            # Per exchange (dispatch, combine): the last completed round (direct)
            # or GIN signal value. See _wait_kernel.
            self._state = torch.tensor(
                self._signal_base, dtype=torch.int64, device=t_device(self.window)
            )
            provider.barrier()
            t = self.window.tensor
        rows = t[self.o_in : self.o_in + 2 * cap * self.row_bytes].view(
            2 * cap, self.row_bytes
        )
        self._in_tokens = rows[:, : model_dim // 2]
        self._in_scales = rows[:, model_dim // 2 : model_dim // 2 + model_dim // 32]
        self._ids = (
            t[self.o_ids : self.o_ids + 2 * cap * topk * 4]
            .view(torch.int32)
            .view(2 * cap, topk)
        )
        self._w = (
            t[self.o_w : self.o_w + 2 * cap * topk * 4]
            .view(torch.float32)
            .view(2 * cap, topk)
        )
        self._send_ids = (
            t[self.o_send_ids : self.o_send_ids + cap * topk * 4]
            .view(torch.int32)
            .view(cap, topk)
        )
        self._out = (
            t[self.o_out : self.o_out + 2 * cap * model_dim * 2]
            .view(torch.bfloat16)
            .view(2 * cap, model_dim)
        )
        self._crecv = (
            t[self.o_crecv : self.o_crecv + cap * model_dim * 2]
            .view(torch.bfloat16)
            .view(cap, model_dim)
        )

    # transport ---------------------------------------------------------------

    def _init_mori(self, nbytes):
        """MORI-CCO communicator over the world, one window, cross-node GDA."""
        from iris.host.memory.tensor_utils import tensor_from_ptr
        from mori.cco import (
            GDA_CONNECTION_CROSSNODE,
            CCODevCommRequirements,
            Communicator,
        )
        from mori.cco.cco import UniqueId

        payload = [bytes(Communicator.get_unique_id()) if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        nbytes = _align(nbytes, 2 << 20)
        self.mcomm = Communicator.init(
            dist.get_world_size(),
            self.rank,
            UniqueId.frombytes(payload[0]),
            per_rank_vmm=max(nbytes, 256 << 20),
        )
        mem = self.mcomm.alloc_mem(nbytes)
        self.mwin = self.mcomm.register_window(mem.ptr, mem.size)
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_CROSSNODE
        reqs.gda_context_count = 3  # one QP context per dispatch segment
        reqs.gda_signal_count = 2
        reqs.gda_counter_count = 0
        self.mdc = self.mcomm.create_dev_comm(reqs)
        self.direct = False
        self._mori_send, self._mori_wait = _mori_kernels()
        self._lib = _mori().get_extern_libs()
        self._mori_flush = os.environ.get("MORI_PUT_FLUSH", "1") == "1"
        t = tensor_from_ptr(
            self.mwin.local_ptr,
            nbytes,
            torch.uint8,
            f"cuda:{torch.cuda.current_device()}",
        )
        t[: self.o_in].zero_()
        torch.cuda.synchronize()
        self.mcomm.barrier()
        return t

    def _send(self, segments, flag_off, signal):
        segs = list(segments) + [(0, 0, 0)] * (3 - len(segments))
        if self.backend == "mori":
            self._mori_send[(len(segments),)](
                self.mdc.ptr, self.mwin.handle, self.peer, *segs[0], *segs[1], *segs[2], signal,
                FLUSH=self._mori_flush, num_warps=1, extern_libs=self._lib,
            )  # fmt: skip
            return
        w = self.window
        _send_kernel[(len(segments), 16 if self.direct else 1)](
            w.tensor, w.address_map.peer_bases, self.comm.ptr, w.handle, self.rank, self.peer,
            *segs[0], *segs[1], *segs[2], signal,
            DIRECT=self.direct, NSEG=len(segments), CHUNKS=16 if self.direct else 1, BLOCK=1024,
            num_warps=4, extern_libs=self._lib,
        )  # fmt: skip
        if self.direct:
            _notify_kernel[(1,)](w.tensor, w.address_map.peer_bases, flag_off, self.rank, self.peer,
                                 self._state[signal:], num_warps=1)  # fmt: skip

    def _wait(self, flag_off, signal, puts):
        if self.backend == "mori":
            self._mori_wait[(1,)](
                self.mdc.ptr, signal, puts, num_warps=1, extern_libs=self._lib
            )
            return
        inc = 1 if self.direct else puts
        _wait_kernel[(1,)](self.window.tensor, self.comm.ptr, flag_off, signal, self._state[signal:], inc,
                           DIRECT=self.direct, num_warps=1, extern_libs=self._lib)  # fmt: skip

    # forward -------------------------------------------------------------------

    def forward(
        self,
        x,
        topk_ids,
        topk_weights,
        weights,
        events=None,
        num_tokens=None,
        overlap_fn=None,
    ):
        """x: [n, model_dim] bf16; topk_ids: [n, topk] int32 global ids;
        topk_weights: [n, topk] fp32; weights: Petit (w13, w2, s13, s2) for this
        rank's ``E_node / local_world`` experts. ``num_tokens`` (>= n, default
        n) is the row count every rank exchanges and must be the same on both
        ranks of a pair; rows [n, num_tokens) are padding. Returns [n, model_dim] bf16.
        """
        n = x.shape[0]
        T = n if num_tokens is None else int(num_tokens)
        if not n <= T <= self.max_tokens:
            raise ValueError(
                f"need tokens {n} <= padded {T} <= capacity {self.max_tokens}"
            )
        if overlap_fn is not None:
            overlap_fn()
        if T == 0:
            # Every rank of the pair sees T == 0 (it is a global maximum): no
            # exchange, but Petit is collective inside the node.
            return self.petit.run(
                weights, 0, out=self._out[:0], inputs=self.pk_views(0, 0)
            )
        self._epoch += 1
        rb, K = self.row_bytes, self.topk
        rec = (
            (
                lambda: events.append(torch.cuda.Event(enable_timing=True))
                or events[-1].record()
            )
            if events is not None
            else (lambda: None)
        )  # noqa: E731

        rec()
        if n:
            self.petit.config.quantize(x.contiguous(), out=self.pk_views(0, n))
        _route_kernel[(triton.cdiv(T * K, 1024),)](
            topk_ids, topk_weights, self._ids, self._send_ids, self._w, n * K, T * K, self.node, self.peer_node,
            E_NODE=self.e_node, BLOCK=1024, num_warps=4,
        )  # fmt: skip
        rec()
        self._mark("routed")
        # dispatch: my rows/ids/weights -> the peer's [T, 2T) rows
        if self.skip_exchange:
            # Ablation (wrong results, for timing only): no cross-node traffic.
            # Fill the peer half locally with this rank's own rows and peer-node
            # ids, so Petit still sees 2T rows with a realistic routing mix.
            self._in_tokens[T : 2 * T].copy_(self._in_tokens[:T])
            self._in_scales[T : 2 * T].copy_(self._in_scales[:T])
            self._ids[T : 2 * T].copy_(self._send_ids[:T])
            self._w[T : 2 * T].copy_(self._w[:T])
        else:
            self._exchange_dispatch(T, rb, K)
        rec()
        both = self.pk_views(0, 2 * T)
        self.petit.clear_route_outputs(2 * T)
        self.petit.run(weights, 2 * T, out=self._out[: 2 * T], inputs=both)
        self._mark("petit")
        rec()
        if not self.skip_exchange:
            self._exchange_combine(T)
        rec()
        y = self._out[:n] + self._crecv[:n]
        rec()
        return y

    def _exchange_dispatch(self, T, rb, K):
        self._send(
            [
                (self.o_in, self.o_in + T * rb, T * rb),
                (self.o_send_ids, self.o_ids + T * K * 4, T * K * 4),
                (self.o_w, self.o_w + T * K * 4, T * K * 4),
            ],
            self.o_flags,
            0,
        )
        self._mark("sent")
        self._wait(self.o_flags, 0, puts=3)
        self._mark("dispatched")

    def _exchange_combine(self, T):
        # PETIT_RETURN_PUTS (1-3, default 1) splits the return into several puts
        # (MORI: one QP context each) so a large combine is not limited to one
        # queue pair's bandwidth.
        row = self.model_dim * 2
        parts = min(self.return_puts, T)
        bounds = [T * i // parts for i in range(parts + 1)]
        segments = [
            (self.o_out + (T + a) * row, self.o_crecv + a * row, (b - a) * row)
            for a, b in zip(bounds, bounds[1:])
        ]
        self._send(segments, self.o_flags + 64, 1)
        self._wait(self.o_flags + 64, 1, puts=len(segments))

    def _mark(self, msg):
        if os.environ.get("DIAG") == "2":
            torch.cuda.synchronize()
            print(f"[mark r{self.rank}] epoch {self._epoch} {msg}", flush=True)

    def pk_views(self, start, stop):
        return self.petit.pk.MegaMoeInputViews(
            tokens=self._in_tokens[start:stop],
            scales=self._in_scales[start:stop],
            expert_ids=self._ids[start:stop],
            expert_weights=self._w[start:stop],
        )

    def close(self):
        if self.backend == "mori":
            self.mcomm.barrier()
            self.mcomm.destroy()
            return
        self.provider.barrier()
        self.provider.destroy_device_comm(self.comm)
        self.provider.free_window(self.window)


def env_int(name, default):
    return int(os.environ.get(name, default))


class PairExchange:
    """Fixed-size packed-row swap between rank (n, l) and its pair (1-n, l).

    Two directions share one GIN window: ``dispatch`` (signal 0) and ``combine``
    (signal 1). Each call sends ``rows`` packed rows of ``row_bytes`` from this
    rank's send buffer into the pair's receive buffer and waits for the pair's
    rows. Both ranks of a pair must call with the same ``rows``. Device-side
    rounds keep it CUDA-graph capturable.
    """

    def __init__(
        self, provider, node_group, *, max_rows, dispatch_row_bytes, combine_row_bytes
    ):
        self.rank = dist.get_rank()
        self.local_world = dist.get_world_size(node_group)
        node = self.rank // self.local_world
        self.peer = (1 - node) * self.local_world + self.rank % self.local_world
        self.max_rows = max_rows
        self.skip = os.environ.get("PETIT_SKIP_EXCHANGE") == "1"
        off = 0

        def region(nbytes):
            nonlocal off
            start = off
            off = _align(off + nbytes)
            return start

        self.o_flags = region(256)
        self.o_dsend = region(max_rows * dispatch_row_bytes)
        self.o_drecv = region(max_rows * dispatch_row_bytes)
        self.o_csend = region(max_rows * combine_row_bytes)
        self.o_crecv = region(max_rows * combine_row_bytes)
        self.drb, self.crb = dispatch_row_bytes, combine_row_bytes
        self.provider = provider
        self.window = provider.allocate_window(off)
        self.direct = bool(self.window.address_map.direct[self.peer])
        if not self.direct and not provider.gin_available:
            raise RuntimeError(
                f"peer {self.peer} is not load/store reachable and RCCL has no GIN backend"
            )
        self.comm = provider.device_comm(signal_count=2)
        base = (
            provider.read_signals(self.comm, count=2).tolist()
            if provider.gin_available
            else [0, 0]
        )
        self.window.tensor.zero_()
        self._lib = rccl_gin.extern_libs()
        self._state = torch.tensor(
            base, dtype=torch.int64, device=t_device(self.window)
        )
        provider.barrier()
        t = self.window.tensor
        self.dsend = t[self.o_dsend : self.o_dsend + max_rows * self.drb].view(
            max_rows, self.drb
        )
        self.drecv = t[self.o_drecv : self.o_drecv + max_rows * self.drb].view(
            max_rows, self.drb
        )
        self.csend = t[self.o_csend : self.o_csend + max_rows * self.crb].view(
            max_rows, self.crb
        )
        self.crecv = t[self.o_crecv : self.o_crecv + max_rows * self.crb].view(
            max_rows, self.crb
        )

    def _swap(self, src_off, dst_off, nbytes, flag_off, signal):
        w = self.window
        _send_kernel[(1, 16 if self.direct else 1)](
            w.tensor, w.address_map.peer_bases, self.comm.ptr, w.handle, self.rank, self.peer,
            src_off, dst_off, nbytes, 0, 0, 0, 0, 0, 0, signal,
            DIRECT=self.direct, NSEG=1, CHUNKS=16 if self.direct else 1, BLOCK=1024,
            num_warps=4, extern_libs=self._lib,
        )  # fmt: skip
        if self.direct:
            _notify_kernel[(1,)](w.tensor, w.address_map.peer_bases, flag_off, self.rank, self.peer,
                                 self._state[signal:], num_warps=1)  # fmt: skip
        _wait_kernel[(1,)](w.tensor, self.comm.ptr, flag_off, signal, self._state[signal:], 1,
                           DIRECT=self.direct, num_warps=1, extern_libs=self._lib)  # fmt: skip

    def dispatch(self, rows):
        """Send ``dsend[:rows]``; return the pair's rows in ``drecv[:rows]``."""
        if self.skip:  # timing ablation: no cross-node traffic (wrong results)
            self.drecv[:rows].copy_(self.dsend[:rows])
        elif rows:
            self._swap(self.o_dsend, self.o_drecv, rows * self.drb, self.o_flags, 0)
        return self.drecv[:rows]

    def combine(self, rows):
        """Send ``csend[:rows]``; return the pair's rows in ``crecv[:rows]``."""
        if self.skip:
            self.crecv[:rows].copy_(self.csend[:rows])
        elif rows:
            self._swap(
                self.o_csend, self.o_crecv, rows * self.crb, self.o_flags + 64, 1
            )
        return self.crecv[:rows]
