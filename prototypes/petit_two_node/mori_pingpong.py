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

"""MORI-CCO GDA put latency and bandwidth between paired ranks on two nodes
(the same measurement as gin_pingpong.py, for comparison).

Rank (0, l) and (1, l) ping-pong one put_signal of S bytes (latency = half the
round trip), then rank (0, l) streams S bytes split into C puts over C
contexts (bandwidth). All pairs run at once, as in the MoE exchange.
"""

import os

import torch
import torch.distributed as dist

if os.environ.get("PLAIN_TRITON") == "1":
    # The image's own Triton, as MORI's examples use, to separate MORI from tokenspeed-triton.
    import contextlib

    import triton
    import triton.language as tl

    redirect_triton_to_tokenspeed_triton = contextlib.nullcontext
else:
    from tokenspeed_kernel._triton import (
        redirect_triton_to_tokenspeed_triton,
        tl,
        triton,
    )

with redirect_triton_to_tokenspeed_triton():
    from mori.cco import GDA_CONNECTION_CROSSNODE, CCODevCommRequirements, Communicator
    from mori.ir.triton import cco


FLUSH = tl.constexpr(int(os.environ.get("MORI_PUT_FLUSH", "1")))


@triton.jit
def _put(dc, win, peer, off, nbytes, ctx, sig):
    cco.Gda.put(
        dc,
        peer,
        win,
        off,
        win,
        off,
        nbytes,
        ctx=ctx,
        signal_op=cco.SignalOp.INC,
        signal_id=sig,
        coop=cco.CoopScope.WARP,
    )
    if FLUSH:
        # MORI's own examples flush after every put; without it the first
        # exchange hung (the post may only progress when its CQ is polled).
        cco.Gda.flush_peer(dc, peer, ctx=ctx, coop=cco.CoopScope.WARP)


@triton.jit
def _probe(dc, win, peer, out, nbytes, PUT: tl.constexpr):
    """One symmetric put (optional), then poll signal 0 a bounded number of times."""
    if PUT:
        _put(dc, win, peer, 0, nbytes, 0, 0)
    v = cco.Gda.read_signal(dc, 0)
    for i in range(2000000):
        v = cco.Gda.read_signal(dc, 0)
    tl.store(out, v)


@triton.jit
def _wait(dc, sig, count):
    # MORI's wait_signal consumes: it waits until the signal has moved `count`
    # past a per-signal shadow, then advances the shadow by `count`. (RCCL
    # GIN's wait_signal compares against an absolute value instead.)
    cco.Gda.wait_signal(dc, sig, count, coop=cco.CoopScope.WARP)


@triton.jit(do_not_specialize=["nbytes", "least"])
def _pingpong(comm, win, peer, nbytes, state, rounds, FIRST: tl.constexpr):
    """``rounds`` round trips: the first mover sends then waits; the other waits
    then answers. Signal 0 counts arrivals here; ``*state`` is its last value."""
    base = tl.load(state)
    for r in range(rounds):
        if FIRST:
            _put(comm, win, peer, 0, nbytes, 0, 0)
            _wait(comm, 0, 1)
        else:
            _wait(comm, 0, 1)
            _put(comm, win, peer, 0, nbytes, 0, 0)
    tl.store(state, base + rounds)


@triton.jit(do_not_specialize=["chunk"])
def _stream(comm, win, peer, chunk, SEND: tl.constexpr):
    """Program c puts bytes [c*chunk, (c+1)*chunk) on context c, signal 1+c."""
    c = tl.program_id(0)
    if SEND:
        off = tl.cast(c, tl.int64) * chunk
        _put(comm, win, peer, off, chunk, c, 1 + c)


@triton.jit
def _stream_wait(comm, state, rounds):
    c = tl.program_id(0)
    _wait(comm, 1 + c, rounds)


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_world = world // 2
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    node, local = divmod(rank, local_world)
    peer = (1 - node) * local_world + local
    contexts = int(os.environ.get("CONTEXTS", "4"))
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    from mori.cco.cco import UniqueId

    mcomm = Communicator.init(
        world, rank, UniqueId.frombytes(payload[0]), per_rank_vmm=256 << 20
    )
    mem = mcomm.alloc_mem(64 << 20)
    w = mcomm.register_window(mem.ptr, mem.size)
    reqs = CCODevCommRequirements()
    reqs.gda_connection_type = GDA_CONNECTION_CROSSNODE
    reqs.gda_context_count = contexts
    reqs.gda_signal_count = 1 + contexts
    reqs.gda_counter_count = 0
    dc = mcomm.create_dev_comm(reqs)
    lib = cco.get_extern_libs()
    mcomm.barrier()
    # Signals start at 0 on a fresh device communicator.
    state0 = torch.zeros(1, dtype=torch.int64, device="cuda")
    states = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(contexts)]
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    log(
        f"mori cco gda contexts={contexts} triton={triton.__name__} {triton.__version__}"
    )

    def timed(fn, reps=5):
        fn()
        torch.cuda.synchronize()
        dist.barrier()
        best = []
        for _ in range(reps):
            dist.barrier()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            best.append(s.elapsed_time(e) * 1000)
        t = torch.tensor([sorted(best)[len(best) // 2]], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    if os.environ.get("MORI_PROBE") == "1":
        out = torch.zeros(1, dtype=torch.int64, device="cuda")
        _probe[(1,)](
            dc.ptr, w.handle, peer, out, 64, PUT=True, num_warps=1, extern_libs=lib
        )
        torch.cuda.synchronize()
        print(
            f"[probe] rank {rank} -> peer {peer}: signal0 after symmetric put = {out.item()}",
            flush=True,
        )
        dist.barrier()
        _probe[(1,)](
            dc.ptr, w.handle, peer, out, 64, PUT=node == 0, num_warps=1, extern_libs=lib
        )
        torch.cuda.synchronize()
        print(
            f"[probe] rank {rank}: signal0 after one-sided put from node 0 = {out.item()}",
            flush=True,
        )
        return
    rounds = 50
    for size in (
        64,
        4096,
        44 << 10,
        160 << 10,
        640 << 10,
        1280 << 10,
        2560 << 10,
        5 << 20,
    ):
        t = timed(lambda: _pingpong[(1,)](dc.ptr, w.handle, peer, size, state0, rounds, FIRST=node == 0,
                                          num_warps=1, extern_libs=lib))  # fmt: skip
        one_way = t / rounds / 2
        log(f"  pingpong S={size} done")
        res = f"S={size:>8}  one-way {one_way:7.1f} us"
        for c in sorted({1, contexts}):
            chunk = (size + c - 1) // c
            chunk = (chunk + 255) // 256 * 256
            st = torch.cat(states[:c]) if c > 1 else states[0]

            def stream():
                for _ in range(10):
                    _stream[(c,)](
                        dc.ptr,
                        w.handle,
                        peer,
                        chunk,
                        SEND=node == 0,
                        num_warps=1,
                        extern_libs=lib,
                    )
                    if node == 1:
                        _stream_wait[(c,)](dc.ptr, st, 1, num_warps=1, extern_libs=lib)

            # Only node 1 waits; node 0's time is the issue time. Report node 1.
            tt = timed(stream) / 10
            for i in range(c):
                states[i].copy_(st[i : i + 1])
            res += f"   {c} ctx: {tt:7.1f} us ({size / tt / 1e3:5.1f} GB/s)"
        log(res)
    mcomm.barrier()
    mcomm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
