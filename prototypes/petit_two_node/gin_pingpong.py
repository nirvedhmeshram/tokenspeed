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

"""GIN put latency and bandwidth between paired ranks on two nodes.

Rank (0, l) and (1, l) ping-pong one put_signal of S bytes (latency = half the
round trip), then rank (0, l) streams S bytes split into C puts over C
contexts (bandwidth). All pairs run at once, as in the MoE exchange.
"""

import os

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton, tl, triton

with redirect_triton_to_tokenspeed_triton():
    from iris.experimental import rccl_gin
    from iris.experimental.rccl_gin import RcclGinProvider


@triton.jit(do_not_specialize=["nbytes", "least"])
def _pingpong(comm, win, peer, nbytes, state, rounds, FIRST: tl.constexpr):
    """``rounds`` round trips: the first mover sends then waits; the other waits
    then answers. Signal 0 counts arrivals here; ``*state`` is its last value."""
    base = tl.load(state)
    for r in range(rounds):
        if FIRST:
            rccl_gin.put_signal(comm, 0, peer, win, 0, win, 0, nbytes, 0, 1)
            rccl_gin.wait_signal(comm, 0, 0, base + r + 1)
        else:
            rccl_gin.wait_signal(comm, 0, 0, base + r + 1)
            rccl_gin.put_signal(comm, 0, peer, win, 0, win, 0, nbytes, 0, 1)
    tl.store(state, base + rounds)


@triton.jit(do_not_specialize=["chunk"])
def _stream(comm, win, peer, chunk, SEND: tl.constexpr):
    """Program c puts bytes [c*chunk, (c+1)*chunk) on context c, signal 1+c."""
    c = tl.program_id(0)
    if SEND:
        off = tl.cast(c, tl.int64) * chunk
        rccl_gin.put_signal(comm, c, peer, win, off, win, off, chunk, 1 + c, 1)


@triton.jit
def _stream_wait(comm, state, rounds):
    c = tl.program_id(0)
    target = tl.load(state + c) + rounds
    rccl_gin.wait_signal(comm, c, 1 + c, target)
    tl.store(state + c, target)


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_world = world // 2
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    node, local = divmod(rank, local_world)
    peer = (1 - node) * local_world + local
    contexts = int(os.environ.get("CONTEXTS", "4"))
    provider = RcclGinProvider()
    w = provider.allocate_window(64 << 20)
    dc = provider.device_comm(signal_count=1 + contexts, context_count=contexts)
    lib = rccl_gin.extern_libs()
    sig = provider.read_signals(dc, count=1 + contexts).cuda()
    state0 = sig[:1].clone()
    states = [
        provider.read_signals(dc, context=c, count=1 + contexts)[1 + c : 2 + c].cuda()
        for c in range(contexts)
    ]
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    log(f"gin={provider.gin_type_name} contexts={contexts}")

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
    provider.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
