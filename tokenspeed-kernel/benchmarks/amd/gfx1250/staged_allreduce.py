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

"""TokenSpeed's staged one-shot all-reduce on gfx1250: the current kernel vs the phase-1 fix.

    torchrun --nproc_per_node=4 staged_allreduce.py [--json out.json]

On MI450 this is the all-reduce TokenSpeed runs today: IrisAllReduce's tuned shapes and its
two-stage path are both gated on CDNA4, so gfx1250 falls through to
iris_stage_one_shot_allreduce_kernel. `current` is that kernel, copied verbatim from
tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/iris.py. `fixed` is phase 1:

  * each rank pushes its ready epoch into every peer's flag array (sys-scope release) and
    polls its *own* array, instead of every peer polling the rank's flag across the fabric;
  * the poll is an acquire, so the peer-data reads that follow are ordered after it;
  * no .wt on the staging store and no .cg on the peer loads: the release/acquire pair
    is what makes the staged data visible;
  * the wait is bounded: on timeout the kernel records the block in an error buffer and
    stops waiting instead of hanging silently.

Correctness is checked against RCCL on fresh random data every call, including runs with no
host barrier between calls, which is how the kernel is used and where a stale slot would show.
"""

import argparse
import json
import os
import time

import iris
import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def current_kernel(
    input_ptr,
    input_sym_ptr,
    output_ptr,
    ready_flags,
    heap_bases,
    NUMEL,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    # Verbatim from tokenspeed's iris_stage_one_shot_allreduce_kernel (docstring dropped).
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    flag_offset = block_id * WORLD_SIZE
    local_ready = ready_flags + flag_offset + RANK
    epoch = tl.load(local_ready).to(tl.int32) + 1
    slot = (epoch % NUM_SLOTS) * SLOT_STRIDE

    local = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(input_sym_ptr + slot + offsets, local, mask=mask, cache_modifier=".wt")
    tl.debug_barrier()

    tl.atomic_xchg(local_ready, epoch, sem="release", scope="sys")
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            seen = tl.full((), 0, dtype=tl.int32)
            while seen < epoch:
                seen = iris.load(
                    ready_flags + flag_offset + peer,
                    RANK,
                    peer,
                    heap_bases,
                    cache_modifier=".cv",
                    volatile=True,
                )

    acc = local.to(tl.float32)
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            acc += iris.load(
                input_sym_ptr + slot + offsets,
                RANK,
                peer,
                heap_bases,
                mask=mask,
                other=0.0,
                cache_modifier=".cg",
                hint=BLOCK_SIZE,
            ).to(tl.float32)
    tl.store(output_ptr + offsets, acc.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def fixed_kernel(
    input_ptr,
    input_sym_ptr,
    output_ptr,
    ready_flags,  # [blocks, WORLD_SIZE] int32: [b, p] = latest epoch rank p published to *this* rank
    heap_bases,
    error_ptr,  # int32: set to block_id + 1 if a wait timed out
    NUMEL,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    MAX_SPINS: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    flag_offset = block_id * WORLD_SIZE
    # Our own column is written by our own publish below, so it still counts our epochs.
    epoch = tl.load(ready_flags + flag_offset + RANK).to(tl.int32) + 1
    slot = (epoch % NUM_SLOTS) * SLOT_STRIDE

    local = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(input_sym_ptr + slot + offsets, local, mask=mask)
    tl.debug_barrier()

    # Publish into every rank's array (ours included); the sys-scope release orders the
    # staging store before the flag for every reader.
    for peer in tl.static_range(0, WORLD_SIZE):
        iris.atomic_xchg(
            ready_flags + flag_offset + RANK,
            epoch,
            RANK,
            peer,
            heap_bases,
            sem="release",
            scope="sys",
        )

    # Wait on our own memory; the acquire orders the peer-data reads below after it.
    timed_out = False
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            seen = tl.atomic_add(
                ready_flags + flag_offset + peer, 0, sem="acquire", scope="sys"
            )
            spins = 0
            while (seen < epoch) & (spins < MAX_SPINS):
                seen = tl.atomic_add(
                    ready_flags + flag_offset + peer, 0, sem="acquire", scope="sys"
                )
                spins += 1
            timed_out = timed_out | (seen < epoch)
    if timed_out:
        tl.atomic_xchg(error_ptr, block_id + 1)

    acc = local.to(tl.float32)
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            acc += iris.load(
                input_sym_ptr + slot + offsets,
                RANK,
                peer,
                heap_bases,
                mask=mask,
                other=0.0,
                hint=BLOCK_SIZE,
            ).to(tl.float32)
    tl.store(output_ptr + offsets, acc.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def fixed2_kernel(
    input_ptr,
    input_sym_ptr,
    output_ptr,
    ready_flags,  # [blocks, WORLD_SIZE] int32: [b, p] = latest epoch rank p published to *this* rank
    heap_bases,
    error_ptr,  # int32: set to block_id + 1 if a wait timed out
    NUMEL,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    MAX_SPINS: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    flag_offset = block_id * WORLD_SIZE
    # Our own column is written by our own publish below, so it still counts our epochs.
    epoch = tl.load(ready_flags + flag_offset + RANK).to(tl.int32) + 1
    slot = (epoch % NUM_SLOTS) * SLOT_STRIDE

    local = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(input_sym_ptr + slot + offsets, local, mask=mask)
    tl.debug_barrier()

    # Publish into every rank's array (ours included). Only the first store is a release: it
    # writes the staged data back once (global_wb), and the relaxed stores after it are issued
    # only once that has completed. Each release costs a system-wide L2 write-back.
    for i in tl.static_range(0, WORLD_SIZE):
        peer = (RANK + i) % WORLD_SIZE
        if i == 0:
            iris.atomic_xchg(
                ready_flags + flag_offset + RANK,
                epoch,
                RANK,
                peer,
                heap_bases,
                sem="release",
                scope="sys",
            )
        else:
            iris.atomic_xchg(
                ready_flags + flag_offset + RANK,
                epoch,
                RANK,
                peer,
                heap_bases,
                sem="relaxed",
                scope="sys",
            )

    # Wait on our own memory with relaxed polls (an acquire per spin would invalidate the caches
    # on every iteration), then one acquire to order the peer-data reads below after the flags.
    timed_out = False
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            seen = tl.atomic_add(
                ready_flags + flag_offset + peer, 0, sem="relaxed", scope="sys"
            )
            spins = 0
            while (seen < epoch) & (spins < MAX_SPINS):
                seen = tl.atomic_add(
                    ready_flags + flag_offset + peer, 0, sem="relaxed", scope="sys"
                )
                spins += 1
            timed_out = timed_out | (seen < epoch)
    tl.atomic_add(ready_flags + flag_offset + RANK, 0, sem="acquire", scope="sys")
    if timed_out:
        tl.atomic_xchg(error_ptr, block_id + 1)

    acc = local.to(tl.float32)
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            acc += iris.load(
                input_sym_ptr + slot + offsets,
                RANK,
                peer,
                heap_bases,
                mask=mask,
                other=0.0,
                hint=BLOCK_SIZE,
            ).to(tl.float32)
    tl.store(output_ptr + offsets, acc.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def fixed2c_kernel(
    input_ptr,
    input_sym_ptr,
    output_ptr,
    ready_flags,  # [blocks, WORLD_SIZE] int32: [b, p] = latest epoch rank p published to *this* rank
    heap_bases,
    error_ptr,  # int32: set to block_id + 1 if a wait timed out
    NUMEL,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    MAX_SPINS: tl.constexpr,
    STORE_CM: tl.constexpr,
    LOAD_CM: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    flag_offset = block_id * WORLD_SIZE
    # Our own column is written by our own publish below, so it still counts our epochs.
    epoch = tl.load(ready_flags + flag_offset + RANK).to(tl.int32) + 1
    slot = (epoch % NUM_SLOTS) * SLOT_STRIDE

    local = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(input_sym_ptr + slot + offsets, local, mask=mask, cache_modifier=STORE_CM)
    tl.debug_barrier()

    # Publish into every rank's array (ours included). Only the first store is a release: it
    # writes the staged data back once (global_wb), and the relaxed stores after it are issued
    # only once that has completed. Each release costs a system-wide L2 write-back.
    for i in tl.static_range(0, WORLD_SIZE):
        peer = (RANK + i) % WORLD_SIZE
        if i == 0:
            iris.atomic_xchg(
                ready_flags + flag_offset + RANK,
                epoch,
                RANK,
                peer,
                heap_bases,
                sem="release",
                scope="sys",
            )
        else:
            iris.atomic_xchg(
                ready_flags + flag_offset + RANK,
                epoch,
                RANK,
                peer,
                heap_bases,
                sem="relaxed",
                scope="sys",
            )

    # Wait on our own memory with relaxed polls (an acquire per spin would invalidate the caches
    # on every iteration), then one acquire to order the peer-data reads below after the flags.
    timed_out = False
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            seen = tl.atomic_add(
                ready_flags + flag_offset + peer, 0, sem="relaxed", scope="sys"
            )
            spins = 0
            while (seen < epoch) & (spins < MAX_SPINS):
                seen = tl.atomic_add(
                    ready_flags + flag_offset + peer, 0, sem="relaxed", scope="sys"
                )
                spins += 1
            timed_out = timed_out | (seen < epoch)
    tl.atomic_add(ready_flags + flag_offset + RANK, 0, sem="acquire", scope="sys")
    if timed_out:
        tl.atomic_xchg(error_ptr, block_id + 1)

    acc = local.to(tl.float32)
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            acc += iris.load(
                input_sym_ptr + slot + offsets,
                RANK,
                peer,
                heap_bases,
                mask=mask,
                other=0.0,
                hint=BLOCK_SIZE,
                cache_modifier=LOAD_CM,
            ).to(tl.float32)
    tl.store(output_ptr + offsets, acc.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def fixed3_kernel(
    input_ptr,
    input_sym_ptr,
    output_ptr,
    ready_flags,  # [blocks, WORLD_SIZE] int32: [b, p] = latest epoch rank p published to *this* rank
    heap_bases,
    error_ptr,  # int32: set to block_id + 1 if a wait timed out
    NUMEL,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    MAX_SPINS: tl.constexpr,
    STORE_CM: tl.constexpr,
    LOAD_CM: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    flag_offset = block_id * WORLD_SIZE
    # Our own column is written by our own publish below, so it still counts our epochs.
    epoch = tl.load(ready_flags + flag_offset + RANK).to(tl.int32) + 1
    slot = (epoch % NUM_SLOTS) * SLOT_STRIDE

    local = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(input_sym_ptr + slot + offsets, local, mask=mask, cache_modifier=STORE_CM)
    tl.debug_barrier()

    # Publish into every rank's array (ours included). Only the first store is a release: it
    # writes the staged data back once (global_wb), and the relaxed stores after it are issued
    # only once that has completed. Each release costs a system-wide L2 write-back.
    for i in tl.static_range(0, WORLD_SIZE):
        peer = (RANK + i) % WORLD_SIZE
        if i == 0:
            iris.atomic_xchg(
                ready_flags + flag_offset + RANK,
                epoch,
                RANK,
                peer,
                heap_bases,
                sem="release",
                scope="sys",
            )
        else:
            iris.atomic_xchg(
                ready_flags + flag_offset + RANK,
                epoch,
                RANK,
                peer,
                heap_bases,
                sem="relaxed",
                scope="sys",
            )

    # Wait on our own memory with relaxed polls (an acquire per spin would invalidate the caches
    # on every iteration), then one acquire to order the peer-data reads below after the flags.
    timed_out = False
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            # Plain uncached loads: polling with sys-scope atomics sends every spin to memory.
            seen = tl.load(
                ready_flags + flag_offset + peer, cache_modifier=".cv", volatile=True
            )
            spins = 0
            while (seen < epoch) & (spins < MAX_SPINS):
                seen = tl.load(
                    ready_flags + flag_offset + peer,
                    cache_modifier=".cv",
                    volatile=True,
                )
                spins += 1
            timed_out = timed_out | (seen < epoch)
    tl.atomic_add(ready_flags + flag_offset + RANK, 0, sem="acquire", scope="sys")
    if timed_out:
        tl.atomic_xchg(error_ptr, block_id + 1)

    acc = local.to(tl.float32)
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            acc += iris.load(
                input_sym_ptr + slot + offsets,
                RANK,
                peer,
                heap_bases,
                mask=mask,
                other=0.0,
                hint=BLOCK_SIZE,
                cache_modifier=LOAD_CM,
            ).to(tl.float32)
    tl.store(output_ptr + offsets, acc.to(output_ptr.type.element_ty), mask=mask)


class Staged:
    def __init__(
        self,
        ctx,
        kind,
        max_numel,
        block_size=2048,
        num_warps=4,
        slots=2,
        max_spins=50_000_000,
    ):
        self.ctx, self.kind, self.bs, self.nw, self.slots, self.max_spins = (
            ctx, kind, block_size, num_warps, slots, max_spins)  # fmt: skip
        world = ctx.get_num_ranks()
        self.max_numel = max_numel
        self.input_buf = ctx.zeros((slots, max_numel), dtype=torch.bfloat16)
        self.flags = ctx.zeros(
            (triton.cdiv(max_numel, block_size), world), dtype=torch.int32
        )
        self.error = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.heap_bases = ctx.get_heap_bases()
        torch.cuda.synchronize()
        ctx.barrier()

    def __call__(self, tensor):
        n = tensor.numel()
        common = dict(
            RANK=self.ctx.get_rank(),
            WORLD_SIZE=self.ctx.get_num_ranks(),
            BLOCK_SIZE=self.bs,
            SLOT_STRIDE=self.max_numel,
            NUM_SLOTS=self.slots,
            num_warps=self.nw,
        )
        grid = (triton.cdiv(n, self.bs),)
        t = tensor.view(-1)
        if self.kind == "current":
            current_kernel[grid](
                t, self.input_buf.view(-1), t, self.flags, self.heap_bases, n, **common
            )
        else:
            cms = {"fixed2_wt": dict(STORE_CM=".wt", LOAD_CM=""), "fixed2_wt_cg": dict(STORE_CM=".wt", LOAD_CM=".cg"),
                   "fixed3": dict(STORE_CM="", LOAD_CM=""), "fixed3_wt": dict(STORE_CM=".wt", LOAD_CM="")}  # fmt: skip
            if self.kind in cms:
                (fixed3_kernel if self.kind.startswith("fixed3") else fixed2c_kernel)[grid](
                    t, self.input_buf.view(-1), t, self.flags, self.heap_bases, self.error, n,
                    MAX_SPINS=self.max_spins, **cms[self.kind], **common,
                )  # fmt: skip
                return tensor
            kernel = fixed_kernel if self.kind == "fixed" else fixed2_kernel
            kernel[grid](
                t, self.input_buf.view(-1), t, self.flags, self.heap_bases, self.error, n,
                MAX_SPINS=self.max_spins, **common,
            )  # fmt: skip
        return tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--tokens", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--b2b", type=int, default=100)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    nccl = dist.new_group(backend="nccl")
    ctx = iris.iris(4 << 30)
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    tokens = [int(x) for x in a.tokens.split(",")]
    max_numel = max(tokens) * a.hidden
    kinds = os.environ.get("KINDS", "current,fixed,fixed2").split(",")
    impls = {k: Staged(ctx, k, max_numel) for k in kinds}
    results = []

    def check(fn, n, calls, fenced, seed):
        """Run `calls` all-reduces on fresh data (fenced or back to back); return max relative error."""
        xs = [torch.randn(n, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed + i * 977 + rank)).bfloat16() for i in range(calls)]  # fmt: skip
        refs = []
        for x in xs:
            r = x.float().clone()
            dist.all_reduce(r, group=nccl)
            refs.append(r)
        outs = [x.clone() for x in xs]
        torch.cuda.synchronize()
        ctx.barrier()
        for o in outs:
            if fenced:
                torch.cuda.synchronize()
                ctx.barrier()
            fn(o)
        torch.cuda.synchronize()
        ctx.barrier()
        err = max(
            float(((o.float() - r).abs() / (r.abs() + 1.0)).max())
            for o, r in zip(outs, refs)
        )
        e = torch.tensor([err])
        dist.all_reduce(e, op=dist.ReduceOp.MAX)
        return e.item()

    def timed(fn, x, iters, fenced):
        samples = []
        for _ in range(iters if fenced else 10):
            torch.cuda.synchronize()
            ctx.barrier()
            t0 = time.perf_counter()
            for _ in range(1 if fenced else a.b2b):
                fn(x)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) / (1 if fenced else a.b2b))
        samples.sort()
        return samples[len(samples) // 2] * 1e6

    rccl = lambda x: dist.all_reduce(x, group=nccl)  # noqa: E731
    for tok in tokens:
        n = tok * a.hidden
        rec = dict(tokens=tok, hidden=a.hidden, numel=n, world=world)
        for name, fn in (*impls.items(), ("rccl", rccl)):
            if name != "rccl":
                rec[f"{name}_err_fenced"] = check(fn, n, 20, True, 1)
                rec[f"{name}_err_b2b"] = check(fn, n, 50, False, 2)
            x = torch.randn(n, device="cuda").bfloat16()
            # A sum of bf16 values grows every call; reset between calls isn't timed, so keep x small.
            x.mul_(1e-6)
            timed(fn, x, 5, True)
            rec[f"{name}_us"] = timed(fn, x, a.iters, True)
            rec[f"{name}_b2b_us"] = timed(fn, x, 0, False)
        names = [k for k in impls if k != "current"]
        err = torch.tensor([int(impls[k].error.item()) for k in names])
        dist.all_reduce(err, op=dist.ReduceOp.MAX)
        for k, e in zip(names, err.tolist()):
            rec[f"{k}_timeout_block"] = e
        results.append(rec)
        if rank == 0:
            parts = [f"{tok:4d} tok"]
            for k in (*impls, "rccl"):
                p = f"{k} {rec[k + '_us']:6.1f}/{rec[k + '_b2b_us']:5.1f}"
                if k != "rccl":
                    p += f" e{max(rec[k + '_err_fenced'], rec[k + '_err_b2b']):.0e}"
                if k + "_timeout_block" in rec and rec[k + "_timeout_block"]:
                    p += " TIMEOUT"
                parts.append(p)
            print("  ".join(parts), flush=True)
    if rank == 0 and a.json:
        json.dump(results, open(a.json, "w"), indent=2)
    ctx.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
