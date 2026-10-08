# Staged one-shot all-reduce on gfx1250 (MI450)

On gfx1250, `IrisAllReduce` always runs `iris_stage_one_shot_allreduce_kernel`: its tuned
shapes and its two-stage path are both gated on CDNA4. `staged_allreduce.py` compares that
kernel, copied verbatim, with candidate fixes, checking every call against RCCL.

    torchrun --nproc_per_node=4 staged_allreduce.py [--tokens 1,2,4,...] [--json out.json]
    KINDS=current,fixed2 torchrun --nproc_per_node=4 staged_allreduce.py

It is standalone: it imports `iris` and `triton` directly rather than through
`tokenspeed_kernel`, so it runs with a plain iris install.

## Variants

| Kind | Change from `current` |
|---|---|
| `current` | Verbatim. Each rank publishes its ready epoch in its own memory; every peer polls it across the fabric with volatile `.cv` loads; staging store `.wt`, peer loads `.cg`; unbounded wait. |
| `fixed` | Each rank pushes its epoch into every peer's flag array with a system-scope release and polls its own array with an acquire; no `.wt` / `.cg`; bounded wait that records a timeout in an error buffer. |
| `fixed2` | As `fixed`, but only the first epoch push is a release (the rest are relaxed), and the poll is relaxed with a single acquire after it. |
| `fixed2_wt`, `fixed2_wt_cg` | `fixed2` with the `.wt` store / `.cg` loads put back. |

## Findings so far (bf16, hidden 7168, 4 ranks)

* All variants are correct against RCCL, including 50 calls back to back with no host barrier.
* `fixed` is slower than `current` from 8 tokens up and its cost grows with size: every
  system-scope release compiles to an L2 write-back (`global_wb`), and an acquire inside the spin
  loop to a cache invalidate (`global_inv`) on every iteration.
* `fixed2` is faster than `current` from 2 to 64 tokens (about 1.2–1.7x back to back, 1.1–1.3x
  with a barrier between calls) but slower at 128–256 tokens.
* The `.wt` / `.cg` hints make little difference up to 64 tokens.
* `current` varies a lot from run to run; use a few hundred iterations.

## Open

* A variant that polls with plain uncached loads instead of system-scope atomics, to test
  whether atomic polling is what costs `fixed2` at large sizes (many more blocks than CUs).
* A louder timeout: Triton's `device_assert` is compiled out without debug mode, so the kernel
  currently reports through an error buffer instead of trapping.
