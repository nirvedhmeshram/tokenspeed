# Two-node Gluon Petit MegaMoE (prototype)

This prototype runs DeepSeek V4.1 Flash expert parallelism across two nodes. Gluon
Petit MegaMoE stays unchanged inside each node, and the tokens move between the nodes
over a GPU-initiated RDMA transport. Two transports are supported:

- **RCCL GIN**, through Iris's `RcclGinProvider`
  ([ROCm/iris#644](https://github.com/ROCm/iris/pull/644)).
- **MORI-CCO GDA**, through MORI's Triton bindings (`amd-mori`).

It is a research prototype, not a supported TokenSpeed path.

## How it works

Petit's MegaMoE is a single fused dispatch / expert / combine kernel over a
node-local VMM symmetric heap, so on its own it cannot reach a GPU on another
node. `petit_hier.HierPetitMoE` adds a second, inter-node level on top of it.

**Expert placement**

- Node `n` owns experts `[n * E/2, (n + 1) * E/2)`.
- Inside the node, Petit runs EP over the local GPUs with `E/2` experts.
- The prototype registers one extra Petit geometry for this: 192 experts, top-6.

**Pairing.** Rank `(n, l)` is paired with rank `(1 - n, l)`.

**Each forward**

1. The rank quantizes its rows to MXFP4 directly into the transport window.
2. A small kernel splits the global top-k ids into this node's local ids and the
   peer node's local ids. An expert that is not on a node gets `-1`.
3. Three puts carry the rows, the peer-local ids and the weights to rows `[T, 2T)`
   of the peer's window. Each put carries a signal.
4. Each rank calls Petit once on `[own rows | peer rows]`. Petit's dispatch already
   skips expert ids `>= kNumExperts`, which covers the `-1` ids.
5. Petit writes its output straight into the window.
6. The peer's rows go back in up to three puts, and the origin adds them to its own.

**Synchronization and buffer reuse**

- Every rank exchanges the global maximum row count, padding its own rows with
  routes that go nowhere, so the two ranks of a pair always agree on `T`.
- The forward has no host synchronization.
- Completion state lives on the device, so a decode step can be captured in a CUDA
  graph.
  - GIN signals are compared against an absolute value, tracked per exchange.
  - MORI's `wait_signal` consumes a count past its own shadow.
- No buffer needs a barrier before reuse. A rank overwrites its peer's regions for
  step `k + 1` only after it has received the peer's combine for step `k`.

**Petit workarounds**

- **Combine sums skipped routes.** Petit's combine adds every top-k route row of a
  token, including routes it skipped. The prototype therefore zeroes those rows
  before each call.
- **Heap uses the default process group.** Petit's VMM heap exchanges handles over
  `torch.distributed`'s default group. `node_local_dist` points that exchange at the
  node-local group.

### Large batches

Petit's per-rank capacity (1024 rows) has to hold `[own | peer]`, so one two-node
call carries at most 512 rows per GPU. The TokenSpeed hook splits larger (prefill)
batches into several calls. The number of calls follows the global maximum, so both
ranks of a pair make the same calls.

## TokenSpeed integration

The TokenSpeed integration is opt-in through environment variables:

- `TOKENSPEED_PETIT_NODE_SIZE=<GPUs per node>` makes `ops/moe/gluon.py` build a
  `HierPetitMoE` workspace when the world spans two nodes. It also lifts the
  one-node check in `layers/moe/expert.py`.
- `TOKENSPEED_PETIT_TRANSPORT=gin|mori` selects the transport. The default is
  `gin`.

`petit_hier.py` must be importable, for example by putting this directory on
`PYTHONPATH`. The GIN transport also needs an Iris checkout with `iris.experimental.rccl_gin`.

The serving layout is the one Petit already requires: attention TP1 with attention
DP over every GPU, and EP equal to the world size. For example, on two nodes with
four GPUs each:

```
--tensor-parallel-size 1 --data-parallel-size 8 --expert-parallel-size 8
--world-size 8 --nprocs-per-node 4 --nnodes 2 --node-rank <0|1> --dist-init-addr <node0>:<port>
--moe-backend gluon_petit --all2all-backend gluon_petit
--engram-host-table --chunked-prefill-size 1024 --max-prefill-tokens 1024
```

## Files

| File | Purpose |
|---|---|
| `petit_hier.py` | `HierPetitMoE`, transport kernels (GIN and MORI), node-local Petit heap |
| `bench_hier.py` | Kernel-level check and timing of single-node Petit against the two-node version. Uses Petit's sparse exact-oracle weights; supports CUDA-graph timing |
| `gin_pingpong.py`, `mori_pingpong.py` | Raw put latency / bandwidth between paired ranks |
| `loadgen.py` | Closed-loop OpenAI-completions load generator (TTFT, TPOT, tokens/s) |
| `Dockerfile.rocm101` | TokenSpeed on the ROCm 10.1 PyTorch image (RCCL's GIN device API needs HIP >= 7.12) |

## Environment notes

**RCCL GIN**
- Needs `NCCL_CUMEM_ENABLE=1`, set before any RCCL communicator is created.

**TokenSpeed + Petit serving**
- Needs `TOKENSPEED_JIT_COMPILE_CHECK=off`; see
  [lightseekorg/tokenspeed#3505](https://github.com/lightseekorg/tokenspeed/issues/3505).
- On TheRock-layout ROCm images, `tokenspeed_kernel.platform` dlopens the
  unversioned `libamdhip64.so`. Point that name at the core runtime that torch
  loads.

**MORI 1.2.5 in a ROCm 10.1 container**
- It needs `libdrm2` and `libibverbs-dev`.
- The wheel loads `rocm_smi`, which ROCm 10.1 no longer ships. `MORI_ROCM_SMI_PATH`
  can point at another copy.
- Without an `/etc/libibverbs.d/ionic.driver` entry, libibverbs lists no Pollara
  devices. MORI then silently runs without RDMA.
- It dlopens the unversioned `libionic.so`.

**Benchmark launch.** The benchmarks run under `torchrun` with node-major ranks, for
example:

```
torchrun --nnodes 2 --node-rank <0|1> --nproc-per-node 4 --master-addr <node0> --master-port <port> bench_hier.py
```

With `LOCAL_WORLD=4`, `BACKEND=gin|mori` selects the transport.
