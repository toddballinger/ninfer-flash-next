# RTX 5080 Runtime Strategy

## 1. Target

Primary target:

```text
GPU              NVIDIA GeForce RTX 5080
architecture     Blackwell / SM120
physical VRAM    approximately 15,841 MiB
clean free VRAM  approximately 15,540 MiB
```

NInfer baseline capabilities include CUDA Graph support and Blackwell-oriented low-precision execution.

The production goal is Qwen4Exp / Qwen3.8-Flash-Next inference for local OpenClaw and programming-agent workloads.

## 2. Optimisation objective

Optimise the combined product of correctness, usable context, sustained decode speed, and runtime reliability.

Do not optimise only for maximum VRAM consumption, maximum model-weight residency, maximum context allocation, maximum prefill throughput, or one short decode benchmark.

## 3. Operating headroom

Do not intentionally design steady-state operation around every available byte of the 16 GB-class card.

An initial practical planning envelope is approximately 14.5 to 15.0 GiB observed device usage under the largest normal workload. This is a measured-policy target, not a hard-coded allocation constant. If real peak behaviour requires more headroom, preserve more headroom.

## 4. GPU-memory categories

Account separately for:

- core persistent weights;
- shared-expert persistent weights;
- resident routed experts;
- packed/transformed persistent weights;
- expert device cache;
- expert staging;
- GDN state;
- PLE state;
- QSA ordinary KV;
- QSA raw indexer keys;
- position/state metadata;
- persistent activation buffers;
- prefill scratch;
- decode scratch;
- CUDA Graph resources;
- CUDA/driver overhead;
- safety margin.

Do not estimate usable context from model file size alone.

## 5. Key Flash-Next memory characteristic

Flash-Next combines four-stream HyperConnection activations, stateful GDN layers, stateful PLE, QSA ordinary KV, QSA raw indexer-key history, 512 routed experts, Top-10 routing, and a shared expert.

Runtime memory strategy must consider both persistent model weights and persistent sequence state. Context expansion directly competes with expert GPU residency.

## 6. Earlier reconnaissance

Initial hardware reconnaissance indicated roughly 5.9 GiB projected device use and roughly 9.5 GiB remaining capacity for the early arrangement.

Treat this only as reconnaissance. Actual final allocations must be measured after the real Flash-Next operators and state planes are implemented.

The important implication is that meaningful additional device residency may be available.

## 7. Residency priority

Initial GPU-residency order should be:

1. always-hot core path — HyperConnection weights, normalization, attention/GDN core projections, QSA indexer, router, final mixer, and other high-frequency small tensors;
2. shared expert — it executes independently on every MoE block invocation and therefore has materially higher reuse than arbitrary routed experts;
3. routed-expert working set — use remaining device budget for permanently resident routed experts, routed-expert cache, and staging buffers;
4. context/state headroom — do not consume routed-expert capacity so aggressively that useful context disappears, prefill OOMs, graph instantiation becomes unsafe, or state growth leaves no margin.

## 8. Exact selected-expert streaming

The frozen reference provides an important runtime property:

```text
router
  -> exact Top-10 expert IDs
  -> expert execution
```

Therefore NInfer can know the exact experts needed before executing them.

On the RTX 5080 this enables:

```text
route
  -> look up ten experts
  -> identify resident hits
  -> transfer only cold misses
  -> execute
```

This is an exact optimization. No expert pruning or approximate routing is required.

## 9. Initial expert policy

Begin with a deterministic policy:

```text
core weights          GPU
shared expert         GPU where practical
router                GPU
routed experts        partial GPU residency
remaining experts     host backing
expert device cache   fixed allocation
staging buffers       fixed allocation
```

Do not initially add complicated expert-popularity prediction. First measure routing locality.

## 10. Routed-expert cache requirements

A device expert cache should have a fixed maximum allocation, stable addresses, no per-token device allocation, explicit expert-to-slot mapping, deterministic replacement initially, transfer counters, hit/miss counters, and byte-transfer counters.

Correctness must be identical regardless of cache hit/miss behavior.

## 11. Host backing strategies

Benchmark rather than assume.

Candidate mechanisms:

- pageable host memory — simple correctness baseline;
- pinned host memory — potentially useful for asynchronous transfer of cold experts, but do not pin excessive memory without evidence;
- CUDA mapped host memory — potentially useful for selected access patterns, but large expert GEMM weight traffic over PCIe can be bandwidth limited, so zero-copy is not automatically faster;
- explicit staged device cache — copy cold experts to reusable device slots, likely useful where routed-expert reuse exists.

All remain implementation options until measured.

## 12. Expert locality measurement

Collect routing statistics from realistic workloads:

- expert frequency;
- expert frequency per layer;
- expert reuse distance;
- unique experts/token;
- unique experts/turn;
- cache hit rate;
- cold transfer rate;
- bytes transferred/token.

Compare coding prompts, prose, agent/tool workflows, and long-context continuation. Do not assume 512 experts are uniformly selected.

## 13. Transfer overlap

After synchronous correctness is established, test overlap between cold expert H2D transfer, execution of resident experts, and other independent runtime work.

Use explicit CUDA stream/event dependencies. Do not introduce races merely to gain theoretical overlap.

## 14. Decode is the main optimisation target

For every residency strategy capture decode tok/s, median token latency, long-tail token latency, expert-cache hit rate, expert H2D bytes/token, and peak VRAM.

A strategy that improves prefill while badly reducing sustained decode is unlikely to be the final production configuration.

## 15. Prefill controls peak memory

Long-context prefill can require more temporary memory than T=1 decode.

Test short, medium, 32K-class, 64K-class, and 128K-class prompts where feasible. Record peak VRAM, not merely post-load free memory.

A configuration that fits decode but OOMs during realistic prefill is invalid.

## 16. GDN persistent state

Every GDN layer retains causal-convolution state and recurrent matrix state.

These are context/session state rather than model weights. Allocations should be explicit, reusable, included in session-memory reporting, and included in snapshot/fork accounting. Avoid repeated allocation during decode.

## 17. PLE persistent state

PLE retains nine-step dilated-convolution history and two-token lexical history.

These allocations are small relative to model weights but essential to exact continuation semantics. Never omit them from sequence cloning merely because their memory cost is small.

## 18. QSA state

Each QSA layer retains ordinary attention KV and raw indexer-token keys.

The raw indexer keys are an additional context-scaled state plane and must be included when estimating 32K/64K/128K memory requirements.

The QSA token budget of 2048 reduces attention compute visibility but does not automatically eliminate the history needed for index selection. Do not confuse sparse compute with zero cache cost.

## 19. QSA GPU selection

QSA block scoring and TopK should remain GPU resident.

Avoid a per-token round-trip such as GPU keys -> CPU scoring -> CPU TopK -> GPU token IDs.

The intended path is GPU raw keys -> GPU compression/scoring -> GPU TopK -> GPU selected IDs -> GPU sparse KV gather -> GPU attention.

This is essential for decode latency.

## 20. QSA scratch reuse

Preallocate reusable buffers for compressed block keys, block scores, TopK values, TopK IDs, expanded selected token IDs, and sparse-gather metadata.

Size buffers against configured maximum context and QSA geometry where practical. Avoid per-token dynamic allocation.

## 21. Four-stream HyperConnection memory

Internal HyperConnection width is `4 * 2560 = 10240`.

Transient activation planning must reflect this width. Avoid retaining unnecessary full-layer four-stream intermediates once their lifetime has ended. Use lifetime-aware buffer reuse where NInfer's graph/runtime permits it.

## 22. Low-precision weights

Preserve quantized/low-precision storage as far into execution as practical.

Avoid repeatedly expanding quantized host/device weights into large dequantized temporaries where an exact existing NInfer SM120 kernel can directly consume the stored format.

This matters especially for routed experts because any expansion multiplies residency and transfer cost.

## 23. Blackwell / SM120 audit

For every major projection record stored format, execution format, kernel, tensor-core path, temporary conversion, bytes read, and measured latency.

Priority audit targets:

1. routed expert gate/up;
2. routed expert down;
3. shared expert;
4. attention/GDN projections;
5. QSA projections;
6. output head.

Use existing exact kernels before creating new ones.

## 24. CUDA Graph strategy

First establish stable decode without graphs.

Then require stable device pointers, fixed scratch allocation, stable state buffers, expert-cache slots with stable addresses, and graph-safe kernels.

Dynamic expert IDs do not necessarily prohibit graphs if dispatch metadata can change while addresses/topology remain graph compatible.

Benchmark graph OFF vs ON under identical workloads.

## 25. Context progression

Do not jump directly to 128K.

Use 4K/8K -> 32K -> 64K -> 128K. Each gate requires correctness and peak-memory measurement.

## 26. 32K gate

At 32K require correct prefill, correct decode, stable state, no unexplained memory growth, working selected-expert policy, and usable agent behavior.

This is the first production-style gate.

## 27. 64K gate

64K is a strong target for coding-agent use.

Compare configurations such as more routed experts resident + 64K context versus fewer routed experts resident + more context headroom.

Measure actual decode and agent usefulness. Do not automatically choose the configuration with more resident weights.

## 28. 128K gate

128K is the main long-context target.

It is acceptable to reduce permanent routed-expert residency to achieve stable 128K if decode remains usable, expert transfer does not become pathological, prefill remains inside the safe VRAM envelope, and state replay remains exact.

Allocation success alone is not enough; the model must remain practically usable.

## 29. Runtime buffer policy

Allocate once where practical:

- routing logits/probabilities;
- Top-10 IDs;
- Top-10 weights;
- expert token dispatch;
- expert staging;
- expert output accumulation;
- QSA compression;
- QSA scoring;
- QSA TopK;
- QSA gather metadata;
- attention scratch;
- GDN scratch;
- PLE scratch.

A stable steady-state decode allocator is a design objective.

## 30. NUMA relevance

The RTX 5080 is attached to one host NUMA locality.

If expert host-to-device traffic becomes material, benchmark pinned-memory allocation locality, CPU worker affinity, and host-buffer NUMA placement. Do not add NUMA complexity before transfer profiling demonstrates value.

## 31. Runtime OOM policy

Prefer early capacity validation.

Before launching a large-context session estimate persistent GPU weights + resident expert/cache budget + context-scaled state + runtime scratch + graph resources + safety margin.

If the requested configuration cannot fit, fail clearly. Do not silently reduce user-requested context unless an explicit auto-fit mode is being used.

## 32. Benchmark matrix

Minimum useful matrix:

| Test | Context | Expert policy | Graph |
|---|---:|---|---|
| correctness | small | baseline | off |
| A | 32K | baseline | off |
| B | 32K | tuned | off |
| C | 32K | tuned | on |
| D | 64K | tuned | on |
| E | 128K | context-prioritised | off |
| F | 128K | context-prioritised | on |

Capture prompt tok/s, decode tok/s, persistent VRAM, peak VRAM, host RAM, expert hit/miss, H2D bytes, and correctness.

## 33. OpenClaw validation

The final production test should include large system instructions, long code context, repeated code-generation turns, tool-call/tool-result cycles, context growth, snapshot/fork behavior if used, and prolonged decode.

A configuration that performs well only for a one-shot benchmark is not the production winner.

## 34. Recommended optimisation order

Use this order:

1. exact small-context inference;
2. complete state semantics;
3. memory accounting;
4. stable decode allocations;
5. QSA entirely on GPU;
6. core + shared expert residency;
7. simple routed-expert streaming;
8. routed-expert cache tuning;
9. exact SM120 kernel audit;
10. transfer/compute overlap;
11. CUDA Graph decode;
12. 32K;
13. 64K;
14. 128K;
15. advanced adaptive residency only if justified.

## 35. Success criteria

The RTX 5080 runtime is successful when exact Qwen4Exp semantics are retained, state replay is complete, no unexplained allocation growth exists, QSA selection remains GPU resident, expert cold misses do not allocate per token, VRAM placement is explicit, 32K and 64K are stable, 128K is stable if the measured budget permits, decode remains interactive for OpenClaw, prefill does not OOM at the selected operating point, and Blackwell kernels/CUDA Graphs are used only where measurement supports them.
