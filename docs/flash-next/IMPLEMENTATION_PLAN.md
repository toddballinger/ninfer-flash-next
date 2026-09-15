# Flash-Next Implementation Plan

## 1. Purpose

This document defines the staged implementation of Qwen4Exp / Qwen3.8-Flash-Next support in NInfer.

The implementation must proceed as small validated batches. Do not treat Flash-Next support as one monolithic coding task.

## 2. Mandatory pre-edit documents

Before modifying implementation source, read completely:

1. `docs/flash-next/REFERENCE_FORMULAS.md`
2. `docs/flash-next/BASELINE.md`
3. `docs/flash-next/ARCHITECTURE.md`
4. `docs/flash-next/IMPLEMENTATION_PLAN.md`
5. `docs/flash-next/RTX5080_RUNTIME_STRATEGY.md`

If any are missing, stop.

## 3. General rules

For every implementation batch:

1. inspect;
2. map existing NInfer machinery;
3. make the smallest coherent change;
4. build;
5. test;
6. compare against the frozen formulas;
7. record findings;
8. commit only a verified state.

Do not combine unrelated refactoring with Flash-Next support. Do not optimize unverified mathematics. Do not silently approximate unsupported semantics.

## 4. Batch 0 — repository reconnaissance

Objective: map the current NInfer tree to every Flash-Next subsystem before editing.

Source edits: none.

Find existing implementations for artifact architecture identification, metadata/config extraction, tensor binding, model instance/config objects, model graph/layer construction, RMSNorm/group RMSNorm, RoPE, causal Conv1D, recurrent state, gated-delta/GDN-like operations, ordinary attention, sparse attention, GPU TopK, KV cache, sequence snapshot/fork/replay, MoE routing, grouped expert dispatch, shared expert, quantized gate/up/down projections, Blackwell kernels, host/device mapped weight support, materialisation, CUDA Graphs, runtime memory planning, server/model command-line paths, tests, and benchmarks.

Required output:

```text
Flash-Next requirement
    -> existing file/type/function
    -> REUSE / EXTEND / NEW / UNKNOWN
```

Also identify likely files for Batch 1. No implementation edits before this map is complete.

## 5. Batch 1 — architecture identity and config

Objective: recognize Qwen4Exp and produce a validated configuration object.

Scope only:

- model-family identification;
- architecture metadata parsing;
- layer-type schedule;
- HyperConnection geometry;
- PLE config;
- GDN/QSA config;
- MoE config;
- shared-expert config;
- diagnostics.

Out of scope: new execution kernels, MoE runtime changes, attention optimization, memory-placement optimization.

Validation: load real artifact metadata and verify canonical target values. Incorrect or incompatible metadata must fail clearly. Existing architectures must continue to identify correctly.

Suggested commit: `flash-next: add Qwen4Exp architecture configuration`.

## 6. Batch 2 — tensor inventory and binder

Objective: resolve every required tensor without yet requiring complete inference.

Inventory categories: token embedding, HyperConnection, PLE, GDN, QSA indexer, main attention, router, routed experts, shared expert, shared-expert gate, norms, final HyperConnection mixer, output head.

Validate expected/discovered tensor counts, shapes, storage types, layer index, and expert index. Missing required tensors are hard failures.

## 7. Batch 3 — HyperConnection reference path

Objective: implement four-stream model state and exact HyperConnection semantics.

Required operations:

- initial 2560 -> 10240 repeat;
- grouped RMSNorm;
- low-rank read/mix;
- per-block injection weights;
- four-stream residual update;
- final global mixer.

Use synthetic tests against an independent implementation of the frozen equations. Do not proceed if HyperConnection is approximate.

## 8. Batch 4 — persistent state framework

Objective: establish complete Flash-Next sequence-state ownership before stateful operators expand.

State image must cover GDN convolution state, GDN recurrent state, PLE convolution state, PLE lexical token history, QSA ordinary KV, QSA raw indexer keys, and full required position history.

Validate allocate, reset, snapshot, restore, fork, and replay.

## 9. Batch 5 — PLE lexical hash path

Implement exact PLE token-history and n-gram hash semantics.

Verify one-based to zero-based layer mapping, EOS left padding, EOS history boundary, n=2 hashing, n=3 hashing, per-head prime vocabulary, offsets, sixteen heads, 160 dimensions/head, and 2560 concatenated output.

## 10. Batch 6 — PLE injection and convolution

Implement the exact PLE interaction with four-stream HyperConnection state.

Verify key projection, value projection, grouped RMSNorm, per-stream dot gate, `sqrt(2560)` scale, signed square root, sigmoid, four-stream broadcast, flatten to 10240, grouped RMSNorm, depthwise conv, kernel 4, dilation 3, nine-step history, SiLU, and residual addition before attention/GDN HyperConnection read.

Test both prefill and T=1 continuation.

## 11. Batch 7 — GDN reference path

Implement exact GDN semantics.

Verify `a` projection, `b` projection, beta sigmoid, FP32 discretisation, A_log handling, softplus with dt_bias, head repetition, causal-convolution state, recurrent matrix state, chunk/prefill path, and T=1 recurrent decode path.

Critical test: prefill N tokens followed by single-token decode must match equivalent reference execution. Snapshot/restore must reproduce the same next state and output.

## 12. Batch 8 — QSA indexer cache

Implement indexer projection and raw-key persistence.

Verify four query heads, one KV head, 128 head dimension, query RMSNorm, query RoPE, raw key storage, and position history.

Do not yet optimize block selection.

## 13. Batch 9 — QSA block compression

Implement exact visible-history compression.

For every query verify causal visibility, complete groups of four, FP32 four-key mean, cast semantics, RMSNorm, first-token RoPE position, and incomplete-tail preservation.

Build deterministic boundary tests around history lengths 0, 1, 2, 3, 4, 5, 7, 8, and 9.

## 14. Batch 10 — QSA GPU scoring and TopK

Implement production sparse-selection semantics.

Formula:

```text
per-head score = dot(q_head, block_key)
block score = sum(ReLU(per-head score)) / sqrt(128)
```

Scoring must preserve required FP32 semantics.

Selection uses `block TopK = min(512, complete block count)`. Expand selected blocks into four token IDs and append all incomplete visible-tail IDs.

Production requirement: scoring and TopK stay on GPU. Do not construct a dense CPU boolean mask.

## 15. Batch 11 — sparse QSA attention

Execute main attention only over selected token IDs.

Verify Q/K/V projections, learned Q gate, Q/K RMSNorm, RoPE, selected-KV gather/addressing, FP32 softmax semantics, and output projection.

Compare against dense reference semantics on short contexts where practical.

## 16. Batch 12 — MoE router reference path

Prove routing exactness before expert-runtime optimization.

Verify 512 logits, FP32 softmax, TopK 10, selected-only renormalization, and cast only after renormalization.

For deterministic inputs expose router logits, selected expert IDs, selected probabilities, and final route weights.

## 17. Batch 13 — routed experts

Execute selected expert gate/up/down formulas correctly:

```text
gate_up
SiLU(gate) * up
down
route-weighted accumulation
```

Test one selected expert, multiple selected experts, multiple tokens sharing one expert, and multiple tokens selecting different experts.

Reuse existing grouped expert infrastructure only where exact.

## 18. Batch 14 — shared expert

Add independent shared-expert contribution.

Verify shared expert, sigmoid shared gate, `shared_gate * shared`, then routed + shared. Confirm no routed probability is applied to the shared branch.

## 19. Batch 15 — complete decoder ordering

Assemble one complete decoder layer in exact frozen order:

```text
optional PLE
attention HyperConnection read
GDN or QSA
attention injection
MLP HyperConnection read
routed MoE + shared expert
MLP injection
```

Then assemble all 48 layers according to the configured schedule and finalize through the global HyperConnection mixer.

## 20. Batch 16 — first end-to-end inference

Obtain mathematically correct generation using the simplest runtime policy.

Prefer conservative memory placement and disable optional performance complexity if necessary.

Verify no NaN/Inf, deterministic output, prefill, T=1 decode, repeated decode, reset, snapshot/restore, and fork/replay where enabled.

Only after this batch passes should runtime placement become the main focus.

## 21. Batch 17 — memory accounting

Instrument/report model/artifact storage, persistent GPU weights, transformed persistent weights, host weights, persistent sequence state, QSA caches, GDN state, PLE state, expert cache/staging, prefill scratch, decode scratch, graph resources, and peak device use.

Do not tune residency without these measurements.

## 22. Batch 18 — selected-expert streaming

Exploit the fact that exact routing completes before routed expert execution:

```text
route
  -> exact Top-10 IDs
  -> resolve expert residency
  -> transfer cold misses only
  -> execute selected experts
```

No semantic approximation is required. Initially use a deterministic residency/cache policy.

## 23. Batch 19 — RTX 5080 residency tuning

Prioritize GPU residency for the core always-used model path, shared expert, router, then high-value routed experts/cache while preserving runtime state and context headroom.

Measure trade-offs between routed-expert residency, context capacity, and staging/cache capacity. Do not optimize solely for model-weight residency.

## 24. Batch 20 — Blackwell kernel audit

For heavy operations identify tensor storage format, execution format, current kernel, SM120 path, temporary conversions, tensor-core usage, and bandwidth limitations.

Prioritize decode-dominant operations. Any new kernel must retain the existing correct path for comparison.

## 25. Batch 21 — CUDA Graph decode

Attempt only after stable buffer addresses and residency behavior exist. Compare identical workloads with graphs OFF vs ON. Retain graph mode only if real decode improves.

## 26. Batch 22 — context progression

Progress deliberately through 4K/8K, 32K, 64K, then 128K.

At each step validate correctness, prefill, decode, peak VRAM, host RAM, state growth, and repeated turns. Do not jump directly to maximum context.

## 27. Batch 23 — OpenClaw workload validation

Test representative large system prompts, source-tree/code context, programming questions, tool-call loops, long conversation history, and repeated agent turns.

Measure prompt throughput, sustained decode, latency, context, expert cache behavior, peak GPU memory, host traffic, and stability.

This workload is more important than a short synthetic benchmark.

## 28. Stop conditions

The implementation agent must stop and report evidence if any mandatory planning document is missing, the frozen formulas are ambiguous, actual artifact tensors contradict the frozen plan, required tensor mapping is uncertain, an existing operator cannot be proven mathematically equivalent, a shared-code change breaks existing architectures, optimization changes output unexpectedly, state replay is not exact, OOM blocks required validation, or unrelated working-tree changes are present.

Do not invent around a blocked condition.

## 29. Commit discipline

Prefer one coherent verified batch per commit. Examples:

```text
flash-next: add Qwen4Exp architecture configuration
flash-next: bind canonical model tensors
flash-next: add HyperConnection reference path
flash-next: add complete sequence state image
flash-next: implement PLE
flash-next: implement GDN reference path
flash-next: add QSA GPU selection
flash-next: implement sparse QSA attention
flash-next: add exact MoE routing
flash-next: add routed and shared experts
flash-next: enable end-to-end Qwen4Exp inference
flash-next: add selected-expert streaming
flash-next: tune RTX 5080 residency
flash-next: validate 128K runtime
```

Do not mix correctness fixes and speculative optimizations when they can be separated.

## 30. Benchmark record

Record commit, model artifact, GPU, driver, CUDA, context, prompt length, generated length, batch, microbatch, graph mode, expert residency policy, persistent VRAM, peak VRAM, host RAM, prompt tok/s, decode tok/s, expert transfers, expert cache hits/misses, correctness result, and notes.

Performance numbers without runtime configuration are not comparable.

## 31. Definition of completion

Flash-Next support is complete only when Qwen4Exp config is validated, canonical tensors bind, HyperConnection/PLE/GDN/QSA/MoE/shared-expert semantics match reference, all persistent state participates in sequence semantics, end-to-end deterministic inference succeeds, existing architectures regress cleanly, RTX 5080 memory use is understood, 32K and 64K are stable, 128K is validated if the measured hardware budget permits, and sustained OpenClaw-style decode is practically usable.
