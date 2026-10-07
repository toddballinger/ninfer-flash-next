# NInfer6000 reuse plan for RTX 5080 Flash-Next

## Decision

`lkarlslund/ninfer6000` is the primary external implementation reference for **native NInfer Qwen3.8-Flash-Next execution**.

The goal of `ninfer-flash-next` is no longer to independently rediscover every Flash-Next execution mechanism. The project should reuse/adapt proven native execution where practical and concentrate engineering effort on the unsolved **RTX 5080 16 GB memory-placement problem**.

Primary tracker: issue #9.

## Why

ninfer6000 currently demonstrates a working Flash-Next stack on RTX PRO 6000 Blackwell with:

- NVFP4 routed experts;
- optional FP8 non-expert projections;
- optional FP8 PLE;
- QSA;
- HyperConnection;
- Flash-Next MoE fusion/overlap work;
- MTP3;
- `--lm-head-draft`;
- adaptive draft widths;
- compact verification through 16 tokens;
- INT8 group-64 KV and FP8 KV;
- Flash-Next CUDA Graph support;
- optional NVFP4 MTP experts.

Its target has 96 GB VRAM, so its memory-placement assumptions are **not** the solution for RTX 5080 16 GB. Its model semantics, bindings, converter/artifact choices, MTP/graph behavior and operator composition are nevertheless high-value reuse candidates.

## Project boundary

### Reuse/adapt first

Before independently implementing these areas, compare against ninfer6000:

1. Flash-Next model binding and artifact mapping;
2. QSA;
3. HyperConnection;
4. routed/shared MoE execution;
5. Flash-Next MTP drafter and verification;
6. `lm-head-draft`;
7. adaptive draft-width policy;
8. FP8 projection and PLE formats;
9. NVFP4 MTP experts;
10. Flash-Next INT8/FP8 KV;
11. small-token verification kernels;
12. Flash-Next CUDA Graph capture/update behavior.

### Benchmark before porting exact tuning

Do not assume PRO 6000 kernel schedules transfer to RTX 5080. Re-benchmark:

- tile sizes;
- CTA counts;
- occupancy choices;
- exact shape dispatch;
- graph family widths;
- prefill chunk sizes;
- memory residency decisions.

### Remains project-specific

The 16 GB target still requires:

- device expert-residency planning;
- adaptive hot-expert placement;
- CPU fallback for expert misses;
- CPU/GPU overlap;
- host memory bandwidth accounting;
- PCIe traffic minimization;
- host-backed/windowed KV;
- phase-specific VRAM reuse;
- lower-bit host experts only if NVFP4 is measurably too expensive;
- 128K+ OpenClaw correctness and latency validation.

## Implementation order

1. **Reconcile first — issue #9.**
   Map active Batch 3C work against ninfer6000. Preserve the existing dirty worktree; do not reset/stash/clean/discard.
2. Establish the smallest native Flash-Next baseline that reuses already-solved execution work.
3. **Expert residency / CPU fallback — issue #2.**
4. **Host-backed KV / resident window — issue #3.**
5. **MTP adaptation/tuning — issue #4.**
   Prefer ninfer6000 adaptive MTP/lm-head-draft mechanisms over independent redesign.
6. **Phase-specific VRAM reuse — issue #6.**
7. **Low-bit host experts — issue #5.**
   Only promote after native NVFP4 host execution is measured and shown insufficient.
8. **Alternative KV research — issue #7.**
   Promote only on measured Pareto improvement.

## Validation target

All promoted work should be measured on the actual production target:

- RTX 5080 16 GB;
- large host DDR4 ECC memory;
- configured context 131,072 minimum;
- short prompts plus genuinely occupied ~118K–120K prompts;
- repeated-prefix/OpenClaw follow-up turns;
- prefill / TTFT;
- 512-token and 2048-token sustained decode where appropriate;
- correctness/tool-call gates;
- VRAM breakdown;
- host RAM bandwidth;
- PCIe traffic;
- expert hit/miss rate;
- CPU/GPU overlap;
- MTP acceptance and verification cost.

## Reuse decision rule

Continue an independent implementation only when evidence shows at least one of:

- ninfer6000 does not implement the required behavior;
- its code cannot be cleanly adapted to current NInfer;
- its behavior is tied to large-VRAM assumptions;
- it regresses correctness or performance on RTX 5080;
- the local implementation already has stronger validated evidence.

Otherwise, reuse/adapt rather than reinvent.

## Upstream watch

The Flash-Next Dual-Repo Watch should explicitly monitor:

- `lkarlslund/ninfer6000` for Flash-Next changes;
- `Neroued/ninfer` for upstream adoption or conflicting changes;
- `toddballinger/ninfer-flash-next` for local issue/PR/dependency state;
- `toddballinger/ninfer-5080` and `toddballinger/Strata-5080` for portable hot-path ideas.

Relevant ninfer6000 changes should be reconciled against issue #9 first, then folded into #2–#7 only where they materially change the 16 GB hot path, dependency order, or benchmark plan.
