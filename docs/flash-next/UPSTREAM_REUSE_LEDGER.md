# Flash-Next upstream reuse ledger

This ledger is the practical checklist for reconciling `lkarlslund/ninfer6000` into
`toddballinger/ninfer-flash-next` without duplicating solved work.

Status values:
- **ADOPT** — architecture/semantics should be reused unless review finds a blocker.
- **BENCHMARK** — reuse concept/code path but retune exact implementation on RTX 5080.
- **LOCAL** — remains specific to the 16 GB project.
- **BLOCKED** — needs evidence before implementation continues.

| Area | Upstream source | Status | Local owner | Evidence / next action |
|---|---|---|---|---|
| Common ancestry | `e360c4c06773ae08bddfda0c6c67ce834d4f584a` | ADOPT | #9 | Both repos share this exact base; reconcile by range rather than greenfield port. |
| Model family/runtime | `src/models/qwen3_8_flash_next/` | ADOPT | #9 | Bring in family/runtime after prerequisite source mapping. |
| 125B-A6B package | `src/models/qwen3_8_flash_next_125b_a6b/` | ADOPT | #9 | Reuse config/load-plan/package structure. |
| RMSNorm epsilon | ninfer6000 config = `1e-6`; local contract = `1e-5` | BLOCKED | #8/#9 | Verify exact source checkpoint and correct contract/oracle together if required. |
| HyperConnection semantics | `include/ninfer/ops/hyperconnection.h`, `src/ops/launcher/hyperconnection.cu` | ADOPT + VERIFY | #8/#9 | Keep local oracle as independent conformance test; do not discard it. |
| HyperConnection small-T fusion | `1cd1b917...` plus preceding fusion work | BENCHMARK | #8/#9 | Test exact schedule on 5080 after semantic equivalence. |
| QSA | `include/ninfer/ops/flash_next_qsa.h`, `src/ops/softmax_attention/qsa/flash_next_qsa.cu` | ADOPT | #9 | Include correctness fixes before tuning. |
| QSA compact-batch tuning | `a92df9a8...`, `0f31336e...`, `1f3ab2c5...` | BENCHMARK | #9 | Re-benchmark on 5080. |
| PLE model semantics | `include/ninfer/ops/flash_next_ple.h`, model PLE integration | ADOPT | #9 | Reuse file-mapped selected-row architecture. |
| PLE host placement | file mapping + page cache + gather workers | ADOPT + BENCHMARK | #9 | Measure cold/warm page-cache and NUMA placement on production host. |
| FP8 PLE | `6565a2b2...` | ADOPT | #9 | Useful 51.2 GB host representation; quality gate already documented upstream, reproduce selectively. |
| Sparse MoE semantics | `include/ninfer/ops/flash_next_moe.h` and flash-next sparse_moe path | ADOPT | #9 | Reuse routing/shared-expert semantics. |
| MoE one-token fusion | `e6c1c3f2...`, `42fd08d7...`, `647be071...` | BENCHMARK | #9 | Architecture portable; exact crossover may differ. |
| MoE 2–8 verify fusion | `97a033a2...`, `d24f9c2c...` | BENCHMARK | #4/#9 | Relevant to MTP verify width. |
| Main routed experts | NVFP4 bank layout | ADOPT | #2/#9 | First representation to test under partial residency. |
| FP8 non-expert projections | `87163868...`, `d2f42756...`, `bd7c391c...`, `9b80b069...` | ADOPT + BENCHMARK | #9 | Use bindings/format; benchmark exact schedules. |
| MTP baseline | native Flash-Next MTP | ADOPT | #4/#9 | Do not create parallel speculative runtime. |
| lm-head-draft | ninfer6000 native path | ADOPT | #4 | Benchmark memory/speed on 5080. |
| Adaptive drafts | `532778a5...` | ADOPT + BENCHMARK | #4 | Port policy; tune with agent/code/tool traces. |
| Compact verify <=16 | `2fa4c756...` | ADOPT + BENCHMARK | #4 | Important for adaptive drafts and concurrency. |
| Adaptive timing fix | `d96e6315...` | ADOPT | #4 | Treat as baseline correctness of policy measurement. |
| NVFP4 MTP experts | `8df59b0b...` | ADOPT + BENCHMARK | #4/#9 | Upstream reports ~3.4 GiB device-weight saving. Especially relevant to 16 GB. |
| INT8 group-64 KV | `fef9f976...` | ADOPT | #3/#9 | Reuse QSA integration, then compare host-residency variants. |
| FP8 KV | existing + Flash-Next integration | ADOPT | #3 | Keep in benchmark matrix. |
| Concurrent PLE decode fix | `8d89297a...` | ADOPT | #9 | Baseline correctness. |
| Batched MTP tensor fix | `a70d5ec0...` | ADOPT | #9 | Baseline correctness. |
| Concurrent prefill KV fix | `b74add31...` | ADOPT | #9 | Baseline correctness. |
| Ordinary graph replay fix | `68409b45...` | ADOPT | #9 | Baseline correctness. |
| Adaptive capture sizing fix | `742a38e7...` | ADOPT | #4/#9 | Baseline correctness. |
| Expert residency | no 16 GB solution upstream | LOCAL | #2 | Primary project-specific implementation. |
| CPU expert fallback | no 16 GB solution upstream | LOCAL | #2 | Benchmark NVFP4 before lower-bit formats. |
| CPU/GPU expert overlap | no 16 GB solution upstream | LOCAL | #2 | Measure useful work per wall clock. |
| Host-backed KV window | no 16 GB solution upstream | LOCAL | #3 | Trade KV VRAM for expert residency. |
| Phase-specific VRAM reuse | no 16 GB solution upstream | LOCAL | #6 | Must preserve graph-address invariants. |
| GSQ/IQ3 host experts | alternative external path | DEFER | #5 | Only promote if NVFP4 host path fails measured target. |
| Hadamard Q4 KV | alternative external path | DEFER | #7 | Only promote after #3 proves a KV bottleneck and current formats lose frontier. |

## Reconciliation handoff checklist

Before a code implementation PR is opened:

- [ ] exact common-base and target upstream SHA recorded;
- [ ] local Batch 3C work preserved, not reset/stashed/discarded;
- [ ] RMS epsilon discrepancy resolved from source checkpoint evidence;
- [ ] model family/package dependency set enumerated;
- [ ] correctness-fix commits included in the baseline plan;
- [ ] local HyperConnection oracle retained as independent validation;
- [ ] direct-reuse versus benchmark-first files identified;
- [ ] no PRO 6000 memory-residency assumption copied into the 16 GB planner;
- [ ] 5080 GPU validation plan attached;
- [ ] 128K/OpenClaw benchmark plan attached;
- [ ] rollback / production NInfer service protection plan attached.

## Review rule

A future upstream ninfer6000 commit should first be classified here or in issue #9 as:

1. **baseline correctness** — fold immediately into reconciliation;
2. **portable execution architecture** — adopt;
3. **performance tuning** — benchmark first;
4. **large-HBM-specific** — do not port;
5. **16 GB relevant but unsolved** — fold into #2/#3/#6;
6. **format fallback** — fold into #5/#7 only after prerequisite evidence.

This ledger should be updated by the Flash-Next Dual-Repo Watch when a new upstream change materially
changes one of those classifications.
