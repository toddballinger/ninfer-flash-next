# Batch 3C1 — Qwen4Exp HyperConnection primitives

- Base: `97a5a9b7acb26aa3a852ed8504d77ad5c9a71bea`
- Branch: `flash-next/batch3c-hyperconnection-reference`

## Scope

Implement only the reusable mathematical primitives required by the
Qwen4Exp HyperConnection reference equations.

No Qwen4Exp execution/program integration in 3C1.

## Semantics

- initial repeat `[H,T] -> [4H,T]`
- four independent RMSNorm groups of width `H`
- `SiLU(x / hc_count)`
- read/mix:
  `mean_streams(sigmoid(logits) * normalized)`
- injection:
  `hyper += block_output * (2 * sigmoid(inject_logits / hc_count))`

Canonical Flash-Next geometry:

- `hc_count = 4`
- `hidden_size = 2560`
- `expanded_width = 10240`
- `hc_lowrank = 320`

## Files

- `include/ninfer/ops/hyper_connection.h`
- `src/ops/launcher/hyper_connection.h`
- `src/ops/launcher/hyper_connection.cu`
- `src/ops/wrapper/hyper_connection.cpp`
- `src/ops/basic_sources.cmake`
- `tests/ops/test_hyper_connection.cpp`
- `tests/ops/tests.cmake`

## Explicitly deferred to 3C2

- bound HyperConnection parameter preparation
- `linear()` composition for down/up/block-inject
- Qwen4Exp layer execution
- final model-level mixer integration
- PLE/GDN/QSA/MoE execution
- full decoder/program ownership

## GPU policy

Compile while `ninfer-local-model.service` remains running.

Before CUDA execution tests:
1. stop `ninfer-local-model.service`
2. run bounded HyperConnection GPU tests
3. restart service
4. verify service/model health

## Validation

- source hygiene (`git diff --check`): PASS
- HyperConnection primitive CUDA oracle tests: PASS
- canonical Flash-Next geometry (`C=4`, `H=2560`, `R=320`): PASS
- CUDA Graph capture/instantiate/replay: PASS
- GPU test performed with live NInfer service stopped
- `ninfer-local-model.service` restarted successfully
- `/v1/models` health check after restoration: PASS
- ccache enabled for C, C++, and CUDA compilation

## Publication state

- Implementation status: COMPLETE
- Validation status: COMPLETE
- Commit: NOT DONE before publication
- Push: NOT DONE before publication
- Next milestone: Batch 3C2 — Qwen4Exp HyperConnection executor (deferred; not started)

## Publication result

- Batch 3C1 implementation commit: c67f31f71b13d4a0fba199fe86348378e8743fbe
- GitHub publication: PASS
- Published branch: flash-next/batch3c-hyperconnection-reference
- Next batch: Batch 3C2 — Qwen4Exp HyperConnection executor
- Next batch status: deferred; not started
## RESUMPTION CHECKPOINT

- CHECKPOINT_TIMESTAMP: 2026-09-18T13:52:41+00:00
- REMOTE_HOST: brain
- REPOSITORY: /home/toddballinger/ninfer-flash-next
- ORIGIN: https://github.com/toddballinger/ninfer-flash-next.git
- BRANCH: flash-next/batch3c-hyperconnection-reference
- HEAD: 2509aaff019a1b308dd8414abd8021f3f2924674
- WORKTREE_STATE: DIRTY_EXPECTED ? eight-path 3C2a prerequisite diff preserved
- ACTIVE_BATCH: 3C2 ? Qwen4Exp HyperConnection executor; prerequisite sub-batch 3C2a
- CURRENT_STATE: 3C2a BF16 linear-shape prerequisite partially implemented; no commit or push
- CURRENT_BLOCKER: B1_IMPLEMENTATION ? validation found a missing test namespace reference and invalid BF16 schedule/static-assert choices for n4_k10240, n10240_k320, and n320_k10240; repair was not completed
- LAST_COMPLETED_ACTION: guarded 3C2a implementation produced the expected eight-path dirty scope; focused validation ran and reported compile/static-assert failures
- VALIDATION_STATE: BLOCKED ? git diff --check PASS; build/test repair incomplete; GPU validation NOT RUN
- REVIEW_STATE: no Astra review for 3C2a; no Sol escalation
- GPU_STATE: LOCAL_READY; no exclusive GPU window active
- SERVICE_STATE: ninfer-local-model.service ACTIVE; /v1/models HEALTHY with local-model
- ACTIVE_CHILD_STATE: none; 3C2a repair/validation worker cancelled cleanly at user request
- NEXT_REQUIRED_ACTION: re-run the mandatory Brain handoff, verify this checkpoint and exact dirty scope, then dispatch one bounded local-worker repair/validation task for the existing 3C2a files
- DO_NOT_REPEAT: do not reset, restore, stash, clean, discard, switch branch, start 3C2b, run GPU work, commit, or push before 3C2a build/tests pass
- RESUME_COMMAND_INTENT: Resume Flash-Next autonomous roadmap mode from this checkpoint. Re-run the mandatory fresh Brain handoff, verify this checkpoint against the live worktree/docs, then continue from NEXT_REQUIRED_ACTION. Do not redo completed work unless repository evidence shows it is necessary.

## RESUMPTION CHECKPOINT (PAUSED 2026-09-18)

- CHECKPOINT_TIMESTAMP: 2026-09-18T14:09:33+00:00
- REMOTE_HOST: brain
- REPOSITORY: /home/toddballinger/ninfer-flash-next
- ORIGIN: https://github.com/toddballinger/ninfer-flash-next.git
- BRANCH: flash-next/batch3c-hyperconnection-reference
- HEAD: 2509aaff019a1b308dd8414abd8021f3f2924674
- WORKTREE_STATE: DIRTY_EXPECTED; 9-path 3C2a scope preserved
- ACTIVE_BATCH: 3C2 / 3C2a HyperConnection linear prerequisite
- CURRENT_STATE: bounded read-only validation completed; no new files changed by the worker
- CURRENT_BLOCKER: B1_IMPLEMENTATION; focused test has a confirmed missing ninfer::test namespace; CUDA schedule validity still requires NVCC validation
- LAST_COMPLETED_ACTION: five-turn local-worker validation/logging task
- VALIDATION_STATE: git diff --check PASS; focused build/test repair incomplete; GPU validation NOT RUN
- REVIEW_STATE: no Astra review for 3C2a; no Sol escalation
- GPU_STATE: LOCAL_READY
- SERVICE_STATE: ninfer-local-model.service ACTIVE; /v1/models HEALTHY with local-model
- NEXT_REQUIRED_ACTION: re-run the mandatory Brain handoff, then dispatch one bounded local-worker repair/validation task for the existing 3C2a scope

## RESUMPTION CHECKPOINT

- CHECKPOINT_TIMESTAMP=2026-09-18T14:26:31+00:00
- REMOTE_HOST=brain
- REPOSITORY=/home/toddballinger/ninfer-flash-next
- BRANCH=flash-next/batch3c-hyperconnection-reference
- HEAD=2509aaff019a1b308dd8414abd8021f3f2924674
- WORKTREE_STATE=DIRTY_EXPECTED (9-path 3C2a scope preserved)
- ACTIVE_BATCH=3C2 / 3C2a HyperConnection-required BF16 linear prerequisite
- CURRENT_STATE=local-worker eight-turn validation window completed; no further project action
- CURRENT_BLOCKER=B1 implementation validation incomplete
- LAST_COMPLETED_ACTION=read-only diagnosis of namespace and BF16 schedule blockers
- VALIDATION_STATE=git diff --check PASS; focused build/test not completed
- REVIEW_STATE=not started
- GPU_STATE=LOCAL_READY
- SERVICE_STATE=ACTIVE; /v1/models healthy with local-model
- NEXT_REQUIRED_ACTION=guarded local-worker repair/validation of the existing 3C2a diff
- DO_NOT_REPEAT=do not recreate the 3C2a files, reset the worktree, or start 3C2b before validation passes
- RESUME_COMMAND_INTENT=Resume Flash-Next autonomous roadmap mode from this checkpoint. Re-run the mandatory fresh Brain handoff, verify this checkpoint against the live worktree/docs, then continue from NEXT_REQUIRED_ACTION. Do not redo completed work unless repository evidence shows it is necessary.

## RESUMPTION CHECKPOINT (PAUSED 2026-09-18T15:29:57Z)

- CHECKPOINT_TIMESTAMP=2026-09-18T15:29:57Z
- REMOTE_HOST=brain
- REPOSITORY=/home/toddballinger/ninfer-flash-next
- ORIGIN=https://github.com/toddballinger/ninfer-flash-next.git
- BRANCH=flash-next/batch3c-hyperconnection-reference
- HEAD=2509aaff019a1b308dd8414abd8021f3f2924674
- WORKTREE_STATE=DIRTY_EXPECTED (9-path 3C2a scope preserved)
- ACTIVE_BATCH=3C2 / 3C2a HyperConnection-required BF16 linear prerequisite
- CURRENT_STATE=six-turn local-worker window completed; read-only inspection only; no project files changed
- CURRENT_BLOCKER=B1 implementation validation incomplete
- LAST_COMPLETED_ACTION=guarded six-turn local-worker inspection of all 3C2a files
- VALIDATION_STATE=git diff --check PASS; internal consistency PASS; build/tests not run
- REVIEW_STATE=not started; Sol not used
- GPU_STATE=LOCAL_READY
- SERVICE_STATE=API healthy with local-model; no GPU transition performed
- NEXT_REQUIRED_ACTION=guarded local-worker repair/validation of the existing 3C2a diff
- DO_NOT_REPEAT=do not recreate files, reset, stash, clean, discard, switch branch, start 3C2b, run GPU work, commit, or push before 3C2a validation passes
- RESUME_COMMAND_INTENT=Resume Flash-Next autonomous roadmap mode from this checkpoint. Re-run the mandatory fresh Brain handoff, verify this checkpoint against the live worktree/docs, then continue from NEXT_REQUIRED_ACTION. Do not redo completed work unless repository evidence shows it is necessary.

## RESUMPTION CHECKPOINT — bounded four-turn validation window
CHECKPOINT_TIMESTAMP=2026-09-19
REMOTE_HOST=brain
REPOSITORY=/home/toddballinger/ninfer-flash-next
BRANCH=flash-next/batch3c-hyperconnection-reference
HEAD=2509aaff019a1b308dd8414abd8021f3f2924674
WORKTREE_STATE=DIRTY_EXPECTED — 9-path 3C2a scope preserved
ACTIVE_BATCH=3C2 / 3C2a prerequisite
CURRENT_STATE=local-worker four-turn validation completed; no mutation
CURRENT_BLOCKER=B1 validation incomplete
LAST_COMPLETED_ACTION=four-turn local-worker inspection/logging window
VALIDATION_STATE=diff-check PASS; CUDA/build validation still pending
REVIEW_STATE=not started for 3C2a
GPU_STATE=LOCAL_READY
SERVICE_STATE=active; API healthy local-model
NEXT_REQUIRED_ACTION=guarded local-worker repair/validation
DO_NOT_REPEAT=do not redo four-turn inspection unless repository evidence changes
RESUME_COMMAND_INTENT=Resume Flash-Next autonomous roadmap mode from this checkpoint. Re-run the mandatory fresh Brain handoff, verify this checkpoint against the live worktree/docs, then continue from NEXT_REQUIRED_ACTION. Do not redo completed work unless repository evidence shows it is necessary.

## RESUMPTION CHECKPOINT — 3C2a final documentation (2026-09-19)

- CHECKPOINT_TIMESTAMP=2026-09-19 (local-worker 3C2A-FINALDOC-20260919)
- REMOTE_HOST=brain
- REPOSITORY=/home/toddballinger/ninfer-flash-next
- ORIGIN=https://github.com/toddballinger/ninfer-flash-next.git
- BRANCH=flash-next/batch3c-hyperconnection-reference
- HEAD=2509aaff019a1b308dd8414abd8021f3f2924674
- WORKTREE_STATE=DIRTY_EXPECTED — exact 9-path 3C2a scope preserved; documentation-only append to docs/flash-next/CURRENT_BATCH.md
- ACTIVE_BATCH=3C2 / 3C2a prerequisite (complete); parent 3C2 executor still pending, not started
- CURRENT_STATE=3C2a prerequisite COMPLETE: B2 blocker resolved via narrow BF16 reference GEMV fallback; full 3C2a diff preserved; no commit or push
- B2_DISCOVERY=K=320 (the [10240,320] low-rank-up shape) is divisible by no phase width (128/256/512) that the phase-based gemv/SIMT ladders require, so the shape has no feasible phase schedule at any token count T — classified B2_MISSING_PREREQUISITE (missing phase-schedule capability), not a B1 implementation defect
- SOL_E3_DECISION=Sol designated option E3: serve [10240,320] exclusively by a narrow generic/reference GEMV (src/ops/linear/bf16/shapes/n10240_k320.cu): one thread per output row, serial K=320 loop, scalar BF16 loads; the existing specialized [320,10240] and [4,10240] paths are unchanged
- FROZEN_SEMANTICS_PRESERVED=reference path uses per-element FP32 FMA accumulation (fmaf on __bfloat162float scalars) and the standard BF16 store epilogue (__float2bfloat16_rn), matching the existing linear epilogue/BF16-conversion semantics; reference covers the whole serving interval at every T; no frozen contract changed
- VALIDATION_STATE=static/CUDA/GPU validation PASS for the 3C2a scope: static hygiene (git diff --check) PASS; NVCC static-schedule validation PASS; focused 3C2a oracle tests PASS in the MAIN two-phase GPU window, each test reporting exact OK output; no new blockers
- GPU_STATE=GPU window executed by MAIN per two-phase handoff (local worker made no GPU contact); RTX 5080 window returned; no local-worker GPU action
- SERVICE_STATE=ninfer-local-model.service restored to ACTIVE after the GPU window; /v1/models API health PASS with local-model; ninfer-serve untouched by this documentation task
- REVIEW_STATE=3C2a documentation finalized; final 3C2a review/publication owned by MAIN
- NEXT_REQUIRED_ACTION=MAIN: review finalized 3C2a docs, then commit/push the 9-path 3C2a diff; start the parent 3C2 Qwen4Exp HyperConnection executor batch after 3C2a publication
- DO_NOT_REPEAT=do not re-run the 3C2a GPU tests or re-stop the local-model service; do not modify the 3C2a source/test files further; do not start 3C2 before 3C2a is committed and pushed
- RESUME_COMMAND_INTENT=Resume Flash-Next autonomous roadmap mode from this checkpoint. Re-run the mandatory fresh Brain handoff, verify this checkpoint against the live worktree/docs, then continue from NEXT_REQUIRED_ACTION. Do not redo completed work unless repository evidence shows it is necessary.

## RESUMPTION CHECKPOINT — 3C2a final evidence (2026-09-19)

- CHECKPOINT_TIMESTAMP=2026-09-19 (local-worker 3C2A-FINAL-EVIDENCE-20260919)
- REMOTE_HOST=brain
- REPOSITORY=/home/toddballinger/ninfer-flash-next
- ORIGIN=https://github.com/toddballinger/ninfer-flash-next.git
- BRANCH=flash-next/batch3c-hyperconnection-reference
- HEAD=2509aaff019a1b308dd8414abd8021f3f2924674
- WORKTREE_STATE=DIRTY_EXPECTED — exact 10-path 3C2a scope preserved; docs-only append to docs/flash-next/CURRENT_BATCH.md
- ACTIVE_BATCH=3C2 / 3C2a prerequisite (numerical revalidation complete); parent 3C2 Qwen4Exp HyperConnection executor STILL PENDING, not started
- CURRENT_STATE=3C2a prerequisite numerically re-validated PASS after the n4 repair; full 10-path 3C2a diff preserved; no commit or push

- N4_ROOT_CAUSE=the block-inject [4,10240] initial GPU numerical run produced non-finite (non-finite/NaN-inf) output at T=16 and T=1024; the pre-repair T>=5 path routed to the specialized SIMT C8 schedule, whose [4,10240] geometry wrote only the first 8 token columns, leaving later token columns with uninitialized/undefined output that read back non-finite
- N4_REPAIR=n4 T>=5 is now served by a narrow generic/reference GEMV (src/ops/linear/bf16/shapes/n4_k10240.cu: bf16_ref_gemv_n4_k10240_kernel): one thread per output row, a serial K=10240 loop with scalar BF16 loads (__bfloat162float) and per-element FP32 FMA accumulation (fmaf), then the standard BF16 store epilogue (__float2bfloat16_rn); reference covers the whole serving interval at every T; N=4 cannot meet the MMA/phase geometry asserts, so the reference is the narrow generic fallback, mirroring the [10240,320] reference
- FROZEN_SEMANTICS_PRESERVED=reference path reuses the existing linear epilogue/BF16-conversion semantics (per-element FP32 FMA + __float2bfloat16_rn store); no frozen contract, formula, or schedule is changed; T=1 gemv and T=2..4 SIMT ladder for n4 are unchanged

- FOCUSED_NUMERICAL_METHODOLOGY=ctest -R ninfer_linear_hyper_numerical_test (tests/ops/linear/test_linear_hyper_numerical.cpp) drives the three 3C2a shapes through the real BF16 linear() dispatch (A16Only) at sampled token counts and compares guarded BF16 outputs against the FP64 oracle using the existing {relative_l2, gross_absolute, gross_relative_to_max_reference} reduction; per-output finiteness (std::isfinite) and guard-ring integrity are verified; the block-inject [4,10240] case covers T in {1,2,3,4,5,16,1024} eager and {2,4,16,1024} CUDA-Graph replay, so T=16 and T=1024 (the non-finite cases) are directly exercised
- FOCUSED_NUMERICAL_RESULT=after the n4 reference repair, the rerun is 100% pass (1/1 test, OK, non-finite/guard/finiteness checks all clean), 3.02 sec wall-clock; down [320,10240] and up [10240,320] numerical results preserved/unchanged
- SCOPE_EVIDENCE=exact 10-path 3C2a diff, verified via git status/diff --stat: 5 new untracked (src/ops/linear/bf16/shapes/n320_k10240.cu, n10240_k320.cu, n4_k10240.cu; tests/ops/linear/test_linear_hyper_numerical.cpp, test_linear_hyper_shapes.cpp) + 5 tracked (docs/flash-next/CURRENT_BATCH.md; src/ops/linear/bf16/bf16_dispatch.cpp, bf16_shapes.h, sources.cmake; tests/ops/linear/tests.cmake); 139 insertions
- HYGIENE_STATE=git diff --check PASS (no whitespace errors)
- GPU_STATE=two-phase MAIN GPU handoff completed (narrow-reference rerun); local worker made no GPU contact; RTX 5080 window returned
- SERVICE_STATE=ninfer-local-model.service restored to ACTIVE; /v1/models API health PASS with local-model; ninfer-serve untouched
- DOCS_UPDATED=docs/flash-next/CURRENT_BATCH.md only (this checkpoint appended); no frozen formulas, no unrelated files touched
- REVIEW_STATE=3C2a evidence finalized; final 3C2a review/publication owned by MAIN
- NEXT_REQUIRED_ACTION=MAIN: review the finalized 3C2a evidence, then commit/push the 10-path 3C2a diff; start the parent 3C2 Qwen4Exp HyperConnection executor batch (still pending) after 3C2a publication
- DO_NOT_REPEAT=do not re-run the 3C2a GPU tests or re-stop the local-model service; do not modify the 3C2a source/test files further; do not start 3C2 before 3C2a is committed and pushed
- RESUME_COMMAND_INTENT=Resume Flash-Next autonomous roadmap mode from this checkpoint. Re-run the mandatory fresh Brain handoff, verify this checkpoint against the live worktree/docs, then continue from NEXT_REQUIRED_ACTION. Do not redo completed work unless repository evidence shows it is necessary.

## RESUMPTION CHECKPOINT — Batch 3C2 M1 (2026-09-19)

- TASK=Batch 3C2 milestone M1 (Astra-approved): freeze the bounded HyperConnection
  executor API/state contract + add the independent numerical-oracle scaffolding /
  fixtures for later recurrence work (M2/M3). M1 is **API/oracle/docs only**; no
  kernels, no weight/`linear()` production binding, no shared-runtime expansion,
  no optimization, no formula changes, no parent integration, and **no production
  C++ class testing** (that is M2 work).
- REMOTE_HOST=brain
- REPOSITORY=/home/toddballinger/ninfer-flash-next
- BRANCH=flash-next/batch3c-hyperconnection-reference
- HEAD (base, guard-verified)=8a73b5a6a7030192a5e03ff8eccff66b73f101ba
- WORKTREE=CLEAN at guard; M1 adds the 3 new files + 2 tracked edits below
  (six-path M1 scope)
- M1_FILES:
  - include/ninfer/ops/hyper_connection_executor.h (bounded executor API +
    HyperConnectionExecutorState: read / callback / inject / final_mixer /
    initialize / reset; per-forward readiness lifecycle with token-consistency
    gate, `prior_read` authority, and **call-bound** module `norm_weight`
    (block vs final); contract-fixed normalization epsilon `1e-5`)
  - src/ops/hyper_connection_executor_oracle.h (independent FP64 oracle:
    initialize / group_rmsnorm / linear_down / linear_up / read_mix /
    block_inject_logits / inject / read / inject_update / final_mixer, plus the
    `HcExecutorLifecycle` contract model and the `prior_read` reference check)
  - tests/ops/linear_add/test_hyper_connection_executor_contract.cpp (CPU-runnable
    contract/oracle/fixture test with analytically independent, hand-derived
    references)
  - tests/ops/linear_add/tests.cmake (register
    ninfer_hyper_connection_executor_contract_test)
  - docs/flash-next/HYPERCONNECTION_EXECUTOR_CONTRACT.md (frozen contract doc)
- FROZEN_CONTRACT (M1):
  - read -> mixed_input [H,T] + preserved prior_read [CH,T]; read-only hyper;
    no block-inject logits computed by read.
  - callback (same stream) -> block [H,T]; block is independent, no alias to
    hyper state / live scratch; block consumes mixed_input only.
  - inject -> compute raw block-inject logits (BlockInject(Hn)); apply
    alpha = 2*sigmoid(raw / hc_count) exactly once; update hyper in place;
    zero inject logits yield alpha=1.
  - final_mixer -> no block_inject; collapse four streams to [H,T]; bound to
    the final mixer's own (call-bound) norm_weight.
  - per-forward state initializes deterministically (repeat(embedding, C)) and
    is never silently reused across forwards; read/inject/final_mixer before
    initialize or after reset are contract errors; a call whose token extent
    differs from the bound forward T is a contract error, resolved only by
    reset + re-initialize; reset clears hyper/ready and re-init is fresh.
  - prior_read (state-owned preserved prior) is the sole authority for the
    inject reference; externally supplied references are accepted only when
    they match the state-owned prior exactly (mismatches are contract
    errors / unsupported).
  - normalization epsilon is contract-fixed at 1e-5 for every module/call;
    norm_weight is call-bound and module-owned (block vs final may differ).
- ORACLE_EVIDENCE (M1, CPU): independent FP64 reference (header-only; no
  production helper/kernel/linear() helper) on a small well-conditioned
  geometry (H=8, C=4, R=5; H != T). Fixtures use analytically independent,
  hand-derived references (distinct streams/tokens, nonzero logits, nonuniform
  norm weights, identifiable projection entries); exact compare rejects
  nonfinite operands. Detects missing /C scaling, missing sigmoid, [rows,T]
  transposition, and broken stream grouping. The `HcExecutorLifecycle`
  readiness/token-consistency validation is the **contract model only** —
  M1 does not test the production C++ class (M2 work), and M1 performs **no**
  production weight/kernel validation (M2/M3 work).
- VALIDATION (M1, CPU): git diff --check clean; direct binary +
  ctest -R ninfer_hyper_connection_executor_contract_test pass on CPU;
  no GPU needed.
- REMAINING_WORK (next milestone, M2): implement and bind the production
  `linear()` + 3C1-primitive kernels for read/callback/inject/final_mixer per
  the contract doc s.8 weight mapping, bind the production
  HyperConnectionExecutorState (readiness lifecycle, call-bound norm_weight,
  prior_read authority), and validate it against this M1 oracle (CPU first,
  then the two-phase GPU handoff per the established policy); M3 then covers
  recurrence, parent-decoder integration, and GPU numerical validation.
- DO_NOT_REPEAT: do not implement M2/M3 kernels, run GPU work, or integrate
  the parent model in M1; do not commit/push (MAIN owns) before M1
  validation + review; M1 remains API/oracle/docs-only and binds no
  production weights or class behavior.


## R6A REPAIR CHECKPOINT — Batch 3C2 M1 (2026-09-20)

- TASK_ID=3C2A-M1-R6A
- SOL_ESCALATION_REASON=E1 (two acquired local-worker R6A attempts timed out
  without source mutation; Sol repaired only the five Astra findings)
- BASE_GUARD=PASS: origin, branch
  `flash-next/batch3c-hyperconnection-reference`, HEAD
  `8a73b5a6a7030192a5e03ff8eccff66b73f101ba`, and exact six-path dirty scope
  re-verified before mutation
- LAYOUT_REPAIR=production BF16 `linear()` physical weight layout is frozen as
  contiguous input lanes per output (`n*K+k`): down `r*CH+j`, up `j*R+r`,
  inject `c*CH+j`; activation is token-major `t*K+k`. Contract, oracle, and
  asymmetric offset fixtures now agree.
- FIXTURE_REPAIR=read/final/inject projections use small signed,
  nonsaturating weights; injection logits are hand-calculated independently
  from `block_inject_logits`; the offset test probes distinct nonsymmetric
  physical slots.
- LIFECYCLE_REPAIR=initialize is exactly once per forward until reset;
  consuming calls validate caller T, lifecycle T, supplied-state T/storage,
  readiness, and prior validity. Negative coverage spans read, inject, and
  final_mixer; successful inject consumes the pending prior snapshot.
- PRIOR_REPAIR=`HcState` owns `prior_read` plus validity. Read captures it;
  inject logits use live hyper while the additive base is the preserved
  snapshot. A divergence fixture accepts only the matching snapshot and
  rejects live-hyper/external mismatch.
- NORM_OWNERSHIP_REPAIR=block and final module structs own their distinct
  `norm_weight` values; complete modules are call-bound; executor state owns
  no module weights; contradictory unweighted/state-rebinding wording removed.
- VALIDATION=`cmake --build build-3c2a --target
  ninfer_hyper_connection_executor_contract_test -j 4` PASS; direct CPU
  contract/oracle binary PASS; `ctest --output-on-failure -R
  ninfer_hyper_connection_executor_contract_test` PASS (1/1);
  `git diff --check` PASS. No GPU or service action.
- SCOPE=exact six M1 paths preserved; no production kernels, parent
  integration, formula changes, commit, or push.
- REVIEW_STATE=R6A source/contract repair complete; fresh Astra review is
  required before M2 resumes.

### Batch 3C2 M1 — R7 repair checkpoint (2026-09-20)

Astra R7 findings (contradictory `inject` header equation; circular
`test_prior_authority` prior-authority divergence assertion) repaired; exact
six M1 paths and the frozen invariants preserved.

- R7_FINDING_1=`include/ninfer/ops/hyper_connection_executor.h` `inject`
  doc block re-derived to state unambiguously that the in-place update base
  is the **state-owned preserved `prior_read`** (the `read`-time snapshot),
  not the live `hyper`:
  `hyper[c,h,t] = prior_read[c,h,t] + block[h,t] * alpha[c,t]`. The
  contradictory `hyper[c,h,t] += block[h,t]*alpha[c,t]` wording and the
  "sole additive/reference authority" phrasing are removed; block-inject
  logits remain a function of the live, normalized `hyper`; a missing
  snapshot is a contract error; an external reference is unsupported unless
  it matches the state-owned `prior_read` exactly (s.6.3).
- R7_FINDING_2=`tests/ops/linear_add/test_hyper_connection_executor_contract.cpp`
  `test_prior_authority` circular prior-authority divergence assertion
  (which compared two oracle-derived vectors, `via_ref` vs `via_state`,
  making the check tautological) replaced with an **independently
  hand-derived** fixture: the `read` snapshot is preserved, the **live
  `hyper` is then mutated by +7 at a single lane**, and the expected updated
  state (prior base) plus the live-base substitution are hand-derived from
  that base (per-stream normalization + inject projection, no oracle
  `inject`/`logit` helper for the expected value). The successful `inject`
  must match the independent prior-base expectation and must **not** match
  the live-base substitution (the +7 lane-0 shift is the observable
  divergence). Mismatching external live-hyper reference rejection and
  snapshot consumption are retained.
- VALIDATION=`cmake --build build-3c2a --target
  ninfer_hyper_connection_executor_contract_test` PASS; direct CPU
  contract/oracle binary PASS; `ctest -R
  ninfer_hyper_connection_executor_contract_test` PASS (1/1); `git
  diff --check` PASS. No GPU or service action.
- SCOPE=exact six M1 paths preserved; no production kernels, parent
  integration, formula changes, commit, or push.
- REVIEW_STATE=R7 repair complete; the two prior Astra R7 findings are
  closed; fresh Astra re-review is required before M2 resumes.
## RESUMPTION CHECKPOINT — M2 handoff / M1 closure (2026-09-21)

- TASK_ID=3C2-M1-ASTRA-CLOSURE-R13-HANDOFF
- REMOTE_HOST=brain
- REPOSITORY=/home/toddballinger/ninfer-flash-next
- BRANCH=flash-next/batch3c-hyperconnection-reference
- HEAD=8a73b5a6a7030192a5e03ff8eccff66b73f101ba
- WORKTREE_STATE=DIRTY_EXPECTED — exact six M1 paths preserved; docs-only append to docs/flash-next/CURRENT_BATCH.md
- MILESTONE=M1 closure: Astra R13 fresh review returns M1 milestone PASS; FRESH_REVIEW_ID=3C2-M1-ASTRA-CLOSURE-20260920-R13
- R10A=lifecycle tests PASS (read/callback/inject/final_mixer readiness, token-consistency, and prior-authority lifecycle coverage)
- R10B3=configure/build/direct/CTest all RC=0 (configure=0, build=0, direct=0, ctest -R ninfer_hyper_connection_executor_contract_test=0)
- R11=evidence labels corrected (oracle/oracle-label vs production-class labeling reconciled; M1 is API/oracle-only, no production-class claims)
- M1_GATE=CLEAR (M1 milestone gate cleared; M1 is complete and review-closed)
- NEXT_TASK_ID=3C2-M2-KICKOFF-20260921
- M2_SCOPE=API/production executor composition only (bind production linear() + 3C1-primitive kernels for read/callback/inject/final_mixer; bind the production HyperConnectionExecutorState; no recurrence, no parent integration, no GPU until M2 CPU validation passes)
- M2_ALLOWED_MUTATIONS (Sol permit):
  - src/ops/launcher/hyper_connection_executor.h
  - src/ops/launcher/hyper_connection_executor.cu
  - src/ops/wrapper/hyper_connection_executor.cpp
  - src/ops/basic_sources.cmake
  - tests/ops/test_hyper_connection_executor.cpp
  - tests/ops/tests.cmake
  - tests/ops/linear_add/test_hyper_connection_executor_contract.cpp
  - docs/flash-next/CURRENT_BATCH.md
- CPU_VALIDATION=CPU validation is REQUIRED before any GPU work; M2 CPU validation (build + contract/oracle ctest on CPU) must pass before the two-phase GPU handoff
- GPU_HANDOFF_REQUIRED_LATER=YES
- GPU_STATE=NO GPU or service action now; M2 CPU validation precedes any GPU window (local worker makes no GPU contact; MAIN owns the two-phase handoff)
- SERVICE_STATE=ninfer-local-model.service ACTIVE; /v1/models HEALTHY with local-model; ninfer-serve untouched by this documentation task
- HYGIENE_STATE=git diff --check PASS (no whitespace errors)
- REVIEW_STATE=M1 closure review (R13) complete; M2 kickoff is next
- DOCS_UPDATED=docs/flash-next/CURRENT_BATCH.md only (this checkpoint appended); no frozen formulas, no unrelated files touched
- DO_NOT_REPEAT=do not run GPU work, do not start the M2 GPU window, do not modify 3C2a/M1 source beyond the M2 allowlist, and do not commit or push before M2 CPU validation + fresh review; M1 prior text is preserved as-is above
- RESUME_COMMAND_INTENT=Resume Flash-Next autonomous roadmap mode from this checkpoint. Re-run the mandatory fresh Brain handoff, verify this checkpoint against the live worktree/docs, then continue from NEXT_TASK_ID=3C2-M2-KICKOFF-20260921. Do not redo completed work unless repository evidence shows it is necessary.
## RESUMPTION CHECKPOINT — WorkspaceArena amendment + M1 R13 closure (2026-09-21)

- TASK_ID=3C2-M1-WORKSPACEARENA-AMEND-R13
- REMOTE_HOST=brain
- REPOSITORY=/home/toddballinger/ninfer-flash-next
- BRANCH=flash-next/batch3c-hyperconnection-reference
- HEAD=8a73b5a6a7030192a5e03ff8eccff66b73f101ba
- WORKTREE_STATE=DIRTY_EXPECTED — exact six M1 paths preserved plus the already-authorized amendment; docs-only append to docs/flash-next/CURRENT_BATCH.md
- M1_GATE=CLEAR (Astra R13 fresh review returns M1 milestone PASS; M1 complete and review-closed)
- WORKSPACEARENA_AMENDMENT=approved and applied
- SIGNATURES=read/inject/final_mixer each take WorkspaceArena& before cudaStream_t
- WORKSPACE_QUERY=hyper_connection_executor_workspace_capacity_bytes(C,H,R,T)
- WORKSPACE_FORMULA=max(2*T*(2*C*H+R), 2*T*(C*H+C)) bytes
- M2_ALLOWED_MUTATIONS (expanded allowlist = original eight paths plus the two amended paths):
  - src/ops/launcher/hyper_connection_executor.h
  - src/ops/launcher/hyper_connection_executor.cu
  - src/ops/wrapper/hyper_connection_executor.cpp
  - src/ops/basic_sources.cmake
  - tests/ops/test_hyper_connection_executor.cpp
  - tests/ops/tests.cmake
  - tests/ops/linear_add/test_hyper_connection_executor_contract.cpp
  - docs/flash-next/CURRENT_BATCH.md
  - include/ninfer/ops/hyper_connection_executor.h
  - docs/flash-next/HYPERCONNECTION_EXECUTOR_CONTRACT.md
- CPU_VALIDATION=CPU validation REQUIRED before any GPU handoff; no GPU/service/commit/push yet
- GPU_HANDOFF_REQUIRED_LATER=YES (M2 CPU validation precedes any two-phase GPU handoff)
- DOCS_UPDATED=docs/flash-next/CURRENT_BATCH.md only (this checkpoint appended); no source/tests/CMake/oracle touched; no frozen formulas changed
- DO_NOT_REPEAT=no GPU window, no commit/push, no M1 source edits outside the amended allowlist
- RESUME_COMMAND_INTENT=Continue after M2 CPU validation passes; re-verify this checkpoint against the live worktree/docs before resuming autonomous Flash-Next roadmap mode.
