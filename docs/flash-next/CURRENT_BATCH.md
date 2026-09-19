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