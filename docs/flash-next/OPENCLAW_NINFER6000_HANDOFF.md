# OpenClaw handoff — ninfer6000 reconciliation

Use this only when local build/GPU execution is authorized. Static reconciliation and documentation
are already prepared in issue #9 and draft PR #10.

## Goal

Create a clean, reviewable integration branch that brings the native NInfer Flash-Next
implementation from `lkarlslund/ninfer6000` into this repository **without touching the active
Batch 3C worktree**, then validate the merged baseline before starting 16 GB-specific work.

## Hard safety constraints

- Do not reset, stash, clean, discard, overwrite or switch the existing dirty
  `flash-next/batch3c-hyperconnection-reference` worktree.
- Do not merge the reconciliation branch into master automatically.
- Do not stop the production `ninfer-local-model.service` except for a bounded GPU test window.
- Restore the production service and verify `/v1/models` after every GPU window.
- Do not invent new Flash-Next execution paths before checking ninfer6000.
- Treat issue #9 as the anti-duplication gate and issue #1 as the priority queue.

## Pre-read

Read, in order:

1. `docs/flash-next/NINFER6000_REUSE_PLAN.md`
2. `docs/flash-next/NINFER6000_RECONCILIATION_REPORT.md`
3. `docs/flash-next/UPSTREAM_REUSE_LEDGER.md`
4. `docs/flash-next/NINFER6000_PORT_MANIFEST.md`
5. issue #9
6. issue #8
7. current `docs/flash-next/CURRENT_BATCH.md` on the Batch 3C branch

## Stage 1 — inventory only

Report before mutation:

- current repo path, branch, HEAD and worktree state;
- current master HEAD;
- active Batch 3C branch HEAD and exact dirty-path inventory;
- existing remotes;
- whether `ninfer6000` remote already exists;
- common-base proof with
  `e360c4c06773ae08bddfda0c6c67ce834d4f584a`;
- current `ninfer6000/master` HEAD;
- ahead/behind counts.

If the Batch 3C dirty inventory differs materially from issue #8, stop and update the issue before
any integration mutation.

## Stage 2 — create isolated integration worktree

Use a **new worktree**. Do not reuse the Batch 3C directory.

Suggested shape:

```bash
git remote add ninfer6000 https://github.com/lkarlslund/ninfer6000.git   # only if absent
git fetch ninfer6000
git fetch origin

git worktree add ../ninfer-flash-next-reconcile -b reconcile/ninfer6000-core origin/master
cd ../ninfer-flash-next-reconcile
git merge --no-ff ninfer6000/master
```

If the merge conflicts, resolve only in the new worktree. Do not use the dirty Batch 3C tree as a
conflict-resolution scratchpad.

## Stage 3 — classify conflicts

For every conflict, classify it:

- **LOCAL-DOC** — preserve repository-local project docs/issue references.
- **UPSTREAM-GENERIC** — prefer current ninfer6000/upstream NInfer behavior unless local evidence says otherwise.
- **FLASH-NATIVE** — prefer ninfer6000 native model/runtime semantics.
- **5080-SPECIFIC** — preserve local implementation only with explicit evidence.
- **UNKNOWN** — stop and request review.

Record the resolution and reason.

## Stage 4 — CPU/build qualification

Before GPU work:

- configure Release;
- build product targets;
- run non-real-model CTest;
- run Flash-Next converter/unit tests that do not require the 125B artifact;
- run HyperConnection CPU/oracle tests;
- run `git diff --check`.

Do not “fix” failures by deleting native Flash-Next source or disabling tests without root-cause evidence.

## Stage 5 — resolve RMS epsilon

This is a mandatory semantic gate.

Current discrepancy:

```text
local Batch 3C contract: 1e-5
ninfer6000 125B config:   1e-6
```

Verify the exact source checkpoint `rms_norm_eps`.

If it is `1e-6`:

- correct the local reference formula/contract/oracle as one bounded change;
- retain the independent oracle structure;
- run CPU validation;
- prepare GPU validation;
- update issues #8 and #9 with source evidence and the exact corrective diff.

Do not change only the production kernel and leave the oracle stale.

## Stage 6 — bounded GPU component qualification

Request/obtain a GPU window, then stop the production service.

Run the smallest useful component gates first:

1. HyperConnection eager/oracle;
2. HyperConnection CUDA Graph replay;
3. QSA;
4. PLE;
5. Flash-Next MoE;
6. MTP target verification;
7. eager versus graph equivalence.

On completion, restart `ninfer-local-model.service` and verify `/v1/models` returns `local-model`.

If any component fails, stop at the first root-cause boundary. Do not continue to a full model test.

## Stage 7 — artifact qualification

Start with the simplest supported native artifact source.

Preferred order:

1. RadixArk NVFP4;
2. Swift 1.5 FP8 PLE + FP8 projections;
3. optional NVFP4 MTP experts.

Record:

- converter command;
- source checkpoint revision;
- artifact file/shard sizes;
- conversion report;
- binding/load-plan tests;
- host PLE mapping behavior.

Do not expect the full native model to fit 16 GB.

## Stage 8 — establish the 16 GB failure boundary

Attempt only enough native loading/execution to quantify why the large-GPU baseline does not fit.

Return:

- device weight allocation;
- fixed runtime allocations;
- PLE GPU allocation (should remain selected-row only);
- KV allocation;
- MTP allocation;
- available VRAM before expert residency work;
- exact first OOM or planner rejection point.

This evidence becomes the input to issue #2.

## Stage 9 — hand back before new architecture

Stop after a correct native baseline and memory inventory.

Do **not** start expert residency/CPU fallback until the reconciliation report is reviewed.

Expected handoff:

```text
RECONCILIATION_HEAD=
NINFER6000_HEAD=
COMMON_BASE=
MERGE_STATUS=
CONFLICTS=
BUILD=
CTEST=
RMS_EPS_SOURCE=
HYPER_ORACLE=
GPU_COMPONENTS=
ARTIFACT=
NATIVE_VRAM_BREAKDOWN=
PRODUCTION_SERVICE_RESTORED=
NEXT_BOUNDED_TASK=
```

The next bounded task should normally be issue #2 unless a prerequisite correctness blocker remains.
