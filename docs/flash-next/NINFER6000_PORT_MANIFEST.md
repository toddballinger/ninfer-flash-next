# NInfer6000 port manifest

Tracking: #9

This file defines the **lowest-friction integration method** for bringing the native Flash-Next
implementation into this repository without disturbing the active Batch 3C work.

## Preferred Git strategy

Because both repositories share commit
`e360c4c06773ae08bddfda0c6c67ce834d4f584a`, do **not** manually cherry-pick 80 upstream commits.

Use a clean integration branch/worktree and merge the related history:

```text
toddballinger/ninfer-flash-next master
        \
         common base e360c4c0
        /
lkarlslund/ninfer6000 master
```

Recommended local execution sequence:

```bash
git remote add ninfer6000 https://github.com/lkarlslund/ninfer6000.git
git fetch ninfer6000

# Start from the current local master, not the dirty Batch 3C branch.
git switch master
git pull --ff-only
git switch -c reconcile/ninfer6000-core

# Merge the related history. Do not squash at this stage; preserve provenance.
git merge --no-ff ninfer6000/master
```

If the remote already exists, update its URL rather than creating a duplicate.

### Why merge the lineage instead of cherry-picking

- the histories have a proven common base;
- the Flash-Next model/runtime depends on generic NInfer changes made after that base;
- the broad v3/Flash-Next integration commit touches many shared interfaces;
- cherry-picking only model files risks missing artifact/runtime/frontend/CMake prerequisites;
- retaining upstream commits preserves provenance and makes future watches/diffs much easier.

This merge belongs on a **reconciliation branch only** until the build, tests, model conversion and
RTX 5080 gates pass.

## Do not merge into the active Batch 3C branch

The active `flash-next/batch3c-hyperconnection-reference` work has independent oracle/contract value
and may contain local WIP. Keep it untouched.

After the clean ninfer6000 integration builds, reconcile local Batch 3C work by purpose:

### Preserve as validation assets

Prefer porting/reworking:

- independent HyperConnection FP64 oracle;
- lifecycle/shape fixtures that still represent source-model semantics;
- 5080-specific numerical tests;
- useful contract documentation after correcting stale semantics.

### Do not automatically preserve as production implementation

Do not automatically carry forward:

- duplicate production HyperConnection launchers;
- duplicate model lifecycle abstractions;
- duplicate projection paths already implemented natively upstream.

Production duplication should survive only when it has stronger measured correctness/performance.

## Integration phases

### Phase 1 — source merge and compile ownership

Goal: get the merged source tree to configure and compile without a model artifact.

Required checks:

- CMake configure;
- Release build;
- unit tests not requiring a real model;
- no duplicate source ownership;
- no duplicate symbol/linker errors;
- docs links valid.

Likely generic dependencies introduced since the common base include frontend/Jinja and runtime
changes. Resolve them as upstream dependencies rather than deleting Flash-Next source until the
actual dependency is understood.

### Phase 2 — semantic blockers

Before declaring model correctness:

1. resolve the local `1e-5` versus native `1e-6` RMS epsilon discrepancy;
2. compare local HyperConnection oracle against native represented boundaries;
3. check PLE/QSA/GDN state semantics against source checkpoint;
4. verify model/package fixed dimensions.

### Phase 3 — artifact conversion

Use one supported native source profile first.

Recommended order:

1. RadixArk NVFP4, because its PLE is already FP8 and the artifact path is simpler;
2. Swift 1.5 with FP8 PLE + FP8 projections;
3. optional NVFP4 MTP experts after the baseline artifact works.

Do not start by inventing a 16 GB-specific artifact format.

### Phase 4 — native GPU smoke

Before constrained-memory work, demonstrate native correctness on whatever subset can run:

- Op-level HyperConnection;
- QSA;
- PLE;
- sparse MoE;
- MTP;
- graph replay/eager equivalence.

The full native artifact will not fit the 16 GB target; this phase can use bounded Op tests and
component fixtures to establish execution correctness.

### Phase 5 — constrained-VRAM architecture

Now implement the project-specific layer:

- #2 expert residency / CPU fallback;
- #3 host-backed KV;
- #6 phase memory reuse.

The first expert representation should be native NVFP4.

## Upstream file groups

These are ownership groups, not a promise that copying these directories alone will compile.

### Model family

```text
src/models/qwen3_8_flash_next/
src/models/qwen3_8_flash_next_125b_a6b/
```

The family CMake currently owns frontend, PLE table, prefix identity, state, round state, visual
scatter and vision control. The 125B package owns package/load-plan/model/variant.

### Public Flash-Next Ops

```text
include/ninfer/ops/flash_next_gdn.h
include/ninfer/ops/flash_next_moe.h
include/ninfer/ops/flash_next_ple.h
include/ninfer/ops/flash_next_qsa.h
include/ninfer/ops/hyperconnection.h
```

### Core launch/runtime Ops

Key implementation locations include:

```text
src/ops/launcher/hyperconnection.cu
src/ops/launcher/flash_next_ple.cu
src/ops/softmax_attention/qsa/flash_next_qsa.cu
src/ops/sparse_moe/flash_next/
src/ops/linear_attention/flash_next_gdn.cu
src/ops/linear/bf16/flash_next/
src/ops/linear/fp8/shapes/flash_next.cu
src/ops/linear/fp8/flash_next_launch.h
```

### Artifact/converter

```text
tools/convert/qwen3_8_flash_next_125b_a6b/
docs/maintainer/qwen3.8-flash-next-125b-a6b-artifact.md
docs/maintainer/qwen3.8-flash-next-125b-a6b-model.md
```

The load plan and converter must stay aligned. Do not port one without its binding/inventory tests.

## Known baseline correctness commits

Ensure the integrated branch contains the effects of:

| Commit | Reason |
|---|---|
| `8d89297a...` | concurrent PLE decode views |
| `a70d5ec0...` | batched MTP predictor tensor layout |
| `b74add31...` | concurrent prefill KV row selection |
| `68409b45...` | ordinary CUDA Graph replay |
| `742a38e7...` | adaptive MTP capture sizing |

These are not optional benchmark experiments.

## First optional performance set after correctness

Only after native correctness:

| Commit / area | 5080 treatment |
|---|---|
| `1cd1b917...` HyperConnection <=16 fusion | benchmark |
| `e6c1c3f2...` one-token FP8 router/shared fusion | benchmark |
| `97a033a2...` FP8 MoE verify fusion | benchmark |
| `d24f9c2c...` routing/shared overlap | benchmark |
| `a92df9a8...`, `0f31336e...` QSA compact verify | benchmark |
| `532778a5...` adaptive MTP | adopt policy, tune |
| `2fa4c756...` compact verify <=16 | adopt architecture, tune |
| `8df59b0b...` NVFP4 MTP experts | high-priority memory benchmark |

## Handoff output expected from OpenClaw/local execution

The implementation handoff should return:

- merge commit SHA on `reconcile/ninfer6000-core`;
- exact conflict list and resolutions;
- build/CTest summary;
- source-checkpoint epsilon evidence;
- HyperConnection oracle comparison;
- component GPU test results;
- artifact conversion status;
- VRAM failure point for native full-model load on RTX 5080;
- resulting next bounded task (#2 or prerequisite fix).

Do not merge the reconciliation branch into master merely because the source merge compiles.
