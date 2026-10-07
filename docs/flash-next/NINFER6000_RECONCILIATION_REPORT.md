# NInfer6000 reconciliation report

Date: 2026-10-08

Tracking issue: #9

## Executive conclusion

`toddballinger/ninfer-flash-next` and `lkarlslund/ninfer6000` share the exact common commit
`e360c4c06773ae08bddfda0c6c67ce834d4f584a` (`build: organize explicit cmake sources by component`).

From that common point, `ninfer6000/master` is currently 80 commits ahead. That makes this a
much easier reconciliation problem than integrating an unrelated runtime: the repositories share
the same NInfer codebase and the Flash-Next implementation was added after the common base.

The recommended implementation strategy is therefore:

1. preserve the current Batch 3C work as independent validation evidence;
2. reconcile against the native Flash-Next implementation in `ninfer6000`;
3. reuse the model/runtime/operator work that is already solved;
4. retain local code only where it is stronger, independently useful as an oracle, or required for
   RTX 5080 16 GB;
5. put new engineering effort into constrained-VRAM expert/KV placement rather than reimplementing
   the Flash-Next model.

Do **not** wholesale cherry-pick the 80-commit range. It includes generic upstream evolution,
frontend work, benchmarks and large model/runtime additions. Reconcile in bounded bands.

## Proven common base

| Repository | Relevant ref |
|---|---|
| `toddballinger/ninfer-flash-next` master | `e360c4c06773ae08bddfda0c6c67ce834d4f584a` |
| `lkarlslund/ninfer6000` history | same commit |
| `ninfer6000/master` versus common base | 80 commits ahead |

This is the key anti-reinvention fact: model execution can be ported/rebased from a directly related
NInfer lineage instead of reconstructed from external implementations.

## Current local state versus ninfer6000

The active local branch `flash-next/batch3c-hyperconnection-reference` has a bounded
HyperConnection implementation/oracle effort, but it does **not** yet contain the complete native
Flash-Next family. In particular, at the time of this review the active branch does not contain:

- `src/models/qwen3_8_flash_next/`;
- `src/models/qwen3_8_flash_next_125b_a6b/`;
- `include/ninfer/ops/flash_next_qsa.h`;
- `include/ninfer/ops/flash_next_moe.h`;
- `include/ninfer/ops/flash_next_ple.h`.

By contrast, `ninfer6000` has a complete family/runtime and a registered 125B-A6B package.

## Reconciliation map

| Area | Local state | ninfer6000 state | Recommendation |
|---|---|---|---|
| Model family/runtime | not integrated | complete `qwen3_8_flash_next` family/runtime | **Adopt as baseline architecture** |
| 125B-A6B package/load plan | not integrated | complete package/config/load plan | **Adopt** |
| QSA | reference/formula work only | native Op + runtime integration + compact-batch tuning | **Adopt then benchmark 5080 schedules** |
| HyperConnection | independent primitives/executor/oracle | integrated runtime Op + fused small-token path | **Use ninfer6000 runtime, preserve local oracle/tests** |
| PLE | formula/design only | file-mapped table, gather workers, decode/prefill integration | **Adopt architecture** |
| Sparse MoE | not integrated | Flash-Next-specific routing/shared-expert paths | **Adopt model semantics; retune schedules if needed** |
| MTP | generic NInfer MTP exists locally | Flash-Next MTP, adaptive drafts, lm-head draft, compact verify | **Port/reuse, do not redesign** |
| FP8 projections | generic format support exists | Flash-Next role bindings + kernels + converter | **Adopt bindings/roles; benchmark exact kernels** |
| NVFP4 experts | generic support exists | native routed-expert bank binding and W4A4 path | **Use as first expert representation** |
| INT8/FP8 KV | generic codecs exist | Flash-Next QSA integration and INT8 group-64 path | **Reuse integration; keep #3 for host placement** |
| Expert residency | not solved | assumes large HBM residency | **Project-specific #2** |
| Host-backed KV window | not solved for Flash-Next | not a 16 GB solution | **Project-specific #3** |
| Phase-specific VRAM reuse | not solved | large-GPU assumptions | **Project-specific #6** |

## Critical semantic discrepancy: RMSNorm epsilon

This reconciliation found one issue that should be treated as a blocker before extending the local
HyperConnection implementation.

The local Batch 3C contract currently freezes grouped RMSNorm epsilon at:

```text
1e-5
```

The native ninfer6000 125B-A6B config defines:

```cpp
static constexpr float rms_epsilon = 1.0e-6F;
```

and its integrated HyperConnection kernel also uses a `1.0e-6F` normalization term.

This discrepancy is material. It affects every HyperConnection read/inject/final-mixer numerical
oracle and therefore must be resolved against the exact source checkpoint/config before the local
contract is treated as normative.

### Required action

- verify the source checkpoint's `rms_norm_eps`;
- if the source says `1e-6`, update the local reference formula, contract and oracle fixtures
  together rather than patching only production code;
- re-run the HyperConnection numerical oracle after the correction;
- record the corrected source-of-truth in issue #8 and the contract document.

Until that is resolved, the local HyperConnection work should be treated as **valuable independent
scaffolding, not yet authoritative model semantics**.

## HyperConnection design reconciliation

The implementations are structurally different but potentially mathematically reconcilable.

### Local Batch 3C shape

The local executor models an explicit lifecycle:

```text
read -> caller block -> inject
```

It preserves `prior_read`, computes the block, then computes injection logits from normalized live
`hyper`, and writes:

```text
next = prior_read + block * alpha
```

### ninfer6000 shape

The integrated runtime forms the normalized mix and injection weights together before the block,
then later commits the pending block into the four-stream residual state. Its small-token path fuses
commit + grouped RMSNorm + Down/Up + gate mix.

Because `hyper` is not supposed to mutate between mix/injection-weight formation and branch commit,
the two forms may be numerically equivalent at the represented boundaries. Their lifecycle/capture
semantics differ, however, and should not be assumed equivalent without a test.

### Recommended use of local work

Do **not** discard the local executor/oracle. Convert it into an independent conformance harness:

1. correct the epsilon/source-model discrepancy;
2. feed identical BF16 tensors/weights to the local oracle and ninfer6000-equivalent path;
3. compare:
   - mixed block input;
   - per-stream injection logits/alpha;
   - committed four-stream state;
   - final mixer output;
4. include T=1, 2, 4, 8, 16 and representative prefill widths;
5. compare eager and CUDA Graph replay.

If it passes, the local work becomes high-value regression coverage while the production runtime
uses the already integrated ninfer6000 implementation.

## Upstream implementation bands

### Band A — model/runtime foundation

Primary source is the ninfer6000 integration around:

- `bc9ce9381009ee5c5e2845f4eb0bea18130d7601`
  — `feat(artifact): integrate upstream v3 with Flash-Next support`.

This commit is very broad and must **not** be cherry-picked wholesale into the active Batch 3C
branch. It spans generic upstream changes and the Flash-Next model/runtime.

Extract/reconcile these logical groups instead:

- `src/models/qwen3_8_flash_next/`;
- `src/models/qwen3_8_flash_next_125b_a6b/`;
- Flash-Next public Ops;
- artifact/package bindings;
- converter/inventory;
- model tests.

### Band B — correctness fixes that should be included before performance work

Relevant commits include:

- `8d89297a2eef3eb0dba508bcc0f82f60994100ca`
  — concurrent PLE decode-view fix;
- `a70d5ec0a69b3998280e17bafb7988ee5ca32fa4`
  — batched MTP predictor tensor flattening;
- `b74add31e0353390e2700d7bc7e17ac5d33158a4`
  — concurrent prefill KV row selection fix;
- `68409b45ffcabca693949ea9f552ef3a8ff8671b`
  — ordinary CUDA Graph replay correction;
- `742a38e70d93e9797ecbbd3f44d2ea8f9d6086e6`
  — adaptive MTP capture sizing fix.

These should be treated as part of the baseline, not optional optimizations.

### Band C — memory/quality representation

Relevant commits/features:

- Swift 1.5 artifact support:
  `2dcc5b13019eb3860a5aeb275b26f9999cc64740`;
- FP8 PLE conversion:
  `6565a2b2cbc2f3895583c215ae115a4a97397941`;
- weight-only FP8 attention/GDN projections:
  `871638683b295a68606dcaf9c0c4752b8555c651`;
- FP8 output/router/shared projections:
  `d2f427563017a7dde638ba689273b21639555a09`;
- FP8 HyperConnection projections:
  `bd7c391c89988c4ab0fa81bc922f258041186b0f`;
- FP8 MTP/final-mixer projections:
  `9b80b069fd0622eefdffd1d830f27a6849cd520f`;
- NVFP4 MTP experts:
  `8df59b0bf5f5e8b9991ea81e84bf88b01327f8f1`.

The last item is especially relevant to 16 GB because ninfer6000 reports approximately 3.4 GiB less
device weight memory for the MTP drafter with verified target output unchanged.

### Band D — speculative decode hot path

Relevant commits/features:

- adaptive MTP draft count:
  `532778a5a135bc2e9fa74ef09b63ad4a804b2e7e`;
- compact verify up to sixteen tokens:
  `2fa4c75696dbcd36d2b858de057696142766d154`;
- adaptive timing correction:
  `d96e6315f516b64ead25513c78b755251e216712`.

These should replace independent design work in issue #4. The 5080 project should benchmark and
adapt them, not create a second Flash-Next speculative path.

### Band E — decode kernel optimization

Examples:

- fused FP8 router/shared SwiGLU:
  `e6c1c3f25066908bce3350ab20db741619273593`;
- MoE gate/up concurrency:
  `42fd08d70ce660919f839cbb0bce505529aa06bf`;
- one-token routing/shared-down overlap:
  `647be0718286493c75985c4adb042b0ff75403aa`;
- fused FP8 MoE entry for 2–8-token verify:
  `97a033a2cbcf0d9f57132be566b9fac377204b93`;
- routing/shared-down overlap for 2–8 tokens:
  `d24f9c2c4eeac7146f07ec225feb2b61fc0a288b`;
- fused HyperConnection mix:
  `1cd1b91767ac0498643d3971013b9b95f88009c1`;
- QSA compact-batch optimizations:
  `a92df9a8694c2122a261b7d30f817b459fa23afc`,
  `0f31336e6282c7917bac2e82eb59a2460ec2d6a8`.

These are **benchmark-first** on RTX 5080. Preserve the architecture but do not assume exact PRO 6000
launch choices are optimal on the 5080.

## PLE implication for the 5080 target

ninfer6000 does not upload the entire PLE table to the GPU. It file-maps the table and gathers only
the selected rows for each token/chunk.

For the FP8 PLE profile the table is approximately 51.2 GB. A host with sufficient RAM can retain it
in the OS page cache. This is highly relevant to the 5080 target because it avoids consuming scarce
16 GB VRAM for the full PLE table.

Recommended first 5080 baseline:

- reuse the file-mapped PLE architecture;
- place the artifact on fast local storage;
- warm/measure page-cache behavior;
- pin/NUMA-place gather workers if host topology warrants it;
- measure cold versus warm PLE TTFT separately.

Do not invent a separate full-table host allocator until measurements show the page-cache design is
insufficient.

## Expert representation order

The default investigation order should now be:

1. native NVFP4 routed experts;
2. partial GPU residency + host fallback (#2);
3. quantify host DRAM/CPU/PCIe cost;
4. only if that fails the target, investigate IQ3/GSQ lower-bit host experts (#5).

This preserves quality first and avoids paying the complexity/quality cost of a second expert format
before it is demonstrated to be necessary.

## Safe implementation sequence

### Stage 0 — preserve

- do not mutate or clean the active dirty Batch 3C worktree;
- preserve issue #8 evidence;
- use a clean branch/worktree for reconciliation.

### Stage 1 — static reconciliation

- resolve RMS epsilon;
- map local HyperConnection oracle semantics against ninfer6000;
- identify the minimum model/runtime files and prerequisite generic upstream deltas.

### Stage 2 — baseline integration

In a clean reconciliation branch:

- add the native Flash-Next model family/package;
- add required public Ops and converter bindings;
- include known correctness fixes;
- get a clean CPU/build/test baseline before performance tuning.

### Stage 3 — GPU correctness

Run bounded GPU tests for:

- HyperConnection;
- QSA;
- PLE;
- Flash-Next MoE;
- MTP;
- eager/graph equivalence;
- target-model smoke generation.

### Stage 4 — 5080 memory adaptation

Only after native correctness is established:

- issue #2 expert residency/CPU fallback;
- issue #3 host-backed KV;
- issue #6 phase memory reuse.

### Stage 5 — performance tuning

Then benchmark/adapt ninfer6000's exact fused kernels and adaptive MTP choices for RTX 5080.

## Stop conditions for autonomous/static work

The following require local build/GPU/model evidence and should be handed to OpenClaw/local workers
or a human reviewer rather than guessed from GitHub:

- final choice of exact CUDA launch schedule;
- numerical acceptance of corrected HyperConnection contract;
- model conversion success against actual source checkpoints;
- VRAM fit and residency counts on RTX 5080;
- host expert throughput;
- page-cache/NUMA performance;
- end-to-end 128K generation;
- merge decision for production code.

Everything before those points can be advanced through documentation, issue reconciliation, bounded
source mapping and reviewable integration planning.
