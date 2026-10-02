# Qwen4Exp HyperConnection Executor Contract (Batch 3C2 — Milestone M1)

This document freezes the bounded HyperConnection **executor** contract that
later recurrence work (M2/M3) composes from the 3C1 primitives
(`include/ninfer/ops/hyper_connection.h`) and the 3C2a BF16 `linear()`
projections. It is normative for the M2/M3 kernels and their tests; the
equations it restates are the frozen `docs/flash-next/REFERENCE_FORMULAS.md`
sections 1 and 2.

M1 adds **no** kernels, **no** shared-runtime expansion, **no**
optimization, **no** formula changes, and **no** parent-model integration.
M1 is **API/oracle/docs only**: it delivers (1) the bounded executor
API/state contract header, (2) the independent FP64 numerical-oracle
scaffolding, and (3) a CPU-runnable contract/oracle/fixture test. M1
binds **no production weights, no production lifecycle class, and no GPU
kernels**: those are M2/M3 work.

---

## 1. Logical tensors, shapes, and layouts

All tensors are BF16.

Index layout (frozen, normative): activation/result tensors are **token-major, channel-last** - the element at channel `f` and token `t` is stored at flat offset `t*rows + f`: `x[t*K+k]`, `h[t*CH+j]`, `r[t*R+q]`, `inject[t*C+c]`. Projection weights use the
NInfer **output-row-major** physical layout `w[n*K+k]` (contiguous input
lanes per output row): `down[r*CH+j]`, `up[j*R+r]`, `inject[c*CH+j]`
(section 8 binds the exact per-weight offsets). A projection weight
`w` with logical shape `[N, K]` stores the element at output lane `n`
and input lane `k` at flat offset `n*K + k`: each output row contains
contiguous input lanes, exactly matching the production BF16 `linear()`
`Weight` payload (see section 8).

Canonical Flash-Next geometry: `H = 2560` (hidden), `C = 4` (streams),
`R = 320` (low-rank), `CH = C*H = 10240`.

| Tensor | Shape | Role |
| ------ | ----- | ---- |
| `embedding`   | `[H, T]`   | per-forward token embedding (initialization source) |
| `hyper`       | `[CH, T]`  | four-stream residual state; the in-place injection target |
| `mixed_input` | `[H, T]`   | block input produced by `read` |
| `prior_read`  | `[CH, T]`  | preserved prior `hyper` token, `prior[t*CH+j]`; the `inject` reference (state-owned) |
| `block`       | `[H, T]`   | block output produced by the caller `callback` |
| `final`       | `[H, T]`   | final mixer output (four-stream collapse) |
| `norm_weight` | `[CH]`      | grouped-RMSNorm weight (one value per expanded lane) |
| `mix_down`    | `[R, CH]`  | low-rank down-projection weight |
| `mix_up`      | `[CH, R]`  | low-rank up-projection + sigmoid weight |
| `inject`      | `[C, CH]`  | per-stream block-injection logit weight (block modules only) |

### 1.1 Normalization epsilon (fixed and normative)

The grouped-RMSNorm `+eps` term is **`1e-5`** (matching the reference
model's RMSNorm epsilon and the production 3C1 `hyper_connection_group_rmsnorm`
call site). This value is **fixed in the contract**: the oracle binds **no**
hidden epsilon constant; every `read` / `final_mixer` / `inject` oracle call
supplies the module's epsilon explicitly, and M1 fixes that module value at
`1e-5` for all modules. M2/M3 must pass this value at every production
call site; a different epsilon is a contract violation, not a tuning knob.

### 1.2 `norm_weight` ownership and module rebinding

`norm_weight` is **owned by the module, not by the state**:

- the **block** module (`HyperConnectionBlock`) owns `norm_weight`,
  `mix_down`, `mix_up`, and `inject`;
- the **final** module (`HyperConnectionFinalMixer`) owns its own
  `norm_weight`, `mix_down`, and `mix_up`;
- the state (`HyperConnectionExecutorState`) owns no module weights.

The complete owning module is supplied at each call. This mirrors the 3C1
primitive's call-bound `weight` argument without placing a duplicate binding
in state. Block and final norm weights may differ and must not be shared
implicitly.

---

## 2. `read` — mixed input + preserved prior (no block-inject logits)

`read(hyper, T) -> (mixed_input, prior_read)`:

```text
Hn          = GroupRMSNorm(hyper, block.norm_weight, eps=1e-5)
D           = SiLU( LinearDown(Hn, mix_down) / C )
up          = LinearUp(D, mix_up)              # pre-sigmoid
mixed_input[h,t] = (1/C) * sum_c  sigmoid(up[c,h,t]) * Hn[c,h,t]
prior_read[c,h,t] = hyper[c,h,t]               # read-only copy
```

- `mixed_input` and `prior_read` are **independent** storage; neither
  aliases `hyper` or any live scratch buffer.
- `read` **does not** compute block-inject logits. Those are produced
  exclusively by `inject`.
- `read` binds the **block** module's `norm_weight` and `mix_down` / `mix_up`.
- Requires `ready` and token-consistency (section 6); not-ready, reset, or
  token-mismatched calls are contract errors.

---

## 3. `callback` — caller block execution, same stream, no alias

`callback(mixed_input, T) -> block` (all on the **same** stream):

```text
block[h,t] = <caller block: attention/GDN/QSA/MoE>( mixed_input[h,t] )
```

- The block consumes `mixed_input` **only**; it must not read `hyper`
  directly.
- `block` is **independent** BF16 `[H,T]` storage with **no alias** to
  `hyper` state or to any live scratch buffer.
- `callback` is the only step the executor does not own; the caller
  supplies the `mixed -> block` functor.

---

## 4. `inject` — block-inject logits, `alpha` exactly once, in-place update

`inject(block, hyper, T)` (in-place four-stream update):

```text
Hn        = GroupRMSNorm(hyper, block.norm_weight, eps=1e-5)
raw[c,t]  = sum_j  inject[c*CH + j] * Hn[j,t]          # BlockInject(Hn)
alpha[c,t] = 2 * sigmoid( raw[c,t] / C )               # applied EXACTLY ONCE
next[c,h,t] = prior_read[c,h,t] + block[h,t] * alpha[c,t] # in-place result
```

- `alpha` is a function of the **normalized live** `hyper`, never of
  `block` or `mixed_input`; the additive base is the preserved
  state-owned `prior_read`.
- `block` is consumed **read-only** by `inject`; `block` is never aliased
  to `hyper`.
- **Zero** block-inject logits (`raw[c,t] = 0`) give `alpha[c,t] = 1`
  (`2*sigmoid(0)`); this is a frozen fixture the contract must satisfy.
- `inject` binds the **block** module's `norm_weight` and `inject` weight.
- Requires `ready` and token-consistency; `prior_read` authority per
  section 6.3.

---

## 5. `final_mixer` — collapse four streams, **no** block-inject

`final_mixer(hyper, T) -> final`:

```text
final[h,t] = (1/C) * sum_c  sigmoid(up[c,h,t]) * GroupRMSNorm(hyper, final.norm_weight, 1e-5)[c,h,t]
```

- Identical read path to `read`, but bound to the **final mixer's own**
  `norm_weight`, `mix_down`, `mix_up`; the final mixer has **no**
  `block_inject` and returns only the collapsed `mixed_input`, four
  streams -> 2560.
- Requires `ready` and token-consistency.

---

## 6. Per-forward state, readiness lifecycle, prior_read authority, reset

`HyperConnectionExecutorState` (see the header) is the **transient
per-forward** four-stream image, not a persistent sequence-state plane.

### 6.1 Readiness lifecycle (fixed)

- **Initialize** (exactly once per forward, deterministic): `hyper[c*H + h, t] =
  embedding[h, t]` (REFERENCE_FORMULAS.md s.2: `repeat(embedding, C)`).
  No learned stream transform occurs before layer 0. `ready` is set
  `true` only on success.
- **Not ready**: `read` / `inject` / `final_mixer` on a not-ready state
  (before `initialize`, or after `reset`) is a **contract error** and must
  be rejected, never a silent no-op or default-value path.
- **Token consistency**: each consuming call carries the forward's token
  extent `T`. A call whose `T` differs from the bound forward token count
  is a **contract error** (the state must be reset and re-initialized
  with the new `T`). There is **no** in-place token adaptation: a
  changed `T` is resolved only by `reset` + `initialize`. Any second
  `initialize` before reset, including one with the same `T`, is rejected.
- **Phase preconditions**: after successful `initialize`, the state is phase
  **I** (ready with no valid pending `prior_read`). `read` and `final_mixer`
  require phase I. A successful `read` captures the state-owned prior and
  transitions to phase **P**; `inject` requires phase P and consumes that
  snapshot. A phase-P `read` or `final_mixer` is rejected without overwriting
  or consuming the pending snapshot.
- **Reset**: clears `ready` (and the `hyper` binding) for the next
  forward; the prior forward's `hyper` is not carried forward silently.
  After `reset`, the state is not-ready again until `initialize`.

The M1 lifecycle model is **state-owned**: `HcState` in
`src/ops/hyper_connection_executor_oracle.h` exclusively owns `ready`,
`tokens`, `prior_read`, and `prior_read_valid`; the CPU test validates
those state-bound readiness, token-mismatch, changed-T re-initialization,
and post-reset rules. `HcExecutorLifecycle` is a stateless, non-owning
validation façade over the supplied `HcState`. M1 does **not** test the
production C++ class `HyperConnectionExecutorState` — its behavioral
validation remains M2 work.

### 6.2 Ownership and aliasing

`mixed`, `block`, and `prior_read` are each **independent** storage;
neither aliases `hyper` state nor any live scratch buffer. `hyper` is
consumed read-only by `read` / `final_mixer` and updated in place by
`inject`.

### 6.3 `prior_read` authority (fixed)

`prior_read` (the state-owned preserved prior `hyper` token) is the
**sole authority** for the `inject` reference:

- the state **owns** the `inject` reference; `read` snapshots live `hyper`
  into `prior_read` and marks that snapshot valid;
- `inject` computes logits from live `hyper` but uses `prior_read` as its
  sole additive base; a missing/invalid snapshot is rejected, and a
  successful inject consumes/invalidates the snapshot before the next block;
- an **externally supplied** `inject` reference is accepted **only** when
  it matches the valid state-owned `prior_read` exactly, even if live
  `hyper` has since diverged;
- a mismatching external reference is **contract-unsupported and must be
  rejected** (the oracle models this as `HcLifecycleError`; production
  `inject` simply has no external-reference parameter at all, which is
  the stronger form of the same rule).

### 6.4 Per-forward sequence

```text
1. (state)    initialize(embedding)  -> hyper      [CH, T]
2. (per blk)  read(hyper)            -> mixed_input [H, T]  (+ preserved prior)
3. (per blk)  callback(mixed_input)  -> block       [H, T]   (caller)
4. (per blk)  inject(block, hyper)   -> hyper       [CH, T]   (in-place)
   after the final layer's block:
5. (final)    final_mixer(hyper)     -> final       [H, T]
```

---

## 7. Independent numerical oracle and CPU fixtures (scope, exact)

`src/ops/hyper_connection_executor_oracle.h` re-derives the frozen
equations in **plain FP64**, independent of any production helper, kernel,
3C1 primitive, or `linear()` helper. The CPU test
`tests/ops/linear_add/test_hyper_connection_executor_contract.cpp` runs
the oracle on a small well-conditioned geometry (`H=8`, `C=4`, `R=5`,
`H != T`) on CPU only (no GPU), and validates — **and nothing more**:

- the `initialize` repeat mapping, hand-derived (s.2);
- grouped RMSNorm with nonuniform per-lane weights, hand-derived (s.1);
- the full `read` pipeline vs a hand-derived reference (distinct
  streams/tokens, identifiable projection weights) — this detects a
  missing `/C` scaling, a missing `sigmoid`, a `[rows,T]` transposition,
  and broken stream-group normalization;
- `inject` with distinct nonzero logits and identifiable per-stream
  weights, hand-derived per-stream `alpha` rule; zero-logit `alpha == 1`
  fixture;
- the **lifecycle gate as a contract model** (readiness, token
  consistency, changed-T re-initialization, post-reset rejection) — the
  model only, not the production class; plus the **lifecycle conformance**
  fixture (a second
ead in phase P is rejected with the pending
  prior_read snapshot byte-identical; inject consumes the preserved
  snapshot; same-state
eset + re-initialization at the same T clears
  stale prior/phase state);
- `prior_read` authority (state-owned anchor; mismatching external
  reference rejected);
- module-bound weight rebinding (block vs final `norm_weight`);
- `final_mixer` fed the **updated** (post-inject) state with the final
  module's weights — a **component fixture** of the read path, not a full
  decoder-block validation;
- exact compare that **rejects nonfinite operands**.

The fixtures use analytically independent, hand-derived references
(distinct stream/token values, nonzero logits, nonuniform norm weights,
identifiable weight entries) rather than self-comparison of two oracle
code paths.

---

## 8. Production weight binding (normative for M2/M3; not done in M1)

The three projection weights are production BF16 `linear()` weights.
For `linear(x=[K,T], w=[N,K], out=[N,T])`:

- activation `x[k,t]` is stored at `t*K + k` (one contiguous input vector
  per token);
- weight `w[n,k]` is stored at `n*K + k` (one contiguous input row per
  output);
- output `out[n,t]` is stored at `t*N + n`.

These are distinct physical layouts. The binding is:

| Module weight | Logical `linear` weight | Activation `x` (in) | `out` | Physical weight offset |
| ------------- | ------------------------ | -------------------- | ----- | ---------------------- |
| `mix_down`    | `[R, CH]`                | `[CH, T]` (Hn)      | `[R, T]`  | `r*CH + j` |
| `mix_up`      | `[CH, R]`                | `[R, T]` (D)        | `[CH, T]` | `j*R + r` |
| `inject`      | `[C, CH]`                | `[CH, T]` (Hn)      | `[C, T]`  | `c*CH + j` |

The oracle and fixtures use these exact nonsymmetric physical offsets.
The weight-table offsets are the NInfer **output-row-major** physical layout
`w[n*K+k]` (contiguous input lanes per output row); activation/result
tensors are token-major, channel-last (`t*rows + f`). No transposition
or re-layout is allowed when M2/M3 binds the production weights. M1
states and CPU-checks the mapping only; production `linear()` execution
remains M2 work.

---

## 9. M1 scope boundary (exact)

M1 is **API/oracle/docs only**. It does **not**:

- implement or bind any production kernel, `linear()` weight, or
  `Weight` payload (M2/M3);
- test the production `HyperConnectionExecutorState` C++ class or its
  GPU behavior (M2+);
- change `REFERENCE_FORMULAS.md`, the 3C1 `hyper_connection.h`
  primitives, or any 3C2a source;
- introduce M2 recurrence, parent-model integration, or GPU work.

**Next milestone (M2)**: implement and bind the production kernels for
`read` / `callback` / `inject` / `final_mixer` from the 3C1 primitives
plus the 3C2a `linear()` projections (per the section-8 binding), bind
the production `HyperConnectionExecutorState` (readiness lifecycle,
`norm_weight` module rebinding, `prior_read` authority), and validate
against this M1 oracle (CPU, then the two-phase GPU handoff per the
established policy).

### 9.5 Caller workspace exposure (normative for M2/M3)

The production `read` / `inject` / `final_mixer` calls each take a
**caller-owned, call-scoped** `WorkspaceArena&` immediately before the
`cudaStream_t`. The arena is **owned by the caller and scoped to the
forward**: the caller constructs, sizes, and resets it per forward and
supplies it to every consuming call; the executor performs **no internal
device allocation** (it does not own or allocate the workspace). Each
arena buffer **must not alias** `hyper`, `mixed_input`, `block`,
`prior_read`, or any `Weight` plane. The `initialize` and `reset`
methods allocate no scratch and therefore take no workspace.

The caller must provision the arena with **at least**
`hyper_connection_executor_workspace_capacity_bytes(C, H, R, T)` bytes
of the **maximum** bound; a smaller arena is a contract error and is
rejected at the call. `T` must equal the bound forward token extent.
BF16 is 2 bytes/element; the bound is the maximum of the two path
scratch needs (read/final vs. inject):

```text
read / final_mixer : 2 * T * (2 * C * H + R) bytes
inject             : 2 * T * (C * H + C)       bytes
workspace          : max(read/final, inject)   bytes
```

where `C` = `hc_count` (`streams`), `H` = `hidden`, `R` = `hc_lowrank`
(`lowrank`), `T` = `tokens`. The `read` / `final_mixer` term carries the
`2 * C * H` GroupRMSNorm-normalized expanded-stream working set plus the
`R`-wide low-rank projection scratch; the `inject` term is the `C * H`
normalized `hyper` working set plus the `C`-wide block-inject-logit
scratch. `C`, `H`, `R`, and `T` are the actual supplied geometry and
token extent, not the canonical 4/2560/320 constants.