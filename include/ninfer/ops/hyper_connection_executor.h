#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>

namespace ninfer::ops {

/**
 * @brief Exact device scratch capacity the caller must provision in the
 *        caller-owned `WorkspaceArena` for one `streams` x `hidden` x
 *        `lowrank` x `tokens` forward, for the `read` / `final_mixer` and
 *        `inject` paths (all scratch is BF16, 2 bytes/element). The `inject`
 *        path needs the per-stream `+streams` block-inject-logit scratch in
 *        addition to the read path; the bound is the maximum of the two.
 * @param[in] streams Stream count `C` (`hc_count`).
 * @param[in] hidden  Hidden width `H`.
 * @param[in] lowrank Low-rank width `R`.
 * @param[in] tokens  Forward token extent `T`.
 * @return Maximum BF16 scratch bytes (2 per element) required in the
 *         caller-owned workspace for the forward.
 */
[[nodiscard]] std::size_t hyper_connection_executor_workspace_capacity_bytes(
    std::int32_t streams, std::int32_t hidden, std::int32_t lowrank,
    std::int32_t tokens);

/**
 * @brief Bounded, state-ful HyperConnection executor contract for Qwen4Exp
 *        Flash-Next decoder layers (Batch 3C2, milestone M1).
 *
 * @details
 * This header freezes the bounded executor **API and state contract** only.
 * It names the exact logical tensors, their shapes/layouts, ownership and
 * aliasing rules, the per-forward state/token/readiness lifecycle, the
 * `prior_read` authority rule, and the reset behaviour that the later
 * recurrence work (M2/M3) will compose from the existing 3C1 primitives
 * (`ninfer/ops/hyper_connection.h`) and the 3C2a
 * BF16 `linear()` projections.
 *
 * It deliberately **adds no kernels, no per-module workspace arena, no
 * optimization, no formula changes, and no parent-model integration**.
 * It is a pure contract header plus the independent numerical oracle that
 * the oracle tests evaluate (`tests/ops/linear_add/`).
 *
 * The normative equations are `docs/flash-next/REFERENCE_FORMULAS.md`
 * (sections 1 and 2). This header restates their tensor contracts in the
 * production (NInfer) logical-storage convention so the later kernels can be
 * written, bound, and tested against one unambiguous interface.
 *
 * M1 is **API/oracle/docs only**: it binds no production weights, no
 * production lifecycle class, and no GPU kernels; those are M2/M3 work.
 *
 * --- Logical storage convention (NInfer) -------------------------------
 * A logical matrix `tensor` with `ne[0] = rows` and `ne[1] = T` stores
 * dimension zero fastest: the element at feature `f` and token `t` lives at
 * flat offset `t * rows + f`. All shapes below are stated as `[rows, T]`
 * with `T` the token extent.
 *
 *   hidden          -> `H`            (2560)
 *   streams         -> `C` = `hc_count` (4)
 *   expanded width  -> `C * H`        (10240)
 *   low-rank width  -> `R` = `hc_lowrank` (320)
 *
 * Canonical Flash-Next geometry: `H = 2560`, `C = 4`, `R = 320`. The
 * executor validates its supplied geometry rather than hard-coding it, but
 * the contract below is fixed for the four-stream model.
 *
 * --- Fixed normalization epsilon (normative) -------------------------
 * The grouped-RMSNorm `+eps` term is **`1e-5`** (matching the reference
 * model RMSNorm epsilon and the 3C1 `hyper_connection_group_rmsnorm`
 * call site) for **every** module and call (`read`, `inject`,
 * `final_mixer`). This is a contract-fixed value, not a tuning knob:
 * the consuming ops take the module epsilon as a call argument and the
 * caller must pass `1e-5`. See `HYPERCONNECTION_EXECUTOR_CONTRACT.md`
 * sections 1.1 and 7.
 *
 * --- Canonical per-forward executor sequence ----------------------------
 * One decoder block consumes the four-stream `hyper` state through exactly:
 *
 *     1. (state)   initialize(embedding)      -> hyper  [10240, T]
 *     2. (per blk) read(hyper, mix_w)         -> mixed  [H, T]
 *     3. (per blk) callback(mixed)            -> block  [H, T]   (caller)
 *     4. (per blk) inject(block, inject_w)    -> hyper  [10240, T]
 *
 * After the final layer's block, `final_mixer(hyper)` collapses four
 * streams to the 2560-wide model hidden state:
 *
 *     final_mixer(hyper)  -> final  [H, T]
 */

/**
 * A block-injecting HyperConnection module: per-block read, caller block
 * execution, and in-place four-stream injection. The block owns
 * `norm_weight`, `mix_down`, `mix_up`, and `inject`.
 *
 * Module weights are **call-bound and module-owned**: the block module is
 * supplied to `read`/`inject` and the final module is supplied to
 * `final_mixer`. No module weight lives in executor state; block and final
 * `norm_weight` may legitimately differ and must not be silently shared.
 */
struct HyperConnectionBlock {
    /// Grouped-RMSNorm weight, logical `[C*H]`.
    Weight norm_weight;
    /// Low-rank down-projection weight for the read path, logical `[R, C*H]`.
    Weight mix_down;
    /// Low-rank up-projection + sigmoid weight for the read path, `[C*H, R]`.
    Weight mix_up;
    /// Per-stream block-injection logit weight, logical `[C, C*H]`.
    Weight inject;
};

/// The final, model-level HyperConnection mixer. It has **no** block-inject
/// weight and collapses four streams to one 2560-wide hidden state.
struct HyperConnectionFinalMixer {
    /// Grouped-RMSNorm weight, logical `[C*H]`; may differ from the block.
    Weight norm_weight;
    /// Low-rank down-projection weight for the read path, `[R, C*H]`.
    Weight mix_down;
    /// Low-rank up-projection + sigmoid weight for the read path, `[C*H, R]`.
    Weight mix_up;
    /// Absence of `inject` is normative: the final mixer performs no
    /// block-injection weighting.
    bool has_block_inject = false; ///< Must be false for the final mixer.
};

/**
 * @brief Per-forward four-stream HyperConnection state, ownership, and
 *        readiness lifecycle for the Qwen4Exp executor.
 *
 * @details
 * `HyperConnectionExecutorState` is **not** a persistent sequence state
 * plane; it is the transient, per-forward image of the four-stream
 * HyperConnection. Its lifecycle is fixed:
 *
 *   - **Initialize** once per forward (deterministically), producing the
 *     `hyper` four-stream tensor. `ready` is false until this completes.
 *   - **Never silently reused** across forwards: any `read`/`inject`/
 *     `final_mixer` call on a not-ready (or reset) state is a contract
 *     **error** and must be rejected — never a silent no-op or default
 *     path.
 *   - **Token consistency**: each consuming call carries the forward token
 *     extent `T`; a call whose `T` differs from the bound forward token
 *     count is a contract error. A changed token count is resolved
 *     **only** by `reset()` + `initialize()` (no in-place token
 *     adaptation).
 *   - **Reset** clears `hyper` and `ready`; the next `initialize`
 *     re-establishes a deterministic fresh state.
 *
 * The M1 CPU oracle models this lifecycle (`HcExecutorLifecycle` in
 * `src/ops/hyper_connection_executor_oracle.h`). M1 does **not** test this
 * production class; production lifecycle validation is M2 work.
 *
 * Ownership / alias rules (all tensors are BF16):
 *
 *   - `hyper`       `[C*H, T]`   four-stream residual state (the in-place
 *                                injection target). `read` consumes it
 *                                read-only; `inject` updates it in place.
 *   - `mixed`       `[H, T]`     block-input produced by `read`; **independent
 *                                storage, no alias** to `hyper` or to any
 *                                live scratch buffer.
 *   - `block`       `[H, T]`     block-output produced by `callback`;
 *                                independent storage, no alias to `hyper`.
 *   - `prior_read`  `[C*H, T]`   preserved prior `hyper` stream token
 *                                (read-only copy of `hyper` at read time).
 *                                **Sole authority for the `inject`
 *                                reference**: an externally supplied
 *                                reference is contract-unsupported unless it
 *                                matches `prior_read` exactly (contract doc
 *                                s.6.3). `read` never computes
 *                                block-inject logits.
 *
 * `ready` gates every consuming call: `false` until `initialize` succeeds,
 * `true` while the forward is active, cleared by `reset`.
 */
class HyperConnectionExecutorState {
public:
    /// Four-stream residual state, logical `[C*H, T]`.
    Tensor hyper;

    /// Current block-input, logical `[H, T]`. Independent storage; never
    /// aliases `hyper` or a live scratch buffer.
    Tensor mixed_input;

    /// Preserved prior `hyper` stream token, logical `[C*H, T]`; **sole
    /// authority for the `inject` reference** (read-only copy of `hyper` at
    /// read time). An externally supplied reference is contract-unsupported
    /// unless it matches `prior_read` exactly (contract doc s.6.3).
    Tensor prior_read;

    /// Grouped RMSNorm and projection weights are call-bound and
    /// module-owned; none is retained in this state.

    /// True once `initialize` has succeeded for the current forward;
    /// cleared by `reset()`.
    bool ready = false;

    /// Token extent bound by the successful initialize.
    std::int32_t tokens = 0;

    /// True only after `read` captures `prior_read` for the pending inject;
    /// a successful `inject` consumes and clears it.
    bool prior_read_valid = false;

    std::int32_t hc_count = 0;
    std::int32_t hidden_size = 0;
    std::int32_t hc_lowrank = 0;

    HyperConnectionExecutorState() = default;

    /// @par Initialize (per forward, deterministic)
    /// `hyper[c*H + h, t] = embedding[h, t]` (REFERENCE_FORMULAS.md s.2:
    /// `repeat(embedding, hc_count)`). No learned stream transform occurs
    /// before layer 0. On success `mixed_input`, `prior_read` and `ready`
    /// are (re)bound to the forward token extent; the bound `T` anchors the
    /// token-consistency gate of every consuming call. A **changed** token
    /// count is resolved only by `reset()` + `initialize()` — never by
    /// re-initializing an active forward to fix a mismatch. Any second
    /// initialize before reset, even with the same T, is a contract error.
    /// @param[in]  embedding Token embedding, BF16 `[H, T]`.
    /// @param[in]  T Token extent (`embedding.ne[1]`).
    /// @param[out] stream Executed on `stream`.
    void initialize(const Tensor& embedding, std::int32_t T,
                    cudaStream_t stream);

    /// @par Read (mixed_input, no block-inject logits)
    /// Produces `mixed_input[h, t] = mean_c(sigmoid(mix[h,t]) *
    /// GroupRMSNorm(hyper, norm_weight, eps)[c,h,t])` (REFERENCE_FORMULAS.md
    /// s.1 "Read / mix") and preserves `prior_read` = prior `hyper`.
    /// **Does not** compute block-inject logits: those are produced
    /// exclusively by `inject`. `mixed_input` and `prior_read` are
    /// independent of `hyper`.
    /// Requires `ready`, `T` to equal the bound forward token extent, and
    /// phase I only (no valid pending `prior_read` snapshot). A phase-P
    /// `read` is a contract error so it cannot overwrite the prior snapshot
    /// that a subsequent `inject` must consume.
    /// `norm_weight` is the **block** module's grouped-RMSNorm weight
    /// (call-bound; `eps` must be the contract-fixed `1e-5`).
    /// @param[in]  hyper `hyper` state, BF16 `[C*H, T]`.
    /// @param[in]  T Token extent (must equal the initialized forward T).
    /// @param[in]  module Block-owned norm/projection weights.
    /// @param[in]  eps Normalization epsilon (contract-fixed `1e-5`).
    /// @param[out] mixed  Block input, BF16 `[H, T]`.
    /// @param[out] prior  Preserved prior stream token, BF16 `[C*H, T]`.
    /// @param[in,out] workspace Caller-owned, call-scoped transient arena;
    ///                          provision capacity per
    ///                          hyper_connection_executor_workspace_capacity_bytes().
    /// @param[out] stream Executed on `stream`.
    void read(const Tensor& hyper, std::int32_t T,
              const HyperConnectionBlock& module, float eps,
              Tensor& mixed, Tensor& prior, WorkspaceArena& workspace,
              cudaStream_t stream);

    /// @par Callback (caller block execution)
    /// Runs the caller-supplied block (attention/GDN/QSA/MoE) on `mixed_input`
    /// on the **same stream** and produces `block` `[H, T]`. `block` is
    /// independent storage with **no alias** to `hyper` state or to any live
    /// scratch buffer. `callback` is the caller's responsibility: it is the
    /// only step the executor does not own, and it must not read `hyper`
    /// directly (it consumes `mixed_input` only).
    /// @param[in]   mixed   Block input, BF16 `[H, T]`.
    /// @param[out]  block   Block output, BF16 `[H, T]`.
    /// @param[out]  stream Executed on `stream` (same as `read`).
    /// @param[in]   run     Caller-provided `mixed -> block` functor.
    void callback(const Tensor& mixed, Tensor& block, cudaStream_t stream,
                  void (*run)(const Tensor& mixed, Tensor& block,
                              cudaStream_t stream));

    /// @par Inject (block-inject logits + in-place four-stream update)
    /// Computes `alpha[c,t] = 2 * sigmoid(inject_logits[c,t] / C)`, where
    /// `inject_logits = BlockInject(GroupRMSNorm(hyper, norm_weight, eps))`;
    /// applies `alpha` **exactly once**; the block-inject logits are a
    /// function of the **live, normalized** `hyper` (`alpha` never depends
    /// on `block` or `mixed_input`).
    /// The in-place update base is the **state-owned preserved
    /// `prior_read`** (the `read`-time snapshot), never the live `hyper`:
    ///     `hyper[c,h,t] = prior_read[c,h,t] + block[h,t] * alpha[c,t]`
    /// (REFERENCE_FORMULAS.md s.1 "Injection weights"). This is the
    /// normative `prior_read` rule: `read` captures the pre-block stream
    /// into `prior_read`; `inject` then writes the updated four-stream
    /// `hyper` **from** `prior_read + block * alpha` (a substitution, not a
    /// live `hyper +=`). A **missing** `prior_read` snapshot is a contract
    /// error; an **externally supplied** reference is contract-unsupported
    /// unless it matches the state-owned `prior_read` exactly (contract doc
    /// s.6.3).
    /// Requires `ready`, token-consistency, and phase P only (a valid
    /// state-owned `prior_read` snapshot); successful `inject` consumes it.
    /// `norm_weight` is the block module's weight (call-bound).
    /// @param[in]    block        Block output, BF16 `[H, T]`.
    /// @param[in,out] hyper       Four-stream state, BF16 `[C*H, T]`; updated in
    ///                            place (in-place injection target).
    /// @param[in]    T Token extent (must equal the initialized forward T).
    /// @param[in]    module Block-owned norm/projection weights.
    /// @param[in]    eps Normalization epsilon (contract-fixed `1e-5`).
    /// @param[in,out] workspace Caller-owned, call-scoped transient arena;
    ///                          provision capacity per
    ///                          hyper_connection_executor_workspace_capacity_bytes().
    /// @param[out]   stream Executed on `stream`.
    void inject(const Tensor& block, Tensor& hyper, std::int32_t T,
                const HyperConnectionBlock& module, float eps,
                WorkspaceArena& workspace, cudaStream_t stream);

    /// @par Final mixer (collapse four streams, no block-inject)
    /// `final[h, t] = mean_c(sigmoid(mix[h,t]) *
    /// GroupRMSNorm(hyper, norm_weight, eps)[c,h,t])`. The final mixer has
    /// **no** `block_inject`; it returns only the collapsed `mixed_input`,
    /// four streams -> 2560 (REFERENCE_FORMULAS.md s.1 "For the final
    /// model-level HyperConnection mixer").
    /// Requires `ready`, token-consistency, and phase I only (no pending
    /// unconsumed `prior_read` snapshot); phase P is a contract error.
    /// `norm_weight` is the **final mixer's** weight (call-bound; may differ
    /// from the block module's weight).
    /// @param[in]  hyper   Four-stream state, BF16 `[C*H, T]`.
    /// @param[in]  T Token extent (must equal the initialized forward T).
    /// @param[in]  module Final-mixer-owned norm/projection weights.
    /// @param[in]  eps Normalization epsilon (contract-fixed `1e-5`).
    /// @param[out] final   Final hidden state, BF16 `[H, T]`.
    /// @param[in,out] workspace Caller-owned, call-scoped transient arena;
    ///                          provision capacity per
    ///                          hyper_connection_executor_workspace_capacity_bytes().
    /// @param[out] stream Executed on `stream`.
    void final_mixer(const Tensor& hyper, std::int32_t T,
                     const HyperConnectionFinalMixer& module, float eps,
                     Tensor& final, WorkspaceArena& workspace, cudaStream_t stream);

    /// @par Reset (per forward, deterministic)
    /// Clears `hyper`, `prior_read` validity, `tokens`, and `ready` for
    /// the next forward. After `reset` the
    /// state is not-ready again; it must be re-`initialize`d before any
    /// consuming call. The previous forward's `hyper` is not silently
    /// carried forward. A **changed** token count is resolved only through
    /// `reset()` + `initialize()` (no in-place adaptation).
    void reset();
};

} // namespace ninfer::ops