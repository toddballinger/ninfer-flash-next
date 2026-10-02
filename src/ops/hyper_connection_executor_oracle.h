//
// ops/hyper_connection_executor_oracle.h — Independent FP64 numerical oracle
// for the Qwen4Exp HyperConnection executor (Batch 3C2, milestone M1).
//
// This header is the *independent* numerical reference for the bounded
// HyperConnection executor contract declared in
// `ninfer/ops/hyper_connection_executor.h`. It is derived **independently**
// from `docs/flash-next/REFERENCE_FORMULAS.md` (sections 1 and 2) and
// deliberately does **not** call any production helper, launcher, kernel,
// 3C1 primitive, or `linear()` helper. Every formula below is re-stated in
// plain FP64 arithmetic so it can serve as the oracle the later recurrence
// kernels (M2/M3) must match.
//
// The header is header-only (all bodies `inline`) so the CPU oracle/fixture
// test compiles and runs with no separate source or CUDA dependency.
//
// Storage convention: a logical `[rows, T]` matrix uses dimension-zero-fastest
// indexing; the element at feature `f` and token `t` is `matrix[t*rows + f]`.
// All matrices here are the *represented* (dequantized) float values, i.e. the
// FP32/FP64 value the production BF16 tensor carries.
//
//   H  = hidden_size (2560)
//   C  = hc_count    (4)
//   R  = hc_lowrank  (320)
//   CH = C*H         (10240)
//
// Production-weight mapping (M2/M3 binding) and the fixed norm epsilon are
// normative in `docs/flash-next/HYPERCONNECTION_EXECUTOR_CONTRACT.md`
// sections 2 and 7. The oracle itself binds neither weights to production
// `Weight` payloads nor any specific epsilon: both are supplied per call so
// the contract model, not the fixture, pins the semantics.
//
// Lifecycle: the per-forward `ready`/token gate is **state-owned**. `HcState`
// exclusively owns the lifecycle fields `ready`, `tokens`, `prior_read`, and
// `prior_read_valid`; the three forward phases U (uninitialized), I
// (initialized, no valid prior), and P (read, valid prior) are derived
// exclusively from those state-owned fields. `HcExecutorLifecycle` is a
// **stateless, non-owning** compatibility façade: it stores no `ready`/`tokens`
// of its own and performs all validation against the supplied `HcState`, so
// pre-existing callers remain source-compatible.
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <span>
#include <string>
#include <vector>

namespace ninfer::ops::detail {

/// Geometry for one HyperConnection module. `hc_count` is the stream count
/// (C); `hidden_size` is the per-stream width (H); `hc_lowrank` is the low-rank
/// width (R) used by the read/final-mixer low-rank path.
struct HcExecutorGeometry {
    std::int32_t hidden_size = 0;
    std::int32_t hc_count = 0;
    std::int32_t hc_lowrank = 0;

    std::int32_t hidden() const { return hidden_size; }
    std::int32_t streams() const { return hc_count; }
    std::int32_t lowrank() const { return hc_lowrank; }
    std::int32_t expanded() const { return hidden_size * hc_count; }
    std::size_t expanded_elems(std::int32_t tokens) const {
        return static_cast<std::size_t>(expanded()) * tokens;
    }
};

namespace hc_oracle {

[[nodiscard]] inline double fp64_sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

[[nodiscard]] inline double fp64_silu(double x) { return x * fp64_sigmoid(x); }

/// Thrown when a consuming lifecycle call is made in the wrong phase, with a
/// token extent different from the one the state was initialized with, or
/// when an operation is rejected by the frozen phase rules. Mirrors the
/// production contract's "error" wording.
class HcLifecycleError : public std::runtime_error {
public:
    explicit HcLifecycleError(std::string what)
        : std::runtime_error(std::move(what)) {}
};

/// The state-owned per-forward lifecycle phase. Derived solely from
/// `HcState`'s owned fields:
///   U (uninitialized): `ready` is false; no bound token extent; no valid
///       prior. The only state from which `initialize` is legal.
///   I (initialized):   `ready` is true, tokens are bound, and no valid prior
///       read snapshot exists. The legal state for `read` and `final_mixer`.
///   P (read):          `ready` is true, tokens are bound, and a valid prior
///       read snapshot exists. The only legal state for `inject_update`.
/// `HcState` is the *exclusive* owner of the fields that determine the phase;
/// the façade stores none of them.
enum class HcPhase {
    U,  ///< not initialized (no token, no snapshot)
    I,  ///< ready, no prior (bound T)
    P,  ///< ready, valid prior (read has run)
};

/// State owned by one forward. `HcState` is the **exclusive** owner of the
/// lifecycle fields: `ready`, `tokens`, `prior_read`, and
/// `prior_read_valid`. `prior_read` is captured by `read` and is the sole
/// additive/reference authority for the following `inject`. The phase is
/// derived exclusively from these owned fields.
struct HcState {
    std::vector<double> hyper;
    std::vector<double> prior_read;
    std::int32_t tokens = 0;
    bool ready = false;
    bool prior_read_valid = false;

    [[nodiscard]] HcPhase phase() const {
        if (!ready) return HcPhase::U;
        return prior_read_valid ? HcPhase::P : HcPhase::I;
    }
};

/// Stateless, non-owning compatibility façade for the per-forward lifecycle
/// gate. It **does not** store `ready`/`tokens`; every validation is performed
/// against the supplied `HcState`, which exclusively owns the lifecycle fields.
/// Kept so that pre-existing call sites (`life.initialize(...)`,
/// `life.reset(...)`, `read(life, ...)`, ...) remain source-compatible while
/// the authority for all lifecycle state moves to `HcState`.
struct HcExecutorLifecycle {
    // Intentionally stateless: the façade owns no `ready`/`tokens`.

    /// `U -> I` only; any other phase rejects. On accept, binds the token
    /// extent to `T`, validates the supplied hyper extent, and clears the
    /// prior to invalid. On reject, no state field is mutated
    /// (failure-atomic).
    void initialize(HcState& state, std::int32_t T,
                    const HcExecutorGeometry& geo) {
        if (state.phase() != HcPhase::U) {
            throw HcLifecycleError(
                "executor: initialize requires uninitialized state; "
                "re-initialization without reset is illegal");
        }
        if (T <= 0 ||
            state.hyper.size() != geo.expanded_elems(T)) {
            throw HcLifecycleError(
                "executor: initialize extent does not match supplied state");
        }
        state.ready = true;
        state.tokens = T;
        state.prior_read.clear();
        state.prior_read_valid = false;
    }

    /// `any -> U`, idempotent: clears all state-owned lifecycle fields.
    void reset(HcState& state) {
        state.tokens = 0;
        state.ready = false;
        state.hyper.clear();
        state.prior_read.clear();
        state.prior_read_valid = false;
    }

    /// Frozen consuming-gate checks, in the required order: (1) ready,
    /// (2) token equality, (3) phase validity for the operation, (4) the
    /// supplied state's own bound token. A later check may only throw after
    /// every earlier check has passed; the phase check therefore throws
    /// before any shape/argument validation, keeping a rejected operation
    /// failure-atomic.
    void check(const char* op, std::int32_t T, const HcState& state,
               const HcExecutorGeometry& geo) const {
        if (!state.ready) {
            throw HcLifecycleError(std::string("executor: ") + op +
                                   " called before initialize / after reset");
        }
        if (T != state.tokens) {
            throw HcLifecycleError(std::string("executor: ") + op +
                                  " token extent " + std::to_string(T) +
                                  " does not match state tokens " +
                                  std::to_string(state.tokens) +
                                  " (reset then initialize for a changed T)");
        }
        const HcPhase ph = state.phase();
        if (std::string(op) == "inject") {
            if (ph != HcPhase::P) {
                throw HcLifecycleError(std::string("executor: ") + op +
                                       " requires a valid prior (phase P); "
                                       "call read first");
            }
        } else if (std::string(op) == "final_mixer") {
            if (ph != HcPhase::I) {
                throw HcLifecycleError(std::string("executor: ") + op +
                                       " requires phase I (no valid prior); "
                                       "a read has already run");
            }
        }
        // read: phase-restricted -- legal only in I (ready, no valid prior).
        // A phase P (a valid, still-unconsumed read snapshot exists) is
        // rejected before any state mutation so a second read cannot
        // overwrite the unconsumed prior. The phase check therefore throws
        // prior to the shape/argument checks below, keeping a rejected read
        // failure-atomic.
        else if (std::string(op) == "read") {
            if (ph != HcPhase::I) {
                throw HcLifecycleError(std::string("executor: ") + op +
                                       " requires phase I (no valid prior); "
                                       "a read has already run (phase P)");
            }
        }
        if (state.tokens != T) {
            throw HcLifecycleError(std::string("executor: ") + op +
                                   " supplied state is not the initialized "
                                   "forward state");
        }
        if (state.hyper.size() != geo.expanded_elems(state.tokens)) {
            throw HcLifecycleError(std::string("executor: ") + op +
                                   " supplied state storage has wrong extent");
        }
    }
};

// ----------------------------------------------------------------------------
// Reference-formula primitives (FP64, independent)
// ----------------------------------------------------------------------------

/**
 * REFERENCE_FORMULAS.md s.2: `hyper[c*H + h, t] = embedding[h, t]`.
 * `hyper` has `CH*T` lanes; `embedding` has `H*T`.
 */
inline std::vector<double> initialize(std::span<const double> embedding,
                                      const HcExecutorGeometry& geo,
                                      std::int32_t tokens) {
    const std::int32_t H = geo.hidden();
    const std::int32_t C = geo.streams();
    const std::int32_t CH = H * C;
    std::vector<double> hyper(
        static_cast<std::size_t>(CH) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t c = 0; c < C; ++c) {
            for (std::int32_t h = 0; h < H; ++h) {
                hyper[static_cast<std::size_t>(t) * CH +
                      static_cast<std::size_t>(c) * H + h] =
                    embedding[static_cast<std::size_t>(t) * H + h];
            }
        }
    }
    return hyper;
}

/**
 * REFERENCE_FORMULAS.md s.1 "grouped RMSNorm": each `H`-wide stream is
 * normalized independently with per-lane `weight` (width `CH`). `eps` is the
 * caller-supplied normalization epsilon (contract default: 1e-5, see the
 * contract doc); it is never a hidden constant of the oracle.
 */
inline std::vector<double> group_rmsnorm(std::span<const double> input,
                                         std::span<const double> weight,
                                         double eps,
                                         const HcExecutorGeometry& geo,
                                         std::int32_t tokens) {
    const std::int32_t H = geo.hidden();
    const std::int32_t C = geo.streams();
    const std::int32_t CH = H * C;
    std::vector<double> out(
        static_cast<std::size_t>(CH) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t c = 0; c < C; ++c) {
            const std::size_t base =
                static_cast<std::size_t>(t) * CH +
                static_cast<std::size_t>(c) * H;
            double sum = 0.0;
            for (std::int32_t h = 0; h < H; ++h) {
                const double v =
                    input[base + static_cast<std::size_t>(h)];
                sum += v * v;
            }
            const double inverse = 1.0 / std::sqrt(
                sum / static_cast<double>(H) + eps);
            for (std::int32_t h = 0; h < H; ++h) {
                out[base + static_cast<std::size_t>(h)] =
                    input[base + static_cast<std::size_t>(h)] * inverse *
                    weight[static_cast<std::size_t>(c) * H + h];
            }
        }
    }
    return out;
}

/**
 * Low-rank down projection: `out[r,t] = sum_j down_w[r*CH+j]*x[j,t]` (physical offset `r*CH+j`).
 * `down_w` has logical shape `[R, CH]`, output-row-major with contiguous input lanes per output row (`r*CH+j`); `x` has `[CH,T]`;
 * `out` has `[R*T]` (R = geo.lowrank()). Maps to production
 * `linear(x=[CH,T], w=[R,CH]) -> out=[R,T]` (see contract doc s.7).
 */
inline std::vector<double> linear_down(std::span<const double> down_w,
                                       std::span<const double> x,
                                       const HcExecutorGeometry& geo,
                                       std::int32_t tokens) {
    const std::int32_t CH = geo.expanded();
    const std::int32_t R = geo.lowrank();
    std::vector<double> out(
        static_cast<std::size_t>(R) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t r = 0; r < R; ++r) {
            double acc = 0.0;
            for (std::int32_t j = 0; j < CH; ++j) {
                acc += down_w[static_cast<std::size_t>(r) * CH + j] *
                       x[static_cast<std::size_t>(t) * CH + j];
            }
            out[static_cast<std::size_t>(t) * R + r] = acc;
        }
    }
    return out;
}

/**
 * Low-rank up projection: `up[c,h,t] = sum_r up_w[j*R+r]*d[r,t]` (physical offset `j*R+r`).
 * `up_w` has logical shape `[CH, R]`, output-row-major with contiguous input lanes per output row (`j*R+r`); `d` has `[R,T]`; returns `[CH,T]`
 * (pre-sigmoid; the read / final-mixer applies `sigmoid` before the mix).
 * Maps to production `linear(x=[R,T], w=[CH,R]) -> out=[CH,T]`.
 */
inline std::vector<double> linear_up(std::span<const double> up_w,
                                     std::span<const double> d,
                                     const HcExecutorGeometry& geo,
                                     std::int32_t tokens) {
    const std::int32_t CH = geo.expanded();
    const std::int32_t R = geo.lowrank();
    std::vector<double> up(static_cast<std::size_t>(CH) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t j = 0; j < CH; ++j) {
            double acc = 0.0;
            for (std::int32_t r = 0; r < R; ++r) {
                acc += up_w[static_cast<std::size_t>(j) * R + r] *
                       d[static_cast<std::size_t>(t) * R + r];
            }
            up[static_cast<std::size_t>(t) * CH + j] = acc;
        }
    }
    return up;
}

/**
 * REFERENCE_FORMULAS.md s.1 "Read / mix": the block input is the
 * stream-weighted mean of the normalized streams:
 *   mixed[h,t] = (1/C) * sum_c( sigmoid(mix_logit[c,h,t]) * normalized[c,h,t] )
 */
inline std::vector<double> read_mix(std::span<const double> normalized,
                                    std::span<const double> mix_logit,
                                    const HcExecutorGeometry& geo,
                                    std::int32_t tokens) {
    const std::int32_t H = geo.hidden();
    const std::int32_t C = geo.streams();
    const std::int32_t CH = H * C;
    std::vector<double> mixed(static_cast<std::size_t>(H) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t h = 0; h < H; ++h) {
            double sum = 0.0;
            for (std::int32_t c = 0; c < C; ++c) {
                const std::size_t off =
                    static_cast<std::size_t>(t) * CH +
                    static_cast<std::size_t>(c) * H + h;
                sum += fp64_sigmoid(mix_logit[off]) *
                       normalized[off];
            }
            mixed[static_cast<std::size_t>(t) * H + h] =
                sum / static_cast<double>(C);
        }
    }
    return mixed;
}

/**
 * Raw block-inject logits (the `BlockInject` projection): `raw[c,t] =
 * sum_j inject_w[c*CH+j] * Hn[j,t]` (physical offset `c*CH+j`). `inject_w` has logical shape
 * `[C, CH]`; `Hn` has `[CH,T]`; returns `[C*T]` raw logits. Maps to
 * production `linear(x=[CH,T], w=[C,CH]) -> out=[C,T]`.
 */
inline std::vector<double> block_inject_logits(std::span<const double> hn,
                                               std::span<const double> inject_w,
                                               const HcExecutorGeometry& geo,
                                               std::int32_t tokens) {
    const std::int32_t CH = geo.expanded();
    const std::int32_t C = geo.streams();
    std::vector<double> raw(
        static_cast<std::size_t>(C) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t c = 0; c < C; ++c) {
            double acc = 0.0;
            for (std::int32_t j = 0; j < CH; ++j) {
                acc += inject_w[static_cast<std::size_t>(c) * CH + j] *
                       hn[static_cast<std::size_t>(t) * CH + j];
            }
            raw[static_cast<std::size_t>(t) * C + c] = acc;
        }
    }
    return raw;
}

/**
 * REFERENCE_FORMULAS.md s.1 "Injection weights":
 *   alpha[c,t] = 2 * sigmoid(inject_logit[c,t] / C)
 *   next[c,h,t] = hyper[c,h,t] + block[h,t] * alpha[c,t]
 */
inline std::vector<double> inject(std::span<const double> hyper,
                                  std::span<const double> block,
                                  std::span<const double> inject_logit,
                                  const HcExecutorGeometry& geo,
                                  std::int32_t tokens) {
    const std::int32_t H = geo.hidden();
    const std::int32_t C = geo.streams();
    const std::int32_t CH = H * C;
    std::vector<double> next(static_cast<std::size_t>(CH) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t c = 0; c < C; ++c) {
            const double raw =
                inject_logit[static_cast<std::size_t>(t) * C + c];
            const double alpha = 2.0 *
                                 fp64_sigmoid(
                                     raw / static_cast<double>(C));
            for (std::int32_t h = 0; h < H; ++h) {
                const std::size_t hyper_i =
                    static_cast<std::size_t>(t) * CH +
                    static_cast<std::size_t>(c) * H + h;
                const std::size_t block_i =
                    static_cast<std::size_t>(t) * H + h;
                next[hyper_i] =
                    hyper[hyper_i] +
                    block[block_i] * alpha;
            }
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// Bounded executor reference pipeline (contract, independent)
// ----------------------------------------------------------------------------

/**
 * REFERENCE_FORMULAS.md s.1 "Read / mix" pipeline. Produces the block
 * `mixed_input` (`H*T`) and preserves the prior four-stream token
 * (`CH*T`). **No** block-inject logits are computed here. The `norm_weight`
 * span is the *module's own* grouped-RMSNorm weight: `read` binds the block
 * module's weight; `final_mixer` binds the final mixer's weight. `eps` is
 * the caller-supplied normalization epsilon (never a hidden constant).
 *   Hn   = GroupRMSNorm(hyper, norm_weight, eps)
 *   D    = SiLU( LinearDown(Hn, mix_down) / C )
 *   up   = LinearUp(D, mix_up)
 *   mixed = mean_c( sigmoid(up) * Hn )
 */
struct HcRead {
    std::vector<double> mixed_input;  // [H, T]
    std::vector<double> prior_hyper;  // [CH, T]  (preserved prior token)
};

inline HcRead compute_read(const HcState& state,
                           std::span<const double> norm_weight, double eps,
                           std::span<const double> mix_down,
                           std::span<const double> mix_up,
                           const HcExecutorGeometry& geo) {
    const std::int32_t T = state.tokens;
    const std::int32_t C = geo.streams();
    HcRead out;
    out.prior_hyper = state.hyper;  // preserved prior (read-only copy)

    auto Hn = group_rmsnorm(std::span<const double>(state.hyper.data(),
                               state.hyper.size()),
                            std::span<const double>(norm_weight.data(),
                                                    norm_weight.size()),
                            eps, geo, T);
    auto d = linear_down(mix_down, std::span<const double>(Hn.data(),
                         Hn.size()), geo, T);
    // D = SiLU(LinearDown(Hn) / C).
    std::vector<double> d2(d.size());
    for (std::size_t i = 0; i < d.size(); ++i) {
        d2[i] = fp64_silu(d[i] / static_cast<double>(C));
    }
    auto up = linear_up(mix_up, std::span<const double>(d2.data(),
                         d2.size()), geo, T);
    out.mixed_input = read_mix(std::span<const double>(Hn.data(),
                           Hn.size()),
                               std::span<const double>(up.data(),
                                                       up.size()),
                               geo, T);
    return out;
}

inline HcRead read(HcState& state, std::span<const double> norm_weight,
                   double eps, std::span<const double> mix_down,
                   std::span<const double> mix_up,
                   const HcExecutorGeometry& geo) {
    HcRead out = compute_read(state, norm_weight, eps, mix_down, mix_up, geo);
    state.prior_read = out.prior_hyper;
    state.prior_read_valid = true;
    return out;
}

/**
 * REFERENCE_FORMULAS.md s.1 "Injection weights": compute raw block-inject
 * logits from `BlockInject(Hn)` (block module's `norm_weight`), apply
 * `alpha = 2*sigmoid(raw/C)` **exactly once**, and update the four-stream
 * state (by value).
 */
inline std::vector<double> inject_update(const HcState& state,
                                         std::span<const double> block,
                                         std::span<const double> norm_weight,
                                         double eps,
                                         std::span<const double> inject_logit_w,
                                         const HcExecutorGeometry& geo) {
    if (!state.prior_read_valid ||
        state.prior_read.size() != state.hyper.size()) {
        throw HcLifecycleError(
            "executor: inject requires a valid state-owned prior_read");
    }
    const std::int32_t T = state.tokens;
    auto Hn = group_rmsnorm(std::span<const double>(state.hyper.data(),
                               state.hyper.size()),
                            std::span<const double>(norm_weight.data(),
                                                    norm_weight.size()),
                            eps, geo, T);
    auto raw = block_inject_logits(std::span<const double>(Hn.data(),
                              Hn.size()),
                                   std::span<const double>(inject_logit_w.data(),
                                                           inject_logit_w.size()),
                                   geo, T);
    return inject(std::span<const double>(state.prior_read.data(),
                state.prior_read.size()),
                  std::span<const double>(block.data(), block.size()),
                  std::span<const double>(raw.data(), raw.size()), geo, T);
}

/**
 * REFERENCE_FORMULAS.md s.1 "final model-level mixer": identical read path to
 * `read` (bound to the **final mixer's own** `norm_weight`) but **no**
 * block-inject; collapses four streams to the final `H`-wide hidden state.
 */
inline std::vector<double> final_mixer(const HcState& state,
                                       std::span<const double> norm_weight,
                                       double eps,
                                       std::span<const double> mix_down,
                                       std::span<const double> mix_up,
                                       const HcExecutorGeometry& geo) {
    const HcRead rd =
        compute_read(state, norm_weight, eps, mix_down, mix_up, geo);
    return rd.mixed_input;
}

// ----------------------------------------------------------------------------
// Lifecycle-gated consuming entry points (state-owned contract model)
// ----------------------------------------------------------------------------

/// `read` with the frozen readiness/token/phase gate: not-ready, mismatched
/// `T`, or a mismatched supplied state throws `HcLifecycleError`. `read`
/// qualifies only in phase `I` (ready, no valid prior); phase `P` is rejected
/// so a second read cannot overwrite the unconsumed state-owned prior snapshot.
/// A successful read captures that authoritative prior on the I-to-P transition.
inline HcRead read(const HcExecutorLifecycle& life, std::int32_t T,
                   HcState& state,
                   std::span<const double> norm_weight, double eps,
                   std::span<const double> mix_down,
                   std::span<const double> mix_up,
                   const HcExecutorGeometry& geo) {
    life.check("read", T, state, geo);
    return read(state, norm_weight, eps, mix_down, mix_up, geo);
}

/// `inject` with the frozen gate: requires the **P** phase (a valid prior)
/// and the bound token extent, then commits `hyper = next` and, only on the
/// success path, consumes/invalidates the prior snapshot (`P -> I`). A
/// rejected inject leaves `state` unmutated (failure-atomic).
inline std::vector<double> inject_update(const HcExecutorLifecycle& life,
                                         std::int32_t T,
                                         HcState& state,
                                         std::span<const double> block,
                                         std::span<const double> norm_weight,
                                         double eps,
                                         std::span<const double> inject_logit_w,
                                         const HcExecutorGeometry& geo) {
    life.check("inject", T, state, geo);
    // Compute entirely first; `state` is mutated only after every formula has
    // succeeded, so a rejected or throwing inject is failure-atomic.
    std::vector<double> next =
        inject_update(static_cast<const HcState&>(state), block, norm_weight,
                      eps, inject_logit_w, geo);
    state.hyper = next;
    state.prior_read.clear();
    state.prior_read_valid = false;
    return next;
}

/// `final_mixer` with the frozen gate: requires the **I** phase; a **P**
/// phase (an already-run read, valid prior) is rejected per the contract.
inline std::vector<double> final_mixer(const HcExecutorLifecycle& life,
                                       std::int32_t T,
                                       const HcState& state,
                                       std::span<const double> norm_weight,
                                       double eps,
                                       std::span<const double> mix_down,
                                       std::span<const double> mix_up,
                                       const HcExecutorGeometry& geo) {
    life.check("final_mixer", T, state, geo);
    return final_mixer(state, norm_weight, eps, mix_down, mix_up, geo);
}

/**
 * `inject` with an *externally supplied* inject reference. The contract
 * fixes `prior_read` (the state-owned preserved prior `hyper` token) as the
 * sole authority for the `inject` reference: an external reference is
 * accepted only when it matches the state-owned prior exactly; a
 * mismatching external reference is contract-unsupported and throws
 * `HcLifecycleError`. All checks (gate, phase, and reference match) are
 * performed before any state mutation, keeping the operation failure-atomic;
 * a successful lifecycle-gated inject then consumes and invalidates the
 * snapshot after writing the next live `hyper` state.
 */
inline std::vector<double> inject_update_reference(
        const HcExecutorLifecycle& life, std::int32_t T,
        HcState& state,
        const std::vector<double>& external_ref,
        std::span<const double> block,
        std::span<const double> norm_weight, double eps,
        std::span<const double> inject_logit_w,
        const HcExecutorGeometry& geo) {
    life.check("inject", T, state, geo);
    if (!state.prior_read_valid) {
        throw HcLifecycleError(
            "externally supplied inject reference has no valid prior_read");
    }
    bool ref_matches = external_ref.size() == state.prior_read.size();
    for (std::size_t i = 0; ref_matches && i < state.prior_read.size(); ++i) {
        if (external_ref[i] != state.prior_read[i]) ref_matches = false;
    }
    if (!ref_matches) {
        throw HcLifecycleError(
            "externally supplied inject reference does not match the "
            "state-owned prior_read; external references are unsupported");
    }
    return inject_update(life, T, state, block, norm_weight, eps,
                         inject_logit_w, geo);
}

} // namespace hc_oracle

} // namespace ninfer::ops::detail