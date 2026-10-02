#include "ninfer/ops/hyper_connection_executor.h"

#include "ops/launcher/hyper_connection.h"
#include "core/arena.h"
#include "ops/launcher/hyper_connection_executor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

/// Throw `std::invalid_argument` when `cond` is true, with `what`. The
/// frozen lifecycle model (M1 oracle `HcExecutorLifecycle`, contract s.6.1)
/// expresses every per-forward rule as a *contract error* that is rejected,
/// never silently defaulted. `std::invalid_argument` is the repository's
/// existing convention for argument/contract validation (see
/// `src/ops/wrapper/hyper_connection.cpp`).
void reject_lifecycle(bool cond, std::string what) {
    if (cond) {
        throw std::invalid_argument(std::move(what));
    }
}

/// Expanded four-stream width `C * H`. Validated to stay inside the `Tensor`
/// dimension domain, matching the 3C1 wrapper's `expanded_width` convention.
std::int32_t expanded_width(std::int32_t hidden, std::int32_t streams) {
    const auto expanded = static_cast<std::int64_t>(hidden) * streams;
    if (expanded > 2147483647L) {  // std::numeric_limits<int32_t>::max()
        throw std::overflow_error(
            "hyper_connection_executor: expanded width exceeds the Tensor "
            "dimension domain");
    }
    return static_cast<std::int32_t>(expanded);
}

/// Frozen consuming-gate checks, in the required order (M1 oracle
/// `HcExecutorLifecycle::check`, contract s.6.1). A later check may only
/// throw after every earlier check has passed, keeping a rejected call
/// failure-atomic. `T` is only compared against the bound forward token
/// extent here; the consuming launchers re-verify supplied-tensor shapes, so
/// a caller passing a stale `T` (or a mis-shaped buffer) is rejected in
/// either place with the same contract failure. `op` names the call for the
/// diagnostic; the phase encodes the legal-phase rule for that operation:
///   read / final_mixer -> phase I (ready, no valid prior),
///   inject            -> phase P (ready, a valid pending prior snapshot).
void check_consuming(const char* op, std::int32_t T, const Tensor& hyper,
                     const HyperConnectionExecutorState& s) {
    reject_lifecycle(
        !s.ready, std::string("hyper_connection_executor: ") + op +
                     " called on a not-ready / reset state");
    reject_lifecycle(
        T != s.tokens, std::string("hyper_connection_executor: ") + op +
                           " token extent " + std::to_string(T) +
                           " does not match bound forward tokens " +
                           std::to_string(s.tokens) +
                           " (reset then initialize for a changed T)");
    const bool pending = s.prior_read_valid;
    if (std::string(op) == "inject") {
        reject_lifecycle(
            !pending,
            "hyper_connection_executor: inject requires phase P (a valid "
            "pending prior_read); call read first");
    } else {
        // read and final_mixer require phase I (no valid pending prior).
        reject_lifecycle(
            pending, std::string("hyper_connection_executor: ") + op +
                         " requires phase I (no valid pending prior_read); "
                         "a read has already run (phase P)");
    }
    // `hyper` is the caller-rebound [C*H, T] state; confirm its bound extent.
    const std::int32_t ch = expanded_width(s.hidden_size, s.hc_count);
    reject_lifecycle(
        hyper.ne[0] != ch || hyper.ne[1] != T,
        std::string("hyper_connection_executor: ") + op +
            " supplied hyper state is not the bound [C*H, T] forward state");
    // Astra R3-1: the bound low-rank width must be a positive, finite, integer
    // dimension -- capacity and the read/final_mixer scratch are both keyed by
    // R, so an invalid R is host-rejected rather than silently zeroed.
    const std::int64_t r_raw = s.hc_lowrank;
    const bool r_positive = r_raw > 0;
    const bool r_finite =
        std::isfinite(static_cast<double>(s.hc_lowrank));
    const bool r_int =
        r_finite &&
        static_cast<double>(r_raw) == static_cast<double>(std::floor(r_raw));
    reject_lifecycle(
        !r_positive || !r_finite || !r_int,
        std::string("hyper_connection_executor: ") + op +
            " bound hc_lowrank must be a positive, finite, integer "
            "low-rank width (R=" + std::to_string(r_raw) +
            ") for the bound forward");
}

/// Astra R3-1: the supplied `WorkspaceArena` must already cover the exact
/// maximum device-scratch bytes for the bound forward (the read/final_mixer/
/// inject scratch term, contract s.9); an undersized arena is rejected on the
/// host before any CUDA launcher runs.
void check_workspace_capacity(const char* op, std::int32_t T,
                              WorkspaceArena& workspace,
                              const HyperConnectionExecutorState& s) {
    const std::size_t need =
        hyper_connection_executor_workspace_capacity_bytes(
            s.hc_count, s.hidden_size, s.hc_lowrank, T);
    const std::size_t have = workspace.capacity();
    reject_lifecycle(
        have < need,
        std::string("hyper_connection_executor: ") + op +
            " supplied workspace capacity " + std::to_string(have) +
            " is insufficient for the bound forward (" + std::to_string(need) +
            " bytes); provision per "
            "hyper_connection_executor_workspace_capacity_bytes");
}

/// Astra R4: the exact current module projection dimension (`mix_down.n`,
/// the low-rank width the launcher allocates its low-rank scratch from) must
/// match the bound forward's `hc_lowrank`. `check_workspace_capacity` sizes
/// the bound from `s.hc_lowrank`; a mismatched `mix_down.n` therefore
/// understates the launcher's actual scratch need, so a correctly
/// bound-provisioned workspace is undersized for the real allocation.
/// Validated on the host **before** the capacity check or any launcher
/// dispatch (deterministic host gate; no CUDA on the reject path). An
/// unprovisioned `mix_down.n` (`<= 0`) is not a mismatch against the bound
/// width (its low-rank scratch is zero, which the bound already covers), so
/// only a *provisioned* `mix_down.n` that disagrees with the bound is
/// rejected; the consuming launchers re-verify supplied weights regardless.
void check_projection_geometry(const char* op, const Weight& mix_down,
                              const HyperConnectionExecutorState& s) {
    reject_lifecycle(
        mix_down.n > 0 && mix_down.n != s.hc_lowrank,
        std::string("hyper_connection_executor: ") + op +
            " module mix_down.n (low-rank width) " +
            std::to_string(mix_down.n) +
            " does not match the bound hc_lowrank " +
            std::to_string(s.hc_lowrank) +
            " (the launcher allocates low-rank scratch from mix_down.n, but "
            "capacity was sized from hc_lowrank; a mismatch under-provisions "
            "the workspace)");
}

/// Astra R3-2: the supplied hyper must be the bound state's own hyper storage;
/// a right-shape plane from a different allocation (mismatched state/storage)
/// is rejected before any launcher runs.
void check_state_bound_storage(const char* op, const Tensor& hyper_supplied,
                               const Tensor& hyper_bound) {
    if (hyper_bound.data == nullptr) {
        return;  // bound state not initialized/reset: nothing to match.
    }
    if (hyper_supplied.data == nullptr ||
        hyper_supplied.data != hyper_bound.data) {
        throw std::invalid_argument(
            std::string("hyper_connection_executor: ") + op +
                " supplied hyper is not the bound state's hyper storage "
                "(mismatched state/storage: the bound forward four-stream "
                "plane is the only legal hyper)");
    }
}

struct Bf16Plane {
    const void* data;
    std::int32_t rows;
    std::int32_t tokens;
};

struct OccupiedByteRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

/// Represent a logical BF16 [rows, T] plane by its half-open occupied byte
/// range. Null payloads and non-positive extents occupy no bytes, preserving
/// the previous alias gate's null behavior and the contract's zero-extent
/// behavior.
OccupiedByteRange occupied_byte_range(const Bf16Plane& plane) {
    if (plane.data == nullptr || plane.rows <= 0 || plane.tokens <= 0) {
        return {};
    }
    const auto elements = static_cast<std::size_t>(plane.rows) *
                          static_cast<std::size_t>(plane.tokens);
    const auto begin = reinterpret_cast<std::uintptr_t>(plane.data);
    return {begin, begin + elements * sizeof(std::uint16_t)};
}

bool ranges_intersect(const OccupiedByteRange& a,
                      const OccupiedByteRange& b) {
    return a.begin != a.end && b.begin != b.end &&
           a.begin < b.end && b.begin < a.end;
}

/// Astra R3-3 / R4-P2: supplied independent logical BF16 planes must have
/// pairwise-disjoint occupied byte ranges. Comparing full half-open ranges
/// catches both equal-base aliases and partial overlaps before
/// allocation/dispatch (contract s.8 independence rule).
void check_alias_independence(const char* op,
                              const std::vector<Bf16Plane>& planes) {
    for (std::size_t i = 0; i < planes.size(); ++i) {
        const auto a = occupied_byte_range(planes[i]);
        for (std::size_t j = i + 1; j < planes.size(); ++j) {
            const auto b = occupied_byte_range(planes[j]);
            if (ranges_intersect(a, b)) {
                throw std::invalid_argument(
                    std::string("hyper_connection_executor: ") + op +
                        " independent planes are aliased by overlapping "
                        "occupied BF16 byte ranges "
                        "(contract s.8 requires independent storage)");
            }
        }
    }
}

}  // namespace

void HyperConnectionExecutorState::initialize(const Tensor& embedding,
                                              std::int32_t T,
                                              cudaStream_t stream) {
    // Lifecycle gate (frozen, contract s.6.1): `initialize` is legal only
    // from the uninitialized (U) phase. Any second initialize before a
    // reset -- even with the same T -- is a contract error (U -> I only).
    reject_lifecycle(ready, "hyper_connection_executor: initialize requires "
                            "uninitialized state; re-initialization without "
                            "reset is illegal (reset then initialize)");
    if (T <= 0) {
        throw std::invalid_argument(
            "hyper_connection_executor: initialize token extent T must be "
            "positive (T=" + std::to_string(T) + ")");
    }

    // The four-stream geometry the forward is bound to. `hc_count`,
    // `hidden_size` (and `hc_lowrank`) must be provisioned by the caller
    // before the first `initialize`; the state validates its supplied
    // geometry rather than hard-coding the canonical 4/2560/320.
    if (hc_count <= 0 || hidden_size <= 0) {
        throw std::invalid_argument(
            "hyper_connection_executor: initialize requires hc_count and "
            "hidden_size to be provisioned");
    }
    const std::int32_t H = hidden_size;
    const std::int32_t C = hc_count;
    const std::int32_t CH = expanded_width(H, C);

    // `hyper[c*H + h, t] = embedding[h, t]` (REFERENCE_FORMULAS.md s.2:
    // repeat(embedding, C)). The 3C1 `repeat` is a pure GPU kernel over the
    // caller-rebound `hyper` storage; `initialize` allocates no scratch and
    // therefore takes no workspace (contract s.9.5). The long-lived state
    // plane is the caller's plane; a not-provisioned `hyper` is a contract
    // error, rejected (never a silent default).
    reject_lifecycle(hyper.data == nullptr,
                    "hyper_connection_executor: initialize requires a "
                    "provisioned `hyper` [C*H, T] state buffer");
    reject_lifecycle(hyper.ne[0] != CH || hyper.ne[1] != T,
                    "hyper_connection_executor: `hyper` must be [C*H, T] for "
                    "the bound geometry and token extent");
    if (embedding.data == nullptr) {
        throw std::invalid_argument(
            "hyper_connection_executor: initialize `embedding` payload is "
            "null");
    }
    reject_lifecycle(embedding.ne[0] != H || embedding.ne[1] != T,
                    "hyper_connection_executor: `embedding` must be [H, T] "
                    "matching the bound hidden width and token extent");

    // Execute the stream-replicated `hyper` image into the caller plane.
    detail::hyper_connection_repeat_launch(embedding, C, hyper, stream);

    // `ready` / `tokens` / geometry are bound only *after* the launch has
    // committed on `stream`. The forward is fresh (phase I): the prior
    // `read` snapshot validity is cleared (it is set again only by `read`).
    // `mixed_input` / `prior_read` are rebound to the forward extent by the
    // consuming `read` / `inject` calls on their call-scoped arena; the
    // forward anchors them through the bound T.
    ready = true;
    tokens = T;
    prior_read_valid = false;

    // Astra R4: persist the supplied (caller-owned) hyper plane into
    // state-bound storage so consuming calls see the correct reference.
    this->hyper = hyper;
}

void HyperConnectionExecutorState::read(const Tensor& hyper, std::int32_t T,
                                        const HyperConnectionBlock& module,
                                        float eps, Tensor& mixed,
                                        Tensor& prior,
                                        WorkspaceArena& workspace,
                                        cudaStream_t stream) {
    // Frozen consuming gate: ready -> token equality -> phase I (no
    // pending prior). Phase-P `read` is rejected so a second read cannot
    // overwrite the still-unconsumed prior snapshot (contract s.6.1).
    check_consuming("read", T, hyper, *this);
    // Astra R4: the module projection (mix_down.n) must match the bound
    // hc_lowrank before the capacity check, so the workspace bound reflects
    // the launcher's actual low-rank allocation.
    check_projection_geometry("read", module.mix_down, *this);
    // Astra R3: host-side capacity + state-bound-storage + plane-independence
    // gates, all before the consuming launcher (no CUDA on the reject paths).
    check_workspace_capacity("read", T, workspace, *this);
    check_state_bound_storage("read", hyper, this->hyper);
    const std::int32_t ch = expanded_width(hidden_size, hc_count);
    check_alias_independence(
        "read", {{hyper.data, ch, T},
                 {mixed.data, hidden_size, T},
                 {prior.data, ch, T}});
    if (eps != 1.0e-5F) {
        throw std::invalid_argument(
            "hyper_connection_executor: read `eps` must equal the "
            "contract-fixed 1e-5");
    }

    // Preserve the pre-block `hyper` token (read-only copy) before the
    // `inject` that will consume it. `prior` is the caller-provisioned
    // `[C*H, T]` plane; the launcher copies `hyper` into it (it is an
    // lvalue `Tensor&`, so `prior` is bound through itself, not a copy).
    detail::hyper_connection_read_launch(
        hyper, T, hidden_size, hc_count, module.norm_weight, module.mix_down,
        module.mix_up, eps, mixed, prior, workspace, stream);

    mixed_input = mixed;
    prior_read = prior;

    // A successful read captures the state-owned prior on the I->P
    // transition; the snapshot is the sole authority for the `inject`
    // that follows (contract s.6.3).
    prior_read_valid = true;
}

void HyperConnectionExecutorState::callback(const Tensor& mixed,
                                            Tensor& block,
                                            cudaStream_t stream,
                                            void (*run)(const Tensor& mixed,
                                                        Tensor& block,
                                                        cudaStream_t stream)) {
    // The caller block step: the only step the executor does not own. The
    // block consumes `mixed` (the `read` output) only and must not read
    // `hyper` directly (contract s.3). `callback` carries no workspace (it
    // runs the caller's own op) and the caller owns `block`'s independent
    // [H, T] storage, so the executor only dispatches the supplied functor.
    // Contract gate (precomputed, host-only): every `callback` precondition
    // is validated on the host **before** the caller `run` functor runs, so a
    // rejected call never invokes the caller block (contract s.3; the same
    // pre-`run` gate the consuming launchers get from `check_consuming`).
    // `callback` carries no workspace and the caller owns the `block` plane;
    // the block consumes `mixed` (the `read` `mixed_input`) only and must not
    // read `hyper` directly, so `block` requires independent storage (no
    // alias to the bound `hyper` / `mixed_input` / `prior_read`).
    if (run == nullptr) {
        throw std::invalid_argument(
            "hyper_connection_executor: callback requires a non-null "
            "`run` functor");
    }
    if (!ready) {
        throw std::invalid_argument(
            "hyper_connection_executor: callback called on a not-ready / "
            "reset state");
    }
    if (mixed.data == nullptr) {
        throw std::invalid_argument(
            "hyper_connection_executor: callback `mixed` payload is null");
    }
    if (mixed.ne[0] != hidden_size || mixed.ne[1] != tokens) {
        throw std::invalid_argument(
            "hyper_connection_executor: callback `mixed` must be [H, T] "
            "matching the bound hidden width and forward tokens");
    }
    if (block.ne[0] != hidden_size || block.ne[1] != tokens) {
        throw std::invalid_argument(
            "hyper_connection_executor: callback `block` must be [H, T] "
            "matching the bound hidden width and forward tokens");
    }
    if (block.data == nullptr) {
        throw std::invalid_argument(
            "hyper_connection_executor: callback `block` payload is null");
    }
    // `block` must be independent of the bound `hyper`, `mixed_input`, and
    // `prior_read` planes; an aliased `block` is a contract error (contract
    // s.3 / s.6.2). `prior_read.data` may be null (a not-yet-read forward),
    // in which case there is no plane to collide with and the check skips.
    const Bf16Plane block_plane{block.data, hidden_size, tokens};
    const std::int32_t ch = expanded_width(hidden_size, hc_count);
    if (ranges_intersect(occupied_byte_range(block_plane),
                         occupied_byte_range({hyper.data, ch, tokens})) ||
        ranges_intersect(
            occupied_byte_range(block_plane),
            occupied_byte_range({mixed_input.data, hidden_size, tokens})) ||
        ranges_intersect(
            occupied_byte_range(block_plane),
            occupied_byte_range({prior_read.data, ch, tokens}))) {
        throw std::invalid_argument(
            "hyper_connection_executor: callback `block` aliases the bound "
            "hyper / mixed_input / prior_read storage (independent storage "
            "required; contract s.3)");
    }
    run(mixed, block, stream);
}

void HyperConnectionExecutorState::inject(const Tensor& block,
                                          Tensor& hyper, std::int32_t T,
                                          const HyperConnectionBlock& module,
                                          float eps,
                                          WorkspaceArena& workspace,
                                          cudaStream_t stream) {
    // Frozen consuming gate: ready -> token equality -> phase P (a valid
    // pending prior snapshot) -> `hyper` extent. `inject` requires a
    // pending `read`; a missing snapshot is a contract error (s.6.3).
    check_consuming("inject", T, hyper, *this);
    check_workspace_capacity("inject", T, workspace, *this);
    check_state_bound_storage("inject", hyper, this->hyper);
    check_alias_independence(
        "inject", {{block.data, hidden_size, T},
                   {hyper.data, expanded_width(hidden_size, hc_count), T}});
    if (eps != 1.0e-5F) {
        throw std::invalid_argument(
            "hyper_connection_executor: inject `eps` must equal the "
            "contract-fixed 1e-5");
    }
    if (block.data == nullptr) {
        throw std::invalid_argument(
            "hyper_connection_executor: inject `block` payload is null");
    }

    // `hyper` (the in-place four-stream destination) is updated **from**
    // the state-owned `prior_read` snapshot, never from the live `hyper`
    // (REFERENCE_FORMULAS s.1 / contract s.4 / s.6.3):
    //   hyper[c,h,t] = prior_read[c,h,t] + block[h,t] * alpha[c,t].
    //
    // The state-owned `prior_read` view is authoritative: `read` binds
    // it to the caller-provided preserved-prior plane, and this inject
    // consumes that same snapshot as the additive base.
    detail::hyper_connection_inject_launch(
        block, hyper, T, hidden_size, hc_count, module.norm_weight,
        module.inject, eps, prior_read, workspace, stream);

    // A successful inject consumes the pending snapshot: the forward
    // returns to phase I (P -> I) for the next block.
    prior_read_valid = false;
}

void HyperConnectionExecutorState::final_mixer(const Tensor& hyper,
                                               std::int32_t T,
                                               const HyperConnectionFinalMixer&
                                                   module,
                                               float eps, Tensor& final,
                                               WorkspaceArena& workspace,
                                               cudaStream_t stream) {
    // Frozen consuming gate: ready -> token equality -> phase I (no pending
    // prior). The final mixer collapses the four streams to one H-wide
    // hidden state; phase P is a contract error (s.6.1 / s.5).
    check_consuming("final_mixer", T, hyper, *this);
    // Astra R4: the module projection (mix_down.n) must match the bound
    // hc_lowrank before the capacity check, so the workspace bound reflects
    // the launcher's actual low-rank allocation.
    check_projection_geometry("final_mixer", module.mix_down, *this);
    check_workspace_capacity("final_mixer", T, workspace, *this);
    check_state_bound_storage("final_mixer", hyper, this->hyper);
    check_alias_independence(
        "final_mixer",
        {{hyper.data, expanded_width(hidden_size, hc_count), T},
         {final.data, hidden_size, T}});
    if (eps != 1.0e-5F) {
        throw std::invalid_argument(
            "hyper_connection_executor: final_mixer `eps` must equal the "
            "contract-fixed 1e-5");
    }

    // The final mixer has **no** block-inject; it returns only the
    // collapsed `final` [H, T]. Bound to the final mixer's **own** weights
    // (which may differ from the block module's), the identical read path.
    detail::hyper_connection_final_mixer_launch(
        hyper, T, hidden_size, hc_count, module.norm_weight, module.mix_down,
        module.mix_up, eps, final, workspace, stream);
}

void HyperConnectionExecutorState::reset() {
    // `reset` is `any -> U`, idempotent, and allocates no scratch (it takes
    // no workspace). It clears every state-owned lifecycle field for the
    // next forward; the prior forward's `hyper` is **not** carried forward
    // silently (its device plane is the caller's and is freed / recycled by
    // the caller per its own convention). A changed token count is resolved
    // only through reset + initialize (no in-place adaptation).
    ready = false;
    tokens = 0;
    prior_read_valid = false;
    hyper = Tensor();
    mixed_input = Tensor();
    prior_read = Tensor();
    // Geometry (`hc_count`/`hidden_size`/`hc_lowrank`) is the caller's
    // module geometry and is **not** cleared: it is stable per model,
    // while the per-forward lifecycle plane is.
}

}  // namespace ninfer::ops