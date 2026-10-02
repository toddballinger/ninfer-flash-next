#include "ninfer/ops/hyper_connection_executor.h"

#include <cmath>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

//
// Batch 3C2 — milestone M2: CPU / static validation of the production
// HyperConnection *executor* surface (ninfer::ops::HyperConnectionExecutorState)
// plus the documented workspace-capacity geometry.
//
// Scope is deliberately CPU / non-GPU:
//   * Every lifecycle-rejection path validated here throws at the host-side
//     contract gate *before* any CUDA launcher is reached, so no CUDA
//     initialization or kernel launch occurs on the rejection paths.
//   * The workspace capacity bound is validated by link-calling the
//     production `hyper_connection_executor_workspace_capacity_bytes`
//     (defined in the executor launcher; it mirrors the DeviceArena 256-byte
//     aligned placement of the read and inject buffers -- the final arena
//     offset, not the raw element sum -- or 0 for any non-positive dimension)
//     and comparing its return to the independently computed placement
//     sequence at the same geometries.
//   * `initialize`'s *success* path would launch a GPU repeat kernel and is
//     therefore *not* exercised here; only the pure-host rejection gates of
//     initialize (T<=0, unprovisioned geometry, null payload, shape mismatch)
//     are validated, all of which throw before any launch.
//
// This is therefore a genuine non-GPU static/CPU contract test: it compiles
// and links against `ninfer_ops`, exercises the real production methods, and
// requires no CUDA initialization at runtime.
//

namespace {

using namespace ninfer;
using namespace ninfer::ops;

// Small, well-conditioned geometry. The executor validates supplied
// geometry rather than hard-coding it, so the contract holds at any legal
// geometry; we pick a tiny one to keep the test fast and deterministic.
constexpr std::int32_t kStreams = 4;   // C = hc_count
constexpr std::int32_t kHidden  = 8;    // H = hidden_size
constexpr std::int32_t kLowrank = 5;    // R = hc_lowrank
constexpr std::int32_t kCH      = kStreams * kHidden;  // 32
constexpr std::int32_t kTokens  = 3;    // bound forward token extent T
constexpr std::size_t kHiddenBytes =
    static_cast<std::size_t>(kHidden) * kTokens * sizeof(std::uint16_t);
constexpr std::size_t kHyperBytes =
    static_cast<std::size_t>(kCH) * kTokens * sizeof(std::uint16_t);

std::int64_t expanded_width(std::int32_t h, std::int32_t c) {
    return static_cast<std::int64_t>(h) * c;
}

// Independent oracle for the workspace capacity. Mirrors the production
// 256-byte aligned allocation-placement sequence (arena alloc_bytes: align
// each allocation start, add the raw BF16 bytes, advance the running
// offset) and the non-positive -> 0 gate. The sequence and the 256-byte
// constant are re-derived entirely within this test, so the comparison is
// independent, not circular: a genuine production sizing defect still makes
// the link-called production symbol disagree with this independently
// computed bound.
std::size_t expected_capacity(std::int32_t c, std::int32_t h,
                              std::int32_t r, std::int32_t t) {
    if (c <= 0 || h <= 0 || r <= 0 || t <= 0) {
        return 0;  // non-positive geometry: production returns 0
    }
    // Independent oracle for the arena allocation-placement sequence. It
    // re-derives the production alloc_bytes footprint (each allocation
    // start is aligned up to the arena 256-byte boundary, the raw BF16
    // bytes are added, and the running offset advances to the allocation
    // end), so `off` is the true workspace footprint rather than the raw
    // element sum. The sequence and the 256-byte constant are computed
    // entirely within this test -- not link-called from production -- so a
    // genuine production sizing defect still makes the production symbol
    // disagree with this independently computed bound.
    const std::size_t C = static_cast<std::size_t>(c);
    const std::size_t H = static_cast<std::size_t>(h);
    const std::size_t R = static_cast<std::size_t>(r);
    const std::size_t T = static_cast<std::size_t>(t);
    const std::size_t CH = C * H;
    const std::size_t a = 256;  // DeviceArena default allocation alignment
    const std::size_t m = a - 1;  // 255
    auto AU = [m](std::size_t v) -> std::size_t {  // align start up to 256B
        return (v + m) & ~m;
    };
    // Read / final-mixer pass: hn(2*CH*T), down(2*R*T), up(2*CH*T).
    std::size_t read_off = 0;
    read_off = AU(read_off) + 2u * CH * T;
    read_off = AU(read_off) + 2u * R * T;
    read_off = AU(read_off) + 2u * CH * T;
    // Inject pass: hn(2*CH*T), raw(2*C*T).
    std::size_t inject_off = 0;
    inject_off = AU(inject_off) + 2u * CH * T;
    inject_off = AU(inject_off) + 2u * C * T;
    return std::max(read_off, inject_off);
}

// Construct a non-owning WorkspaceArena (a DeviceArena over an already-
// allocated region). Passing a null/zero `DeviceSpan` performs **no**
// `cudaMalloc` and the destructor does not `cudaFree` (owns_ == false), so
// the arena object is pure-CPU: building it never initializes CUDA.
WorkspaceArena make_arena() {
    alignas(16) unsigned char pad[1] = {0};
    DeviceSpan sp{reinterpret_cast<void*>(pad), sizeof(pad)};
    return DeviceArena(sp);  // WorkspaceArena == DeviceArena
}

/// Capacity-adequate arena for the consuming-gate tests, sized to the exact
/// `hyper_connection_executor_workspace_capacity_bytes(C, H, R, T)` bound for
/// the bound forward. This mirrors the caller's contract: a legal call
/// provisions at least that many bytes of workspace. Link-calling the
/// production symbol (rather than recomputing the formula) keeps the
/// consuming-gate tests from masking a production capacity defect, and keeps
/// every preserved lifecycle/eps/reinit test valid under the new capacity
/// gate. Pure-CPU (no cudaMalloc) as the default arena above.
WorkspaceArena make_arena_adequate(std::int32_t c, std::int32_t h,
                                   std::int32_t r, std::int32_t t) {
    std::size_t cap =
        hyper_connection_executor_workspace_capacity_bytes(c, h, r, t);
    if (cap == 0) {
        cap = 1;  // non-positive geometry: keep a nonempty arena
    }
    alignas(16) unsigned char pad[1] = {0};
    DeviceSpan sp{reinterpret_cast<void*>(pad), cap};
    return DeviceArena(sp);  // pure-CPU, no cudaMalloc
}

// A valid `Tensor` with a non-null payload pointer and a chosen 2-D shape
// [rows, T]. The payload buffer is never dereferenced on the rejection paths
// (they throw before any device access), so any non-null host pointer is
// sufficient to satisfy the "payload not null" contract checks.
Tensor make_tensor(std::int32_t rows, std::int32_t T, char* payload) {
    Tensor t;
    t.data = payload;
    t.ne[0] = rows;
    t.ne[1] = T;
    return t;
}

// A default-constructed (null-payload) block / final module. The consuming
// gate rejects before any module weight is read, so default modules suffice
// for the lifecycle cases; the shape gate keys only on the state's geometry.
HyperConnectionBlock default_block() {
    HyperConnectionBlock b;  // all Weight members are null/default
    return b;
}

HyperConnectionFinalMixer default_final() {
    HyperConnectionFinalMixer f;
    f.has_block_inject = false;  // normative: final mixer has no block-inject
    return f;
}

/// `ok` = 0 means the expected behavior held (no counted failure).
/// Each helper returns the number of failed assertions (0 on success).

// (A) A consuming call on a default (not-ready) state is rejected:
//     "called on a not-ready / reset state".
int test_not_ready_reject(int& failures) {
    int f = 0;
    auto ws = make_arena();
    char payload = 0;
    auto hyper = make_tensor(kCH, kTokens, &payload);

    {
        HyperConnectionExecutorState s;  // default: ready=false
        Tensor mixed = make_tensor(kHidden, kTokens, &payload);
        Tensor prior = make_tensor(kCH, kTokens, &payload);
        try {
            s.read(hyper, kTokens, default_block(), 1.0e-5F,
                   mixed, prior, ws, 0u);
            std::cerr << "FAIL not-ready: read did not reject on a "
                         "not-ready state\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "called on a not-ready / reset state") ==
                std::string::npos) {
                std::cerr << "FAIL not-ready: read rejection message = "
                          << e.what() << "\n";
                ++f;
            }
        }
    }
    {
        HyperConnectionExecutorState s;
        Tensor final_t = make_tensor(kHidden, kTokens, &payload);
        try {
            s.final_mixer(hyper, kTokens, default_final(), 1.0e-5F,
                          final_t, ws, 0u);
            std::cerr << "FAIL not-ready: final_mixer did not reject on a "
                         "not-ready state\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "called on a not-ready / reset state") ==
                std::string::npos) {
                std::cerr << "FAIL not-ready: final_mixer message = "
                          << e.what() << "\n";
                ++f;
            }
        }
    }
    failures += f;
    return f;
}

// (B) Token mismatch: a *ready* state bound to kTokens rejects a consuming
//     call carrying a different T: "does not match bound forward tokens".
int test_token_mismatch_reject(int& failures) {
    int f = 0;
    auto ws = make_arena();
    char payload = 0;
    const std::int32_t otherT = kTokens + 1;

    // Ready state bound to kTokens, prior_read_valid=false (phase I).
    auto build_ready = [&]() {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = false;
        return s;
    };

    {
        auto s = build_ready();
        auto hyper = make_tensor(kCH, kTokens, &payload);  // shape ok
        Tensor mixed = make_tensor(kHidden, otherT, &payload);
        Tensor prior = make_tensor(kCH, otherT, &payload);
        try {
            s->read(hyper, otherT, default_block(), 1.0e-5F,
                   mixed, prior, ws, 0u);
            std::cerr << "FAIL token: read did not reject mismatched T\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "does not match bound forward tokens") ==
                std::string::npos) {
                std::cerr << "FAIL token: read token-mismatch message = "
                          << e.what() << "\n";
                ++f;
            }
        }
    }
    {
        auto s = build_ready();
        auto hyper = make_tensor(kCH, kTokens, &payload);
        Tensor final_t = make_tensor(kHidden, otherT, &payload);
        try {
            s->final_mixer(hyper, otherT, default_final(), 1.0e-5F,
                           final_t, ws, 0u);
            std::cerr << "FAIL token: final_mixer did not reject T\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "does not match bound forward tokens") ==
                std::string::npos) {
                std::cerr << "FAIL token: final_mixer message = "
                          << e.what() << "\n";
                ++f;
            }
        }
    }
    failures += f;
    return f;
}

// (C) Phase-P read / final_mixer rejection: a *ready, pending-prior*
//     (phase P) state must reject read/final_mixer ("requires phase I ..."),
//     while inject (phase P) is the legal consuming call in that phase.
int test_phase_p_reject(int& failures) {
    int f = 0;
    auto ws = make_arena();
    char payload = 0;

    auto build_ready_pending = [&]() {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = true;  // phase P: a read has run, pending inject
        return s;
    };

    // read in phase P is rejected ("requires phase I ...").
    {
        auto s = build_ready_pending();
        auto hyper = make_tensor(kCH, kTokens, &payload);
        Tensor mixed = make_tensor(kHidden, kTokens, &payload);
        Tensor prior = make_tensor(kCH, kTokens, &payload);
        try {
            s->read(hyper, kTokens, default_block(), 1.0e-5F,
                   mixed, prior, ws, 0u);
            std::cerr << "FAIL phaseP: read in phase P did not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("requires phase I") ==
                std::string::npos) {
                std::cerr << "FAIL phaseP: read message = " << e.what()
                          << "\n";
                ++f;
            }
        }
    }
    // final_mixer in phase P is rejected ("requires phase I ...").
    {
        auto s = build_ready_pending();
        auto hyper = make_tensor(kCH, kTokens, &payload);
        Tensor final_t = make_tensor(kHidden, kTokens, &payload);
        try {
            s->final_mixer(hyper, kTokens, default_final(), 1.0e-5F,
                           final_t, ws, 0u);
            std::cerr << "FAIL phaseP: final_mixer in phase P did not "
                         "reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("requires phase I") ==
                std::string::npos) {
                std::cerr << "FAIL phaseP: final_mixer message = "
                          << e.what() << "\n";
                ++f;
            }
        }
    }
    failures += f;
    return f;
}

// (D) inject without a prior read: a ready (phase I, no pending prior)
//     state rejects inject ("inject requires phase P (a valid pending
//     prior_read); call read first").
int test_inject_without_prior_reject(int& failures) {
    int f = 0;
    auto ws = make_arena();
    char payload = 0;

    auto s = std::make_unique<HyperConnectionExecutorState>();
    s->ready = true;
    s->tokens = kTokens;
    s->hc_count = kStreams;
    s->hidden_size = kHidden;
    s->hc_lowrank = kLowrank;
    s->prior_read_valid = false;  // phase I: no pending prior -> inject illegal

    auto hyper = make_tensor(kCH, kTokens, &payload);
    auto block = make_tensor(kHidden, kTokens, &payload);
    try {
        s->inject(block, hyper, kTokens, default_block(), 1.0e-5F, ws, 0u);
        std::cerr << "FAIL inject-no-prior: inject did not reject on a "
                     "not-pending state\n";
        ++f;
    } catch (const std::invalid_argument& e) {
        if (std::string(e.what()).find(
                "inject requires phase P") == std::string::npos) {
            std::cerr << "FAIL inject-no-prior: message = " << e.what()
                      << "\n";
            ++f;
        }
    }
    failures += f;
    return f;
}

// (E) Token-match gate ordering: with a ready, *pending* state but a
//     mismatched T, the token check fires *before* any phase check (gate
//     order: ready -> token -> phase). Confirms the frozen failure-atomic
//     ordering rejects on token extent, not on the still-valid phase.
int test_gate_order_token_before_phase(int& failures) {
    int f = 0;
    auto ws = make_arena();
    char payload = 0;

    // Ready, phase P (pending prior), bound to kTokens; we inject with a
    // *different* T so the token gate (2nd) rejects before the (legal)
    // phase-P inject would otherwise pass.
    auto s = std::make_unique<HyperConnectionExecutorState>();
    s->ready = true;
    s->tokens = kTokens;
    s->hc_count = kStreams;
    s->hidden_size = kHidden;
    s->hc_lowrank = kLowrank;
    s->prior_read_valid = true;

    auto hyper = make_tensor(kCH, kTokens, &payload);
    auto block = make_tensor(kHidden, kTokens + 1, &payload);
    try {
        s->inject(block, hyper, kTokens + 1, default_block(), 1.0e-5F,
                  ws, 0u);
        std::cerr << "FAIL gate-order: inject with mismatched T did not "
                     "reject at the token gate\n";
        ++f;
    } catch (const std::invalid_argument& e) {
        if (std::string(e.what()).find(
                "does not match bound forward tokens") ==
            std::string::npos) {
            std::cerr << "FAIL gate-order: token gate fired on = "
                      << e.what() << "\n";
            ++f;
        }
    }
    failures += f;
    return f;
}

// (F) eps rejection: after the lifecycle/shape gate passes, every epsilon
//     other than the exact contract-fixed 1e-5 is rejected before a launcher.
//     This includes non-positive/non-finite values and positive finite values
//     that would have passed the weaker pre-M2 predicate.
int test_eps_reject(int& failures) {
    auto ws = make_arena_adequate(kStreams, kHidden, kLowrank, kTokens);
    alignas(16) char hyper_storage[kHyperBytes] = {};
    alignas(16) char hidden_storage[kHiddenBytes] = {};
    alignas(16) char prior_storage[kHyperBytes] = {};
    int f = 0;

    auto build_ready_pending = [&]() {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = true;
        return s;
    };

    // read with a non-finite (NaN) eps, ready + phase I -> eps check fires.
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = false;  // phase I: read legal; only eps blocks
        auto hyper = make_tensor(kCH, kTokens, hyper_storage);
        const float nanEps = std::nan("");
        Tensor mixed = make_tensor(kHidden, kTokens, hidden_storage);
        Tensor prior = make_tensor(kCH, kTokens, prior_storage);
        try {
            s->read(hyper, kTokens, default_block(), nanEps,
                   mixed, prior, ws, 0u);
            std::cerr << "FAIL eps: read with NaN eps did not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "read `eps` must equal the contract-fixed 1e-5") ==
                std::string::npos) {
                std::cerr << "FAIL eps: read message = " << e.what() << "\n";
                ++f;
            }
        }
    }
    // inject with a negative eps on a pending (phase P) ready state.
    {
        auto s = build_ready_pending();
        auto hyper = make_tensor(kCH, kTokens, hyper_storage);
        auto block = make_tensor(kHidden, kTokens, hidden_storage);
        try {
            s->inject(block, hyper, kTokens, default_block(), -1.0F,
                      ws, 0u);
            std::cerr << "FAIL eps: inject with negative eps did not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "inject `eps` must equal the contract-fixed 1e-5") ==
                std::string::npos) {
                std::cerr << "FAIL eps: inject message = " << e.what()
                          << "\n";
                ++f;
            }
        }
    }
    // final_mixer with zero eps (ready, phase I).
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = false;
        auto hyper = make_tensor(kCH, kTokens, hyper_storage);
        Tensor final_t = make_tensor(kHidden, kTokens, hidden_storage);
        try {
            s->final_mixer(hyper, kTokens, default_final(), 0.0F,
                           final_t, ws, 0u);
            std::cerr << "FAIL eps: final_mixer with eps=0 did not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "final_mixer `eps` must equal the contract-fixed 1e-5") ==
                std::string::npos) {
                std::cerr << "FAIL eps: final_mixer message = " << e.what()
                          << "\n";
                ++f;
            }
        }
    }
    // Each consuming operation must also reject a positive, finite but wrong
    // epsilon. These paths pass the lifecycle and tensor-shape gates, then
    // stop at the host epsilon gate before any CUDA launcher is reached.
    constexpr float wrongPositiveEps = 2.0e-5F;
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = false;
        auto hyper = make_tensor(kCH, kTokens, hyper_storage);
        Tensor mixed = make_tensor(kHidden, kTokens, hidden_storage);
        Tensor prior = make_tensor(kCH, kTokens, prior_storage);
        try {
            s->read(hyper, kTokens, default_block(), wrongPositiveEps,
                    mixed, prior, ws, 0u);
            std::cerr << "FAIL eps: read accepted positive wrong eps\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "read `eps` must equal the contract-fixed 1e-5") ==
                std::string::npos) {
                std::cerr << "FAIL eps: positive-wrong read message = "
                          << e.what() << "\n";
                ++f;
            }
        }
    }
    {
        auto s = build_ready_pending();
        auto hyper = make_tensor(kCH, kTokens, hyper_storage);
        auto block = make_tensor(kHidden, kTokens, hidden_storage);
        try {
            s->inject(block, hyper, kTokens, default_block(),
                      wrongPositiveEps, ws, 0u);
            std::cerr << "FAIL eps: inject accepted positive wrong eps\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "inject `eps` must equal the contract-fixed 1e-5") ==
                std::string::npos) {
                std::cerr << "FAIL eps: positive-wrong inject message = "
                          << e.what() << "\n";
                ++f;
            }
        }
    }
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = false;
        auto hyper = make_tensor(kCH, kTokens, hyper_storage);
        Tensor final_t = make_tensor(kHidden, kTokens, hidden_storage);
        try {
            s->final_mixer(hyper, kTokens, default_final(),
                           wrongPositiveEps, final_t, ws, 0u);
            std::cerr << "FAIL eps: final_mixer accepted positive wrong eps\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "final_mixer `eps` must equal the contract-fixed 1e-5") ==
                std::string::npos) {
                std::cerr << "FAIL eps: positive-wrong final_mixer message = "
                          << e.what() << "\n";
                ++f;
            }
        }
    }
    failures += f;
    return f;
}

// (G) Repeated initialize rejection: a *ready* state rejects a second
//     `initialize` ("re-initialization without reset is illegal"). This is
//     the pure-host U->I-only gate and fires before any launch, so it is
//     non-GPU.
int test_reinit_reject(int& failures) {
    auto ws = make_arena();
    char payload = 0;
    int f = 0;

    // A ready state (bound forward) is re-initialized -> rejected at the
    // ready gate, before any T/geometry/launch handling.
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;  // already initialized this forward
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        auto hyper = make_tensor(kCH, kTokens, &payload);
        auto emb = make_tensor(kHidden, kTokens, &payload);
        try {
            s->initialize(emb, kTokens, 0u);
            std::cerr << "FAIL reinit: re-initialize on a ready state did "
                         "not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "re-initialization without reset is illegal") ==
                std::string::npos) {
                std::cerr << "FAIL reinit: message = " << e.what() << "\n";
                ++f;
            }
        }
    }
    (void)ws;
    failures += f;
    return f;
}

// (H) reset() clears every lifecycle field and returns the state to
//     not-ready (phase U): ready=false, tokens=0, prior_read_valid=false,
//     and the state plane is re-defaulted. After reset, consuming calls are
//     again rejected as not-ready.
int test_reset_clears(int& failures) {
    int f = 0;
    auto ws = make_arena();
    char payload = 0;

    // Start ready + pending (an active forward that has read, about to
    // inject) to prove reset clears *both* readiness and the pending prior.
    auto s = std::make_unique<HyperConnectionExecutorState>();
    s->ready = true;
    s->tokens = kTokens;
    s->hc_count = kStreams;
    s->hidden_size = kHidden;
    s->hc_lowrank = kLowrank;
    s->prior_read_valid = true;
    // Give the plane a live view so reset visibly clears it to default.
    s->hyper = make_tensor(kCH, kTokens, &payload);

    s->reset();

    if (s->ready) {
        std::cerr << "FAIL reset: ready not cleared by reset\n";
        ++f;
    }
    if (s->tokens != 0) {
        std::cerr << "FAIL reset: tokens not cleared by reset\n";
        ++f;
    }
    if (s->prior_read_valid) {
        std::cerr << "FAIL reset: prior_read_valid not cleared by reset\n";
        ++f;
    }
    if (s->hyper.data != nullptr) {
        std::cerr << "FAIL reset: hyper plane not cleared by reset\n";
        ++f;
    }
    if (s->prior_read.data != nullptr) {
        std::cerr << "FAIL reset: prior_read plane not cleared by reset\n";
        ++f;
    }

    // After reset, a consuming call is again rejected as not-ready.
    {
        auto hyper = make_tensor(kCH, kTokens, &payload);
        Tensor mixed = make_tensor(kHidden, kTokens, &payload);
        Tensor prior = make_tensor(kCH, kTokens, &payload);
        try {
            s->read(hyper, kTokens, default_block(), 1.0e-5F,
                   mixed, prior, ws, 0u);
            std::cerr << "FAIL reset: read after reset did not reject "
                         "(should be not-ready)\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "called on a not-ready / reset state") ==
                std::string::npos) {
                std::cerr << "FAIL reset: post-reset read message = "
                          << e.what() << "\n";
                ++f;
            }
        }
    }
    // reset() on an already-not-ready default state is a legal no-op (no
    // throw) and leaves the state not-ready.
    {
        auto s2 = std::make_unique<HyperConnectionExecutorState>();
        s2->reset();
        if (s2->ready || s2->tokens != 0 || s2->prior_read_valid) {
            std::cerr << "FAIL reset: default-state reset left non-clear "
                         "fields set\n";
            ++f;
        }
    }
    failures += f;
    return f;
}

// (I) initialize pure-host rejection gates (all throw before any launch):
//     T<=0, unprovisioned geometry, null embedding payload, and shape
//     mismatch of `hyper`/`embedding`. The *success* path (GPU repeat
//     launch) is intentionally not exercised to keep the test non-GPU.
int test_initialize_reject_gates(int& failures) {
    auto ws = make_arena();
    char payload = 0;
    int f = 0;

    auto provisioned = [&](std::int32_t c, std::int32_t h) {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->hc_count = c;
        s->hidden_size = h;
        s->hc_lowrank = kLowrank;
        return s;
    };

    // T<=0
    {
        auto s = provisioned(kStreams, kHidden);
        auto emb = make_tensor(kHidden, kTokens, &payload);
        auto hyper = make_tensor(kCH, kTokens, &payload);
        s->hyper = hyper;
        try {
            s->initialize(emb, 0, 0u);
            std::cerr << "FAIL init: T<=0 did not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "initialize token extent T must be positive") ==
                std::string::npos) {
                std::cerr << "FAIL init: T<=0 message = " << e.what() << "\n";
                ++f;
            }
        }
    }
    // unprovisioned geometry
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();  // 0/0
        auto emb = make_tensor(kHidden, kTokens, &payload);
        s->hyper = make_tensor(kCH, kTokens, &payload);
        try {
            s->initialize(emb, kTokens, 0u);
            std::cerr << "FAIL init: unprovisioned geometry did not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "initialize requires hc_count and hidden_size to be "
                    "provisioned") == std::string::npos) {
                std::cerr << "FAIL init: geometry message = " << e.what()
                          << "\n";
                ++f;
            }
        }
    }
    // null embedding payload
    {
        auto s = provisioned(kStreams, kHidden);
        s->hyper = make_tensor(kCH, kTokens, &payload);
        Tensor nullEmb;  // default: data = nullptr
        nullEmb.ne[0] = kHidden;
        nullEmb.ne[1] = kTokens;
        try {
            s->initialize(nullEmb, kTokens, 0u);
            std::cerr << "FAIL init: null embedding payload did not "
                         "reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "initialize `embedding` payload is null") ==
                std::string::npos) {
                std::cerr << "FAIL init: null-embedding message = "
                          << e.what() << "\n";
                ++f;
            }
        }
    }
    // `hyper` shape mismatch
    {
        auto s = provisioned(kStreams, kHidden);
        s->hyper = make_tensor(kCH + 1, kTokens, &payload);  // ne[0] wrong
        auto emb = make_tensor(kHidden, kTokens, &payload);
        try {
            s->initialize(emb, kTokens, 0u);
            std::cerr << "FAIL init: hyper shape mismatch did not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "`hyper` must be [C*H, T]") == std::string::npos) {
                std::cerr << "FAIL init: hyper-shape message = " << e.what()
                          << "\n";
                ++f;
            }
        }
    }
    // `embedding` shape mismatch
    {
        auto s = provisioned(kStreams, kHidden);
        s->hyper = make_tensor(kCH, kTokens, &payload);
        auto badEmb = make_tensor(kHidden + 1, kTokens, &payload);
        try {
            s->initialize(badEmb, kTokens, 0u);
            std::cerr << "FAIL init: embedding shape mismatch did not "
                         "reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "`embedding` must be [H, T]") == std::string::npos) {
                std::cerr << "FAIL init: embedding-shape message = "
                          << e.what() << "\n";
                ++f;
            }
        }
    }
    (void)ws;
    failures += f;
    return f;
}

// (J) Production workspace capacity: link-call the real
//     `hyper_connection_executor_workspace_capacity_bytes` and compare its
//     return to the independently computed `expected_capacity` at several
//     geometries (normal, monotone, and non-positive). This proves the
//     production symbol matches the documented formula, not the reverse.
int test_capacity_definition(int& failures) {
    int f = 0;

    // (J.1) Normal: the production symbol must equal the independent
    //       expected value at the canonical and a few small geometries.
    {
        const std::size_t got =
            hyper_connection_executor_workspace_capacity_bytes(4, 2560, 320, 128);
        const std::size_t exp = expected_capacity(4, 2560, 320, 128);
        if (got != exp) {
            std::cerr << "FAIL capacity: production != expected at canonical "
                         "geometry (got " << got << " expected " << exp << ")\n";
            ++f;
        }
    }
    {
        const std::size_t got =
            hyper_connection_executor_workspace_capacity_bytes(4, 8, 5, 3);
        const std::size_t exp = expected_capacity(4, 8, 5, 3);
        if (got != exp) {
            std::cerr << "FAIL capacity: production != expected (C4 H8 R5 T3): "
                         "got " << got << " expected " << exp << "\n";
            ++f;
        }
    }
    {
        const std::size_t got =
            hyper_connection_executor_workspace_capacity_bytes(2, 8, 5, 3);
        const std::size_t exp = expected_capacity(2, 8, 5, 3);
        if (got != exp) {
            std::cerr << "FAIL capacity: production != expected (C2 H8 R5 T3): "
                         "got " << got << " expected " << exp << "\n";
            ++f;
        }
    }

    // (J.2) Monotone: the bound is monotone non-decreasing under aligned placement.
    if (!(hyper_connection_executor_workspace_capacity_bytes(2, 8, 5, 3) <
          hyper_connection_executor_workspace_capacity_bytes(4, 8, 5, 3))) {
        std::cerr << "FAIL capacity: not strictly increasing in C\n";
        ++f;
    }
    if (!(hyper_connection_executor_workspace_capacity_bytes(4, 8, 5, 3) <
          hyper_connection_executor_workspace_capacity_bytes(4, 16, 5, 3))) {
        std::cerr << "FAIL capacity: not strictly increasing in H\n";
        ++f;
    }
    if (!(hyper_connection_executor_workspace_capacity_bytes(4, 8, 5, 3) <=
          hyper_connection_executor_workspace_capacity_bytes(4, 8, 7, 3))) {
        std::cerr << "FAIL capacity: not strictly increasing in R\n";
        ++f;
    }
    if (!(hyper_connection_executor_workspace_capacity_bytes(4, 8, 5, 3) <
          hyper_connection_executor_workspace_capacity_bytes(4, 8, 5, 6))) {
        std::cerr << "FAIL capacity: not strictly increasing in T\n";
        ++f;
    }

    // (J.3) Non-positive geometry: any dimension <= 0 yields a zero bound
    //       (no forward), mirroring the production non-positive gate.
    {
        const std::size_t got =
            hyper_connection_executor_workspace_capacity_bytes(4, 8, 5, 0);
        if (got != expected_capacity(4, 8, 5, 0) || got != 0) {
            std::cerr << "FAIL capacity: T=0 should give zero capacity (got "
                      << got << ")\n";
            ++f;
        }
    }
    {
        const std::size_t got =
            hyper_connection_executor_workspace_capacity_bytes(0, 8, 5, 3);
        if (got != expected_capacity(0, 8, 5, 3) || got != 0) {
            std::cerr << "FAIL capacity: C=0 should give zero capacity (got "
                      << got << ")\n";
            ++f;
        }
    }
    {
        const std::size_t got =
            hyper_connection_executor_workspace_capacity_bytes(4, 0, 5, 3);
        if (got != expected_capacity(4, 0, 5, 3) || got != 0) {
            std::cerr << "FAIL capacity: H=0 should give zero capacity (got "
                      << got << ")\n";
            ++f;
        }
    }
    {
        const std::size_t got =
            hyper_connection_executor_workspace_capacity_bytes(4, 8, 0, 3);
        if (got != expected_capacity(4, 8, 0, 3) || got != 0) {
            std::cerr << "FAIL capacity: R=0 should give zero capacity (got "
                      << got << ")\n";
            ++f;
        }
    }
    // (J.4) Boundary: at the int32 maximum geometry the workspace
    //       capacity overflows `size_t`. The production symbol must
    //       *reject* the overflow (throw) rather than silently wrapping
    //       the 2x element-width doublings to a small value. This is the
    //       focused boundary assertion for the checked-doubling fix.
    {
        const std::int32_t max32 =
            std::numeric_limits<std::int32_t>::max();
        bool threw = false;
        try {
            (void)hyper_connection_executor_workspace_capacity_bytes(
                max32, max32, max32, max32);
        } catch (const std::overflow_error&) {
            threw = true;
        }
        if (!threw) {
            std::cerr << "FAIL capacity: max-int32 geometry overflow was "
                         "not rejected (wrapped value returned)\n";
            ++f;
        }
    }
    failures += f;
    return f;
}

// (K) Astra R3-1: an undersized WorkspaceArena is rejected on the host
//     before any CUDA launcher, for every consuming op, keyed by the exact
//     production capacity bound for the bound forward.
int test_undersized_workspace_reject(int& failures) {
    int f = 0;
    char payload = 0;
    const std::size_t cap =
        hyper_connection_executor_workspace_capacity_bytes(
            kStreams, kHidden, kLowrank, kTokens);
    if (cap == 0) {
        std::cerr << "FAIL undersized: capacity bound computed 0\n";
        ++f;
    }
    // Deliberately 1 byte (the default make_arena capacity) so 1 < cap is
    // guaranteed for any non-positive-free geometry.
    alignas(16) unsigned char pad[1] = {0};
    DeviceSpan sp{reinterpret_cast<void*>(pad), 1};
    WorkspaceArena tiny{sp};

    // read (phase I) with an undersized arena.
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = false;
        auto hyper = make_tensor(kCH, kTokens, &payload);
        Tensor mixed = make_tensor(kHidden, kTokens, (char*)pad);
        Tensor prior = make_tensor(kCH, kTokens, (char*)pad + 1);
        try {
            s->read(hyper, kTokens, default_block(), 1.0e-5F,
                   mixed, prior, tiny, 0u);
            std::cerr << "FAIL undersized: read with undersized arena "
                         "did not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "insufficient for the bound forward") ==
                std::string::npos) {
                std::cerr << "FAIL undersized: read message = "
                             << e.what() << "\n";
                ++f;
            }
        }
    }
    // final_mixer (phase I) with an undersized arena.
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = false;
        auto hyper = make_tensor(kCH, kTokens, &payload);
        Tensor final_t = make_tensor(kHidden, kTokens, (char*)pad + 2);
        try {
            s->final_mixer(hyper, kTokens, default_final(), 1.0e-5F,
                           final_t, tiny, 0u);
            std::cerr << "FAIL undersized: final_mixer with undersized "
                         "arena did not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "insufficient for the bound forward") ==
                std::string::npos) {
                std::cerr << "FAIL undersized: final_mixer message = "
                             << e.what() << "\n";
                ++f;
            }
        }
    }
    // inject (phase P) with an undersized arena.
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = true;  // phase P: inject is legal
        auto hyper = make_tensor(kCH, kTokens, &payload);
        auto block = make_tensor(kHidden, kTokens, (char*)pad + 3);
        try {
            s->inject(block, hyper, kTokens, default_block(), 1.0e-5F,
                      tiny, 0u);
            std::cerr << "FAIL undersized: inject with undersized arena "
                         "did not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "insufficient for the bound forward") ==
                std::string::npos) {
                std::cerr << "FAIL undersized: inject message = "
                             << e.what() << "\n";
                ++f;
            }
        }
    }
    failures += f;
    return f;
}

// (L) Astra R3-2: a `hyper` plane that is the right shape yet a different
//     allocation than the bound state's `s.hyper` is rejected (mismatched
//     state/storage), on every consuming op.
int test_storage_mismatch_reject(int& failures) {
    int f = 0;
    alignas(16) unsigned char boundpad[1] = {0};
    alignas(16) unsigned char supp_pad[1] = {0};

    // read: bound state's hyper is `boundpad`; supplied is `supp_pad`.
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = false;
        s->hyper = make_tensor(kCH, kTokens, (char*)boundpad);
        auto ws = make_arena_adequate(kStreams, kHidden, kLowrank, kTokens);
        auto supp = make_tensor(kCH, kTokens, (char*)supp_pad);  // mismatched
        Tensor mixed = make_tensor(kHidden, kTokens, (char*)supp_pad);
        Tensor prior = make_tensor(kCH, kTokens, (char*)boundpad);
        try {
            s->read(supp, kTokens, default_block(), 1.0e-5F,
                   mixed, prior, ws, 0u);
            std::cerr << "FAIL storage: read mismatched storage did "
                         "not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "mismatched state/storage") == std::string::npos) {
                std::cerr << "FAIL storage: read message = " << e.what()
                             << "\n";
                ++f;
            }
        }
    }
    // final_mixer with a mismatched supplied hyper.
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = false;
        s->hyper = make_tensor(kCH, kTokens, (char*)boundpad);
        auto ws = make_arena_adequate(kStreams, kHidden, kLowrank, kTokens);
        auto supp = make_tensor(kCH, kTokens, (char*)supp_pad);
        Tensor final_t = make_tensor(kHidden, kTokens, (char*)boundpad);
        try {
            s->final_mixer(supp, kTokens, default_final(), 1.0e-5F,
                           final_t, ws, 0u);
            std::cerr << "FAIL storage: final_mixer mismatched storage "
                         "did not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "mismatched state/storage") == std::string::npos) {
                std::cerr << "FAIL storage: final_mixer message = "
                             << e.what() << "\n";
                ++f;
            }
        }
    }
    failures += f;
    return f;
}

// (M) Astra R3-3: aliased independent planes are rejected before
//     allocation/dispatch, for every consuming op whose contract requires
//     plane independence.
int test_alias_reject(int& failures) {
    int f = 0;
    alignas(16) char hyper_storage[kHyperBytes] = {};
    alignas(16) char prior_storage[kHyperBytes] = {};

    // read: `mixed` and `hyper` have the same base -> independent planes
    // aliased -> rejected. `prior` uses a genuinely distinct allocation.
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = false;
        s->hyper = make_tensor(kCH, kTokens, hyper_storage);
        auto ws = make_arena_adequate(kStreams, kHidden, kLowrank, kTokens);
        auto hyper = make_tensor(kCH, kTokens, hyper_storage);  // supplied hyper = bound storage (passes storage gate)
        Tensor mixed = make_tensor(kHidden, kTokens, hyper_storage);  // mixed aliases bound hyper (distinct from prior)
        Tensor prior = make_tensor(kCH, kTokens, prior_storage);  // distinct
        try {
            s->read(hyper, kTokens, default_block(), 1.0e-5F,
                   mixed, prior, ws, 0u);
            std::cerr << "FAIL alias: read with aliased mixed/hyper did "
                         "not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("independent") ==
                std::string::npos) {
                std::cerr << "FAIL alias: read message = " << e.what()
                             << "\n";
                ++f;
            }
        }
    }
    // final_mixer: `hyper` and `final` alias the same byte.
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = false;
        s->hyper = make_tensor(kCH, kTokens, hyper_storage);  // bound plane
        auto ws = make_arena_adequate(kStreams, kHidden, kLowrank, kTokens);
        auto hyper = make_tensor(kCH, kTokens, hyper_storage);  // supplied hyper = bound storage (passes storage gate)
        Tensor final_t =
            make_tensor(kHidden, kTokens, hyper_storage);  // final aliases bound hyper
        try {
            s->final_mixer(hyper, kTokens, default_final(), 1.0e-5F,
                           final_t, ws, 0u);
            std::cerr << "FAIL alias: final_mixer with aliased planes did "
                         "not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("independent") ==
                std::string::npos) {
                std::cerr << "FAIL alias: final_mixer message = " << e.what()
                             << "\n";
                ++f;
            }
        }
    }
    // read: `mixed` begins one BF16 element inside the occupied `hyper`
    // range. Distinct base pointers are still an overlap and must reject.
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = false;
        s->hyper = make_tensor(kCH, kTokens, hyper_storage);
        auto ws = make_arena_adequate(kStreams, kHidden, kLowrank, kTokens);
        auto hyper = make_tensor(kCH, kTokens, hyper_storage);
        Tensor mixed = make_tensor(
            kHidden, kTokens, hyper_storage + sizeof(std::uint16_t));
        Tensor prior = make_tensor(kCH, kTokens, prior_storage);
        try {
            s->read(hyper, kTokens, default_block(), 1.0e-5F,
                    mixed, prior, ws, 0u);
            std::cerr << "FAIL alias: read with partial occupied-range "
                         "overlap did not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "overlapping occupied BF16 byte ranges") ==
                std::string::npos) {
                std::cerr << "FAIL alias: partial-overlap message = "
                          << e.what() << "\n";
                ++f;
            }
        }
    }

    failures += f;
    return f;
}

// (L) Sentinel infrastructure for callback host-rejection coverage:
//     a run functor that counts its invocations, so every rejection test
//     can prove the caller block was *not* dispatched (the callback gate
//     is pre-run; a rejected call must never invoke the caller block).
//     The sentinel is a function pointer (matching the frozen callback
//     signature) and increments a counter; no CUDA, no CUDA runtime, no
//     kernel launch — pure host-side proof.

struct SentinelRun {
    int calls = 0;
};

/// File-scope sentinel: the `sentinel_run_fn` functor increments this;
/// the tests read `calls` to prove the block was (or was not) dispatched.
static SentinelRun g_sentinel;

void sentinel_run_fn(const Tensor& /*mixed*/, Tensor& /*block*/,
                     cudaStream_t /*stream*/) {
    ++g_sentinel.calls;
}

/// Build a ready, phase-I (no pending prior) state with the bound geometry,
/// mirroring the consuming-gate helper but with the state's planes set so
/// the callback's alias/shape gates have a defined bound storage to test
/// against. The bound `hyper` plane is a 1-byte payload; `mixed_input` and
/// `prior_read` are set per-case to create the specific collision.
std::unique_ptr<HyperConnectionExecutorState> make_callback_state(
    char* hyper_payload, char* mixed_payload = nullptr,
    char* prior_payload = nullptr) {
    auto s = std::make_unique<HyperConnectionExecutorState>();
    s->ready = true;
    s->tokens = kTokens;
    s->hc_count = kStreams;
    s->hidden_size = kHidden;
    s->hc_lowrank = kLowrank;
    s->prior_read_valid = false;  // phase I
    s->hyper = make_tensor(kCH, kTokens, hyper_payload);
    if (mixed_payload != nullptr) {
        s->mixed_input = make_tensor(kHidden, kTokens, mixed_payload);
    }
    if (prior_payload != nullptr) {
        s->prior_read = make_tensor(kCH, kTokens, prior_payload);
    }
    return s;
}

/// Reset the sentinel counter to 0 (call before each sub-test so each
/// rejection is proven independently).
void reset_sentinel() { g_sentinel.calls = 0; }

/// Assert that the sentinel counter is `expected` (typically 0 for a
/// rejection: the block was never dispatched). Returns 1 on mismatch.
int check_sentinel(int expected, int& f, const char* label) {
    if (g_sentinel.calls != expected) {
        std::cerr << "FAIL " << label
                  << ": sentinel expected " << expected
                  << " calls, got " << g_sentinel.calls << "\n";
        ++f;
    }
    return (g_sentinel.calls != expected) ? 1 : 0;
}

// (M) Callback on a not-ready / reset state: the callback's first gate
//     (not-ready / reset) rejects before the run functor dispatches.
//     The sentinel proves the functor is never called.
int test_callback_not_ready_reset(int& failures) {
    int f = 0;
    alignas(16) char mixed_storage[kHiddenBytes] = {};
    alignas(16) char block_storage[kHiddenBytes] = {};
    auto mixed = make_tensor(kHidden, kTokens, mixed_storage);
    auto block = make_tensor(kHidden, kTokens, block_storage);

    // (1) Default (never-initialized) state: ready=false -> not-ready reject.
    reset_sentinel();
    {
        HyperConnectionExecutorState s;  // default: ready=false
        try {
            s.callback(mixed, block, 0u, sentinel_run_fn);
            std::cerr << "FAIL cb-notready: callback on default state did "
                         "not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "called on a not-ready / reset state") ==
                std::string::npos) {
                std::cerr << "FAIL cb-notready: default-state message = "
                          << e.what() << "\n";
                ++f;
            }
        }
        f += check_sentinel(0, f, "cb-notready/default");
    }

    // (2) Reset state (ready was cleared by reset()): the same not-ready
    //     gate rejects; the sentinel proves no dispatch.
    reset_sentinel();
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->reset();  // clears ready -> not-ready
        try {
            s->callback(mixed, block, 0u, sentinel_run_fn);
            std::cerr << "FAIL cb-notready: callback on reset state did "
                         "not reject\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "called on a not-ready / reset state") ==
                std::string::npos) {
                std::cerr << "FAIL cb-notready: reset-state message = "
                          << e.what() << "\n";
                ++f;
            }
        }
        f += check_sentinel(0, f, "cb-notready/reset");
    }
    failures += f;
    return f;
}

// (N) Callback `mixed` shape/token gate: a ready state rejects a `mixed`
//     whose shape or token extent differs from the bound [H, T].
int test_callback_mixed_gate(int& failures) {
    int f = 0;
    char payload = 0;
    char mixed_payload = 0;
    char block_payload = 0;
    auto block = make_tensor(kHidden, kTokens, &block_payload);

    // (1) Mixed with wrong rows (ne[0] != hidden_size).
    reset_sentinel();
    {
        auto s = make_callback_state(&payload, &mixed_payload,
                                     &block_payload);
        auto bad_mixed = make_tensor(kHidden + 1, kTokens, &mixed_payload);
        try {
            s->callback(bad_mixed, block, 0u, sentinel_run_fn);
            std::cerr << "FAIL cb-mixed: wrong-shape mixed accepted\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("callback `mixed` must be") ==
                std::string::npos) {
                std::cerr << "FAIL cb-mixed: shape message = " << e.what()
                          << "\n";
                ++f;
            }
        }
        f += check_sentinel(0, f, "cb-mixed/shape");
    }
    // (2) Mixed with wrong token extent (ne[1] != bound tokens).
    reset_sentinel();
    {
        auto s = make_callback_state(&payload, &mixed_payload,
                                     &block_payload);
        auto bad_mixed =
            make_tensor(kHidden, kTokens + 1, &mixed_payload);
        try {
            s->callback(bad_mixed, block, 0u, sentinel_run_fn);
            std::cerr << "FAIL cb-mixed: wrong-T mixed accepted\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("callback `mixed` must be") ==
                std::string::npos) {
                std::cerr << "FAIL cb-mixed: T message = " << e.what()
                          << "\n";
                ++f;
            }
        }
        f += check_sentinel(0, f, "cb-mixed/token");
    }
    // (3) Mixed with null payload (data == nullptr).
    reset_sentinel();
    {
        auto s = make_callback_state(&payload, &mixed_payload,
                                     &block_payload);
        Tensor null_mixed;  // data=nullptr, ne={0,0}
        try {
            s->callback(null_mixed, block, 0u, sentinel_run_fn);
            std::cerr << "FAIL cb-mixed: null mixed accepted\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "callback `mixed` payload is null") ==
                std::string::npos) {
                std::cerr << "FAIL cb-mixed: null message = " << e.what()
                          << "\n";
                ++f;
            }
        }
        f += check_sentinel(0, f, "cb-mixed/null");
    }
    failures += f;
    return f;
}

// (O) Callback `block` shape/shape-null gate: a ready state rejects a
//     `block` with wrong shape or null payload before the run dispatch.
int test_callback_block_gate(int& failures) {
    int f = 0;
    char payload = 0;
    char mixed_payload = 0;
    char block_payload = 0;
    auto mixed = make_tensor(kHidden, kTokens, &mixed_payload);

    // (1) Block with wrong shape (ne[0] != hidden_size).
    reset_sentinel();
    {
        auto s = make_callback_state(&payload, &mixed_payload,
                                     &block_payload);
        auto bad_block =
            make_tensor(kHidden + 1, kTokens, &block_payload);
        try {
            s->callback(mixed, bad_block, 0u, sentinel_run_fn);
            std::cerr << "FAIL cb-block: wrong-shape block accepted\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("callback `block` must be") ==
                std::string::npos) {
                std::cerr << "FAIL cb-block: shape message = " << e.what()
                          << "\n";
                ++f;
            }
        }
        f += check_sentinel(0, f, "cb-block/shape");
    }
    // (2) Block with wrong token extent (ne[1] != bound tokens).
    reset_sentinel();
    {
        auto s = make_callback_state(&payload, &mixed_payload,
                                     &block_payload);
        auto bad_block =
            make_tensor(kHidden, kTokens + 1, &block_payload);
        try {
            s->callback(mixed, bad_block, 0u, sentinel_run_fn);
            std::cerr << "FAIL cb-block: wrong-T block accepted\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("callback `block` must be") ==
                std::string::npos) {
                std::cerr << "FAIL cb-block: T message = " << e.what()
                          << "\n";
                ++f;
            }
        }
        f += check_sentinel(0, f, "cb-block/token");
    }
    // (3) Block with null payload (data == nullptr).
    reset_sentinel();
    {
        auto s = make_callback_state(&payload, &mixed_payload,
                                     &block_payload);
        Tensor null_block = make_tensor(kHidden, kTokens, nullptr);  // valid [H,T] shape, null payload
        try {
            s->callback(mixed, null_block, 0u, sentinel_run_fn);
            std::cerr << "FAIL cb-block: null block accepted\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find(
                    "callback `block` payload is null") ==
                std::string::npos) {
                std::cerr << "FAIL cb-block: null message = " << e.what()
                          << "\n";
                ++f;
            }
        }
        f += check_sentinel(0, f, "cb-block/null");
    }
    failures += f;
    return f;
}

// (P) Callback `block` alias gate: a `block` whose storage collides with
//     the bound hyper, mixed_input, or prior_read is rejected before the
//     run dispatch. The sentinel proves the block was never invoked.
int test_callback_alias_gate(int& failures) {
    int f = 0;
    alignas(16) char hyper_payload[kHyperBytes] = {};
    alignas(16) char mixed_payload[kHiddenBytes] = {};
    alignas(16) char prior_payload[kHyperBytes] = {};

    // (1) Block aliases the bound hyper storage.
    reset_sentinel();
    {
        auto s = make_callback_state(hyper_payload, mixed_payload,
                                     prior_payload);
        auto mixed = make_tensor(kHidden, kTokens, mixed_payload);
        auto bad_block =
            make_tensor(kHidden, kTokens, hyper_payload);  // == s->hyper.data
        try {
            s->callback(mixed, bad_block, 0u, sentinel_run_fn);
            std::cerr << "FAIL cb-alias: block aliasing hyper accepted\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("aliases the bound") ==
                std::string::npos) {
                std::cerr << "FAIL cb-alias: hyper message = " << e.what()
                          << "\n";
                ++f;
            }
        }
        f += check_sentinel(0, f, "cb-alias/hyper");
    }
    // (2) Block aliases the bound mixed_input storage.
    reset_sentinel();
    {
        auto s = make_callback_state(hyper_payload, mixed_payload,
                                     prior_payload);
        auto mixed = make_tensor(kHidden, kTokens, mixed_payload);
        auto bad_block =
            make_tensor(kHidden, kTokens, mixed_payload);  // == s->mixed.data
        try {
            s->callback(mixed, bad_block, 0u, sentinel_run_fn);
            std::cerr << "FAIL cb-alias: block aliasing mixed accepted\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("aliases the bound") ==
                std::string::npos) {
                std::cerr << "FAIL cb-alias: mixed message = " << e.what()
                          << "\n";
                ++f;
            }
        }
        f += check_sentinel(0, f, "cb-alias/mixed");
    }
    // (3) Block aliases the bound prior_read storage (phase-P state where
    //     prior_read.data is non-null).
    reset_sentinel();
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = true;  // phase P: prior_read is live
        s->hyper = make_tensor(kCH, kTokens, hyper_payload);
        s->mixed_input = make_tensor(kHidden, kTokens, mixed_payload);
        s->prior_read = make_tensor(kCH, kTokens, prior_payload);
        auto mixed = make_tensor(kHidden, kTokens, mixed_payload);
        auto bad_block =
            make_tensor(kHidden, kTokens, prior_payload);  // == s->prior.data
        try {
            s->callback(mixed, bad_block, 0u, sentinel_run_fn);
            std::cerr << "FAIL cb-alias: block aliasing prior accepted\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("aliases the bound") ==
                std::string::npos) {
                std::cerr << "FAIL cb-alias: prior message = " << e.what()
                          << "\n";
                ++f;
            }
        }
        f += check_sentinel(0, f, "cb-alias/prior");
    }
    failures += f;
    return f;
}

// (M) Correctly-bound read/inject reach the later eps gate, not the alias
//     gate: the fixed state.hyper identity means the alias check sees only
//     distinct planes, so a legal bound call passes the alias stage and is
//     rejected (or accepted) by the *next* gate in the sequence.
//     This is the Astra R4 regression: before the fix, the alias gate
//     rejected every correctly bound call because it included
//     this->hyper.data alongside hyper.data (identical pointers).
int test_correctly_bound_reach_eps(int& failures) {
    int f = 0;
    alignas(16) char hyper_storage[kHyperBytes] = {};
    alignas(16) char mixed_storage[kHiddenBytes] = {};
    alignas(16) char prior_storage[kHyperBytes] = {};
    alignas(16) char block_storage[kHiddenBytes] = {};
    auto ws = make_arena_adequate(kStreams, kHidden, kLowrank, kTokens);

    // (M.1) read: state.hyper == supplied hyper (correctly bound). The
    //         alias gate must NOT fire (state.hyper is not redundantly
    //         included). Instead, a wrong eps rejects at the eps gate.
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = false;
        s->hyper = make_tensor(kCH, kTokens, hyper_storage);
        auto hyper = make_tensor(kCH, kTokens, hyper_storage);  // == state.hyper
        Tensor mixed = make_tensor(kHidden, kTokens, mixed_storage);
        Tensor prior = make_tensor(kCH, kTokens, prior_storage);
        // Use wrong eps (2e-5) so the call passes the alias gate (no
        // collision) and is rejected at the eps gate.
        try {
            s->read(hyper, kTokens, default_block(), 2.0e-5F,
                   mixed, prior, ws, 0u);
            std::cerr << "FAIL M1: read with correct bind + wrong eps "
                         "did not reject at eps gate\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            auto msg = std::string(e.what());
            if (msg.find("independent planes are aliased") !=
                std::string::npos) {
                std::cerr << "FAIL M1: read correctly-bound call was "
                             "rejected by ALIAS gate (the bug)\n";
                ++f;
            } else if (msg.find("read `eps` must equal") ==
                       std::string::npos) {
                std::cerr << "FAIL M1: read message = " << e.what() << "\n";
                ++f;
            }
            // else: correctly rejected at eps gate (expected)
        }
    }

    // (M.2) inject: state.hyper == supplied hyper (correctly bound).
    //         Same reasoning: alias gate passes, eps gate rejects.
    {
        auto s = std::make_unique<HyperConnectionExecutorState>();
        s->ready = true;
        s->tokens = kTokens;
        s->hc_count = kStreams;
        s->hidden_size = kHidden;
        s->hc_lowrank = kLowrank;
        s->prior_read_valid = true;  // phase P: inject is legal
        s->hyper = make_tensor(kCH, kTokens, hyper_storage);
        auto hyper = make_tensor(kCH, kTokens, hyper_storage);  // == state.hyper
        auto block = make_tensor(kHidden, kTokens, block_storage);
        try {
            s->inject(block, hyper, kTokens, default_block(), -1.0F,
                      ws, 0u);
            std::cerr << "FAIL M2: inject with correct bind + wrong eps "
                         "did not reject at eps gate\n";
            ++f;
        } catch (const std::invalid_argument& e) {
            auto msg = std::string(e.what());
            if (msg.find("independent planes are aliased") !=
                std::string::npos) {
                std::cerr << "FAIL M2: inject correctly-bound call was "
                             "rejected by ALIAS gate (the bug)\n";
                ++f;
            } else if (msg.find("inject `eps` must equal") ==
                       std::string::npos) {
                std::cerr << "FAIL M2: inject message = " << e.what() << "\n";
                ++f;
            }
        }
    }
    failures += f;
    return f;
}

} // namespace

int main() {
    std::cout
        << "=== Batch 3C2 / M2: HyperConnection executor CPU/static "
           "validation (non-GPU) ===\n";
    int failures = 0;

    failures += test_not_ready_reject(failures);
    failures += test_token_mismatch_reject(failures);
    failures += test_phase_p_reject(failures);
    failures += test_inject_without_prior_reject(failures);
    failures += test_gate_order_token_before_phase(failures);
    failures += test_eps_reject(failures);
    failures += test_reinit_reject(failures);
    failures += test_reset_clears(failures);
    failures += test_initialize_reject_gates(failures);
    failures += test_capacity_definition(failures);
    failures += test_undersized_workspace_reject(failures);
    failures += test_storage_mismatch_reject(failures);
    failures += test_alias_reject(failures);
    failures += test_callback_not_ready_reset(failures);
    failures += test_callback_mixed_gate(failures);
    failures += test_callback_block_gate(failures);
    failures += test_callback_alias_gate(failures);
    failures += test_correctly_bound_reach_eps(failures);

    std::cout << (failures == 0 ? "OK" : "FAIL")
              << " hyper_connection_executor M2 CPU/static validation\n";
    return failures == 0 ? 0 : 1;
}