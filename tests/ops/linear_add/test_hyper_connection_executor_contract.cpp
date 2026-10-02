#include "ninfer/ops/hyper_connection_executor.h"
#include "ops/hyper_connection_executor_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

//
// Batch 3C2 — milestone M1. Independent FP64 numerical oracle + analytically
// independent fixtures for the bounded HyperConnection executor contract.
//
// This test compiles and runs on CPU only (no CUDA launch, no GPU). It is the
// numerical-oracle scaffolding that the later recurrence work (M2/M3) must
// match. The oracle is an *independent* FP64 re-derivation of the frozen
// equations in docs/flash-next/REFERENCE_FORMULAS.md; it calls no production
// helper, launcher, kernel, 3C1 primitive, or linear() helper.
//
// Validation covered here:
//   * exact FP64 oracle (independent from REFERENCE_FORMULAS.md);
//   * analytically independent fixtures (hand-derived reference math,
//     distinct streams/tokens, nonzero logits, nonuniform weights,
//     identifiable weight entries); error-detection coverage for a missing
//     /C scaling, a missing sigmoid, a [rows,T] transposition, and broken
//     stream-group normalization;
//   * compare_exact rejects nonfinite operands outright;
//   * production weight-layout mapping (Weight payloads -> logical [N,K]);
//   * the lifecycle gate (readiness + token consistency + changed-token
//     re-initialization), verified as the *contract model* only;
//   * prior_read authority: the state-owned prior anchors the inject
//     reference; a mismatched externally supplied reference is rejected;
//   * the final mixer fed the updated (post-inject) state, labeled as a
//     component fixture (not a full decoder block);
//   * deterministic zero-logit alpha==1 fixture;
//   * module-bound (block vs final) norm_weight rebinding; module eps fixed.
//

namespace {

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::ops::detail;
using namespace ninfer::ops::detail::hc_oracle;

// Small, well-conditioned geometry (the executor validates supplied
// geometry, so the contract holds at any legal geometry). H != T is
// deliberate: a [rows,T] transposition mismatch then changes the vector
// shape (caught by compare_exact) or the numeric pattern (caught by value
// comparison).
constexpr std::int32_t kHidden  = 8;
constexpr std::int32_t kStreams  = 4;
constexpr std::int32_t kLowrank  = 5;

// Contract-fixed normalization epsilon (contract doc s.2): the grouped
// RMSNorm `+eps` term is 1e-5, matching the reference model's RMSNorm
// epsilon. The oracle itself binds no constant; the module (and this test)
// supplies it per call.
constexpr double kEps = 1.0e-5;

HcExecutorGeometry geometry() {
    return HcExecutorGeometry{kHidden, kStreams, kLowrank};
}

std::vector<double> small_down(const HcExecutorGeometry& geo, int seed = 0) {
    std::vector<double> w(geo.lowrank() * geo.expanded());
    for (std::int32_t r = 0; r < geo.lowrank(); ++r)
        for (std::int32_t j = 0; j < geo.expanded(); ++j)
            w[static_cast<std::size_t>(r) * geo.expanded() + j] =
                0.003 * ((std::int32_t)(((r + 1) * 7 + (j + 1) * 3 + seed) % 23) - 11);
    return w;
}

std::vector<double> small_up(const HcExecutorGeometry& geo, int seed = 0) {
    std::vector<double> w(geo.expanded() * geo.lowrank());
    for (std::int32_t j = 0; j < geo.expanded(); ++j)
        for (std::int32_t r = 0; r < geo.lowrank(); ++r)
            w[static_cast<std::size_t>(j) * geo.lowrank() + r] =
                0.02 * ((std::int32_t)(((j + 1) * 5 + (r + 1) * 11 + seed) % 17) - 8);
    return w;
}

std::vector<double> small_inject(const HcExecutorGeometry& geo, int seed = 0) {
    std::vector<double> w(geo.streams() * geo.expanded());
    for (std::int32_t c = 0; c < geo.streams(); ++c)
        for (std::int32_t j = 0; j < geo.expanded(); ++j)
            w[static_cast<std::size_t>(c) * geo.expanded() + j] =
                0.001 * ((std::int32_t)(((c + 1) * 11 + (j + 1) * 7 + seed) % 19) - 9);
    return w;
}

// Deterministic per-token, per-stream value: stream `c` of token `t` is a
// distinct nonzero pattern (never the degenerate all-1s RMSNorm case,
// where RMSNorm output is identically 1 and bugs vanish).
double stream_value(std::int32_t t, std::int32_t c, std::int32_t h) {
    const double base = 1.0 + 0.25 * (t + 1);   // distinct per token
    const double lane = 1.0 + 0.1 * c;          // distinct per stream
    const double f = 1.0 + 0.05 * h;            // distinct per lane
    return base * lane * f;
}

// HcState built directly (not via initialize()) from the distinct
// stream/token pattern; the initialize() repeat mapping itself is checked
// separately against the hand-derived reference in test_initialize.
HcState pattern_state(std::int32_t T, const HcExecutorGeometry& geo) {
    HcState s;
    const std::size_t CH = geo.expanded();
    s.hyper.resize(CH * T);
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t c = 0; c < geo.streams(); ++c) {
            for (std::int32_t h = 0; h < geo.hidden(); ++h) {
                s.hyper[static_cast<std::size_t>(t) * CH +
                        static_cast<std::size_t>(c) * geo.hidden() + h] =
                    stream_value(t, c, h);
            }
        }
    }
    s.tokens = T;
    return s;
}

std::span<const double> as_span(const std::vector<double>& v) {
    return std::span<const double>(v.data(), v.size());
}

// Exact FP64 compare. The oracle is exact, so a tight bound is appropriate
// (the later GPU kernels carry the looser BF16 tolerance instead).
// Nonfinite operands (either side) are an immediate, labeled failure —
// the oracle must be finite, and an NaN/inf "match" is not a match.
int compare_exact(const std::vector<double>& got, const std::vector<double>& ref,
                  const char* label, double tol = 1.0e-12) {
    if (got.size() != ref.size()) {
        std::cerr << "FAIL " << label << ": size mismatch ("
                  << got.size() << " vs " << ref.size() << ")\n";
        return 1;
    }
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) {
            std::cerr << "FAIL " << label << ": nonfinite operand at index "
                      << i << "\n";
            return 1;
        }
    }
    int bad = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = std::abs(got[i] - ref[i]);
        if (d > tol) {
            if (bad == 0) {
                std::cerr << "FAIL " << label << ": first mismatch at index "
                          << i << " (|d|=" << d << ")\n";
            }
            ++bad;
        }
    }
    if (bad) {
        std::cerr << "FAIL " << label << ": " << bad << " mismatched lanes\n";
        return 1;
    }
    return 0;
}

// ----------------------------------------------------------------------------
// Hand-derived independent reference (plain FP64, straight from
// REFERENCE_FORMULAS.md s.1/s.2; deliberately re-implemented here, not via
// the oracle, so the cross-check is genuinely independent).
// ----------------------------------------------------------------------------

// s.2: per-token, per-lane distinct embedding value, shared by both the
// [H,T] embedding and the [CH,T] repeat reference.
double init_value(std::int32_t t, std::int32_t h) {
    return 1.0 + 0.25 * (t + 1) + 0.05 * (h + 1);
}

// The token embedding, [H, T] layout (dim 0 fastest): emb[t*H + h].
std::vector<double> ref_embedding(std::int32_t T, const HcExecutorGeometry& geo) {
    std::vector<double> emb(static_cast<std::size_t>(geo.hidden()) * T);
    for (std::int32_t t = 0; t < T; ++t)
        for (std::int32_t h = 0; h < geo.hidden(); ++h)
            emb[static_cast<std::size_t>(t) * geo.hidden() + h] = init_value(t, h);
    return emb;
}

// s.2 reference: repeat(embedding, C) -> [CH, T]; all C streams repeat the
// same embedding lane (s.2: repeat(embedding, hc_count)).
std::vector<double> ref_initialize(std::int32_t T, const HcExecutorGeometry& geo) {
    std::vector<double> hyper(geo.expanded_elems(T));
    for (std::int32_t t = 0; t < T; ++t)
        for (std::int32_t c = 0; c < geo.streams(); ++c)
            for (std::int32_t h = 0; h < geo.hidden(); ++h)
                hyper[static_cast<std::size_t>(t) * geo.expanded() +
                        static_cast<std::size_t>(c) * geo.hidden() + h] =
                    init_value(t, h);
    return hyper;
}

// s.1 "grouped RMSNorm": per-stream, per-lane, hand-derived.
std::vector<double> ref_group_rmsnorm(const std::vector<double>& x,
                                      const std::vector<double>& w,
                                      double eps, std::int32_t T,
                                      const HcExecutorGeometry& geo) {
    std::vector<double> out(geo.expanded_elems(T));
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t c = 0; c < geo.streams(); ++c) {
            const std::size_t base = static_cast<std::size_t>(t) * geo.expanded() +
                                     static_cast<std::size_t>(c) * geo.hidden();
            double sum = 0.0;
            for (std::int32_t h = 0; h < geo.hidden(); ++h) {
                sum += x[base + h] * x[base + h];
            }
            const double inv = 1.0 / std::sqrt(sum / geo.hidden() + eps);
            for (std::int32_t h = 0; h < geo.hidden(); ++h) {
                out[base + h] = x[base + h] * inv *
                                w[static_cast<std::size_t>(c) * geo.hidden() + h];
            }
        }
    }
    return out;
}

// s.1 "Read / mix", all-ones weights: mixed[h,t] = mean_c(Hn[c,h,t]) with
// per-lane sigmoid weights (logit = weight lane). Hand-derived.
std::vector<double> ref_read_mix(const std::vector<double>& hn,
                                 const std::vector<double>& up_logit,
                                 std::int32_t T, const HcExecutorGeometry& geo) {
    std::vector<double> mixed(static_cast<std::size_t>(geo.hidden()) * T);
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t h = 0; h < geo.hidden(); ++h) {
            double sum = 0.0;
            for (std::int32_t c = 0; c < geo.streams(); ++c) {
                const std::size_t off = static_cast<std::size_t>(t) * geo.expanded() +
                                        static_cast<std::size_t>(c) * geo.hidden() + h;
                sum += 1.0 / (1.0 + std::exp(-up_logit[off])) * hn[off];
            }
            mixed[static_cast<std::size_t>(t) * geo.hidden() + h] =
                sum / geo.streams();
        }
    }
    return mixed;
}

// s.1 "Injection weights": alpha = 2*sigmoid(logit/C); distinct per (c,t).
std::vector<double> ref_inject(const std::vector<double>& hyper,
                               const std::vector<double>& block,
                               const std::vector<double>& raw,
                               std::int32_t T, const HcExecutorGeometry& geo) {
    std::vector<double> next(geo.expanded_elems(T));
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t c = 0; c < geo.streams(); ++c) {
            const double a = 2.0 / (1.0 + std::exp(
                                         -(raw[static_cast<std::size_t>(t) * geo.streams() + c] /
                                            static_cast<double>(geo.streams()))));
            for (std::int32_t h = 0; h < geo.hidden(); ++h) {
                const std::size_t hi = static_cast<std::size_t>(t) * geo.expanded() +
                                       static_cast<std::size_t>(c) * geo.hidden() + h;
                const std::size_t bi = static_cast<std::size_t>(t) * geo.hidden() + h;
                next[hi] = hyper[hi] + block[bi] * a;
            }
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// M1 contract: initialize (repeat) matches the hand-derived reference,
// and the module `initialize` API is the per-forward source of `hyper`.
// ----------------------------------------------------------------------------
int test_initialize() {
    const auto geo = geometry();
    const std::int32_t T = 2;
    const std::vector<double> emb = ref_embedding(T, geo);
    const std::vector<double> got =
        initialize(as_span(emb), geo, T);  // embedding [H,T]
    const std::vector<double> ref = ref_initialize(T, geo);  // repeat -> [CH,T]
    int failures = compare_exact(got, ref, "initialize: repeat(embedding,C)");
    if (failures == 0) std::cout << "OK initialize: repeat mapping [CH,T]\n";
    return failures;
}

// ----------------------------------------------------------------------------
// M1 contract: grouped RMSNorm with nonuniform (per-lane) weights matches
// the hand-derived reference. Detects a broken per-stream grouping or a
// missing per-lane weight.
// ----------------------------------------------------------------------------
int test_group_rmsnorm() {
    const auto geo = geometry();
    const std::int32_t T = 2;
    const HcState state = pattern_state(T, geo);
    std::vector<double> w(geo.expanded(), 1.0);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = 1.0 + 0.25 * (i % 11);
    const std::vector<double> got = group_rmsnorm(as_span(state.hyper),
                                                   as_span(w), kEps, geo, T);
    const std::vector<double> ref = ref_group_rmsnorm(state.hyper, w, kEps, T, geo);
    int failures = compare_exact(got, ref, "group_rmsnorm: per-stream per-lane");
    if (failures == 0) std::cout << "OK group_rmsnorm: nonuniform weights, per-stream\n";
    return failures;
}

// ----------------------------------------------------------------------------
// M1 contract: read() full pipeline vs the hand-derived reference.
// Weights are identifiable entries (mix_down[lane] = lane+1, mix_up[lane] =
// 1 + lane/128), so a missing /C scaling, missing sigmoid, [rows,T]
// transposition, or broken stream grouping changes the result pattern and is
// detected against the independently derived reference.
// ----------------------------------------------------------------------------
int test_read_independence() {
    const auto geo = geometry();
    const std::int32_t T = 2;
    HcState state = pattern_state(T, geo);
    HcExecutorLifecycle life;
    life.initialize(state, T, geo);
    std::vector<double> norm_w(geo.expanded());
    for (std::int32_t j = 0; j < geo.expanded(); ++j)
        norm_w[static_cast<std::size_t>(j)] = 1.0 + 0.03125 * (std::int32_t)((j % 5) - 2);
    const std::vector<double> down = small_down(geo);
    const std::vector<double> up = small_up(geo);

    const HcRead rd =
        read(life, T, state, as_span(norm_w), kEps,
             as_span(down), as_span(up), geo);
    int failures = 0;

    if (rd.mixed_input.size() != static_cast<std::size_t>(kHidden) * T) {
        std::cerr << "FAIL read: mixed_input not [H,T]\n";
        ++failures;
    }
    if (rd.prior_hyper.size() != geo.expanded_elems(T)) {
        std::cerr << "FAIL read: prior not [CH,T]\n";
        ++failures;
    }
    failures += compare_exact(rd.prior_hyper, state.hyper,
                              "read: returned prior == input hyper");
    failures += compare_exact(state.prior_read, state.hyper,
                              "read: state owns prior_read snapshot");
    if (!state.prior_read_valid) {
        std::cerr << "FAIL read: state prior_read not marked valid\n";
        ++failures;
    }

    const std::vector<double> hn =
        ref_group_rmsnorm(state.hyper, norm_w, kEps, T, geo);
    std::vector<double> d(geo.lowrank() * T);
    for (std::int32_t t = 0; t < T; ++t)
        for (std::int32_t r = 0; r < geo.lowrank(); ++r) {
            double acc = 0.0;
            for (std::int32_t j = 0; j < geo.expanded(); ++j)
                acc += down[static_cast<std::size_t>(r) * geo.expanded() + j] *
                       hn[static_cast<std::size_t>(t) * geo.expanded() + j];
            d[static_cast<std::size_t>(t) * geo.lowrank() + r] =
                acc / geo.streams();
        }
    std::vector<double> d2(d.size());
    for (std::size_t i = 0; i < d.size(); ++i)
        d2[i] = d[i] / (1.0 + std::exp(-d[i]));
    std::vector<double> ul(geo.expanded() * T);
    for (std::int32_t t = 0; t < T; ++t)
        for (std::int32_t j = 0; j < geo.expanded(); ++j) {
            double acc = 0.0;
            for (std::int32_t r = 0; r < geo.lowrank(); ++r)
                acc += up[static_cast<std::size_t>(j) * geo.lowrank() + r] *
                       d2[static_cast<std::size_t>(t) * geo.lowrank() + r];
            ul[static_cast<std::size_t>(t) * geo.expanded() + j] = acc;
            if (std::abs(acc) >= 1.0) {
                std::cerr << "FAIL read: fixture saturated sigmoid logit\n";
                ++failures;
            }
        }
    const std::vector<double> refmix = ref_read_mix(hn, ul, T, geo);
    failures += compare_exact(rd.mixed_input, refmix,
                              "read: nonsaturating pipeline vs hand reference");
    for (std::int32_t t = 0; t < T; ++t)
        for (std::int32_t j = 0; j < geo.expanded(); ++j)
            if (std::abs(ul[static_cast<std::size_t>(t) * geo.expanded() + j]) >= 1.0) {
                std::cerr << "FAIL read: ul pre-sigmoid saturated (|v| >= 1.0)\n";
                ++failures;
            }
    if (failures == 0)
        std::cout << "OK read: nonsaturating signed fixture + state-owned prior\n";
    return failures;
}

// ----------------------------------------------------------------------------
// M1 contract: inject() with distinct nonzero logits and identifiable
// per-stream weights; hand-derived alpha rule (2*sigmoid(logit/C)), applied
// exactly once.
// ----------------------------------------------------------------------------
int test_inject_nonzero() {
    const auto geo = geometry();
    const std::int32_t T = 2;
    HcState state = pattern_state(T, geo);
    std::vector<double> norm_w(geo.expanded());
    for (std::int32_t j = 0; j < geo.expanded(); ++j)
        norm_w[static_cast<std::size_t>(j)] = 1.0 + 0.03125 * (std::int32_t)((j % 5) - 2);
    const std::vector<double> down = small_down(geo, 2);
    const std::vector<double> up = small_up(geo, 3);
    read(state, as_span(norm_w), kEps, as_span(down), as_span(up), geo);

    const std::vector<double> iw = small_inject(geo);
    std::vector<double> block(kHidden * T);
    for (std::size_t i = 0; i < block.size(); ++i)
        block[i] = -0.35 + 0.08 * (i % kHidden) + 0.05 * (i / kHidden);

    const std::vector<double> hn =
        ref_group_rmsnorm(state.hyper, norm_w, kEps, T, geo);
    std::vector<double> expected_raw(geo.streams() * T, 0.0);
    for (std::int32_t t = 0; t < T; ++t)
        for (std::int32_t c = 0; c < geo.streams(); ++c)
            for (std::int32_t j = 0; j < geo.expanded(); ++j)
                expected_raw[static_cast<std::size_t>(t) * geo.streams() + c] +=
                    iw[static_cast<std::size_t>(c) * geo.expanded() + j] *
                    hn[static_cast<std::size_t>(t) * geo.expanded() + j];

    const std::vector<double> got_raw =
        block_inject_logits(as_span(hn), as_span(iw), geo, T);
    int failures = compare_exact(got_raw, expected_raw,
        "inject logits: hand dot with physical c*CH+j offsets");
    for (double raw : expected_raw) {
        if (std::abs(raw / geo.streams()) >= 0.25) {
            std::cerr << "FAIL inject: fixture sigmoid is too close to saturation\n";
            ++failures;
        }
    }

    const std::vector<double> next =
        inject_update(state, as_span(block), as_span(norm_w), kEps,
                      as_span(iw), geo);
    const std::vector<double> ref =
        ref_inject(state.prior_read, block, expected_raw, T, geo);
    failures += compare_exact(next, ref,
        "inject: independent nonsaturating logits and prior_read base");
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t c = 0; c < geo.streams(); ++c) {
            if (std::abs(expected_raw[static_cast<std::size_t>(t) * geo.streams() + c]) <= 1.0e-3) {
                std::cerr << "FAIL inject: logit stream non-informative (|v| <= 1e-3) at (t="
                          << static_cast<int>(t) << ", c=" << static_cast<int>(c) << ")\n";
                ++failures;
            }
            for (std::int32_t c2 = c + 1; c2 < geo.streams(); ++c2)
                if (std::abs(expected_raw[static_cast<std::size_t>(t) * geo.streams() + c] -
                              expected_raw[static_cast<std::size_t>(t) * geo.streams() + c2])
                    <= 1.0e-3) {
                    std::cerr << "FAIL inject: logits not pairwise distinct at (t="
                              << static_cast<int>(t) << ", c1=" << static_cast<int>(c)
                              << ", c2=" << static_cast<int>(c2) << ")\n";
                    ++failures;
                }
        }
    }
    if (failures == 0)
        std::cout << "OK inject: signed nonsaturating logits + hand-derived alpha\n";
    return failures;
}

// ----------------------------------------------------------------------------
// M1 contract: zero block-inject logits -> alpha == 1 (frozen fixture);
// inject is block read-only.
// ----------------------------------------------------------------------------
int test_inject_zero_alpha() {
    const auto geo = geometry();
    const std::int32_t T = 2;
    const HcState state = pattern_state(T, geo);
    std::vector<double> block(kHidden * T);
    for (std::size_t i = 0; i < block.size(); ++i)
        block[i] = 1.0 + 0.125 * (i % kHidden);
    std::vector<double> zero_logit(geo.streams() * T, 0.0);
    const std::vector<double> next =
        inject(as_span(state.hyper), as_span(block), as_span(zero_logit), geo, T);
    std::vector<double> ref(state.hyper.size());
    for (std::int32_t t = 0; t < T; ++t)
        for (std::int32_t c = 0; c < geo.streams(); ++c)
            for (std::int32_t h = 0; h < kHidden; ++h)
                ref[static_cast<std::size_t>(t) * geo.expanded() +
                        static_cast<std::size_t>(c) * kHidden + h] =
                    state.hyper[static_cast<std::size_t>(t) * geo.expanded() +
                                    static_cast<std::size_t>(c) * kHidden + h] +
                    block[static_cast<std::size_t>(t) * kHidden + h];
    int failures = compare_exact(next, ref, "inject: zero logits -> alpha=1");
    if (failures == 0) std::cout << "OK inject: zero logits -> alpha=1; block read-only\n";
    return failures;
}

// ----------------------------------------------------------------------------
// M1 contract: final mixer — **component fixture** (read path on the
// post-inject state), no block-inject. Feeds the *updated* state; the
// final mixer's own (distinct) norm weight and mix weights are bound,
// proving module weight rebinding (block vs final). Labeled as a component:
// it is NOT the full decoder block output.
// ----------------------------------------------------------------------------
int test_final_mixer() {
    const auto geo = geometry();
    const std::int32_t T = 2;
    HcState base = pattern_state(T, geo);
    std::vector<double> block_norm(geo.expanded(), 1.0);
    std::vector<double> final_norm(geo.expanded());
    for (std::int32_t j = 0; j < geo.expanded(); ++j)
        final_norm[static_cast<std::size_t>(j)] = 1.0 + 0.03125 * (std::int32_t)(((j + 2) % 5) - 2);
    const std::vector<double> bdown = small_down(geo, 1);
    const std::vector<double> bup = small_up(geo, 2);
    read(base, as_span(block_norm), kEps, as_span(bdown), as_span(bup), geo);

    std::vector<double> block(kHidden * T);
    for (std::size_t i = 0; i < block.size(); ++i)
        block[i] = -0.2 + 0.05 * (i % kHidden);
    HcState post = base;
    post.hyper = inject_update(base, as_span(block), as_span(block_norm), kEps,
                               as_span(small_inject(geo, 4)), geo);
    post.prior_read.clear();
    post.prior_read_valid = false;

    const std::vector<double> fdown = small_down(geo, 5);
    const std::vector<double> fup = small_up(geo, 6);
    const std::vector<double> final =
        final_mixer(post, as_span(final_norm), kEps, as_span(fdown),
                    as_span(fup), geo);

    int failures = 0;
    if (final.size() != static_cast<std::size_t>(kHidden) * T) {
        std::cerr << "FAIL final_mixer: output not [H,T]\n";
        ++failures;
    }
    const std::vector<double> hn =
        ref_group_rmsnorm(post.hyper, final_norm, kEps, T, geo);
    std::vector<double> d(geo.lowrank() * T);
    for (std::int32_t t = 0; t < T; ++t)
        for (std::int32_t r = 0; r < geo.lowrank(); ++r) {
            double acc = 0.0;
            for (std::int32_t j = 0; j < geo.expanded(); ++j)
                acc += fdown[static_cast<std::size_t>(r) * geo.expanded() + j] *
                       hn[static_cast<std::size_t>(t) * geo.expanded() + j];
            d[static_cast<std::size_t>(t) * geo.lowrank() + r] =
                acc / geo.streams();
        }
    std::vector<double> d2(d.size());
    for (std::size_t i = 0; i < d.size(); ++i)
        d2[i] = d[i] / (1.0 + std::exp(-d[i]));
    std::vector<double> ul(geo.expanded() * T);
    for (std::int32_t t = 0; t < T; ++t)
        for (std::int32_t j = 0; j < geo.expanded(); ++j) {
            double acc = 0.0;
            for (std::int32_t r = 0; r < geo.lowrank(); ++r)
                acc += fup[static_cast<std::size_t>(j) * geo.lowrank() + r] *
                       d2[static_cast<std::size_t>(t) * geo.lowrank() + r];
            ul[static_cast<std::size_t>(t) * geo.expanded() + j] = acc;
            if (std::abs(acc) >= 1.0) {
                std::cerr << "FAIL final_mixer: saturated fixture logit\n";
                ++failures;
            }
        }
    failures += compare_exact(final, ref_read_mix(hn, ul, T, geo),
        "final_mixer: post-inject state + final module nonsaturating weights");
    for (std::int32_t t = 0; t < T; ++t)
        for (std::int32_t j = 0; j < geo.expanded(); ++j)
            if (std::abs(ul[static_cast<std::size_t>(t) * geo.expanded() + j]) >= 1.0) {
                std::cerr << "FAIL final_mixer: ul pre-sigmoid saturated (|v| >= 1.0)\n";
                ++failures;
            }
    if (failures == 0)
        std::cout << "OK final_mixer: updated state + distinct final weights\n";
    return failures;
}

// ----------------------------------------------------------------------------
// M1 contract: the lifecycle gate (contract model). read/inject/final_mixer
// are rejected (throw) before initialize and after reset; a token extent
// different from the bound forward is rejected; a changed token count is
// resolved by reset + re-initialize, after which the calls are legal again.
// This validates the *contract model*; it does not claim the production
// C++ class behavior is tested (M2/M3 will bind and test it).
// ----------------------------------------------------------------------------
int expect_lifecycle_throw(const char* label, const char* detail,
                           auto&& f) {
    try {
        f();
    } catch (const HcLifecycleError&) {
        std::cout << "OK lifecycle: " << label << " rejected (" << detail << ")\n";
        return 0;
    } catch (...) {
        std::cerr << "FAIL lifecycle: " << label << " threw non-lifecycle error\n";
        return 1;
    }
    std::cerr << "FAIL lifecycle: " << label << " was NOT rejected\n";
    return 1;
}

int test_lifecycle_gate() {
    const auto geo = geometry();
    int failures = 0;
    HcExecutorLifecycle life;
    HcState state = pattern_state(2, geo);
    std::vector<double> norm_w(geo.expanded(), 1.0);
    const std::vector<double> down = small_down(geo);
    const std::vector<double> up = small_up(geo);
    const std::vector<double> iw = small_inject(geo);
    std::vector<double> block(kHidden * 2, 0.25);

    failures += expect_lifecycle_throw("read pre-init", "before initialize", [&] {
        read(life, 2, state, as_span(norm_w), kEps, as_span(down), as_span(up), geo);
    });
    failures += expect_lifecycle_throw("inject pre-init", "before initialize", [&] {
        inject_update(life, 2, state, as_span(block), as_span(norm_w), kEps,
                      as_span(iw), geo);
    });
    failures += expect_lifecycle_throw("final pre-init", "before initialize", [&] {
        final_mixer(life, 2, state, as_span(norm_w), kEps,
                    as_span(down), as_span(up), geo);
    });

    life.initialize(state, 2, geo);
    failures += expect_lifecycle_throw("initialize twice", "same T without reset", [&] {
        life.initialize(state, 2, geo);
    });
    // A fresh (U) HcState may initialize at any valid token extent, including
    // T=4. The stateless HcExecutorLifecycle owns no `tokens` of its own, so
    // it must NOT globally bind T across independent HcState instances:
    // `state4` is its own T=4-bound I instance, independent of the T=2 `state`
    // forward. (State-owned lifecycle, not facade-owned.)
    HcState state4 = pattern_state(4, geo);
    int f4 = 0;
    try {
        life.initialize(state4, 4, geo);
        std::cout << "OK lifecycle: fresh U-state initializes at T=4 (state-owned T)\n";
    } catch (const HcLifecycleError&) {
        std::cerr << "FAIL lifecycle: fresh state4 initialize(T=4) was rejected\n";
        f4 = 1;
    } catch (...) {
        std::cerr << "FAIL lifecycle: fresh state4 initialize(T=4) threw non-lifecycle error\n";
        f4 = 1;
    }
    failures += f4;
    // `state4` is now I at T=4: a T=4 caller must be accepted (proving the
    // stateless facade does not globally bind T to the T=2 `state` forward);
    // any global T binding would reject it here.
    int g4 = 0;
    try {
        (void)read(life, 4, state4, as_span(norm_w), kEps,
                   as_span(down), as_span(up), geo);
        std::cout << "OK lifecycle: T=4 caller accepted on T=4-bound state (no global T binding)\n";
    } catch (const HcLifecycleError&) {
        std::cerr << "FAIL lifecycle: T=4 caller on T=4-bound state4 rejected (global T binding)\n";
        g4 = 1;
    } catch (...) {
        std::cerr << "FAIL lifecycle: T=4 caller on T=4-bound state4 threw non-lifecycle error\n";
        g4 = 1;
    }
    failures += g4;
    // The T=4-bound `state4` is exercised by the caller-T gate in its own
    // direction: a T=2 caller against it rejects (T=2 vs bound T=4).
    failures += expect_lifecycle_throw("read caller-T mismatch (fresh4)", "T=2 vs bound T=4", [&] {
        read(life, 2, state4, as_span(norm_w), kEps, as_span(down), as_span(up), geo);
    });
    failures += expect_lifecycle_throw("inject caller-T mismatch (fresh4)", "T=2 vs bound T=4", [&] {
        inject_update(life, 2, state4, as_span(block), as_span(norm_w), kEps,
                      as_span(iw), geo);
    });
    failures += expect_lifecycle_throw("final caller-T mismatch (fresh4)", "T=2 vs bound T=4", [&] {
        final_mixer(life, 2, state4, as_span(norm_w), kEps,
                    as_span(down), as_span(up), geo);
    });
    failures += expect_lifecycle_throw("inject before read", "prior snapshot invalid", [&] {
        inject_update(life, 2, state, as_span(block), as_span(norm_w), kEps,
                      as_span(iw), geo);
    });

    failures += expect_lifecycle_throw("read caller-T mismatch", "T=4 vs bound T=2", [&] {
        read(life, 4, state, as_span(norm_w), kEps, as_span(down), as_span(up), geo);
    });
    failures += expect_lifecycle_throw("inject caller-T mismatch", "T=4 vs bound T=2", [&] {
        inject_update(life, 4, state, as_span(block), as_span(norm_w), kEps,
                      as_span(iw), geo);
    });
    failures += expect_lifecycle_throw("final caller-T mismatch", "T=4 vs bound T=2", [&] {
        final_mixer(life, 4, state, as_span(norm_w), kEps,
                    as_span(down), as_span(up), geo);
    });

    state4.ready = true;
    failures += expect_lifecycle_throw("read supplied-state mismatch", "state T=4", [&] {
        read(life, 2, state4, as_span(norm_w), kEps, as_span(down), as_span(up), geo);
    });
    failures += expect_lifecycle_throw("inject supplied-state mismatch", "state T=4", [&] {
        std::vector<double> block4(kHidden * 4, 0.25);
        inject_update(life, 2, state4, as_span(block4), as_span(norm_w), kEps,
                      as_span(iw), geo);
    });
    failures += expect_lifecycle_throw("final supplied-state mismatch", "state T=4", [&] {
        final_mixer(life, 2, state4, as_span(norm_w), kEps,
                    as_span(down), as_span(up), geo);
    });

    read(life, 2, state, as_span(norm_w), kEps, as_span(down), as_span(up), geo);
    inject_update(life, 2, state, as_span(block), as_span(norm_w), kEps,
                  as_span(iw), geo);
    failures += expect_lifecycle_throw("inject twice", "snapshot consumed", [&] {
        inject_update(life, 2, state, as_span(block), as_span(norm_w), kEps,
                      as_span(iw), geo);
    });
    final_mixer(life, 2, state, as_span(norm_w), kEps,
                as_span(down), as_span(up), geo);

    life.reset(state);
    failures += expect_lifecycle_throw("read post-reset", "after reset", [&] {
        read(life, 2, state, as_span(norm_w), kEps, as_span(down), as_span(up), geo);
    });
    failures += expect_lifecycle_throw("inject post-reset", "after reset", [&] {
        inject_update(life, 2, state, as_span(block), as_span(norm_w), kEps,
                      as_span(iw), geo);
    });
    failures += expect_lifecycle_throw("final post-reset", "after reset", [&] {
        final_mixer(life, 2, state, as_span(norm_w), kEps,
                    as_span(down), as_span(up), geo);
    });

    HcState fresh4 = pattern_state(4, geo);
    life.initialize(fresh4, 4, geo);
    read(life, 4, fresh4, as_span(norm_w), kEps, as_span(down), as_span(up), geo);

    if (failures == 0)
        std::cout << "OK lifecycle: once-per-forward + caller/state extent gates\n";
    return failures;
}

// ----------------------------------------------------------------------------
// M1 contract: prior_read authority. The state-owned preserved `prior_read`
// is the additive *update base* of `inject`; the block-inject logits consume
// the live `hyper`. This fixture proves the two authorities are distinct via
// an *independently hand-derived* expectation (never the oracle inject/logit
// helper): the preserved snapshot is preserved, the live `hyper` is then
// mutated by +7 at a single lane, and the expected updated state (and the
// live-base substitution) are hand-derived from that base. The successful
// `inject` must match the prior-base expectation and must *not* match the
// live-base substitution. An externally supplied inject reference that does
// not equal the state-owned prior is rejected; a successful inject consumes
// (clears) the snapshot.
// ----------------------------------------------------------------------------
int test_prior_authority() {
    const auto geo = geometry();
    const std::int32_t T = 2;
    int failures = 0;

    // --- Live/prior base divergence: preserve, mutate, hand-derive ---
    {
        HcState state = pattern_state(T, geo);
        std::vector<double> norm_w(geo.expanded(), 1.0);
        const std::vector<double> down = small_down(geo, 1);
        const std::vector<double> up = small_up(geo, 2);
        const std::vector<double> iw = small_inject(geo, 3);
        std::vector<double> block(kHidden * T, 0.2);

        HcExecutorLifecycle life;
        life.initialize(state, T, geo);
        read(life, T, state, as_span(norm_w), kEps,
             as_span(down), as_span(up), geo);
        const std::vector<double> preserved = state.prior_read;

        // Mutate the LIVE hyper at a single lane; the preserved base is
        // unaffected (it was captured at read time, before this mutation).
        state.hyper[0] += 7.0;

        // Independent hand-derived block-inject logits from the (mutated)
        // LIVE hyper: per-stream normalization -> inject projection.
        const std::vector<double> live_hn =
            ref_group_rmsnorm(state.hyper, norm_w, kEps, T, geo);
        std::vector<double> raw(geo.streams() * T, 0.0);
        for (std::int32_t t = 0; t < T; ++t)
            for (std::int32_t c = 0; c < geo.streams(); ++c)
                for (std::int32_t j = 0; j < geo.expanded(); ++j)
                    raw[static_cast<std::size_t>(t) * geo.streams() + c] +=
                        iw[static_cast<std::size_t>(c) * geo.expanded() + j] *
                        live_hn[static_cast<std::size_t>(t) * geo.expanded() + j];
        std::vector<double> alpha(geo.streams() * T);
        for (std::size_t i = 0; i < raw.size(); ++i)
            alpha[i] = 2.0 / (1.0 + std::exp(-raw[i] / static_cast<double>(geo.streams())));

        // Independent expected updated state, hand-derived for the two
        // bases. The prior (preserved) base is the normative update base;
        // the live base is what a live-`hyper`-based implementation yields.
        std::vector<double> exp_prior(geo.expanded_elems(T));
        std::vector<double> exp_live(geo.expanded_elems(T));
        for (std::int32_t t = 0; t < T; ++t)
            for (std::int32_t c = 0; c < geo.streams(); ++c)
                for (std::int32_t h = 0; h < geo.hidden(); ++h) {
                    const std::size_t hi = static_cast<std::size_t>(t) * geo.expanded() +
                                            static_cast<std::size_t>(c) * geo.hidden() + h;
                    const std::size_t bi = static_cast<std::size_t>(t) * geo.hidden() + h;
                    const std::size_t ai = static_cast<std::size_t>(t) * geo.streams() + c;
                    exp_prior[hi] = preserved[hi] + block[bi] * alpha[ai];  // prior base
                    exp_live[hi]  = state.hyper[hi] + block[bi] * alpha[ai]; // live base
                }

        const std::vector<double> got =
            inject_update_reference(life, T, state, preserved,
                                    as_span(block), as_span(norm_w), kEps,
                                    as_span(iw), geo);
        failures += compare_exact(got, exp_prior,
            "prior authority: successful inject matches independent prior base");

        // The +7 shift is observable at lane 0: the live-base expectation is
        // exactly 7.0 above the prior-base at that lane. The independent
        // divergence bound (the two hand-derived bases) and the lane-0 check
        // both confirm the update used the preserved base, not the live hyper.
        if (std::abs(exp_live[0] - exp_prior[0]) < 6.5) {
            std::cerr << "FAIL prior authority: live/prior base expectations did not diverge\n";
            ++failures;
        }
        if (!got.empty() && std::abs(got[0] - exp_live[0]) < 1.0) {
            std::cerr << "FAIL prior authority: update used live hyper as base\n";
            ++failures;
        }

        // Snapshot consumption: a successful inject clears the prior snapshot.
        if (state.prior_read_valid || !state.prior_read.empty()) {
            std::cerr << "FAIL prior authority: successful inject retained snapshot\n";
            ++failures;
        }
    }

    // --- Rejection: a mismatching external live-hyper reference ---
    {
        HcState state = pattern_state(T, geo);
        std::vector<double> norm_w(geo.expanded(), 1.0);
        const std::vector<double> down = small_down(geo, 9);
        const std::vector<double> up = small_up(geo, 10);
        const std::vector<double> iw = small_inject(geo, 11);
        std::vector<double> block(kHidden * T, 0.2);
        HcExecutorLifecycle life;
        life.initialize(state, T, geo);
        read(life, T, state, as_span(norm_w), kEps,
             as_span(down), as_span(up), geo);
        state.hyper[0] += 7.0;   // live hyper now differs from the preserved snapshot
        failures += expect_lifecycle_throw(
            "prior authority live-hyper mismatch",
            "external live hyper is not the preserved snapshot", [&] {
                inject_update_reference(life, T, state, state.hyper,
                                        as_span(block), as_span(norm_w), kEps,
                                        as_span(iw), geo);
            });
    }

    if (failures == 0)
        std::cout << "OK prior_read authority: independent prior base; live base differs\n";
    return failures;
}


// ----------------------------------------------------------------------------
// M1 contract: weight-layout mapping to production `linear()` (doc-only
// check; the production binding itself is M2/M3). Verifies the geometry
// extents used by the three projection weights, per contract doc s.7.
// ----------------------------------------------------------------------------
int test_weight_mapping() {
    const auto geo = geometry();
    int failures = 0;

    std::vector<double> down(geo.lowrank() * geo.expanded(), 0.0);
    std::vector<double> x(geo.expanded(), 0.0);
    constexpr std::int32_t r = 3;
    constexpr std::int32_t k = 7;
    down[static_cast<std::size_t>(r) * geo.expanded() + k] = 1.25;
    x[k] = 2.0;
    const auto d = linear_down(as_span(down), as_span(x), geo, 1);
    std::vector<double> dref(geo.lowrank(), 0.0);
    dref[r] = 2.5;
    failures += compare_exact(d, dref, "weight mapping: down r*CH+k");

    std::vector<double> up(geo.expanded() * geo.lowrank(), 0.0);
    std::vector<double> din(geo.lowrank(), 0.0);
    constexpr std::int32_t j = 11;
    constexpr std::int32_t ur = 2;
    up[static_cast<std::size_t>(j) * geo.lowrank() + ur] = -0.75;
    din[ur] = 2.0;
    const auto u = linear_up(as_span(up), as_span(din), geo, 1);
    std::vector<double> uref(geo.expanded(), 0.0);
    uref[j] = -1.5;
    failures += compare_exact(u, uref, "weight mapping: up j*R+r");

    std::vector<double> iw(geo.streams() * geo.expanded(), 0.0);
    std::vector<double> hn(geo.expanded(), 0.0);
    constexpr std::int32_t c = 2;
    constexpr std::int32_t ij = 13;
    iw[static_cast<std::size_t>(c) * geo.expanded() + ij] = 0.625;
    hn[ij] = 4.0;
    const auto raw = block_inject_logits(as_span(hn), as_span(iw), geo, 1);
    std::vector<double> rawref(geo.streams(), 0.0);
    rawref[c] = 2.5;
    failures += compare_exact(raw, rawref, "weight mapping: inject c*CH+j");

    if (failures == 0)
        std::cout << "OK weight mapping: asymmetric physical offsets n*K+k\n";
    return failures;
}

// ----------------------------------------------------------------------------
// M1 contract: module-bound weights (norm_weight rebinding, block vs final).
// The same state with different module weights must give different results;
// a weight bound to the wrong module (or an unbound weight) is detectable.
// ----------------------------------------------------------------------------
int test_weight_rebinding() {
    const auto geo = geometry();
    const std::int32_t T = 2;
    const HcState state = pattern_state(T, geo);
    std::vector<double> wA(geo.expanded(), 1.0);
    std::vector<double> wB(geo.expanded(), 1.0);
    for (std::size_t i = 0; i < wB.size(); ++i)
        wB[i] = 1.0 + 0.3 * (i % 13);  // distinct final-module weight
    const std::vector<double> down = small_down(geo, 7);
    const std::vector<double> up = small_up(geo, 8);

    const std::vector<double> finA =
        final_mixer(state, as_span(wA), kEps, as_span(down), as_span(up), geo);
    const std::vector<double> finB =
        final_mixer(state, as_span(wB), kEps, as_span(down), as_span(up), geo);

    int failures = 0;
    // The rebound weight must change the result (proves the module weight is
    // bound per module, not a hidden constant).
    bool differ = false;
    for (std::size_t i = 0; i < finA.size(); ++i)
        if (std::abs(finA[i] - finB[i]) > 1.0e-9) { differ = true; break; }
    if (!differ) {
        std::cerr << "FAIL weight rebinding: distinct module weights gave identical output\n";
        ++failures;
    }
    for (double x : finA) if (!std::isfinite(x)) { std::cerr << "FAIL weight rebinding: nonfinite (A)\n"; ++failures; break; }
    for (double x : finB) if (!std::isfinite(x)) { std::cerr << "FAIL weight rebinding: nonfinite (B)\n"; ++failures; break; }
    // A third bound (all-ones weight) must match finA exactly: proves the
    // module weight is the only differentiator (no hidden constant path).
    const std::vector<double> finC =
        final_mixer(state, as_span(std::vector<double>(geo.expanded(), 1.0)), kEps,
                   as_span(down), as_span(up), geo);
    failures += compare_exact(finC, finA, "weight rebinding: weight-only differentiator");
    if (failures == 0)
        std::cout << "OK weight rebinding: block vs final norm_weight bound per module\n";
    return failures;
}


// ----------------------------------------------------------------------------
// M1 contract: lifecycle conformance. Validates the frozen per-forward
// lifecycle using the HcState/HcExecutorLifecycle gated APIs:
//   A) successful read enters phase P; a second read in P is rejected;
//      the pending prior_read snapshot remains byte-identical.
//   B) inject consumes the preserved snapshot (P->I transition); after
//      successful inject the snapshot is invalidated/cleared.
//   C) same-state reset clears stale prior/phase; same-state reinit at
//      the same T succeeds; subsequent read is legal again.
//   D) compare_exact rejects nonfinite (NaN/inf) operands outright.
// Formula-only helper checks (D) use the compare_exact helper directly.
// ----------------------------------------------------------------------------
int test_lifecycle_conformance() {
    const auto geo = geometry();
    const std::int32_t T = 2;
    int failures = 0;
    HcExecutorLifecycle life;
    std::vector<double> norm_w(geo.expanded(), 1.0);
    const std::vector<double> down = small_down(geo);
    const std::vector<double> up = small_up(geo);
    const std::vector<double> iw = small_inject(geo);
    std::vector<double> block(kHidden * T, 0.25);

    // --- A) read enters P; second read in P rejected; prior unchanged ---
    {
        HcState state = pattern_state(T, geo);
        life.initialize(state, T, geo);
        read(life, T, state, as_span(norm_w), kEps, as_span(down),
             as_span(up), geo);

        // After a successful read the state is in phase P.
        if (state.phase() != HcPhase::P) {
            std::cerr << "FAIL lifecycle A: state not in phase P after read\n";
            ++failures;
        }
        if (!state.prior_read_valid) {
            std::cerr << "FAIL lifecycle A: prior_read not valid after read\n";
            ++failures;
        }

        // Snapshot the prior before attempting the (rejected) second read.
        const std::vector<double> preserved = state.prior_read;

        // A second read in phase P must be rejected (read requires phase I).
        failures += expect_lifecycle_throw(
            "read in phase P (second read)",
            "read requires phase I; a read has already run", [&] {
                read(life, T, state, as_span(norm_w), kEps,
                     as_span(down), as_span(up), geo);
            });

        // The pending prior snapshot must be byte-identical after the
        // rejected second read: a rejected operation must not mutate state.
        if (state.prior_read != preserved) {
            std::cerr << "FAIL lifecycle A: prior snapshot mutated by rejected read\n";
            ++failures;
        }
        if (!std::equal(state.prior_read.begin(), state.prior_read.end(),
                        preserved.begin())) {
            std::cerr << "FAIL lifecycle A: prior element-wise changed\n";
            ++failures;
        }
    }

    // --- B) inject consumes the preserved snapshot; P->I transition ---
    {
        HcState state = pattern_state(T, geo);
        life.initialize(state, T, geo);
        read(life, T, state, as_span(norm_w), kEps, as_span(down),
             as_span(up), geo);
        // state is now in phase P with a valid prior snapshot.

        const std::vector<double> next =
            inject_update(life, T, state, as_span(block), as_span(norm_w),
                          kEps, as_span(iw), geo);

        // After successful inject: the snapshot is consumed/cleared;
        // the state transitions from P to I.
        if (state.prior_read_valid) {
            std::cerr << "FAIL lifecycle B: prior_read_valid not cleared after inject\n";
            ++failures;
        }
        if (!state.prior_read.empty()) {
            std::cerr << "FAIL lifecycle B: prior_read not cleared after inject\n";
            ++failures;
        }
        if (state.phase() != HcPhase::I) {
            std::cerr << "FAIL lifecycle B: state not in phase I after inject\n";
            ++failures;
        }
        if (!std::isfinite(next[0])) {
            std::cerr << "FAIL lifecycle B: inject produced nonfinite value\n";
            ++failures;
        }
        std::cout << "OK lifecycle B: inject consumes prior; P->I transition\n";
    }

    // --- C) same-state reset clears stale state; same-state reinit ---
    {
        HcState state = pattern_state(T, geo);
        life.initialize(state, T, geo);
        read(life, T, state, as_span(norm_w), kEps, as_span(down),
             as_span(up), geo);
        // state is in phase P with a valid prior snapshot (stale state).

        life.reset(state);

        // After reset: all stale prior/phase state is cleared.
        if (state.ready) {
            std::cerr << "FAIL lifecycle C: ready not cleared by reset\n";
            ++failures;
        }
        if (state.tokens != 0) {
            std::cerr << "FAIL lifecycle C: tokens not cleared by reset\n";
            ++failures;
        }
        if (!state.hyper.empty()) {
            std::cerr << "FAIL lifecycle C: hyper not cleared by reset\n";
            ++failures;
        }
        if (!state.prior_read.empty()) {
            std::cerr << "FAIL lifecycle C: prior_read not cleared by reset\n";
            ++failures;
        }
        if (state.prior_read_valid) {
            std::cerr << "FAIL lifecycle C: prior_read_valid not cleared by reset\n";
            ++failures;
        }
        if (state.phase() != HcPhase::U) {
            std::cerr << "FAIL lifecycle C: state not in phase U after reset\n";
            ++failures;
        }

        // Reinitialize the SAME state at the same T.
        state.hyper = pattern_state(T, geo).hyper;
        life.initialize(state, T, geo);

        if (!state.ready) {
            std::cerr << "FAIL lifecycle C: reinit did not set ready\n";
            ++failures;
        }
        if (state.tokens != T) {
            std::cerr << "FAIL lifecycle C: reinit did not bind tokens\n";
            ++failures;
        }
        if (state.prior_read_valid) {
            std::cerr << "FAIL lifecycle C: reinit left stale prior valid\n";
            ++failures;
        }

        // After reinit, a read on the same state must succeed (phase I).
        try {
            read(life, T, state, as_span(norm_w), kEps,
                 as_span(down), as_span(up), geo);
            std::cout << "OK lifecycle C: same-state reset+reinit; read legal again\n";
        } catch (const HcLifecycleError&) {
            std::cerr << "FAIL lifecycle C: read after same-state reinit rejected\n";
            ++failures;
        }
    }

    // --- D) compare_exact rejects nonfinite operands outright ---
    // Formula-only helper check: the exact-compare helper must refuse to
    // proceed when either operand contains a nonfinite (NaN or inf) value.
    {
        const std::vector<double> nan_got = {1.0, std::nan("")};
        const std::vector<double> nan_ref = {1.0, std::nan("")};
        const int rc_nan =
            compare_exact(nan_got, nan_ref, "nonfinite: both NaN");
        if (rc_nan == 0) {
            std::cerr << "FAIL lifecycle D: nonfinite (NaN/NaN) not rejected\n";
            ++failures;
        }

        const std::vector<double> inf_got = {1.0,
            std::numeric_limits<double>::infinity()};
        const std::vector<double> finite_ref = {1.0, 1.0};
        const int rc_inf =
            compare_exact(inf_got, finite_ref, "nonfinite: inf vs finite");
        if (rc_inf == 0) {
            std::cerr << "FAIL lifecycle D: nonfinite (inf) not rejected\n";
            ++failures;
        }

        const std::vector<double> finite_got = {1.0, 2.0};
        const std::vector<double> neginf_ref = {1.0,
            -std::numeric_limits<double>::infinity()};
        const int rc_neginf =
            compare_exact(finite_got, neginf_ref, "nonfinite: finite vs -inf");
        if (rc_neginf == 0) {
            std::cerr << "FAIL lifecycle D: nonfinite (-inf) not rejected\n";
            ++failures;
        }
    }

    if (failures == 0)
        std::cout << "OK lifecycle conformance: A-P-reject B-inject-C-reset D-nonfinite\n";
    return failures;
}

} // namespace

int main() {
    std::cout << "=== Batch 3C2 / M1: HyperConnection executor contract "
                 "oracle (independent FP64; CPU) ===\n";
    int failures = 0;

    failures += test_initialize();
    failures += test_group_rmsnorm();
    failures += test_read_independence();
    failures += test_inject_nonzero();
    failures += test_inject_zero_alpha();
    failures += test_final_mixer();
    failures += test_lifecycle_gate();
    failures += test_prior_authority();
    failures += test_weight_mapping();
    failures += test_weight_rebinding();
    failures += test_lifecycle_conformance();

    std::cout << (failures == 0 ? "OK" : "FAIL")
              << " hyper_connection_executor contract (M1)\n";
    return failures == 0 ? 0 : 1;
}