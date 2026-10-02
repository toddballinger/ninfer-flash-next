#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

/**
 * M2 detail read launcher: composes the existing 3C1 primitives and the BF16
 * `linear()` projection into the four-stream `read` path.
 *
 * Composition (docs HYPERCONNECTION_EXECUTOR_CONTRACT s.2 /
 * REFERENCE_FORMULAS.md s.1 "Read / mix"):
 *
 *   Hn    = GroupRMSNorm(hyper, module norm_weight, eps)   [CH, T]
 *   down  = linear(Hn, mix_down [R,CH])                    [R, T]
 *   down  = SiLU(down / C)   (in place; one low-rank buffer)
 *   up    = linear(down, mix_up [CH,R])                    [CH, T]
 *   mixed = read_mix(Hn, up)                               [H, T]
 *
 * `mixed` (the produced `mixed_input`) and `prior` (the preserved
 * `prior_read` snapshot, an exact read-only copy of `hyper`) are the two
 * output tensors bound by the caller / state wrapper; neither aliases the
 * `hyper` input or any live arena scratch buffer.
 *
 * All transient scratch (Hn, down, up) is allocated from the caller-owned,
 * call-scoped `workspace`; the launcher performs **no** internal device
 * allocation. The three live arena buffers occupy `2*T*(2*C*H + R)` bytes,
 * exactly the frozen `read`/`final_mixer` capacity term: the SiLU step runs
 * in place on the `down` buffer, so the low-rank scratch is a single `R`
 * buffer rather than two, keeping the total within the provisioned bound.
 *
 * @param[in]      hyper     Four-stream state, BF16 `[CH, T]`.
 * @param[in]      T         Forward token extent.
 * @param[in]      hidden    Per-stream width `H`.
 * @param[in]      hc_count  Stream count `C`.
 * @param[in]      norm      Module-owned grouped-RMSNorm weight `[CH]`.
 * @param[in]      mix_down  Low-rank down weight `[R, CH]`.
 * @param[in]      mix_up    Low-rank up weight `[CH, R]`.
 * @param[in]      eps       Normalization epsilon (contract-fixed 1e-5).
 * @param[out]     mixed     Produced block input, BF16 `[H, T]`.
 * @param[out]     prior     Preserved prior stream token, BF16 `[CH, T]`.
 * @param[in, out] workspace Caller-owned, call-scoped transient arena.
 * @param[out]     stream    Executed on `stream`.
 */
void hyper_connection_read_launch(
    const Tensor& hyper, std::int32_t T, std::int32_t hidden,
    std::int32_t hc_count, const Weight& norm, const Weight& mix_down,
    const Weight& mix_up, float eps, Tensor& mixed, Tensor& prior,
    WorkspaceArena& workspace, cudaStream_t stream);

/**
 * M2 detail final-mixer launcher: composes the 3C1 primitives and the BF16
 * `linear()` projection into the four-stream `final_mixer` collapse. It is the
 * identical read path bound to the **final mixer's own** `norm` / `mix_down` /
 * `mix_up`; the final mixer has no `block_inject` and returns only the
 * collapsed `final_mix` (four streams -> `H`).
 *
 * Composition (docs HYPERCONNECTION_EXECUTOR_CONTRACT s.5 /
 * REFERENCE_FORMULAS.md s.1, final-mixer note):
 *
 *   Hn        = GroupRMSNorm(hyper, final.norm_weight, eps)   [CH, T]
 *   down      = linear(Hn, mix_down [R,CH])                    [R, T]
 *   down      = SiLU(down / C)   (in place; one low-rank buffer)
 *   up        = linear(down, mix_up [CH,R])                    [CH, T]
 *   final_mix = read_mix(Hn, up)                               [H, T]
 *
 * Unlike `read`, `final_mix` is the sole produced tensor: no `prior_read`
 * snapshot is written, so the collapse leaves no block-inject reference and
 * does not disturb the caller-owned snapshot plane. All transient scratch
 * (Hn, down, up) is allocated from the caller-owned, call-scoped `workspace`
 * (no internal device allocation); the three live arena buffers occupy
 * `2*T*(2*C*H + R)` bytes, the frozen `read`/`final_mixer` capacity term.
 *
 * @param[in]      hyper     Four-stream state, BF16 `[CH, T]`.
 * @param[in]      T         Forward token extent.
 * @param[in]      hidden    Per-stream width `H`.
 * @param[in]      hc_count  Stream count `C`.
 * @param[in]      norm      Final-mixer grouped-RMSNorm weight `[CH]`.
 * @param[in]      mix_down  Final-mixer low-rank down weight `[R, CH]`.
 * @param[in]      mix_up    Final-mixer low-rank up weight `[CH, R]`.
 * @param[in]      eps       Normalization epsilon (contract-fixed 1e-5).
 * @param[out]     final_mix Produced four-stream collapse, BF16 `[H, T]`.
 * @param[in, out] workspace Caller-owned, call-scoped transient arena.
 * @param[out]     stream    Executed on `stream`.
 */
void hyper_connection_final_mixer_launch(
    const Tensor& hyper, std::int32_t T, std::int32_t hidden,
    std::int32_t hc_count, const Weight& norm, const Weight& mix_down,
    const Weight& mix_up, float eps, Tensor& final_mix,
    WorkspaceArena& workspace, cudaStream_t stream);

/**
 * M2 detail inject launcher: composes the 3C1 GroupRMSNorm and the BF16
 * `linear()` projection of the block-inject logits into the per-stream
 * `alpha` weighting; then applies the one-shot in-place four-stream
 * `hyper` update using the state-owned `prior_read` as the immutable
 * additive base (never the live `hyper`).
 *
 * Composition (docs HYPERCONNECTION_EXECUTOR_CONTRACT s.4 /
 * REFERENCE_FORMULAS.md s.1 "Injection weights"):
 *
 *   Hn          = GroupRMSNorm(hyper, norm_weight, eps)   [CH, T]
 *   raw[c, t]   = linear(Hn, inject [C, CH])               [C, T]
 *   alpha[c, t] = 2 * sigmoid(raw[c, t] / C)   (once)
 *   hyper[c, h, t] = prior_read[c, h, t] + block[h, t] * alpha[c, t]
 *
 * `block` is consumed read-only; `prior_read` is the sole immutable
 * additive base; `hyper` is the in-place destination. All transient
 * scratch (Hn, raw) is allocated from the caller-owned, call-scoped
 * `workspace`; no internal device allocation.
 *
 * @param[in]    block         Block output, BF16 `[H, T]`.
 * @param[in,out] hyper         Four-stream state, BF16 `[CH, T]`; in-place target.
 * @param[in]    T              Forward token extent.
 * @param[in]    hidden         Per-stream width `H`.
 * @param[in]    hc_count       Stream count `C`.
 * @param[in]    norm           Grouped-RMSNorm weight `[CH]`.
 * @param[in]    inject_weight  Per-stream block-inject logit weight `[C, CH]`.
 * @param[in]    eps            Normalization epsilon (contract-fixed 1e-5).
 * @param[in]    prior_read     State-owned preserved prior, BF16 `[CH, T]`; immutable base.
 * @param[in, out] workspace    Caller-owned, call-scoped transient arena.
 * @param[out]   stream         Executed on `stream`.
 */
void hyper_connection_inject_launch(
    const Tensor& block, Tensor& hyper, std::int32_t T,
    std::int32_t hidden, std::int32_t hc_count,
    const Weight& norm, const Weight& inject_weight, float eps,
    const Tensor& prior_read,
    WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
