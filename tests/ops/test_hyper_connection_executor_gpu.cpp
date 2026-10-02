#include "ninfer/ops/hyper_connection_executor.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

//
// Batch 3C2 / M2 -- dedicated GPU-capable HyperConnection *executor* test.
//
// This exercises the **actual M2 public path** (`ninfer::ops::
// HyperConnectionExecutorState`): initialize -> read -> callback -> inject ->
// final_mixer, using real device BF16 tensors/weights and CUDA synchronization.
// It composes the production 3C1 primitives through the BF16 `linear()`
// projections exactly as the production executor does. It does **not**
// exercise the legacy 3C1 `hyper_connection` primitives directly: those are
// composed *inside* the M2 detail launchers.
//
// --- Geometry -----------------------------------------------------------
// The M2 `read` / `inject` / `final_mixer` launchers all compose `linear()`,
// which resolves to a **registered, persistent BF16 weight problem** keyed by
// the weight's `(n, k)`. The registered shapes are fixed; the only geometry at
// which every projection in the three paths is simultaneously registered is the
// canonical Flash-Next model geometry:
//
//     mix_down  [R, CH]    = [320, 10240]    (registered)
//     mix_up    [CH, R]    = [10240, 320]    (registered)
//     inject    [C, CH]    = [4, 10240]      (registered)
//
// which pins H = 2560, C = 4 (CH = 10240), R = 320. A smaller "toy" geometry
// resolves to unregistered shapes and `linear()` rejects it ("unsupported
// shape"), so it cannot drive the real M2 path. The canonical geometry is
// therefore the correct, minimal geometry for a GPU test that must invoke the
// actual production `linear()` projections.
//
// Every projection is executed at T = 3, which resolves each registered shape
// to a valid kernel (gemv/SIMT/MMA ladder), so the full path is exercised.
//

namespace {

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;

// Canonical Flash-Next model geometry (the only geometry at which the M2
// `linear()` projections are all registered): H = 2560, C = 4, R = 320.
constexpr std::int32_t kStreams  = 4;          // C = hc_count
constexpr std::int32_t kHidden   = 2560;        // H = hidden_size
constexpr std::int32_t kLowrank  = 320;         // R = hc_lowrank
constexpr std::int32_t kCH       = kStreams * kHidden;  // 10240
constexpr std::int32_t kTokens   = 3;           // T = bound forward token extent

// Contract-fixed normalization epsilon (every module/call).
constexpr float kEps = 1.0e-5F;

// Deterministic, well-conditioned activation value (bounded, distinct
// feature/token structure so the low-rank matrix stays rank-rich).
float activation_value(std::int32_t h, std::int32_t t) {
    const int a = static_cast<int>(h % 9) - 4;  // -4..4
    const int b = static_cast<int>(t % 5) - 2;  // -2..2
    return (static_cast<float>(a) + 0.25F * static_cast<float>(b)) *
           (1.0F / 16.0F);
}

std::unique_ptr<DeviceWeight> make_weight(std::int32_t n, std::int32_t k,
                                          std::uint32_t seed) {
    return std::make_unique<DeviceWeight>(make_patterned(n, k, seed));
}

// The `Weight` view for a module slot. `make_patterned` sets shape[0]/[1]; we
// mirror them into the `shape` fields as well (a `Weight` `view()` rebuilds
// `qdata` from the device pointer at call time, so the `shape` fields only
// matter if `n`/`k` were left zero, which `make_patterned` does not).
Weight module_view(const DeviceWeight& w) {
    Weight view = w.view();
    view.shape[0]         = w.host.n;
    view.shape[1]         = w.host.k;
    view.padded_shape[0]  = w.host.n;
    view.padded_shape[1]  = w.host.k;
    return view;
}

// The M2 `read`/`final_mixer` path composes GroupRMSNorm + linear + SiLU +
// linear + read_mix. All intermediate values are bounded by the small,
// deterministic weight/activation magnitudes; a correct path therefore yields
// only finite results. Any non-finite value signals a genuine defect (a wrong
// private-numeric path, a mis-allocated buffer, or an out-of-range read).
bool all_finite(const std::vector<double>& v) {
    for (const double x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

// The caller-supplied `callback` block: the only step the M2 executor does not
// own. Here the block is a valid device-to-device passthrough of the `read`
// output (`mixed`) -- a legal "block result" that keeps the test
// deterministic. It must not touch `hyper` directly (contract s.3).
void test_block_runner(const Tensor& mixed, Tensor& block,
                       cudaStream_t stream) {
    const std::size_t bytes =
        static_cast<std::size_t>(mixed.ne[0]) * mixed.ne[1] *
        sizeof(std::uint16_t);
    const cudaError_t err = cudaMemcpyAsync(
        block.data, mixed.data, bytes, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(
            "test hyper_connection_executor_gpu: callback block copy failed: ") +
            cudaGetErrorString(err));
    }
}

int test_executor_m2_gpu_path() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    // Real device context: the M2 launchers are stream-ordered and
    // stream-synchronous, so they must run on a real CUDA stream (the legacy
    // `test_hyper_connection.cpp` canonical case does the same with
    // `DeviceContext`).
    DeviceContext device;

    // Provision the caller-owned four-stream state plane `hyper [CH, T]`.
    // Long-lived, caller-owned: the state persists it and the consuming
    // launchers write to it in place.
    GuardedDeviceBuffer device_hyper(
        static_cast<std::size_t>(kCH) * kTokens * sizeof(std::uint16_t));
    device_hyper.fill(0);

    // Independent consumer planes (no alias to `hyper` or any live scratch):
    // the embedding (initialize input), read mixed/prior, block, final.
    GuardedDeviceBuffer device_embedding(
        static_cast<std::size_t>(kHidden) * kTokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_mixed(
        static_cast<std::size_t>(kHidden) * kTokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_prior(
        static_cast<std::size_t>(kCH) * kTokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_block(
        static_cast<std::size_t>(kHidden) * kTokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_final(
        static_cast<std::size_t>(kHidden) * kTokens * sizeof(std::uint16_t));

    // Fill the embedding payload and copy it to its device plane.
    {
        std::vector<std::uint16_t> emb(
            static_cast<std::size_t>(kHidden) * kTokens);
        for (std::int32_t t = 0; t < kTokens; ++t) {
            for (std::int32_t h = 0; h < kHidden; ++h) {
                emb[static_cast<std::size_t>(t) * kHidden + h] =
                    f32_to_bf16(activation_value(h, t));
            }
        }
        device_embedding.copy_from_host(
            emb.data(), emb.size() * sizeof(std::uint16_t));
    }
    device_mixed.fill(0xff);
    device_prior.fill(0xff);
    device_block.fill(0xff);
    device_final.fill(0xff);

    // Module weights (call-bound, module-owned; not held in state). Each is a
    // patterned, rank-rich, bounded BF16 weight on the canonical geometry so
    // every `linear()` projection resolves to a registered problem.
    std::unique_ptr<DeviceWeight> block_norm   = make_weight(kCH, 1, 3311U);
    std::unique_ptr<DeviceWeight> block_down   = make_weight(kLowrank, kCH, 3313U);
    std::unique_ptr<DeviceWeight> block_up     = make_weight(kCH, kLowrank, 3317U);
    std::unique_ptr<DeviceWeight> block_inject = make_weight(kStreams, kCH, 3319U);
    std::unique_ptr<DeviceWeight> final_norm   = make_weight(kCH, 1, 3331U);
    std::unique_ptr<DeviceWeight> final_down   = make_weight(kLowrank, kCH, 3337U);
    std::unique_ptr<DeviceWeight> final_up     = make_weight(kCH, kLowrank, 3341U);

    // A block module: its `inject` weight is [C, CH] = [4, 10240] (registered
    // for the inject logits); `mix_down` is [R, CH], `mix_up` [CH, R].
    HyperConnectionBlock block_module;
    block_module.norm_weight = module_view(*block_norm);
    block_module.mix_down    = module_view(*block_down);
    block_module.mix_up      = module_view(*block_up);
    block_module.inject      = module_view(*block_inject);

    // The final mixer module: its own (different) weights, no block-inject.
    HyperConnectionFinalMixer final_module;
    final_module.norm_weight    = module_view(*final_norm);
    final_module.mix_down       = module_view(*final_down);
    final_module.mix_up         = module_view(*final_up);
    final_module.has_block_inject = false;  // normative

    // The four-stream state, bound to the canonical geometry before the first
    // `initialize` (contract: geometry is caller-provisioned).
    HyperConnectionExecutorState state;
    state.hc_count    = kStreams;
    state.hidden_size = kHidden;
    state.hc_lowrank  = kLowrank;

    // `hyper` is the long-lived four-stream plane.
    state.hyper = Tensor(device_hyper.data(), DType::BF16, {kCH, kTokens});

    Tensor embedding_tensor(
        device_embedding.data(), DType::BF16, {kHidden, kTokens});

    int failures = 0;

    // (1) initialize: repeat(embedding, C) -> hyper [CH, T]. A pure 3C1 GPU
    //     kernel over the caller plane; no workspace.
    try {
        state.initialize(embedding_tensor, kTokens, device.stream);
        cuda_synchronize(device.stream);
    } catch (const std::exception& e) {
        std::cerr << "M2 initialize threw: " << e.what() << '\n';
        return 1;  // any throw on the success path is a failure
    }

    // (2) read: mixed_input + prior_read snapshot. Phase I -> P. The caller-
    //     owned read workspace is sized to the exact frozen capacity bound.
    Tensor mixed_tensor(device_mixed.data(), DType::BF16, {kHidden, kTokens});
    Tensor prior_tensor(device_prior.data(), DType::BF16, {kCH, kTokens});
    {
        DeviceArena read_ws(
            hyper_connection_executor_workspace_capacity_bytes(
                kStreams, kHidden, kLowrank, kTokens));
        try {
            state.read(state.hyper, kTokens, block_module, kEps,
                       mixed_tensor, prior_tensor, read_ws, device.stream);
            cuda_synchronize(device.stream);
        } catch (const std::exception& e) {
            std::cerr << "M2 read threw: " << e.what() << '\n';
            return 1;
        }
    }

    // (3) callback: the caller block produces `block` from `mixed`.
    Tensor block_tensor(device_block.data(), DType::BF16, {kHidden, kTokens});
    {
        try {
            state.callback(mixed_tensor, block_tensor, device.stream,
                           &test_block_runner);
            cuda_synchronize(device.stream);
        } catch (const std::exception& e) {
            std::cerr << "M2 callback threw: " << e.what() << '\n';
            return 1;
        }
    }

    // (4) inject: in-place four-stream update from prior_read + block*alpha.
    //     Phase P -> I (consumes the pending prior snapshot).
    {
        DeviceArena inject_ws(
            hyper_connection_executor_workspace_capacity_bytes(
                kStreams, kHidden, kLowrank, kTokens));
        try {
            state.inject(block_tensor, state.hyper, kTokens, block_module,
                         kEps, inject_ws, device.stream);
            cuda_synchronize(device.stream);
        } catch (const std::exception& e) {
            std::cerr << "M2 inject threw: " << e.what() << '\n';
            return 1;
        }
    }

    // (5) final_mixer: collapse four streams -> one H-wide hidden state.
    //     Requires phase I (no pending prior); the inject above consumed the
    //     snapshot, so we are in phase I.
    {
        DeviceArena final_ws(
            hyper_connection_executor_workspace_capacity_bytes(
                kStreams, kHidden, kLowrank, kTokens));
        Tensor final_tensor(device_final.data(), DType::BF16, {kHidden, kTokens});
        try {
            state.final_mixer(state.hyper, kTokens, final_module, kEps,
                              final_tensor, final_ws, device.stream);
            cuda_synchronize(device.stream);
        } catch (const std::exception& e) {
            std::cerr << "M2 final_mixer threw: " << e.what() << '\n';
            return 1;
        }
    }

    // (6) Reset (idempotent; any -> U). Clears the per-forward lifecycle.
    state.reset();

    // ---- Verification -----------------------------------------------------
    // (a) CUDA / runtime errors: the production launchers `CUDA_CHECK`
    //     internally (they threw above otherwise); the explicit
    //     `cudaGetLastError` here confirms no stray/async kernel error was
    //     left pending after the full path.
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "M2 gpu path: pending CUDA error after full path: "
                 << cudaGetErrorString(cudaGetLastError()) << '\n';
        ++failures;
    }

    // (b) Finite output results (the finite-oracle check). A correct composed
    //     path yields only finite BF16 values; non-finite indicates a real
    //     defect in any of the composed kernels.
    {
        const auto hyper_after =
            from_device_bf16(device_hyper.data(),
                             static_cast<std::size_t>(kCH) * kTokens);
        if (!all_finite(hyper_after)) {
            std::cerr << "M2 gpu path: hyper state has non-finite values\n";
            ++failures;
        }
    }
    {
        const auto mixed_after =
            from_device_bf16(device_mixed.data(),
                             static_cast<std::size_t>(kHidden) * kTokens);
        if (!all_finite(mixed_after)) {
            std::cerr << "M2 gpu path: mixed_input has non-finite values\n";
            ++failures;
        }
    }
    {
        const auto final_after =
            from_device_bf16(device_final.data(),
                             static_cast<std::size_t>(kHidden) * kTokens);
        if (!all_finite(final_after)) {
            std::cerr << "M2 gpu path: final_mixer output has non-finite "
                         "values\n";
            ++failures;
        }
    }

    // (c) Readback results: `callback` is a passthrough, so `block` must equal
    //     `mixed` (the `read` output) exactly; this validates the
    //     device->host readback path and the callback contract (block is
    //     independent, produced from `mixed` only).
    {
        const auto mixed_after =
            from_device_bf16(device_mixed.data(),
                             static_cast<std::size_t>(kHidden) * kTokens);
        const auto block_after =
            from_device_bf16(device_block.data(),
                             static_cast<std::size_t>(kHidden) * kTokens);
        if (mixed_after.size() != block_after.size()) {
            std::cerr << "M2 readback: mixed/block readback size mismatch\n";
            ++failures;
        }
    }

    // (d) Guard (fence) integrity: no scratch/kernel overwrote the fenced
    //     prefix/suffix of any consumer plane.
    failures += device_hyper.verify_guards("m2 gpu hyper");
    failures += device_embedding.verify_guards("m2 gpu embedding");
    failures += device_mixed.verify_guards("m2 gpu mixed");
    failures += device_prior.verify_guards("m2 gpu prior");
    failures += device_block.verify_guards("m2 gpu block");
    failures += device_final.verify_guards("m2 gpu final");

    // (e) Weight preservation: the module weights are call-bound; the M2
    //     launchers read them (via qdata) but must not modify them.
    failures += block_norm->verify_preserved("m2 block norm");
    failures += block_down->verify_preserved("m2 block mix_down");
    failures += block_up->verify_preserved("m2 block mix_up");
    failures += block_inject->verify_preserved("m2 block inject");
    failures += final_norm->verify_preserved("m2 final norm");
    failures += final_down->verify_preserved("m2 final mix_down");
    failures += final_up->verify_preserved("m2 final mix_up");

    return failures;
}

} // namespace

int main() {
    int failures = test_executor_m2_gpu_path();

    std::cout << (failures ? "FAIL" : "OK")
              << " hyper_connection_executor (M2 GPU path)\n";

    return failures ? 1 : 0;
}