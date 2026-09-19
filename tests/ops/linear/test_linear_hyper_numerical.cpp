#include "ninfer/ops/linear.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;

// BF16 A16 reduction criterion, identical to the existing BF16 linear suite
// (test_bf16_a16.cpp). {relative_l2, gross_absolute, gross_relative_to_max_reference}.
constexpr ReductionCriterion kHyperTolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

// Deterministic nonuniform activation pattern: the (token, column) value is a
// hash of its indices, so magnitudes and signs vary across both axes without relying on a
// constant or ramped input. This is the non-uniform input required to prove
// numerical execution (a constant or linearly-ramped input would alias the
// kernel's private accumulation order).
std::vector<std::uint16_t> make_hyper_activation(std::int32_t hidden, std::int32_t tokens,
                                                 std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(hidden) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < hidden; ++column) {
            std::uint32_t hash = static_cast<std::uint32_t>(column) * 0x9e3779b9U ^
                                 static_cast<std::uint32_t>(token) * 0x85ebca6bU ^
                                 seed * 0xc2b2ae35U;
            hash ^= hash >> 16;
            hash *= 0x7feb352dU;
            hash ^= hash >> 15;
            const int centered = static_cast<int>((hash >> 8) & 0xffU) - 128;
            result[static_cast<std::size_t>(token) * hidden + column] =
                f32_to_bf16(static_cast<float>(centered) * (1.0F / 1024.0F));
        }
    }
    return result;
}

std::vector<float> materialize(std::span<const std::uint16_t> bits) {
    std::vector<float> result(bits.size());
    for (std::size_t index = 0; index < bits.size(); ++index) {
        result[index] = bf16_to_f32(bits[index]);
    }
    return result;
}

std::vector<std::int32_t> sampled_rows(std::int32_t rows) {
    std::vector<std::int32_t> result{0, 1, rows / 4, rows / 2, (3 * rows) / 4, rows - 2, rows - 1};
    if (rows == 10240) {
        result.insert(result.end(), {511, 5120, 5121, 8191, 8192, 10239});
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::int32_t> sampled_tokens(std::int32_t tokens) {
    if (tokens <= 32) {
        std::vector<std::int32_t> result(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            result[static_cast<std::size_t>(token)] = token;
        }
        return result;
    }
    std::vector<std::int32_t> result{0, 1, tokens / 2, tokens - 2, tokens - 1};
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

// Runs one BF16 linear dispatch for the given shape at a token count and compares
// the device output against the trusted FP64 dot-product oracle.
int run_hyper_numerical_case(DeviceWeight& weight, std::int32_t tokens, bool replay = false) {
    const std::int32_t rows   = weight.host.n;
    const std::int32_t hidden = weight.host.k;
    const std::uint32_t seed  = 3207U + static_cast<std::uint32_t>(rows * 31 + hidden * 7);

    std::vector<std::uint16_t> activation_bits = make_hyper_activation(hidden, tokens, seed);
    std::vector<float> activation               = materialize(activation_bits);
    DeviceBuffer device_activation              = to_device(activation_bits);
    GuardedDeviceBuffer guarded_output(static_cast<std::size_t>(rows) * tokens *
                                       sizeof(std::uint16_t));
    guarded_output.fill(0xff);

    Tensor x(device_activation.p, DType::BF16, {hidden, tokens});
    Tensor output(guarded_output.data(), DType::BF16, {rows, tokens});
    const std::size_t capacity = ops::linear_workspace_capacity_bytes(
        QType::BF16, rows, hidden, ops::LinearPolicy::A16Only, tokens, tokens);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace, nullptr);
    cuda_synchronize();

    if (replay) {
        DeviceContext context;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(context.stream, [&] {
            ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace,
                        context.stream);
        });
        graph.instantiate(definition);
        graph.launch(context.stream);
        cuda_synchronize();
        for (auto& bits : activation_bits) bits ^= 0x8000;
        activation = materialize(activation_bits);
        device_activation.copy_from_host(activation_bits.data(), device_activation.bytes);
        guarded_output.fill(0xff);
        cuda_synchronize();
        graph.launch(context.stream);
        cuda_synchronize();
    }
    const std::string suffix = " T=" + std::to_string(tokens) + (replay ? " graph" : " eager");
    int failures             = guarded_output.verify_guards("BF16 Hyper linear output" + suffix);
    const std::vector<std::uint16_t> output_bits =
        from_device<std::uint16_t>(guarded_output.data(), static_cast<std::size_t>(rows) * tokens);
    for (auto& bits : output_bits) {
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "BF16 Hyper linear output" << suffix << " element is not finite\n";
            ++failures;
            break;
        }
    }

    std::vector<double> actual;
    std::vector<double> expected;
    if (tokens == 1) {
        // T=1: the FP64 oracle evaluates every output row (the existing convention).
        actual.reserve(rows);
        expected.reserve(rows);
        for (std::int32_t row = 0; row < rows; ++row) {
            actual.push_back(bf16_to_f32(output_bits[static_cast<std::size_t>(row)]));
            expected.push_back(
                dot_fp64(weight.host, row, std::span<const float>(activation.data(), hidden)));
        }
    } else {
        // T>1: sampled rows x sampled tokens, each compared against the FP64 oracle.
        const std::vector<std::int32_t> sampled       = sampled_rows(rows);
        const std::vector<std::int32_t> token_samples = sampled_tokens(tokens);
        actual.reserve(sampled.size() * token_samples.size());
        expected.reserve(actual.capacity());
        for (const std::int32_t row : sampled) {
            for (const std::int32_t token : token_samples) {
                actual.push_back(
                    bf16_to_f32(output_bits[static_cast<std::size_t>(token) * rows + row]));
                expected.push_back(
                    dot_fp64(weight.host, row,
                            std::span<const float>(
                                activation.data() + static_cast<std::size_t>(token) * hidden,
                                hidden)));
            }
        }
    }
    failures += verify_reduction("BF16 Hyper linear [" + std::to_string(rows) + "," +
                                     std::to_string(hidden) + "]" + suffix,
                                 actual, expected, kHyperTolerance);

    const std::vector<std::uint16_t> activation_after =
        from_device<std::uint16_t>(device_activation, activation_bits.size());
    if (activation_after != activation_bits) {
        std::cerr << "BF16 Hyper linear" << suffix << " modified its activation\n";
        ++failures;
    }
    failures += weight.verify_preserved("BF16 Hyper linear weight" + suffix);
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;

        // down: LinearDown [320,10240] — gemv (T=1), SIMT ladder (T=2..8), MMA (T>=9).
        {
            DeviceWeight weight(make_patterned(320, 10240, 3207U));
            for (std::int32_t tokens : {1, 2, 3, 4, 5, 8, 9, 16, 64, 128, 1024}) {
                failures += run_hyper_numerical_case(weight, tokens);
            }
            for (std::int32_t tokens : {2, 8, 16, 1024}) {
                failures += run_hyper_numerical_case(weight, tokens, /*replay=*/true);
            }
            failures += weight.verify_preserved("BF16 Hyper down weight");
        }

        // up: LinearUp [10240,320] — narrow reference GEMV fallback for every T.
        {
            DeviceWeight weight(make_patterned(10240, 320, 3219U));
            for (std::int32_t tokens : {1, 2, 4, 8, 16, 64, 128, 1024}) {
                failures += run_hyper_numerical_case(weight, tokens);
            }
            for (std::int32_t tokens : {2, 8, 16, 1024}) {
                failures += run_hyper_numerical_case(weight, tokens, /*replay=*/true);
            }
            failures += weight.verify_preserved("BF16 Hyper up weight");
        }

        // block-inject [4,10240] — gemv (T=1), SIMT ladder (T=2..4), reference GEMV (T>=5).
        {
            DeviceWeight weight(make_patterned(4, 10240, 3229U));
            for (std::int32_t tokens : {1, 2, 3, 4, 5, 16, 1024}) {
                failures += run_hyper_numerical_case(weight, tokens);
            }
            for (std::int32_t tokens : {2, 4, 16, 1024}) {
                failures += run_hyper_numerical_case(weight, tokens, /*replay=*/true);
            }
            failures += weight.verify_preserved("BF16 Hyper block-inject weight");
        }

        std::cout << (failures == 0 ? "OK" : "FAIL") << " BF16 HyperConnection linear shapes\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "BF16 HyperConnection linear shapes: " << error.what() << '\n';
        return 1;
    }
}