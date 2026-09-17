#include "ninfer/ops/hyper_connection.h"
#include "ops/op_tester.h"

#include "core/decode_graph.h"
#include "core/device.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr float kEps = 1.0e-6F;

std::vector<std::uint16_t>
encode_bf16(std::vector<float> values) {
    round_to_bf16(values);

    std::vector<std::uint16_t> bits(values.size());

    for (std::size_t i = 0; i < values.size(); ++i) {
        bits[i] = f32_to_bf16(values[i]);
    }

    return bits;
}

std::vector<float>
rounded(std::vector<float> values) {
    round_to_bf16(values);
    return values;
}

constexpr PointwiseCriterion pointwise_criterion() {
    return {/*absolute*/ 4.0e-3, /*relative*/ 8.0e-3};
}

constexpr ReductionCriterion reduction_criterion() {
    return {
        /*relative_l2*/ 4.0e-3,
        /*gross_absolute*/ 8.0e-3,
        /*gross_relative_to_max_reference*/ 8.0e-3
    };
}

int test_repeat() {
    constexpr int hidden = 7;
    constexpr int streams = 4;
    constexpr int tokens = 3;

    std::vector<float> input(hidden * tokens);

    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) / 8.0F;
    }

    input = rounded(input);

    const auto input_bits = encode_bf16(input);

    GuardedDeviceBuffer device_input(input_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_output(
        hidden * streams * tokens * sizeof(std::uint16_t));

    device_input.copy_from_host(input_bits.data(), device_input.bytes());
    device_output.fill(0xff);

    Tensor input_tensor(
        device_input.data(), DType::BF16, {hidden, tokens});

    Tensor output_tensor(
        device_output.data(), DType::BF16, {hidden * streams, tokens});

    ops::hyper_connection_repeat(
        input_tensor, streams, output_tensor, nullptr);

    cuda_synchronize();

    std::vector<double> expected(hidden * streams * tokens);

    for (int t = 0; t < tokens; ++t) {
        for (int c = 0; c < streams; ++c) {
            for (int h = 0; h < hidden; ++h) {
                expected[
                    static_cast<std::size_t>(t) * hidden * streams +
                    c * hidden + h] =
                    input[static_cast<std::size_t>(t) * hidden + h];
            }
        }
    }

    int failures = verify_pointwise(
        "hyper_connection repeat",
        from_device_bf16(device_output.data(), expected.size()),
        expected,
        pointwise_criterion());

    failures += verify_exact(
        "hyper_connection repeat preserves input",
        from_device<std::uint16_t>(
            device_input.data(), input_bits.size()),
        input_bits);

    failures += device_input.verify_guards("repeat input");
    failures += device_output.verify_guards("repeat output");

    return failures;
}

int test_group_rmsnorm() {
    constexpr int hidden = 9;
    constexpr int streams = 4;
    constexpr int tokens = 2;
    constexpr int expanded = hidden * streams;

    std::vector<float> input(expanded * tokens);
    std::vector<float> weight(expanded);

    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] =
            (static_cast<int>(i % 13) - 6) * 0.375F;
    }

    for (std::size_t i = 0; i < weight.size(); ++i) {
        weight[i] =
            0.5F + static_cast<float>(i % 7) * 0.125F;
    }

    input = rounded(input);
    weight = rounded(weight);

    const auto input_bits = encode_bf16(input);
    const auto weight_bits = encode_bf16(weight);

    GuardedDeviceBuffer device_input(
        input_bits.size() * sizeof(std::uint16_t));

    GuardedDeviceBuffer device_weight(
        weight_bits.size() * sizeof(std::uint16_t));

    GuardedDeviceBuffer device_output(
        input_bits.size() * sizeof(std::uint16_t));

    device_input.copy_from_host(input_bits.data(), device_input.bytes());
    device_weight.copy_from_host(weight_bits.data(), device_weight.bytes());
    device_output.fill(0xff);

    Tensor input_tensor(
        device_input.data(), DType::BF16, {expanded, tokens});

    Tensor weight_tensor(
        device_weight.data(), DType::BF16, {expanded});

    Tensor output_tensor(
        device_output.data(), DType::BF16, {expanded, tokens});

    ops::hyper_connection_group_rmsnorm(
        input_tensor,
        weight_tensor,
        hidden,
        streams,
        kEps,
        output_tensor,
        nullptr);

    cuda_synchronize();

    std::vector<double> expected(input.size());

    for (int t = 0; t < tokens; ++t) {
        for (int c = 0; c < streams; ++c) {
            const std::size_t base =
                static_cast<std::size_t>(t) * expanded +
                c * hidden;

            double sum_squares = 0.0;

            for (int h = 0; h < hidden; ++h) {
                const double value = input[base + h];
                sum_squares += value * value;
            }

            const double inverse =
                1.0 / std::sqrt(
                    sum_squares / static_cast<double>(hidden) +
                    static_cast<double>(kEps));

            for (int h = 0; h < hidden; ++h) {
                expected[base + h] =
                    static_cast<double>(input[base + h]) *
                    inverse *
                    static_cast<double>(
                        weight[c * hidden + h]);
            }
        }
    }

    int failures = verify_reduction(
        "hyper_connection grouped rmsnorm",
        from_device_bf16(device_output.data(), expected.size()),
        expected,
        reduction_criterion());

    failures += verify_exact(
        "group rmsnorm preserves input",
        from_device<std::uint16_t>(
            device_input.data(), input_bits.size()),
        input_bits);

    failures += verify_exact(
        "group rmsnorm preserves weight",
        from_device<std::uint16_t>(
            device_weight.data(), weight_bits.size()),
        weight_bits);

    failures += device_input.verify_guards("group rmsnorm input");
    failures += device_weight.verify_guards("group rmsnorm weight");
    failures += device_output.verify_guards("group rmsnorm output");

    return failures;
}

int test_silu_divide() {
    std::vector<float> input{
        -8.0F, -2.0F, -0.5F, 0.0F,
         0.5F,  2.0F,  8.0F, 1.25F
    };

    input = rounded(input);

    const auto input_bits = encode_bf16(input);

    GuardedDeviceBuffer device_input(
        input_bits.size() * sizeof(std::uint16_t));

    GuardedDeviceBuffer device_output(
        input_bits.size() * sizeof(std::uint16_t));

    device_input.copy_from_host(input_bits.data(), device_input.bytes());
    device_output.fill(0xff);

    Tensor input_tensor(
        device_input.data(), DType::BF16,
        {4, static_cast<int>(input.size() / 4)});

    Tensor output_tensor(
        device_output.data(), DType::BF16,
        {4, static_cast<int>(input.size() / 4)});

    ops::hyper_connection_silu_divide(
        input_tensor, 4.0F, output_tensor, nullptr);

    cuda_synchronize();

    std::vector<double> expected(input.size());

    for (std::size_t i = 0; i < input.size(); ++i) {
        const double x =
            static_cast<double>(input[i]) / 4.0;

        expected[i] =
            x / (1.0 + std::exp(-x));
    }

    int failures = verify_pointwise(
        "hyper_connection silu divide",
        from_device_bf16(device_output.data(), expected.size()),
        expected,
        pointwise_criterion());

    failures += verify_exact(
        "silu divide preserves input",
        from_device<std::uint16_t>(
            device_input.data(), input_bits.size()),
        input_bits);

    failures += device_input.verify_guards("silu divide input");
    failures += device_output.verify_guards("silu divide output");

    return failures;
}

int test_read_mix() {
    constexpr int hidden = 5;
    constexpr int streams = 4;
    constexpr int tokens = 3;
    constexpr int expanded = hidden * streams;

    std::vector<float> normalized(expanded * tokens);
    std::vector<float> logits(expanded * tokens);

    for (std::size_t i = 0; i < normalized.size(); ++i) {
        normalized[i] =
            (static_cast<int>(i % 11) - 5) * 0.25F;

        logits[i] =
            (static_cast<int>(i % 9) - 4) * 0.5F;
    }

    normalized = rounded(normalized);
    logits = rounded(logits);

    const auto normalized_bits = encode_bf16(normalized);
    const auto logits_bits = encode_bf16(logits);

    GuardedDeviceBuffer device_normalized(
        normalized_bits.size() * sizeof(std::uint16_t));

    GuardedDeviceBuffer device_logits(
        logits_bits.size() * sizeof(std::uint16_t));

    GuardedDeviceBuffer device_output(
        hidden * tokens * sizeof(std::uint16_t));

    device_normalized.copy_from_host(
        normalized_bits.data(), device_normalized.bytes());

    device_logits.copy_from_host(
        logits_bits.data(), device_logits.bytes());

    device_output.fill(0xff);

    Tensor normalized_tensor(
        device_normalized.data(), DType::BF16,
        {expanded, tokens});

    Tensor logits_tensor(
        device_logits.data(), DType::BF16,
        {expanded, tokens});

    Tensor output_tensor(
        device_output.data(), DType::BF16,
        {hidden, tokens});

    ops::hyper_connection_read_mix(
        normalized_tensor,
        logits_tensor,
        hidden,
        streams,
        output_tensor,
        nullptr);

    cuda_synchronize();

    std::vector<double> expected(hidden * tokens);

    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < hidden; ++h) {
            double sum = 0.0;

            for (int c = 0; c < streams; ++c) {
                const std::size_t offset =
                    static_cast<std::size_t>(t) * expanded +
                    c * hidden + h;

                const double gate =
                    1.0 /
                    (1.0 + std::exp(
                        -static_cast<double>(logits[offset])));

                sum += gate *
                       static_cast<double>(
                           normalized[offset]);
            }

            expected[
                static_cast<std::size_t>(t) * hidden + h] =
                sum / static_cast<double>(streams);
        }
    }

    int failures = verify_pointwise(
        "hyper_connection read mix",
        from_device_bf16(device_output.data(), expected.size()),
        expected,
        pointwise_criterion());

    failures += verify_exact(
        "read mix preserves normalized",
        from_device<std::uint16_t>(
            device_normalized.data(), normalized_bits.size()),
        normalized_bits);

    failures += verify_exact(
        "read mix preserves logits",
        from_device<std::uint16_t>(
            device_logits.data(), logits_bits.size()),
        logits_bits);

    failures += device_normalized.verify_guards("read mix normalized");
    failures += device_logits.verify_guards("read mix logits");
    failures += device_output.verify_guards("read mix output");

    return failures;
}

int test_inject_update() {
    constexpr int hidden = 6;
    constexpr int streams = 4;
    constexpr int tokens = 2;
    constexpr int expanded = hidden * streams;

    std::vector<float> block_output(hidden * tokens);
    std::vector<float> inject_logits(streams * tokens);
    std::vector<float> hyper(expanded * tokens);

    for (std::size_t i = 0; i < block_output.size(); ++i) {
        block_output[i] =
            (static_cast<int>(i % 7) - 3) * 0.5F;
    }

    for (std::size_t i = 0; i < inject_logits.size(); ++i) {
        inject_logits[i] =
            (static_cast<int>(i % 5) - 2) * 1.25F;
    }

    for (std::size_t i = 0; i < hyper.size(); ++i) {
        hyper[i] =
            (static_cast<int>(i % 13) - 6) * 0.125F;
    }

    block_output = rounded(block_output);
    inject_logits = rounded(inject_logits);
    hyper = rounded(hyper);

    const auto block_bits = encode_bf16(block_output);
    const auto inject_bits = encode_bf16(inject_logits);
    const auto hyper_bits = encode_bf16(hyper);

    GuardedDeviceBuffer device_block(
        block_bits.size() * sizeof(std::uint16_t));

    GuardedDeviceBuffer device_inject(
        inject_bits.size() * sizeof(std::uint16_t));

    GuardedDeviceBuffer device_hyper(
        hyper_bits.size() * sizeof(std::uint16_t));

    device_block.copy_from_host(
        block_bits.data(), device_block.bytes());

    device_inject.copy_from_host(
        inject_bits.data(), device_inject.bytes());

    device_hyper.copy_from_host(
        hyper_bits.data(), device_hyper.bytes());

    Tensor block_tensor(
        device_block.data(), DType::BF16,
        {hidden, tokens});

    Tensor inject_tensor(
        device_inject.data(), DType::BF16,
        {streams, tokens});

    Tensor hyper_tensor(
        device_hyper.data(), DType::BF16,
        {expanded, tokens});

    ops::hyper_connection_inject_update(
        block_tensor,
        inject_tensor,
        hidden,
        streams,
        hyper_tensor,
        nullptr);

    cuda_synchronize();

    std::vector<double> expected(hyper.size());

    for (int t = 0; t < tokens; ++t) {
        for (int c = 0; c < streams; ++c) {
            const double logit =
                static_cast<double>(
                    inject_logits[
                        static_cast<std::size_t>(t) * streams + c]);

            const double alpha =
                2.0 /
                (1.0 + std::exp(
                    -logit /
                    static_cast<double>(streams)));

            for (int h = 0; h < hidden; ++h) {
                const std::size_t hyper_index =
                    static_cast<std::size_t>(t) * expanded +
                    c * hidden + h;

                const std::size_t block_index =
                    static_cast<std::size_t>(t) * hidden + h;

                expected[hyper_index] =
                    static_cast<double>(hyper[hyper_index]) +
                    static_cast<double>(
                        block_output[block_index]) *
                    alpha;
            }
        }
    }

    int failures = verify_pointwise(
        "hyper_connection inject update",
        from_device_bf16(device_hyper.data(), expected.size()),
        expected,
        pointwise_criterion());

    failures += verify_exact(
        "inject update preserves block output",
        from_device<std::uint16_t>(
            device_block.data(), block_bits.size()),
        block_bits);

    failures += verify_exact(
        "inject update preserves logits",
        from_device<std::uint16_t>(
            device_inject.data(), inject_bits.size()),
        inject_bits);

    failures += device_block.verify_guards("inject block");
    failures += device_inject.verify_guards("inject logits");
    failures += device_hyper.verify_guards("inject hyper");

    return failures;
}

int test_invalid_geometry() {
    GuardedDeviceBuffer a(16 * sizeof(std::uint16_t));
    GuardedDeviceBuffer b(16 * sizeof(std::uint16_t));

    Tensor x(a.data(), DType::BF16, {4, 4});
    Tensor y(b.data(), DType::BF16, {4, 4});

    try {
        ops::hyper_connection_repeat(x, 0, y, nullptr);
        std::cerr << "FAIL invalid hc_count accepted\n";
        return 1;
    } catch (const std::invalid_argument&) {
    }

    return 0;
}

} // namespace


int test_canonical_flash_next_geometry() {
    constexpr int hidden = 2560;
    constexpr int streams = 4;
    constexpr int lowrank = 320;
    constexpr int tokens = 2;
    constexpr int expanded = hidden * streams;

    std::vector<float> input(hidden * tokens, 1.0F);
    std::vector<float> weight(expanded, 1.0F);
    std::vector<float> logits(expanded * tokens, 0.0F);
    std::vector<float> lowrank_input(lowrank * tokens, 0.0F);
    std::vector<float> block_output(hidden * tokens, 0.5F);
    std::vector<float> inject_logits(streams * tokens, 0.0F);

    input = rounded(input);
    weight = rounded(weight);
    logits = rounded(logits);
    lowrank_input = rounded(lowrank_input);
    block_output = rounded(block_output);
    inject_logits = rounded(inject_logits);

    const auto input_bits = encode_bf16(input);
    const auto weight_bits = encode_bf16(weight);
    const auto logits_bits = encode_bf16(logits);
    const auto lowrank_bits = encode_bf16(lowrank_input);
    const auto block_bits = encode_bf16(block_output);
    const auto inject_bits = encode_bf16(inject_logits);

    GuardedDeviceBuffer device_input(input_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_logits(logits_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_lowrank(lowrank_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_block(block_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_inject(inject_bits.size() * sizeof(std::uint16_t));

    GuardedDeviceBuffer device_hyper(
        static_cast<std::size_t>(expanded) * tokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_norm(
        static_cast<std::size_t>(expanded) * tokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_mixed(
        static_cast<std::size_t>(hidden) * tokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_lowrank_out(
        static_cast<std::size_t>(lowrank) * tokens * sizeof(std::uint16_t));

    device_input.copy_from_host(input_bits.data(), device_input.bytes());
    device_weight.copy_from_host(weight_bits.data(), device_weight.bytes());
    device_logits.copy_from_host(logits_bits.data(), device_logits.bytes());
    device_lowrank.copy_from_host(lowrank_bits.data(), device_lowrank.bytes());
    device_block.copy_from_host(block_bits.data(), device_block.bytes());
    device_inject.copy_from_host(inject_bits.data(), device_inject.bytes());

    device_hyper.fill(0);
    device_norm.fill(0);
    device_mixed.fill(0);
    device_lowrank_out.fill(0);

    Tensor input_tensor(device_input.data(), DType::BF16, {hidden, tokens});
    Tensor weight_tensor(device_weight.data(), DType::BF16, {expanded});
    Tensor logits_tensor(device_logits.data(), DType::BF16, {expanded, tokens});
    Tensor lowrank_tensor(device_lowrank.data(), DType::BF16, {lowrank, tokens});
    Tensor block_tensor(device_block.data(), DType::BF16, {hidden, tokens});
    Tensor inject_tensor(device_inject.data(), DType::BF16, {streams, tokens});

    Tensor hyper_tensor(device_hyper.data(), DType::BF16, {expanded, tokens});
    Tensor norm_tensor(device_norm.data(), DType::BF16, {expanded, tokens});
    Tensor mixed_tensor(device_mixed.data(), DType::BF16, {hidden, tokens});
    Tensor lowrank_out_tensor(
        device_lowrank_out.data(), DType::BF16, {lowrank, tokens});

    DeviceContext device;

    ops::hyper_connection_repeat(
        input_tensor, streams, hyper_tensor, device.stream);

    cuda_synchronize(device.stream);

    const auto repeated_bits =
        from_device<std::uint16_t>(
            device_hyper.data(),
            static_cast<std::size_t>(expanded) * tokens);

    std::vector<std::uint16_t> expected_repeat(repeated_bits.size());
    std::fill(
        expected_repeat.begin(),
        expected_repeat.end(),
        f32_to_bf16(1.0F));

    int failures = verify_exact(
        "hyper_connection canonical repeat",
        repeated_bits,
        expected_repeat);

    ops::hyper_connection_group_rmsnorm(
        hyper_tensor,
        weight_tensor,
        hidden,
        streams,
        kEps,
        norm_tensor,
        device.stream);

    ops::hyper_connection_silu_divide(
        lowrank_tensor,
        static_cast<float>(streams),
        lowrank_out_tensor,
        device.stream);

    ops::hyper_connection_read_mix(
        norm_tensor,
        logits_tensor,
        hidden,
        streams,
        mixed_tensor,
        device.stream);

    ops::hyper_connection_inject_update(
        block_tensor,
        inject_tensor,
        hidden,
        streams,
        hyper_tensor,
        device.stream);

    cuda_synchronize(device.stream);

    const double inverse =
        1.0 / std::sqrt(1.0 + static_cast<double>(kEps));

    std::vector<double> expected_norm(
        static_cast<std::size_t>(expanded) * tokens,
        inverse);

    failures += verify_reduction(
        "hyper_connection canonical grouped rmsnorm",
        from_device_bf16(
            device_norm.data(),
            expected_norm.size()),
        expected_norm,
        reduction_criterion());

    // The read kernel consumes the represented BF16 normalized values.
    const auto represented_norm =
        from_device_bf16(
            device_norm.data(),
            static_cast<std::size_t>(expanded) * tokens);

    std::vector<double> expected_mix(
        static_cast<std::size_t>(hidden) * tokens);

    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < hidden; ++h) {
            double sum = 0.0;

            for (int c = 0; c < streams; ++c) {
                const std::size_t offset =
                    static_cast<std::size_t>(t) * expanded +
                    c * hidden +
                    h;

                sum +=
                    0.5 *
                    static_cast<double>(represented_norm[offset]);
            }

            expected_mix[
                static_cast<std::size_t>(t) * hidden + h] =
                sum / streams;
        }
    }

    failures += verify_pointwise(
        "hyper_connection canonical read mix",
        from_device_bf16(
            device_mixed.data(),
            expected_mix.size()),
        expected_mix,
        pointwise_criterion());

    std::vector<double> expected_lowrank(
        static_cast<std::size_t>(lowrank) * tokens,
        0.0);

    failures += verify_pointwise(
        "hyper_connection canonical silu divide",
        from_device_bf16(
            device_lowrank_out.data(),
            expected_lowrank.size()),
        expected_lowrank,
        pointwise_criterion());

    std::vector<double> expected_hyper(
        static_cast<std::size_t>(expanded) * tokens,
        1.5);

    failures += verify_pointwise(
        "hyper_connection canonical inject update",
        from_device_bf16(
            device_hyper.data(),
            expected_hyper.size()),
        expected_hyper,
        pointwise_criterion());

    failures += device_input.verify_guards("canonical input");
    failures += device_weight.verify_guards("canonical weight");
    failures += device_logits.verify_guards("canonical logits");
    failures += device_lowrank.verify_guards("canonical lowrank");
    failures += device_block.verify_guards("canonical block");
    failures += device_inject.verify_guards("canonical inject");
    failures += device_hyper.verify_guards("canonical hyper");
    failures += device_norm.verify_guards("canonical norm");
    failures += device_mixed.verify_guards("canonical mixed");
    failures += device_lowrank_out.verify_guards("canonical lowrank out");

    return failures;
}

int test_cuda_graph_replay() {
    constexpr int hidden = 17;
    constexpr int streams = 4;
    constexpr int lowrank = 11;
    constexpr int tokens = 3;
    constexpr int expanded = hidden * streams;

    std::vector<float> input(hidden * tokens);
    std::vector<float> weight(expanded);
    std::vector<float> logits(expanded * tokens);
    std::vector<float> lowrank_input(lowrank * tokens);
    std::vector<float> block_output(hidden * tokens);
    std::vector<float> inject_logits(streams * tokens);

    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] =
            (static_cast<int>(i % 9) - 4) * 0.375F;
    }

    for (std::size_t i = 0; i < weight.size(); ++i) {
        weight[i] =
            0.625F +
            static_cast<float>(i % 5) * 0.125F;
    }

    for (std::size_t i = 0; i < logits.size(); ++i) {
        logits[i] =
            (static_cast<int>(i % 7) - 3) * 0.5F;
    }

    for (std::size_t i = 0; i < lowrank_input.size(); ++i) {
        lowrank_input[i] =
            (static_cast<int>(i % 11) - 5) * 0.25F;
    }

    for (std::size_t i = 0; i < block_output.size(); ++i) {
        block_output[i] =
            (static_cast<int>(i % 13) - 6) * 0.125F;
    }

    for (std::size_t i = 0; i < inject_logits.size(); ++i) {
        inject_logits[i] =
            (static_cast<int>(i % 5) - 2) * 0.75F;
    }

    input = rounded(input);
    weight = rounded(weight);
    logits = rounded(logits);
    lowrank_input = rounded(lowrank_input);
    block_output = rounded(block_output);
    inject_logits = rounded(inject_logits);

    const auto input_bits = encode_bf16(input);
    const auto weight_bits = encode_bf16(weight);
    const auto logits_bits = encode_bf16(logits);
    const auto lowrank_bits = encode_bf16(lowrank_input);
    const auto block_bits = encode_bf16(block_output);
    const auto inject_bits = encode_bf16(inject_logits);

    GuardedDeviceBuffer device_input(input_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_logits(logits_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_lowrank(lowrank_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_block(block_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_inject(inject_bits.size() * sizeof(std::uint16_t));

    GuardedDeviceBuffer device_hyper(
        static_cast<std::size_t>(expanded) * tokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_norm(
        static_cast<std::size_t>(expanded) * tokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_mixed(
        static_cast<std::size_t>(hidden) * tokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_lowrank_out(
        static_cast<std::size_t>(lowrank) * tokens * sizeof(std::uint16_t));

    device_input.copy_from_host(input_bits.data(), device_input.bytes());
    device_weight.copy_from_host(weight_bits.data(), device_weight.bytes());
    device_logits.copy_from_host(logits_bits.data(), device_logits.bytes());
    device_lowrank.copy_from_host(lowrank_bits.data(), device_lowrank.bytes());
    device_block.copy_from_host(block_bits.data(), device_block.bytes());
    device_inject.copy_from_host(inject_bits.data(), device_inject.bytes());

    Tensor input_tensor(device_input.data(), DType::BF16, {hidden, tokens});
    Tensor weight_tensor(device_weight.data(), DType::BF16, {expanded});
    Tensor logits_tensor(device_logits.data(), DType::BF16, {expanded, tokens});
    Tensor lowrank_tensor(device_lowrank.data(), DType::BF16, {lowrank, tokens});
    Tensor block_tensor(device_block.data(), DType::BF16, {hidden, tokens});
    Tensor inject_tensor(device_inject.data(), DType::BF16, {streams, tokens});

    Tensor hyper_tensor(device_hyper.data(), DType::BF16, {expanded, tokens});
    Tensor norm_tensor(device_norm.data(), DType::BF16, {expanded, tokens});
    Tensor mixed_tensor(device_mixed.data(), DType::BF16, {hidden, tokens});
    Tensor lowrank_out_tensor(
        device_lowrank_out.data(), DType::BF16, {lowrank, tokens});

    DeviceContext device;

    const auto body = [&] {
        ops::hyper_connection_repeat(
            input_tensor,
            streams,
            hyper_tensor,
            device.stream);

        ops::hyper_connection_group_rmsnorm(
            hyper_tensor,
            weight_tensor,
            hidden,
            streams,
            kEps,
            norm_tensor,
            device.stream);

        ops::hyper_connection_silu_divide(
            lowrank_tensor,
            static_cast<float>(streams),
            lowrank_out_tensor,
            device.stream);

        ops::hyper_connection_read_mix(
            norm_tensor,
            logits_tensor,
            hidden,
            streams,
            mixed_tensor,
            device.stream);

        ops::hyper_connection_inject_update(
            block_tensor,
            inject_tensor,
            hidden,
            streams,
            hyper_tensor,
            device.stream);
    };

    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;

    definition.capture(device.stream, body);
    graph.instantiate(definition);

    // Direct execution establishes the represented BF16 reference for
    // graph replay. Primitive numerical correctness is checked separately
    // against independent CPU oracles above.
    device_hyper.fill(0);
    device_norm.fill(0);
    device_mixed.fill(0);
    device_lowrank_out.fill(0);

    body();
    cuda_synchronize(device.stream);

    const auto direct_hyper =
        from_device<std::uint16_t>(
            device_hyper.data(),
            static_cast<std::size_t>(expanded) * tokens);

    const auto direct_norm =
        from_device<std::uint16_t>(
            device_norm.data(),
            static_cast<std::size_t>(expanded) * tokens);

    const auto direct_mixed =
        from_device<std::uint16_t>(
            device_mixed.data(),
            static_cast<std::size_t>(hidden) * tokens);

    const auto direct_lowrank =
        from_device<std::uint16_t>(
            device_lowrank_out.data(),
            static_cast<std::size_t>(lowrank) * tokens);

    device_hyper.fill(0);
    device_norm.fill(0);
    device_mixed.fill(0);
    device_lowrank_out.fill(0);

    graph.launch(device.stream);
    cuda_synchronize(device.stream);

    int failures = 0;

    failures += verify_exact(
        "hyper_connection graph hyper",
        from_device<std::uint16_t>(
            device_hyper.data(),
            direct_hyper.size()),
        direct_hyper);

    failures += verify_exact(
        "hyper_connection graph norm",
        from_device<std::uint16_t>(
            device_norm.data(),
            direct_norm.size()),
        direct_norm);

    failures += verify_exact(
        "hyper_connection graph mixed",
        from_device<std::uint16_t>(
            device_mixed.data(),
            direct_mixed.size()),
        direct_mixed);

    failures += verify_exact(
        "hyper_connection graph lowrank",
        from_device<std::uint16_t>(
            device_lowrank_out.data(),
            direct_lowrank.size()),
        direct_lowrank);

    failures += device_hyper.verify_guards("graph hyper");
    failures += device_norm.verify_guards("graph norm");
    failures += device_mixed.verify_guards("graph mixed");
    failures += device_lowrank_out.verify_guards("graph lowrank");

    return failures;
}

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    failures += test_repeat();
    failures += test_group_rmsnorm();
    failures += test_silu_divide();
    failures += test_read_mix();
    failures += test_inject_update();
    failures += test_invalid_geometry();
    failures += test_canonical_flash_next_geometry();
    failures += test_cuda_graph_replay();

    std::cout
        << (failures ? "FAIL" : "OK")
        << " hyper_connection\n";

    return failures ? 1 : 0;
}
