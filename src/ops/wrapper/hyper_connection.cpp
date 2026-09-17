#include "ninfer/ops/hyper_connection.h"

#include "ops/launcher/hyper_connection.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_bf16(const Tensor& tensor, const char* label) {
    if (tensor.dtype != DType::BF16) {
        throw std::invalid_argument(
            std::string("hyper_connection: ") + label + " must be BF16");
    }
}

void require_contiguous(const Tensor& tensor, const char* label) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(
            std::string("hyper_connection: ") + label + " must be contiguous");
    }
}

void require_data(const Tensor& tensor, const char* label) {
    if (tensor.data == nullptr) {
        throw std::invalid_argument(
            std::string("hyper_connection: ") + label + " data must be non-null");
    }
}

void require_matrix(const Tensor& tensor, const char* label) {
    if (tensor.ne[0] <= 0 || tensor.ne[1] <= 0 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(
            std::string("hyper_connection: ") + label + " must be rank-2 logical storage");
    }
}

std::int32_t expanded_width(std::int32_t hidden_size,
                            std::int32_t hc_count) {
    if (hidden_size <= 0 || hc_count <= 0) {
        throw std::invalid_argument(
            "hyper_connection: hidden_size and hc_count must be positive");
    }

    const auto width =
        static_cast<std::int64_t>(hidden_size) * hc_count;

    if (width > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error(
            "hyper_connection: expanded width exceeds Tensor dimension domain");
    }

    return static_cast<std::int32_t>(width);
}

void require_common_tensor(const Tensor& tensor, const char* label) {
    require_bf16(tensor, label);
    require_matrix(tensor, label);
    require_contiguous(tensor, label);
    require_data(tensor, label);
}

} // namespace

void hyper_connection_repeat(const Tensor& input,
                             std::int32_t hc_count,
                             Tensor& out,
                             cudaStream_t stream) {
    require_common_tensor(input, "input");
    require_common_tensor(out, "out");

    const auto hidden = input.ne[0];
    const auto expanded = expanded_width(hidden, hc_count);

    if (out.ne[0] != expanded || out.ne[1] != input.ne[1]) {
        throw std::invalid_argument(
            "hyper_connection_repeat: out must have shape [hc_count*hidden,T]");
    }

    detail::hyper_connection_repeat_launch(input, hc_count, out, stream);
}

void hyper_connection_group_rmsnorm(const Tensor& input,
                                    const Tensor& weight,
                                    std::int32_t hidden_size,
                                    std::int32_t hc_count,
                                    float eps,
                                    Tensor& out,
                                    cudaStream_t stream) {
    require_common_tensor(input, "input");
    require_common_tensor(out, "out");

    require_bf16(weight, "weight");
    require_contiguous(weight, "weight");
    require_data(weight, "weight");

    if (weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument(
            "hyper_connection_group_rmsnorm: weight must be 1-D");
    }

    if (!(std::isfinite(eps) && eps > 0.0F)) {
        throw std::invalid_argument(
            "hyper_connection_group_rmsnorm: eps must be positive and finite");
    }

    const auto expanded = expanded_width(hidden_size, hc_count);

    if (input.ne[0] != expanded ||
        out.ne[0] != expanded ||
        weight.ne[0] != expanded ||
        input.ne[1] != out.ne[1]) {
        throw std::invalid_argument(
            "hyper_connection_group_rmsnorm: incompatible geometry");
    }

    detail::hyper_connection_group_rmsnorm_launch(
        input, weight, hidden_size, hc_count, eps, out, stream);
}

void hyper_connection_silu_divide(const Tensor& input,
                                  float divisor,
                                  Tensor& out,
                                  cudaStream_t stream) {
    require_common_tensor(input, "input");
    require_common_tensor(out, "out");

    if (!(std::isfinite(divisor) && divisor > 0.0F)) {
        throw std::invalid_argument(
            "hyper_connection_silu_divide: divisor must be positive and finite");
    }

    if (input.ne[0] != out.ne[0] ||
        input.ne[1] != out.ne[1]) {
        throw std::invalid_argument(
            "hyper_connection_silu_divide: input/out shapes must match");
    }

    detail::hyper_connection_silu_divide_launch(input, divisor, out, stream);
}

void hyper_connection_read_mix(const Tensor& normalized,
                               const Tensor& logits,
                               std::int32_t hidden_size,
                               std::int32_t hc_count,
                               Tensor& out,
                               cudaStream_t stream) {
    require_common_tensor(normalized, "normalized");
    require_common_tensor(logits, "logits");
    require_common_tensor(out, "out");

    const auto expanded = expanded_width(hidden_size, hc_count);

    if (normalized.ne[0] != expanded ||
        logits.ne[0] != expanded ||
        normalized.ne[1] != logits.ne[1] ||
        out.ne[0] != hidden_size ||
        out.ne[1] != normalized.ne[1]) {
        throw std::invalid_argument(
            "hyper_connection_read_mix: incompatible geometry");
    }

    detail::hyper_connection_read_mix_launch(
        normalized, logits, hidden_size, hc_count, out, stream);
}

void hyper_connection_inject_update(const Tensor& block_output,
                                    const Tensor& inject_logits,
                                    std::int32_t hidden_size,
                                    std::int32_t hc_count,
                                    Tensor& hyper,
                                    cudaStream_t stream) {
    require_common_tensor(block_output, "block_output");
    require_common_tensor(inject_logits, "inject_logits");
    require_common_tensor(hyper, "hyper");

    const auto expanded = expanded_width(hidden_size, hc_count);

    if (block_output.ne[0] != hidden_size ||
        inject_logits.ne[0] != hc_count ||
        hyper.ne[0] != expanded ||
        block_output.ne[1] != hyper.ne[1] ||
        inject_logits.ne[1] != hyper.ne[1]) {
        throw std::invalid_argument(
            "hyper_connection_inject_update: incompatible geometry");
    }

    detail::hyper_connection_inject_update_launch(
        block_output, inject_logits, hidden_size, hc_count, hyper, stream);
}

} // namespace ninfer::ops
