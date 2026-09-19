#include "ninfer/ops/linear.h"

#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;


void expect_admitted(QType qtype, std::int32_t n, std::int32_t k, std::int32_t min_t,
                    std::int32_t max_t) {
    ops::linear_workspace_capacity_bytes(qtype, n, k, ops::LinearPolicy::A16Only, min_t, max_t);
}

void expect_rejected(std::int32_t n, std::int32_t k, std::int32_t min_t, std::int32_t max_t) {
    try {
        ops::linear_workspace_capacity_bytes(QType::BF16, n, k, ops::LinearPolicy::A16Only, min_t,
                                             max_t);
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("expected unsupported shape to reject but it did not");
}

int run_hyper_shape_admission() {
    int failures = 0;
    // The three HyperConnection projection geometries required by the frozen formulas:
    //   down  (320, 10240),  up  (10240, 320),  block-inject  (4, 10240).
    // The workspace-capacity selector exercises the BF16 kShapes admission without
    // launching a CUDA kernel, so this is a CPU-runnable oracle: every (N,K) in the
    // serving interval must resolve to a registered launch (no "unsupported shape").
    const std::vector<std::pair<std::int32_t, std::int32_t>> shapes = {
        {320, 10240},   // down
        {10240, 320},   // up
        {4, 10240},     // block-inject
    };
    const std::vector<std::pair<std::int32_t, std::int32_t>> intervals = {
        {1, 1},     // decode
        {1, 32},    // short prefill
        {1, 128},   // prefill
        {1, 1024},  // long prefill
    };
    for (const auto& shape : shapes) {
        for (const auto& interval : intervals) {
            try {
                expect_admitted(QType::BF16, shape.first, shape.second, interval.first,
                                interval.second);
            } catch (const std::exception& error) {
                std::cerr << "BF16 linear shape [" << shape.first << "," << shape.second
                          << "] T=[" << interval.first << "," << interval.second << "] failed to "
                             "admit: " << error.what() << '\n';
                ++failures;
            }
        }
    }
    // Control: an unregistered shape must still be rejected, proving the registry
    // is finite (no generic fallback was silently added).
    try {
        expect_rejected(320, 10241, 1, 1);
    } catch (const std::exception& error) {
        std::cerr << "BF16 linear unregistered-shape control failed: " << error.what() << '\n';
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    try {
        const int failures = run_hyper_shape_admission();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " BF16 HyperConnection linear shapes\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "BF16 HyperConnection linear shapes: " << error.what() << '\n';
        return 1;
    }
}