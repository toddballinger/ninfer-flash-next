#pragma once

#include "models/registry.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::qwen4_exp {

enum class LayerKind { FullAttention, LinearAttention };

struct AttentionConfig {
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    float partial_rotary_factor       = 0;
    float rope_theta                  = 0;
    bool attention_bias               = false;
    float attention_dropout           = 0;
};

struct GdnConfig {
    std::uint32_t linear_num_key_heads   = 0;
    std::uint32_t linear_num_value_heads = 0;
    std::uint32_t linear_key_head_dim    = 0;
    std::uint32_t linear_value_head_dim  = 0;
    std::uint32_t linear_conv_kernel_dim = 0;
    std::string output_gate_type;
};

struct HyperConnectionConfig {
    std::uint32_t hc_count   = 0;
    std::uint32_t hc_lowrank = 0;

    [[nodiscard]] std::uint64_t width(std::uint32_t hidden_size) const noexcept {
        return std::uint64_t(hc_count) * hidden_size;
    }
};

struct MoeConfig {
    std::uint32_t num_experts                     = 0;
    std::uint32_t num_experts_per_tok             = 0;
    std::uint32_t moe_intermediate_size           = 0;
    std::uint32_t shared_expert_intermediate_size = 0;
    bool norm_topk_prob                           = false;
};

struct PleConfig {
    std::uint32_t ngram_size                         = 0;
    std::uint64_t ngram_vocab_size_base              = 0;
    std::uint32_t make_ngram_vocab_size_divisible_by = 0;
    std::uint32_t heads_per_ngram                    = 0;
    std::uint32_t split_ngram_parts                  = 0;
    std::uint32_t ple_embed_dim                      = 0;
    std::uint32_t ple_conv_kernel_size               = 0;
    std::uint32_t seed                               = 0;
    std::vector<std::uint32_t> ple_layer_ids;

    [[nodiscard]] std::uint64_t total_heads() const noexcept {
        return ngram_size > 1 ? std::uint64_t(ngram_size - 1) * heads_per_ngram : 0;
    }

    [[nodiscard]] bool has_ple_at_layer(std::uint32_t zero_based_layer_idx) const noexcept {
        const auto one_based = std::uint64_t(zero_based_layer_idx) + 1;
        return std::find(ple_layer_ids.begin(), ple_layer_ids.end(), one_based) !=
               ple_layer_ids.end();
    }
};

struct IndexerConfig {
    std::uint32_t indexer_budget         = 0;
    std::uint32_t indexer_compress_ratio = 0;
    std::uint32_t indexer_head_dim       = 0;
    std::uint32_t indexer_kv_heads       = 0;
    std::uint32_t indexer_n_heads        = 0;

    [[nodiscard]] std::uint32_t block_budget() const noexcept {
        return indexer_compress_ratio ? indexer_budget / indexer_compress_ratio : 0;
    }
};

struct MtpConfig {
    std::uint32_t mtp_num_hidden_layers = 0;
    bool mtp_use_dedicated_embeddings   = false;
};

struct TextConfig {
    Architecture architecture             = Architecture::Qwen4Exp;
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    std::uint32_t full_attention_interval = 0;
    bool tie_word_embeddings              = false;
    float rms_norm_eps                    = 0;
    std::vector<LayerKind> layer_types;
    AttentionConfig attention;
    GdnConfig gdn;
    HyperConnectionConfig hyper_connection;
    MoeConfig moe;
    PleConfig ple;
    IndexerConfig indexer;
    std::optional<MtpConfig> mtp;
};

struct Config {
    TextConfig text;
};

[[nodiscard]] Config parse_config(const artifact::Directory& directory);

} // namespace ninfer::models::qwen4_exp
