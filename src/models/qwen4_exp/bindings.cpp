#include "models/qwen4_exp/bindings.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::models::qwen4_exp {
namespace {

using artifact::ArtifactError;
using artifact::Binder;
using artifact::Residency;
using artifact::Shape;

[[noreturn]] void contract_error(std::string_view detail) {
    throw ArtifactError("qwen4_exp_text: " + std::string(detail));
}

void require(bool condition, std::string_view detail) {
    if (!condition) { contract_error(detail); }
}

void reject_namespace(const Binder& binder, const std::string& prefix, std::string_view detail) {
    const auto& bindings = binder.reader().directory().bindings;
    const auto found = std::find_if(bindings.begin(), bindings.end(), [&](const auto& entry) {
        return entry.first.starts_with(prefix);
    });
    if (found != bindings.end()) { contract_error(detail); }
}

Weight weight(Binder& binder, const std::string& name, Shape shape,
              Residency residency = Residency::Device,
              std::optional<QType> exact_format = {}) {
    return binder.parameter(name, std::move(shape), residency, exact_format);
}

HyperConnectionWeights bind_hyper_connection(Binder& binder, const std::string& prefix,
                                             std::uint64_t expanded_width,
                                             std::uint32_t low_rank, bool require_inject) {
    HyperConnectionWeights out{
        .norm = weight(binder, prefix + "/norm", {expanded_width}, Residency::Device,
                       QType::BF16),
        .down = weight(binder, prefix + "/down", {low_rank, expanded_width}),
        .up = weight(binder, prefix + "/up", {expanded_width, low_rank}),
    };
    const auto inject_name = prefix + "/block_inject";
    if (require_inject) {
        out.block_inject = weight(binder, inject_name, {4, expanded_width});
    } else if (binder.contains(inject_name)) {
        contract_error("the model-level HyperConnection must not contain block_inject");
    }
    return out;
}

GdnWeights bind_gdn(Binder& binder, const std::string& prefix, const TextConfig& config) {
    const auto key_width =
        artifact::checked_mul(config.gdn.linear_num_key_heads,
                              config.gdn.linear_key_head_dim, "GDN key width");
    const auto value_width =
        artifact::checked_mul(config.gdn.linear_num_value_heads,
                              config.gdn.linear_value_head_dim, "GDN value width");
    const auto expanded_width =
        artifact::checked_add(artifact::checked_mul(2, key_width, "GDN doubled key width"),
                              value_width, "GDN qkv width");
    return {
        .a_log = weight(binder, prefix + "/a_log", {config.gdn.linear_num_value_heads},
                        Residency::Device, QType::FP32),
        .dt_bias = weight(binder, prefix + "/dt_bias", {config.gdn.linear_num_value_heads},
                          Residency::Device, QType::FP32),
        .convolution = weight(binder, prefix + "/convolution",
                              {expanded_width, 1, config.gdn.linear_conv_kernel_dim}),
        .in_proj_a = weight(binder, prefix + "/in_proj_a",
                            {config.gdn.linear_num_value_heads, config.hidden_size}),
        .in_proj_b = weight(binder, prefix + "/in_proj_b",
                            {config.gdn.linear_num_value_heads, config.hidden_size}),
        .qkv = weight(binder, prefix + "/qkv", {expanded_width, config.hidden_size}),
        .z = weight(binder, prefix + "/z", {value_width, config.hidden_size}),
        .norm = weight(binder, prefix + "/norm", {config.gdn.linear_value_head_dim},
                       Residency::Device, QType::BF16),
        .output = weight(binder, prefix + "/output", {config.hidden_size, value_width}),
    };
}

QsaWeights bind_qsa(Binder& binder, const std::string& prefix, const TextConfig& config) {
    const auto query_width = std::uint64_t(config.attention.num_attention_heads) *
                             config.attention.head_dim;
    const auto kv_width = std::uint64_t(config.attention.num_key_value_heads) *
                          config.attention.head_dim;
    const auto index_width = std::uint64_t(config.indexer.indexer_n_heads +
                                           config.indexer.indexer_kv_heads) *
                             config.indexer.indexer_head_dim;
    return {
        .query = weight(binder, prefix + "/query", {2 * query_width, config.hidden_size}),
        .key = weight(binder, prefix + "/key", {kv_width, config.hidden_size}),
        .value = weight(binder, prefix + "/value", {kv_width, config.hidden_size}),
        .output = weight(binder, prefix + "/output", {config.hidden_size, query_width}),
        .query_norm = weight(binder, prefix + "/query_norm", {config.attention.head_dim},
                             Residency::Device, QType::BF16),
        .key_norm = weight(binder, prefix + "/key_norm", {config.attention.head_dim},
                           Residency::Device, QType::BF16),
        .indexer = {
            .query_key = weight(binder, prefix + "/indexer/query_key",
                                {index_width, config.hidden_size}),
            .query_norm = weight(binder, prefix + "/indexer/query_norm",
                                 {config.indexer.indexer_head_dim}, Residency::Device,
                                 QType::BF16),
            .key_norm = weight(binder, prefix + "/indexer/key_norm",
                               {config.indexer.indexer_head_dim}, Residency::Device,
                               QType::BF16),
        },
    };
}

DenseWeights bind_dense(Binder& binder, const std::string& prefix, std::uint32_t hidden_size,
                        std::uint32_t intermediate_size,
                        Residency residency = Residency::Device) {
    return {
        .gate = weight(binder, prefix + "/gate", {intermediate_size, hidden_size}, residency),
        .up = weight(binder, prefix + "/up", {intermediate_size, hidden_size}, residency),
        .down = weight(binder, prefix + "/down", {hidden_size, intermediate_size}, residency),
    };
}

MoeWeights bind_moe(Binder& binder, const std::string& prefix, const TextConfig& config) {
    MoeWeights out{
        .router = weight(binder, prefix + "/router",
                         {config.moe.num_experts, config.hidden_size}),
        .shared_gate = weight(binder, prefix + "/shared_gate", {1, config.hidden_size}),
        .shared_expert = bind_dense(binder, prefix + "/shared", config.hidden_size,
                                    config.moe.shared_expert_intermediate_size),
    };
    out.routed_experts.experts.reserve(config.moe.num_experts);
    for (std::uint32_t expert = 0; expert < config.moe.num_experts; ++expert) {
        out.routed_experts.experts.push_back(
            bind_dense(binder, prefix + "/experts/" + std::to_string(expert),
                       config.hidden_size, config.moe.moe_intermediate_size, Residency::Host));
    }
    return out;
}

std::vector<std::int64_t> bind_i64_values(Binder& binder, const std::string& name,
                                                Shape shape) {
    const auto reference =
        weight(binder, name, std::move(shape), Residency::Values, QType::INT64);
    return binder.values(reference.binding, QType::INT64).integers64();
}

Weight bind_ple_shard(Binder& binder, const std::string& name, std::uint64_t head_dim) {
    const auto found = binder.reader().directory().bindings.find(name);
    if (found == binder.reader().directory().bindings.end()) {
        throw ArtifactError("missing logical parameter " + name);
    }
    require(found->second.elements > 0 && found->second.elements % head_dim == 0,
            "PLE embedding shard must be a non-empty matrix with the configured head width");
    return binder.binding(name, found->second, {found->second.elements / head_dim, head_dim},
                          Residency::Host);
}

PleWeights bind_ple(Binder& binder, const std::string& prefix, const TextConfig& config) {
    const auto expanded_width = config.hyper_connection.width(config.hidden_size);
    const auto head_dim       = config.ple.ple_embed_dim / config.ple.total_heads();
    PleWeights out{
        .convolution = weight(binder, prefix + "/convolution",
                              {expanded_width, 1, config.ple.ple_conv_kernel_size}),
        .key = weight(binder, prefix + "/key", {expanded_width, config.hidden_size}),
        .value = weight(binder, prefix + "/value", {config.ple.ple_embed_dim,
                                                    config.hidden_size}),
        .key_norm = weight(binder, prefix + "/key_norm", {expanded_width}, Residency::Device,
                           QType::BF16),
        .query_norm = weight(binder, prefix + "/query_norm", {expanded_width},
                             Residency::Device, QType::BF16),
        .conv_norm = weight(binder, prefix + "/conv_norm", {expanded_width}, Residency::Device,
                            QType::BF16),
        .layer_multipliers =
            bind_i64_values(binder, prefix + "/layer_multipliers", {config.ple.ngram_size}),
        .ngram_heads_offsets =
            bind_i64_values(binder, prefix + "/ngram_heads_offsets",
                            {config.ple.total_heads()}),
        .ngram_heads_vocab_sizes =
            bind_i64_values(binder, prefix + "/ngram_heads_vocab_sizes",
                            {config.ple.total_heads()}),
    };
    out.embedding_shards.reserve(config.ple.split_ngram_parts);
    for (std::uint32_t shard = 0; shard < config.ple.split_ngram_parts; ++shard) {
        out.embedding_shards.push_back(bind_ple_shard(
            binder, prefix + "/embedding/shards/" + std::to_string(shard), head_dim));
    }
    return out;
}

void validate_frozen_contract(const TextConfig& config) {
    require(config.architecture == Architecture::Qwen4Exp, "wrong architecture identity");
    require(config.hidden_size == 2560 && config.vocab_size == 248320 &&
                config.num_hidden_layers == 48,
            "backbone geometry differs from Flash-Next");
    require(config.layer_types.size() == config.num_hidden_layers,
            "layer_types length differs from num_hidden_layers");
    const auto full = std::count(config.layer_types.begin(), config.layer_types.end(),
                                 LayerKind::FullAttention);
    const auto linear = std::count(config.layer_types.begin(), config.layer_types.end(),
                                   LayerKind::LinearAttention);
    require(full == 12 && linear == 36, "Flash-Next requires 36 GDN and 12 QSA layers");
    require(config.hyper_connection.hc_count == 4 &&
                config.hyper_connection.hc_lowrank == 320,
            "HyperConnection geometry differs from Flash-Next");
    require(config.moe.num_experts == 512 && config.moe.moe_intermediate_size == 640 &&
                config.moe.shared_expert_intermediate_size == 640,
            "MoE geometry differs from Flash-Next");
    require(config.ple.split_ngram_parts == 128 && config.ple.total_heads() == 16 &&
                config.ple.ple_embed_dim == 2560 && config.ple.ple_layer_ids.size() == 1 &&
                config.ple.ple_layer_ids.front() == 2,
            "PLE geometry or one-based placement differs from Flash-Next");
    require(config.ple.ple_embed_dim % config.ple.total_heads() == 0,
            "PLE embedding width must divide evenly across its heads");
}

} // namespace

TextWeights bind_text_weights(Binder& binder, const TextConfig& config) {
    validate_frozen_contract(config);
    const auto expanded_width = config.hyper_connection.width(config.hidden_size);
    TextWeights out{
        .token_embedding = weight(binder, "text/token_embedding",
                                  {config.vocab_size, config.hidden_size}),
        .output_head = weight(binder, "text/output_head",
                              {config.vocab_size, config.hidden_size}),
        .hyper_connection = bind_hyper_connection(binder, "text/hyper_connection",
                                                  expanded_width,
                                                  config.hyper_connection.hc_lowrank, false),
    };
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const auto prefix = "text/layers/" + std::to_string(layer);
        auto attention_hyper_connection = bind_hyper_connection(
            binder, prefix + "/attention_hyper_connection", expanded_width,
            config.hyper_connection.hc_lowrank, true);
        auto mlp_hyper_connection = bind_hyper_connection(
            binder, prefix + "/mlp_hyper_connection", expanded_width,
            config.hyper_connection.hc_lowrank, true);
        MixerWeights mixer = GdnWeights{};
        if (config.layer_types[layer] == LayerKind::LinearAttention) {
            reject_namespace(binder, prefix + "/attention/",
                             "GDN layer contains a QSA attention namespace");
            mixer = bind_gdn(binder, prefix + "/gdn", config);
        } else {
            reject_namespace(binder, prefix + "/gdn/", "QSA layer contains a GDN namespace");
            mixer = bind_qsa(binder, prefix + "/attention", config);
        }
        std::optional<PleWeights> ple;
        if (config.ple.has_ple_at_layer(layer)) {
            ple = bind_ple(binder, prefix + "/ple", config);
        } else {
            reject_namespace(binder, prefix + "/ple/", "PLE tensors occur on a non-PLE layer");
        }
        out.layers.push_back({
            .attention_hyper_connection = std::move(attention_hyper_connection),
            .mlp_hyper_connection = std::move(mlp_hyper_connection),
            .mixer = std::move(mixer),
            .ple = std::move(ple),
            .moe = bind_moe(binder, prefix + "/moe", config),
        });
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp
