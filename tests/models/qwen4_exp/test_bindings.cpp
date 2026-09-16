#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/framing.h"
#include "models/qwen4_exp/bindings.h"
#include "models/qwen4_exp/config.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
namespace qwen = ninfer::models::qwen4_exp;
using namespace ninfer::test::artifact_fixture;

Json canonical_config() {
    Json layers = Json::array();
    for (std::uint32_t layer = 0; layer < 48; ++layer) {
        layers.push_back((layer + 1) % 4 == 0 ? "full_attention" : "linear_attention");
    }
    return {
        {"architectures", {"Qwen4ExpForConditionalGeneration"}},
        {"model_type", "qwen4_exp_text"},
        {"hidden_size", 2560},
        {"vocab_size", 248320},
        {"num_hidden_layers", 48},
        {"max_position_embeddings", 262144},
        {"tie_word_embeddings", false},
        {"rms_norm_eps", 1e-6},
        {"full_attention_interval", 4},
        {"layer_types", std::move(layers)},
        {"num_attention_heads", 24},
        {"num_key_value_heads", 2},
        {"head_dim", 256},
        {"partial_rotary_factor", 0.25},
        {"attention_bias", false},
        {"attention_dropout", 0.0},
        {"rope_parameters", {{"rope_theta", 10000000}, {"partial_rotary_factor", 0.25}}},
        {"linear_num_key_heads", 16},
        {"linear_num_value_heads", 48},
        {"linear_key_head_dim", 128},
        {"linear_value_head_dim", 128},
        {"linear_conv_kernel_dim", 4},
        {"output_gate_type", "sigmoid"},
        {"hc_count", 4},
        {"hc_lowrank", 320},
        {"num_experts", 512},
        {"num_experts_per_tok", 10},
        {"moe_intermediate_size", 640},
        {"shared_expert_intermediate_size", 640},
        {"norm_topk_prob", true},
        {"ngram_size", 3},
        {"ngram_vocab_size_base", 20000000},
        {"make_ngram_vocab_size_divisible_by", 128},
        {"heads_per_ngram", 8},
        {"split_ngram_parts", 128},
        {"ple_embed_dim", 2560},
        {"ple_conv_kernel_size", 4},
        {"ple_layer_ids", {2}},
        {"indexer_budget", 2048},
        {"indexer_compress_ratio", 4},
        {"indexer_head_dim", 128},
        {"indexer_kv_heads", 1},
        {"indexer_n_heads", 4},
        {"mtp_num_hidden_layers", 1},
        {"mtp_use_dedicated_embeddings", false},
    };
}

struct SparseModelFixture {
    std::filesystem::path directory;
    std::filesystem::path entry;
    Json root;
    std::uint64_t payload_bytes = 0;

    explicit SparseModelFixture(bool complete = true) {
        auto pattern = (std::filesystem::temp_directory_path() / "ninfer-qwen4-XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        const char* path = ::mkdtemp(buffer.data());
        if (!path) { throw std::runtime_error("cannot create Qwen4Exp fixture directory"); }
        directory = path;
        entry     = directory / "model.ninfer";
        root      = {
            {"components", {{"text", {{"config", canonical_config()}}}}},
            {"objects", Json::array()},
            {"bindings", Json::object()},
            {"uses", Json::array()},
            {"metadata", {{"name", "synthetic-qwen4-exp-contract"}}},
        };
        build_contract(complete);
    }

    ~SparseModelFixture() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }

    SparseModelFixture(const SparseModelFixture&)            = delete;
    SparseModelFixture& operator=(const SparseModelFixture&) = delete;

    void tensor(const std::string& id, Shape shape, QType format = QType::BF16) {
        const auto geometry = weight_geometry(format, QuantLayout::Contiguous, shape);
        payload_bytes       = (payload_bytes + 255) / 256 * 256;
        root["objects"].push_back({{"id", id},
                                   {"kind", "tensor"},
                                   {"shape", shape},
                                   {"format", format_name(format)},
                                   {"layout", layout_name(QuantLayout::Contiguous)},
                                   {"offset", payload_bytes},
                                   {"bytes", geometry.bytes}});
        payload_bytes += geometry.bytes;
    }

    void parameter(const std::string& name, Shape shape, QType format = QType::BF16) {
        tensor(name, std::move(shape), format);
        root["bindings"][name] = {{"object", name}};
    }

    void sliced(const std::string& name, const std::string& object, std::uint64_t begin,
                std::uint64_t end) {
        root["bindings"][name] = {
            {"parts", Json::array({{{"object", object}, {"range", {begin, end}}}})}};
    }

    void hyper_connection(const std::string& prefix, bool inject) {
        parameter(prefix + "/norm", {10240});
        parameter(prefix + "/down", {320, 10240});
        parameter(prefix + "/up", {10240, 320});
        if (inject) { parameter(prefix + "/block_inject", {4, 10240}); }
    }

    void gdn(const std::string& prefix) {
        parameter(prefix + "/a_log", {48}, QType::FP32);
        parameter(prefix + "/dt_bias", {48}, QType::FP32);
        parameter(prefix + "/convolution", {10240, 1, 4});
        parameter(prefix + "/in_proj_a", {48, 2560});
        parameter(prefix + "/in_proj_b", {48, 2560});
        parameter(prefix + "/qkv", {10240, 2560});
        parameter(prefix + "/z", {6144, 2560});
        parameter(prefix + "/norm", {128});
        parameter(prefix + "/output", {2560, 6144});
    }

    void qsa(const std::string& prefix) {
        parameter(prefix + "/query", {12288, 2560});
        parameter(prefix + "/key", {512, 2560});
        parameter(prefix + "/value", {512, 2560});
        parameter(prefix + "/output", {2560, 6144});
        parameter(prefix + "/query_norm", {256});
        parameter(prefix + "/key_norm", {256});
        parameter(prefix + "/indexer/query_key", {640, 2560});
        parameter(prefix + "/indexer/query_norm", {128});
        parameter(prefix + "/indexer/key_norm", {128});
    }

    void moe(const std::string& prefix, std::uint32_t layer) {
        parameter(prefix + "/router", {512, 2560});
        parameter(prefix + "/shared_gate", {1, 2560});
        parameter(prefix + "/shared/gate", {640, 2560});
        parameter(prefix + "/shared/up", {640, 2560});
        parameter(prefix + "/shared/down", {2560, 640});
        constexpr std::uint64_t projection_elements = 640ULL * 2560;
        constexpr std::uint64_t down_elements       = 2560ULL * 640;
        for (std::uint32_t expert = 0; expert < 512; ++expert) {
            const auto logical = std::uint64_t(layer) * 512 + expert;
            const auto base    = logical * 2 * projection_elements;
            const auto name = prefix + "/experts/" + std::to_string(expert);
            sliced(name + "/gate", "routed-gate-up-storage", base,
                   base + projection_elements);
            sliced(name + "/up", "routed-gate-up-storage", base + projection_elements,
                   base + 2 * projection_elements);
            const auto down = logical * down_elements;
            sliced(name + "/down", "routed-down-storage", down, down + down_elements);
        }
    }

    void ple(const std::string& prefix) {
        parameter(prefix + "/convolution", {10240, 1, 4});
        parameter(prefix + "/key", {10240, 2560});
        parameter(prefix + "/value", {2560, 2560});
        parameter(prefix + "/key_norm", {10240});
        parameter(prefix + "/query_norm", {10240});
        parameter(prefix + "/conv_norm", {10240});
        parameter(prefix + "/layer_multipliers", {3}, QType::INT64);
        parameter(prefix + "/ngram_heads_offsets", {16}, QType::INT64);
        parameter(prefix + "/ngram_heads_vocab_sizes", {16}, QType::INT64);
        constexpr std::uint64_t shard_elements = 2 * 160;
        for (std::uint32_t shard = 0; shard < 128; ++shard) {
            sliced(prefix + "/embedding/shards/" + std::to_string(shard),
                   "ple-embedding-storage", shard * shard_elements,
                   (shard + 1) * shard_elements);
        }
    }

    void build_contract(bool complete) {
        parameter("text/token_embedding", {248320, 2560});
        parameter("text/output_head", {248320, 2560});
        hyper_connection("text/hyper_connection", false);

        if (!complete) {
            hyper_connection("text/layers/0/attention_hyper_connection", true);
            hyper_connection("text/layers/0/mlp_hyper_connection", true);
            gdn("text/layers/0/gdn");
            return;
        }

        tensor("routed-gate-up-storage", {48ULL * 512 * 1280, 2560});
        tensor("routed-down-storage", {48ULL * 512 * 2560, 640});
        tensor("ple-embedding-storage", {128 * 2, 160});

        for (std::uint32_t layer = 0; layer < 48; ++layer) {
            const auto prefix = "text/layers/" + std::to_string(layer);
            hyper_connection(prefix + "/attention_hyper_connection", true);
            hyper_connection(prefix + "/mlp_hyper_connection", true);
            if ((layer + 1) % 4 == 0) {
                qsa(prefix + "/attention");
            } else {
                gdn(prefix + "/gdn");
            }
            moe(prefix + "/moe", layer);
            if (layer == 1) { ple(prefix + "/ple"); }
        }
    }

    void write() {
        root["files"] = Json::array({{{"path", nullptr}, {"payload_bytes", payload_bytes}}});
        const auto text = root.dump();
        std::array<std::byte, 32> header{};
        const std::array<unsigned char, 8> magic{'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
        for (std::size_t i = 0; i < magic.size(); ++i) { header[i] = std::byte(magic[i]); }
        put_word(header, 8, text.size(), 8);
        header[16] = std::byte{0x71};
        std::ofstream file(entry, std::ios::binary | std::ios::trunc);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.write(reinterpret_cast<const char*>(header.data()), header.size());
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
        const auto position = std::uint64_t(header.size()) + text.size();
        const auto aligned  = (position + 4095) / 4096 * 4096;
        const std::string padding(static_cast<std::size_t>(aligned - position), '\0');
        file.write(padding.data(), static_cast<std::streamsize>(padding.size()));
        file.close();
        std::filesystem::resize_file(entry, aligned + payload_bytes);
    }
};

qwen::TextWeights bind(SparseModelFixture& fixture) {
    fixture.write();
    Reader reader(fixture.entry);
    const auto config = qwen::parse_config(reader.directory());
    Binder binder(reader);
    return qwen::bind_text_weights(binder, config.text);
}

void complete_contract() {
    SparseModelFixture fixture;
    const auto weights = bind(fixture);
    require(weights.layers.size() == 48, "the full 48-layer backbone was not bound");
    const auto linear = std::count_if(weights.layers.begin(), weights.layers.end(), [](const auto& l) {
        return std::holds_alternative<qwen::GdnWeights>(l.mixer);
    });
    require(linear == 36 && weights.layers.size() - linear == 12,
            "the 36 GDN / 12 QSA split changed");
    require(!weights.hyper_connection.block_inject &&
                weights.layers[0].attention_hyper_connection.block_inject &&
                weights.layers[0].mlp_hyper_connection.block_inject,
            "global and per-layer HyperConnection inject contracts were conflated");
    const auto& gdn = std::get<qwen::GdnWeights>(weights.layers[0].mixer);
    require(gdn.qkv.shape == Shape{10240, 2560} && gdn.norm.shape == Shape{128},
            "GDN qkv or per-head normalization shape changed");
    const auto& qsa = std::get<qwen::QsaWeights>(weights.layers[3].mixer);
    require(qsa.query.shape == Shape{12288, 2560} &&
                qsa.indexer.query_key.shape == Shape{640, 2560},
            "doubled QSA query or indexer projection width changed");
    require(!weights.layers[0].ple && weights.layers[1].ple && !weights.layers[2].ple,
            "one-based PLE placement was not preserved at physical layer 1");
    const auto& ple = *weights.layers[1].ple;
    require(ple.embedding_shards.size() == 128 &&
                std::all_of(ple.embedding_shards.begin(), ple.embedding_shards.end(),
                            [](const auto& shard) {
                                return shard.residency == Residency::Host &&
                                       shard.shape == Shape{2, 160};
                            }),
            "PLE shards were not independently Host-addressable with head width 160");
    require(ple.layer_multipliers.size() == 3 &&
                ple.ngram_heads_offsets.size() == 16 &&
                ple.ngram_heads_vocab_sizes.size() == 16,
            "PLE semantic INT64 buffers were not retained as owned values");
    for (const auto& layer : weights.layers) {
        require(layer.moe.routed_experts.experts.size() == 512,
                "a layer does not expose all 512 logical experts");
        require(std::all_of(layer.moe.routed_experts.experts.begin(),
                            layer.moe.routed_experts.experts.end(), [](const auto& expert) {
                                return expert.gate.residency == Residency::Host &&
                                       expert.up.residency == Residency::Host &&
                                       expert.down.residency == Residency::Host;
                            }),
                "routed experts were incorrectly required to be Device-resident");
    }
    const auto& expert0 = weights.layers[0].moe.routed_experts.at(0);
    const auto& expert1 = weights.layers[0].moe.routed_experts.at(1);
    require(expert0.gate.name != expert1.gate.name &&
                expert0.gate.binding.parts[0].begin != expert1.gate.binding.parts[0].begin,
            "logical experts are not individually addressable inside fused storage");
}

void invalid_contracts() {
    SparseModelFixture fixture(false);
    const auto reject_current = [&](const char* message) {
        fixture.write();
        rejects(
            [&] {
                Reader reader(fixture.entry);
                const auto config = qwen::parse_config(reader.directory());
                Binder binder(reader);
                (void)qwen::bind_text_weights(binder, config.text);
            },
            message);
    };

    const auto inject = fixture.root["bindings"]["text/layers/0/attention_hyper_connection/block_inject"];
    fixture.root["bindings"].erase("text/layers/0/attention_hyper_connection/block_inject");
    reject_current("missing per-layer HyperConnection inject was accepted");
    fixture.root["bindings"]["text/layers/0/attention_hyper_connection/block_inject"] = inject;

    const auto qkv = fixture.root["bindings"]["text/layers/0/gdn/qkv"];
    fixture.root["bindings"]["text/layers/0/gdn/qkv"] =
        fixture.root["bindings"]["text/layers/0/gdn/z"];
    reject_current("wrong GDN qkv shape was accepted");
    fixture.root["bindings"]["text/layers/0/gdn/qkv"] = qkv;

    fixture.root["bindings"]["text/layers/0/attention/query"] = qkv;
    reject_current("QSA tensor namespace was accepted on a GDN layer");
    fixture.root["bindings"].erase("text/layers/0/attention/query");

    fixture.root["bindings"]["text/layers/0/ple/convolution"] =
        fixture.root["bindings"]["text/layers/0/gdn/qkv"];
    reject_current("PLE tensor was accepted on a non-PLE layer");
    fixture.root["bindings"].erase("text/layers/0/ple/convolution");

    fixture.root["bindings"]["text/hyper_connection/block_inject"] = inject;
    reject_current("model-level HyperConnection inject was accepted");
}

} // namespace

int main() {
    try {
        complete_contract();
        invalid_contracts();
        std::cout << "Qwen4Exp logical weight and normalized artifact binding checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
