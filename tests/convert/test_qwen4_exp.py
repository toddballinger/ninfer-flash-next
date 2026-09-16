from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import struct

import pytest
import torch

from tools.artifact.reader import Artifact
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.model import Model
from tools.convert.qwen4_exp import build_model, text_config
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource


def _config():
    layers = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
        for i in range(48)
    ]
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "vocab_size": 248320,
            "num_hidden_layers": 48,
            "max_position_embeddings": 262144,
            "tie_word_embeddings": False,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 4,
            "layer_types": layers,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rope_parameters": {
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "output_gate_type": "sigmoid",
            "hc_count": 4,
            "hc_lowrank": 320,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "norm_topk_prob": True,
            "ngram_size": 3,
            "ngram_vocab_size_base": 20000000,
            "make_ngram_vocab_size_divisible_by": 128,
            "heads_per_ngram": 8,
            "split_ngram_parts": 128,
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "ple_layer_ids": [2],
            "seed": 1234,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 4,
        },
    }


def _tensor_specs():
    specs = {}

    def add(name, shape, dtype="BF16"):
        specs[name] = [tuple(shape), dtype]

    root = "model.language_model"
    add(root + ".embed_tokens.weight", (248320, 2560))
    add("lm_head.weight", (248320, 2560))
    for field, shape in (
        ("hc_norm.weight", (10240,)),
        ("input_mix_weight_down.weight", (320, 10240)),
        ("input_mix_weight_up.weight", (10240, 320)),
    ):
        add(root + ".hyper_connection_mixer." + field, shape)

    def hc(source):
        for field, shape in (
            ("hc_norm.weight", (10240,)),
            ("input_mix_weight_down.weight", (320, 10240)),
            ("input_mix_weight_up.weight", (10240, 320)),
            ("block_inject_weight.weight", (4, 10240)),
        ):
            add(source + "." + field, shape)

    for layer in range(48):
        source = f"{root}.layers.{layer}"
        hc(source + ".attn_hyper_connection")
        hc(source + ".mlp_hyper_connection")
        if (layer + 1) % 4:
            linear = source + ".linear_attn"
            for field, shape, dtype in (
                ("A_log", (48,), "F32"),
                ("dt_bias", (48,), "F32"),
                ("conv1d.weight", (10240, 1, 4), "BF16"),
                ("in_proj_a.weight", (48, 2560), "BF16"),
                ("in_proj_b.weight", (48, 2560), "BF16"),
                ("in_proj_qkv.weight", (10240, 2560), "BF16"),
                ("in_proj_z.weight", (6144, 2560), "BF16"),
                ("norm.weight", (128,), "BF16"),
                ("out_proj.weight", (2560, 6144), "BF16"),
            ):
                add(linear + "." + field, shape, dtype)
        else:
            attention = source + ".self_attn"
            for field, shape in (
                ("q_proj.weight", (12288, 2560)),
                ("k_proj.weight", (512, 2560)),
                ("v_proj.weight", (512, 2560)),
                ("o_proj.weight", (2560, 6144)),
                ("q_norm.weight", (256,)),
                ("k_norm.weight", (256,)),
                ("indexer.index_qk_proj.weight", (640, 2560)),
                ("indexer.q_layernorm.weight", (128,)),
                ("indexer.k_layernorm.weight", (128,)),
            ):
                add(attention + "." + field, shape)

        mlp = source + ".mlp"
        for field, shape in (
            ("gate.weight", (512, 2560)),
            ("shared_expert_gate.weight", (1, 2560)),
            ("shared_expert.gate_proj.weight", (640, 2560)),
            ("shared_expert.up_proj.weight", (640, 2560)),
            ("shared_expert.down_proj.weight", (2560, 640)),
            ("experts.gate_up_proj", (512, 1280, 2560)),
            ("experts.down_proj", (512, 2560, 640)),
        ):
            add(mlp + "." + field, shape)

    ple = root + ".layers.1.ple"
    for field, shape in (
        ("conv1d.weight", (10240, 1, 4)),
        ("key_proj.weight", (10240, 2560)),
        ("value_proj.weight", (2560, 2560)),
        ("norm_key.weight", (10240,)),
        ("norm_query.weight", (10240,)),
        ("norm_conv.weight", (10240,)),
    ):
        add(ple + "." + field, shape)
    semantic = ple + ".ple_embedding"
    for field, shape in (
        ("layer_multipliers", (3,)),
        ("ngram_heads_offsets", (16,)),
        ("ngram_heads_vocab_sizes", (16,)),
    ):
        add(semantic + "." + field, shape, "I64")
    for shard in range(128):
        add(
            semantic + f".ngram_embedding.shard_{shard}.weight",
            (shard % 3 + 1, 160),
        )
    return specs


def _checkpoint(
    path: Path,
    *,
    remove=None,
    shape_override=None,
):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_config()))
    specs = _tensor_specs()
    if remove is not None:
        del specs[remove]
    if shape_override is not None:
        name, shape = shape_override
        specs[name][0] = tuple(shape)

    word_bytes = {"BF16": 2, "F32": 4, "I64": 8}
    gate_bytes = 512 * 1280 * 2560 * 2
    down_base = gate_bytes
    semantic_base = gate_bytes + 512 * 2560 * 640 * 2
    semantic_names = [
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
    ]
    semantic_offsets = {
        name: semantic_base + index * 256 for index, name in enumerate(semantic_names)
    }
    header = {}
    payload_size = 0
    for name, (shape, dtype) in specs.items():
        if name.endswith("experts.gate_up_proj"):
            begin = 0
        elif name.endswith("experts.down_proj"):
            begin = down_base
        else:
            begin = semantic_offsets.get(name, 0)
        size = word_bytes[dtype]
        for dim in shape:
            size *= dim
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [begin, begin + size],
        }
        payload_size = max(payload_size, begin + size)

    encoded = json.dumps(header, separators=(",", ":")).encode()
    file = path / "model.safetensors"
    with file.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        data_start = 8 + len(encoded)
        stream.truncate(data_start + payload_size)

        def bf16(element, value, base=0):
            raw = (
                torch.tensor([value], dtype=torch.bfloat16)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            stream.seek(data_start + base + element * 2)
            stream.write(raw)

        width = 640 * 2560
        bf16(0, 1)
        bf16(width, 2)
        bf16(1022 * width, 3)
        bf16(1023 * width, 4)
        bf16(0, 5, down_base)
        bf16(511 * width, 6, down_base)
        multipliers = [1, (1 << 31) + 17, -(1 << 40)]
        name = semantic_names[0]
        if name in header:
            stream.seek(data_start + semantic_offsets[name])
            stream.write(struct.pack("<3q", *multipliers))
    return SafetensorsSource(path)


def _write(path, model, prepared):
    with ArtifactWriter(
        path,
        [job.spec for job in prepared.weights],
        components=model.components,
        bindings=prepared.bindings,
        uses=prepared.uses,
    ) as writer:
        for job in prepared.weights:
            job.prepared.produce(TensorOutput(writer, job.spec.id))


def test_config_normalizes_identity_topology_and_one_based_ple():
    raw = _config()
    del raw["text_config"]["layer_types"]
    config = text_config(raw)
    assert config["architectures"] == ["Qwen4ExpForConditionalGeneration"]
    assert config["model_type"] == "qwen4_exp_text"
    assert len(config["layer_types"]) == 48
    assert config["layer_types"].count("linear_attention") == 36
    assert config["layer_types"].count("full_attention") == 12
    assert config["ple_layer_ids"] == [2]

    by_identity = deepcopy(raw["text_config"])
    assert text_config(by_identity)["model_type"] == "qwen4_exp_text"
    by_architecture = deepcopy(raw["text_config"])
    by_architecture["architectures"] = ["Qwen4ExpForConditionalGeneration"]
    by_architecture["model_type"] = "qwen4_exp"
    assert text_config(by_architecture)["model_type"] == "qwen4_exp_text"
    changed = _config()
    changed["text_config"]["layer_types"][0] = "full_attention"
    with pytest.raises(ValueError, match="36-GDN/12-QSA"):
        text_config(changed)


def test_model_exposes_frozen_mapping_and_separate_residencies(tmp_path):
    with _checkpoint(tmp_path / "model") as source:
        model = build_model(source)

        assert model.parameters["text/token_embedding"].shape == (248320, 2560)
        assert model.parameters["text/output_head"].shape == (248320, 2560)
        assert "text/final_norm" not in model.parameters
        assert {
            name.rsplit("/", 1)[-1]
            for name in model.parameters
            if name.startswith("text/hyper_connection/")
        } == {"norm", "down", "up"}

        for layer in range(48):
            prefix = f"text/layers/{layer}"
            for group in ("attention_hyper_connection", "mlp_hyper_connection"):
                assert {
                    model.parameters[f"{prefix}/{group}/{role}"].shape
                    for role in ("norm", "down", "up", "block_inject")
                } == {(10240,), (320, 10240), (10240, 320), (4, 10240)}
            experts = {
                int(name.split("/experts/", 1)[1].split("/", 1)[0])
                for name in model.parameters
                if name.startswith(prefix + "/moe/experts/")
            }
            assert experts == set(range(512))
            assert model.parameters[prefix + "/moe/shared_gate"].shape == (1, 2560)
            assert model.parameters[prefix + "/moe/shared/gate"].shape == (640, 2560)
            assert model.parameters[prefix + "/moe/shared/up"].shape == (640, 2560)
            assert model.parameters[prefix + "/moe/shared/down"].shape == (2560, 640)

        assert model.parameters["text/layers/0/gdn/qkv"].shape == (10240, 2560)
        assert model.parameters["text/layers/0/gdn/norm"].shape == (128,)
        assert model.parameters["text/layers/0/gdn/convolution"].shape == (
            10240,
            1,
            4,
        )
        assert model.parameters["text/layers/3/attention/query"].shape == (
            12288,
            2560,
        )
        assert model.parameters[
            "text/layers/3/attention/indexer/query_key"
        ].shape == (640, 2560)
        gdn_layers = {
            int(name.split("/layers/", 1)[1].split("/", 1)[0])
            for name in model.parameters
            if name.endswith("/gdn/qkv")
        }
        qsa_layers = {
            int(name.split("/layers/", 1)[1].split("/", 1)[0])
            for name in model.parameters
            if name.endswith("/attention/query")
        }
        assert gdn_layers == {i for i in range(48) if (i + 1) % 4}
        assert qsa_layers == {i for i in range(48) if not (i + 1) % 4}

        ple_names = [name for name in model.parameters if "/ple/" in name]
        assert ple_names and all(
            name.startswith("text/layers/1/ple/") for name in ple_names
        )
        shards = [name for name in ple_names if "/embedding/shards/" in name]
        assert len(shards) == 128
        assert all(model.parameters[name].shape[1] == 160 for name in shards)
        assert all(model.parameters[name].residency == "ple_shards" for name in shards)
        experts = [name for name in model.parameters if "/moe/experts/" in name]
        assert experts and all(
            model.parameters[name].residency == "routed_experts" for name in experts
        )
        for role, shape in (
            ("layer_multipliers", (3,)),
            ("ngram_heads_offsets", (16,)),
            ("ngram_heads_vocab_sizes", (16,)),
        ):
            item = model.parameters[f"text/layers/1/ple/{role}"]
            assert item.shape == shape and item.direct_format == "int64"
            assert item.residency == "values"


def test_packed_expert_slices_and_ple_int64_values_are_exact(tmp_path):
    with _checkpoint(tmp_path / "model") as source:
        model = build_model(source)
        expected = {
            "text/layers/0/moe/experts/0/gate": 1,
            "text/layers/0/moe/experts/0/up": 2,
            "text/layers/0/moe/experts/511/gate": 3,
            "text/layers/0/moe/experts/511/up": 4,
            "text/layers/0/moe/experts/0/down": 5,
            "text/layers/0/moe/experts/511/down": 6,
        }
        labels = set()
        for name, value in expected.items():
            parameter = model.parameters[name]
            labels.add(parameter.source.label)
            assert parameter.source.values(0, 1).item() == value
        assert len(labels) == 6

        name = "text/layers/1/ple/layer_multipliers"
        parameter = model.parameters[name]
        assert parameter.source.values().tolist() == [1, (1 << 31) + 17, -(1 << 40)]
        isolated = Model({"text": {"config": model.config}})
        isolated.add(parameter)
        prepared = Recipe(isolated).prepare(device="cpu", rows_per_chunk=2)
        path = tmp_path / "semantic.ninfer"
        _write(path, isolated, prepared)
        with Artifact(path) as artifact:
            obj = artifact.object(prepared.weights[0].spec.id)
            assert obj.format == "int64"
            assert artifact.read_object(obj.id) == struct.pack(
                "<3q", 1, (1 << 31) + 17, -(1 << 40)
            )


def test_batch3a_rejects_unsupported_components_and_bad_sources(tmp_path):
    with _checkpoint(tmp_path / "valid") as source:
        for component in ("mtp", "vision"):
            with pytest.raises(ValueError, match="only components"):
                build_model(source, components=("text", component))

    with _checkpoint(tmp_path / "missing", remove="lm_head.weight") as source:
        with pytest.raises(ValueError, match="missing source tensor.*lm_head"):
            build_model(source)

    packed = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    with _checkpoint(
        tmp_path / "wrong-packed",
        shape_override=(packed, (511, 1280, 2560)),
    ) as source:
        with pytest.raises(ValueError, match="expected source shape"):
            build_model(source)
