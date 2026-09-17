from __future__ import annotations

import json
import struct
from math import prod
from pathlib import Path

from copy import deepcopy

import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.reader import Artifact
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.methods import import_encoded
from tools.convert.model import Model
from tools.convert.official_recipes import qwen4_exp_nvfp4
from tools.convert.qwen4_exp import build_model, text_config
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource

_LOW = [1, 2, 5, 11]
_HIGH = [0, 3, 1, 4]

_WORD = {"BF16": 2, "F32": 4, "I64": 8, "U8": 1, "F8_E4M3": 1}


# --- config (unchanged from Batch 3A) ------------------------------------

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
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
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


# --- fused BF16 checkpoint (Batch 3A regression) -------------------------

def _fused_checkpoint(
    path: Path,
    *,
    remove=None,
    shape_override=None,
) -> SafetensorsSource:
    """Build the Batch 3A fused-BF16 expert checkpoint on disk (sparse header)."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(_config()))
    specs = _tensor_specs()
    if remove is not None:
        del specs[remove]
    if shape_override is not None:
        name, shape = shape_override
        specs[name][0] = tuple(shape)
    gate_bytes = 512 * 1280 * 2560 * 2
    down_base = gate_bytes
    semantic_base = gate_bytes + 512 * 2560 * 640 * 2
    semantic_names = [
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
    ]
    semantic_offsets = {
        name: semantic_base + i * 256 for i, name in enumerate(semantic_names)
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
        size = _WORD.get(dtype, 2)
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [begin, begin + size]}
        payload_size = max(payload_size, begin + size)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    file = path / "model.safetensors"
    with file.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        data_start = 8 + len(encoded)
        stream.truncate(data_start + payload_size)

        def bf16(element, value, base=0):
            raw = torch.tensor([value], dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()
            stream.seek(data_start + base + element * 2)
            stream.write(raw)

        width = 640 * 2560
        bf16(0, 1); bf16(width, 2); bf16(1022 * width, 3); bf16(1023 * width, 4)
        bf16(0, 5, down_base); bf16(511 * width, 6, down_base)
        if semantic_names[0] in header:
            stream.seek(data_start + semantic_offsets[semantic_names[0]])
            stream.write(struct.pack("<3q", 1, (1 << 31) + 17, -(1 << 40)))
    return SafetensorsSource(path)


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
                ("A_log", (48,), "F32"), ("dt_bias", (48,), "F32"),
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
                ("q_proj.weight", (12288, 2560)), ("k_proj.weight", (512, 2560)),
                ("v_proj.weight", (512, 2560)), ("o_proj.weight", (2560, 6144)),
                ("q_norm.weight", (256,)), ("k_norm.weight", (256,)),
                ("indexer.index_qk_proj.weight", (640, 2560)),
                ("indexer.q_layernorm.weight", (128,)),
                ("indexer.k_layernorm.weight", (128,)),
            ):
                add(attention + "." + field, shape)

        mlp = source + ".mlp"
        for field, shape in (
            ("gate.weight", (512, 2560)), ("shared_expert_gate.weight", (1, 2560)),
            ("shared_expert.gate_proj.weight", (640, 2560)),
            ("shared_expert.up_proj.weight", (640, 2560)),
            ("shared_expert.down_proj.weight", (2560, 640)),
            ("experts.gate_up_proj", (512, 1280, 2560)),
            ("experts.down_proj", (512, 2560, 640)),
        ):
            add(mlp + "." + field, shape)

    ple = root + ".layers.1.ple"
    for field, shape in (
        ("conv1d.weight", (10240, 1, 4)), ("key_proj.weight", (10240, 2560)),
        ("value_proj.weight", (2560, 2560)), ("norm_key.weight", (10240,)),
        ("norm_query.weight", (10240,)), ("norm_conv.weight", (10240,)),
    ):
        add(ple + "." + field, shape)
    semantic = ple + ".ple_embedding"
    for field, shape in (
        ("layer_multipliers", (3,)), ("ngram_heads_offsets", (16,)),
        ("ngram_heads_vocab_sizes", (16,)),
    ):
        add(semantic + "." + field, shape, "I64")
    for shard in range(128):
        add(semantic + f".ngram_embedding.shard_{shard}.weight", (shard % 3 + 1, 160))
    return specs


class _ModelOptStore:
    """Lightweight in-memory fake of a ModelOpt split-expert checkpoint.

    It synthesises *metadata* (shape/dtype) for the tensors the Qwen4Exp
    adapter inspects and returns small deterministic payloads on read, so the
    full 512-expert NVFP4 mapping can be exercised without an on-disk
    checkpoint.  ``build_model`` never triggers a byte read (the ModelOpt
    closures are fully lazy), so it only needs ``has``/``describe``/``config``.
    """

    _TMAP = {"U8": torch.uint8, "F32": torch.float32, "BF16": torch.bfloat16,
             "F8_E4M3": torch.float8_e4m3fn, "I64": torch.int64}

    def __init__(self, input_scale: bool = False):
        self.path = "modelopt-split"
        self.config = _config()
        self.input_scale = input_scale
        self._meta = {}      # name -> (shape, dtype)
        self._buffer = {}    # name -> bytes (small synthetic payload)
        self._build()

    # -- metadata synthesis (layer 0; the layer every assertion reads) -------

    def _meta_put(self, name, shape, dtype):
        self._meta[name] = (tuple(shape), dtype)

    def _build(self):
        root = "model.language_model"
        self._meta_put(root + ".embed_tokens.weight", (248320, 2560), "BF16")
        self._meta_put("lm_head.weight", (248320, 2560), "BF16")
        for field, shape in (("hc_norm.weight", (10240,)),
                             ("input_mix_weight_down.weight", (320, 10240)),
                             ("input_mix_weight_up.weight", (10240, 320))):
            self._meta_put(root + ".hyper_connection_mixer." + field, shape, "BF16")
        source = f"{root}.layers.0"
        for tag in ("attn_hyper_connection", "mlp_hyper_connection"):
            for field, shape in (("hc_norm.weight", (10240,)),
                                 ("input_mix_weight_down.weight", (320, 10240)),
                                 ("input_mix_weight_up.weight", (10240, 320)),
                                 ("block_inject_weight.weight", (4, 10240))):
                self._meta_put(f"{source}.{tag}.{field}", shape, "BF16")
        mlp = source + ".mlp"
        for field, shape in (("gate.weight", (512, 2560)),
                             ("shared_expert_gate.weight", (1, 2560)),
                             ("shared_expert.gate_proj.weight", (640, 2560)),
                             ("shared_expert.up_proj.weight", (640, 2560)),
                             ("shared_expert.down_proj.weight", (2560, 640))):
            self._meta_put(f"{mlp}.{field}", shape, "BF16")
        attn = source + ".linear_attn"   # layer 0 -> GDN
        for field, shape, dtype in (("A_log", (48,), "F32"), ("dt_bias", (48,), "F32"),
                                    ("conv1d.weight", (10240, 1, 4), "BF16"),
                                    ("in_proj_a.weight", (48, 2560), "BF16"),
                                    ("in_proj_b.weight", (48, 2560), "BF16"),
                                    ("in_proj_qkv.weight", (10240, 2560), "BF16"),
                                    ("in_proj_z.weight", (6144, 2560), "BF16"),
                                    ("norm.weight", (128,), "BF16"),
                                    ("out_proj.weight", (2560, 6144), "BF16")):
            self._meta_put(f"{attn}.{field}", shape, dtype)
        # ModelOpt split routed experts: every expert x module, NVFP4 layout.
        for expert in range(512):
            for module in ("gate_proj", "up_proj", "down_proj"):
                n, k = (640, 2560) if module in ("gate_proj", "up_proj") else (2560, 640)
                base = f"{mlp}.experts.{expert}.{module}"
                self._meta_put(f"{base}.weight", (n, k // 2), "U8")
                self._meta_put(f"{base}.weight_scale", (n, k // 16), "F8_E4M3")
                self._meta_put(f"{base}.weight_scale_2", (), "F32")
                if self.input_scale:
                    self._meta_put(f"{base}.input_scale", (), "F32")
        # Remaining layers need metadata for build_model's structural checks;
        # only layer 0 carries payloads because encoded-row reads inspect it.
        for layer in range(1, 48):
            source = f"{root}.layers.{layer}"
            for tag in ("attn_hyper_connection", "mlp_hyper_connection"):
                for field, shape in (("hc_norm.weight", (10240,)),
                                     ("input_mix_weight_down.weight", (320, 10240)),
                                     ("input_mix_weight_up.weight", (10240, 320)),
                                     ("block_inject_weight.weight", (4, 10240))):
                    self._meta_put(f"{source}.{tag}.{field}", shape, "BF16")
            if (layer + 1) % 4:
                attn = source + ".linear_attn"
                for field, shape, dtype in (("A_log", (48,), "F32"), ("dt_bias", (48,), "F32"),
                                            ("conv1d.weight", (10240, 1, 4), "BF16"),
                                            ("in_proj_a.weight", (48, 2560), "BF16"),
                                            ("in_proj_b.weight", (48, 2560), "BF16"),
                                            ("in_proj_qkv.weight", (10240, 2560), "BF16"),
                                            ("in_proj_z.weight", (6144, 2560), "BF16"),
                                            ("norm.weight", (128,), "BF16"),
                                            ("out_proj.weight", (2560, 6144), "BF16")):
                    self._meta_put(f"{attn}.{field}", shape, dtype)
            else:
                attn = source + ".self_attn"
                for field, shape in (("q_proj.weight", (12288, 2560)), ("k_proj.weight", (512, 2560)),
                                     ("v_proj.weight", (512, 2560)), ("o_proj.weight", (2560, 6144)),
                                     ("q_norm.weight", (256,)), ("k_norm.weight", (256,)),
                                     ("indexer.index_qk_proj.weight", (640, 2560)),
                                     ("indexer.q_layernorm.weight", (128,)),
                                     ("indexer.k_layernorm.weight", (128,))):
                    self._meta_put(f"{attn}.{field}", shape, "BF16")
            mlp = source + ".mlp"
            for field, shape in (("gate.weight", (512, 2560)), ("shared_expert_gate.weight", (1, 2560)),
                                 ("shared_expert.gate_proj.weight", (640, 2560)),
                                 ("shared_expert.up_proj.weight", (640, 2560)),
                                 ("shared_expert.down_proj.weight", (2560, 640))):
                self._meta_put(f"{mlp}.{field}", shape, "BF16")
            for expert in range(512):
                for module in ("gate_proj", "up_proj", "down_proj"):
                    n, k = (640, 2560) if module != "down_proj" else (2560, 640)
                    base = f"{mlp}.experts.{expert}.{module}"
                    self._meta_put(f"{base}.weight", (n, k // 2), "U8")
                    self._meta_put(f"{base}.weight_scale", (n, k // 16), "F8_E4M3")
                    self._meta_put(f"{base}.weight_scale_2", (), "F32")
        ple = root + ".layers.1.ple"
        for field, shape in (("conv1d.weight", (10240, 1, 4)),
                             ("key_proj.weight", (10240, 2560)),
                             ("value_proj.weight", (2560, 2560)),
                             ("norm_key.weight", (10240,)),
                             ("norm_query.weight", (10240,)),
                             ("norm_conv.weight", (10240,))):
            self._meta_put(f"{ple}.{field}", shape, "BF16")
        semantic = ple + ".ple_embedding"
        for field, shape in (("layer_multipliers", (3,)),
                             ("ngram_heads_offsets", (16,)),
                             ("ngram_heads_vocab_sizes", (16,))):
            self._meta_put(f"{semantic}.{field}", shape, "I64")
        for shard in range(128):
            self._meta_put(f"{semantic}.ngram_embedding.shard_{shard}.weight",
                           (shard % 3 + 1, 160), "BF16")
        self._seed_buffers()

    # -- small synthetic payload for the tensors the read tests inspect ------

    def _seed_buffers(self):
        # Synthetic payloads for the layer-0 expert-0 tensors only (the ones the
        # tests read); every other read is zero-filled by read_flat's ljust.
        mlp = "model.language_model.layers.0.mlp"
        for module in ("gate_proj", "up_proj", "down_proj"):
            base = f"{mlp}.experts.0.{module}"
            k = 2560 if module in ("gate_proj", "up_proj") else 640
            packed = _build_packed(1, k, _LOW, _HIGH)   # one reference row
            self._buffer[base + ".weight"] = packed
            self._buffer[base + ".weight_scale"] = bytes(k // 16)  # all 0x00 (E4M3 +0.0)
            self._buffer[base + ".weight_scale_2"] = struct.pack("<f", 2.0)
            if self.input_scale:
                self._buffer[base + ".input_scale"] = struct.pack("<f", 4.0)

    # --- SafetensorsSource surface (duck-typed; build_model never reads) ---

    def has(self, name: str) -> bool:
        return name in self._meta

    def describe(self, name: str):
        if name not in self._meta:
            raise ValueError(f"{self.path}: missing source tensor {name!r}")
        shape, dtype = self._meta[name]
        info = _ModelOptStore._Info(shape, dtype, _WORD.get(dtype, 1))
        info.bytes = info.word_bytes * (prod(shape) if shape else 1)
        info.offset = 0
        return info

    def read_flat(self, name: str, begin: int = 0, end: int | None = None):
        if name not in self._meta:
            raise ValueError(f"{self.path}: missing source tensor {name!r}")
        shape, dtype = self._meta[name]
        elements = prod(shape) if shape else 1
        end = elements if end is None else end
        if not 0 <= begin <= end <= elements:
            raise ValueError(f"{name}: source element range [{begin},{end}) exceeds {shape}")
        if end == begin:
            return torch.empty(0, dtype=_ModelOptStore._TMAP.get(dtype, torch.uint8))
        word = _WORD.get(dtype, 1)
        buf = self._buffer.get(name)
        if buf is None:
            buf = bytes(word * elements)
        raw = buf[begin * word: end * word]
        if len(raw) < (end - begin) * word:
            raw = raw.ljust((end - begin) * word, b"\x00")
        td = _ModelOptStore._TMAP.get(dtype, torch.uint8)
        return torch.frombuffer(bytearray(raw), dtype=td)

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    class _Info:
        def __init__(self, shape, dtype, word_bytes):
            self.shape = tuple(shape)
            self.dtype = dtype
            self.word_bytes = word_bytes
            self.file = None
            self.offset = 0
            self.bytes = 0


def _build_packed(n: int, k: int, low, high) -> bytes:
    out = bytearray()
    for _ in range(n):
        for col in range(k // 2):
            out.append((high[col % len(high)] << 4) | low[col % len(low)])
    return bytes(out)


# --- 1. fused BF16 expert mapping unchanged (Batch 3A regression) --------

def test_fused_bf16_expert_mapping_unchanged(tmp_path):
    with _fused_checkpoint(tmp_path / "model") as source:
        model = build_model(source)
        for layer in range(48):
            prefix = f"text/layers/{layer}"
            experts = {
                int(name.split("/experts/", 1)[1].split("/", 1)[0])
                for name in model.parameters if name.startswith(prefix + "/moe/experts/")
            }
            assert experts == set(range(512))
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
        # fused path has no encoded rows (plain tensor_source)
        assert model.parameters["text/layers/0/moe/experts/0/gate"].source.read_encoded is None


# --- 2. ModelOpt split routed-expert canonical mapping + residency -------

def test_modelopt_split_expert_mapping_and_residency():
    with _ModelOptStore() as store:
        model = build_model(store)
        layer0 = "text/layers/0/moe"
        assert model.parameters[f"{layer0}/experts/0/gate"].shape == (640, 2560)
        assert model.parameters[f"{layer0}/experts/0/up"].shape == (640, 2560)
        assert model.parameters[f"{layer0}/experts/0/down"].shape == (2560, 640)
        assert model.parameters[f"{layer0}/experts/511/gate"].shape == (640, 2560)
        assert model.parameters[f"{layer0}/experts/511/down"].shape == (2560, 640)
        # canonical mapping: every routed expert 0..511 maps to all three roles
        experts = {
            int(name.split("/experts/", 1)[1].split("/", 1)[0])
            for name in model.parameters if name.startswith(layer0 + "/experts/")
        }
        assert experts == set(range(512))
        # residency: every routed expert maps to "routed_experts"
        routed = [p for n, p in model.parameters.items()
                  if n.startswith(layer0 + "/experts/")]
        assert routed and all(p.residency == "routed_experts" for p in routed)


# --- 3. ModelOpt experts expose NVFP4 encoded rows -----------------------

def test_modelopt_experts_expose_nvfp4_encoded_rows():
    with _ModelOptStore() as store:
        model = build_model(store)
        for name, k in (
            ("text/layers/0/moe/experts/0/gate", 2560),
            ("text/layers/0/moe/experts/0/up", 2560),
            ("text/layers/0/moe/experts/0/down", 640),
        ):
            rows = model.parameters[name].source.read_encoded(0, 1)
            assert rows.format == "nvfp4"
            assert rows.codes.shape[1] == k // 2
            assert rows.scales.shape[1] == k // 16
            # weight_divisor = FP32(1 / 2.0) = 0.5 (synthetic scale_2 is 2.0)
            assert rows.weight_divisor == struct.pack("<f", 0.5)


# --- 4. non-routed weights remain existing BF16/value sources -------------

def test_non_routed_weights_keep_existing_sources():
    with _ModelOptStore() as store:
        model = build_model(store)
        for name in ("text/layers/0/moe/shared/gate", "text/layers/0/moe/shared/up",
                     "text/layers/0/moe/shared/down", "text/layers/0/moe/router"):
            source = model.parameters[name].source
            assert source.read_encoded is None
            assert model.parameters[name].residency != "routed_experts"
        assert model.parameters["text/token_embedding"].residency == "text"
        assert model.parameters["text/output_head"].residency == "text"


# --- 5. ModelOpt encoded-row byte-exactness on the split layout ----------

def test_modelopt_encoded_rows_exact():
    with _ModelOptStore() as store:
        model = build_model(store)
        gate = model.parameters["text/layers/0/moe/experts/0/gate"]
        rows = gate.source.read_encoded(0, 1)
        # the packed payload is the reference _build_packed bytes for one 2560 row
        expect = _build_packed(1, 2560, _LOW, _HIGH)
        assert rows.codes.view(torch.uint8).numpy().tobytes() == expect
        expect_scale = bytes(160)   # 2560 // 16 = 160 zero bytes
        assert rows.scales.view(torch.uint8).numpy().tobytes() == expect_scale


# --- 6. Batch 3A config normalization (topology + one-based PLE) ----------

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


# --- 7. Batch 3A frozen mapping + separate residencies --------------------

def test_frozen_mapping_and_separate_residencies(tmp_path):
    with _fused_checkpoint(tmp_path / "model") as source:
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
        assert all(
            model.parameters[name].residency == "ple_shards" for name in shards
        )
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


# --- 8. Batch 3A packed slices + PLE int64 exact bytes ---------------------

def test_packed_expert_slices_and_ple_int64_values_are_exact(tmp_path):
    with _fused_checkpoint(tmp_path / "model") as source:
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


# --- 9. Batch 3A rejects unsupported components / malformed sources -------

def test_batch3a_rejects_unsupported_components_and_bad_sources(tmp_path):
    with _fused_checkpoint(tmp_path / "valid") as source:
        for component in ("mtp", "vision"):
            with pytest.raises(ValueError, match="only components"):
                build_model(source, components=("text", component))

    with _fused_checkpoint(
        tmp_path / "missing", remove="lm_head.weight"
    ) as source:
        with pytest.raises(ValueError, match="missing source tensor.*lm_head"):
            build_model(source)

    packed = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    with _fused_checkpoint(
        tmp_path / "wrong-packed",
        shape_override=(packed, (511, 1280, 2560)),
    ) as source:
        with pytest.raises(ValueError, match="expected source shape"):
            build_model(source)


# --- Batch 3B2: official recipe + activation-use contract -----------------

def _single_parameter_model(model, name):
    """Create a minimal model containing one real Qwen4Exp Parameter."""
    minimal = Model({"text": {"config": model.config}})
    minimal.add(model.parameters[name])
    return minimal


def test_batch3b2_official_recipe_assignments_and_mixed_activation_policy():
    # Layer 0 receives ModelOpt input_scale in this fixture; later layers
    # intentionally do not.  One checkpoint therefore exercises both policies.
    with _ModelOptStore(input_scale=True) as store:
        model = build_model(store)
        recipe = Recipe(model)
        qwen4_exp_nvfp4(model, recipe, {"base": store})

        def selection(name):
            choices = recipe.selections[name]
            assert len(choices) == 1
            return choices[0]

        assert selection("text/token_embedding").format == "q8_g32_fp16"
        assert selection("text/token_embedding").method.__name__ == "grouped_absmax"

        assert selection("text/output_head").format == "q6_g64_fp16"
        assert selection("text/output_head").method.__name__ == "grouped_absmax"

        expert0 = "text/layers/0/moe/experts/0/gate"
        expert1 = "text/layers/1/moe/experts/0/gate"

        assert selection(expert0).format == "nvfp4"
        assert selection(expert0).method is import_encoded
        assert selection(expert1).format == "nvfp4"
        assert selection(expert1).method is import_encoded

        # Frozen direct-BF16 exceptions.
        for name in (
            "text/layers/0/gdn/in_proj_a",
            "text/layers/0/gdn/in_proj_b",
            "text/layers/0/moe/router",
            "text/layers/0/moe/shared_gate",
        ):
            assert selection(name).format == "bf16"
            assert selection(name).method.__name__ == "cast_direct"

        # Ordinary non-routed projection.
        assert selection("text/layers/0/gdn/qkv").format == "q8_g32_fp16"
        assert selection("text/layers/0/gdn/qkv").method.__name__ == "grouped_absmax"

        # PLE remains direct, despite containing matrix-valued tensors.
        assert selection("text/layers/1/ple/key").format == "bf16"
        assert selection("text/layers/1/ple/key").method.__name__ == "cast_direct"

        input0 = "text/layers/0/ffn_input"
        input1 = "text/layers/1/ffn_input"

        assert model.parameters[expert0].inputs == (input0,)
        assert model.parameters[expert1].inputs == (input1,)

        assert recipe.policies[(expert0, input0)] == "AllowA4"
        assert recipe.policies[(expert1, input1)] == "A16Only"


def test_batch3b2_routed_expert_mathematical_inputs():
    with _ModelOptStore() as store:
        model = build_model(store)

        for role in ("gate", "up"):
            name = f"text/layers/0/moe/experts/17/{role}"
            assert model.parameters[name].inputs == ("text/layers/0/ffn_input",)

        down = "text/layers/0/moe/experts/17/down"
        assert model.parameters[down].inputs == (
            "text/layers/0/moe/experts/17/product",
        )


def test_batch3b2_input_divisor_presence_and_exact_reciprocal():
    name = "text/layers/0/moe/experts/0/gate"

    with _ModelOptStore(input_scale=True) as store:
        model = build_model(store)
        source = model.parameters[name].source

        assert source.input_divisor is not None
        assert source.input_divisor() == struct.pack("<f", 0.25)

        encoded = source.read_encoded(0, 1)
        assert encoded.format == "nvfp4"
        assert encoded.weight_divisor == struct.pack("<f", 0.5)

    with _ModelOptStore(input_scale=False) as store:
        model = build_model(store)
        source = model.parameters[name].source

        # This is the Batch 3B2 semantic correction: absence stays absence,
        # rather than a callback that fails only when invoked.
        assert source.input_divisor is None


def test_batch3b2_allowa4_emits_use_and_activation_auxiliary():
    name = "text/layers/0/moe/experts/0/gate"
    input_name = "text/layers/0/ffn_input"

    with _ModelOptStore(input_scale=True) as store:
        full = build_model(store)
        model = _single_parameter_model(full, name)
        source = model.parameters[name].source

        recipe = Recipe(model)
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=source,
            activation_policy="AllowA4",
        )

        prepared = recipe.prepare(device="cpu", rows_per_chunk=128)

        uses = [
            use
            for use in prepared.uses
            if use["parameter"] == name and use["input"] == input_name
        ]
        assert len(uses) == 1
        use = uses[0]

        assert use["activation_policy"] == "AllowA4"
        assert set(use["auxiliaries"]) == {"activation_input_divisor"}

        aux_id = use["auxiliaries"]["activation_input_divisor"]["object"]
        auxiliary = {
            spec.id: (spec, data) for spec, data in prepared.auxiliaries
        }[aux_id]

        spec, data = auxiliary
        assert spec.format == "fp32"
        assert spec.shape == ()
        assert data == struct.pack("<f", 0.25)


def test_batch3b2_a16only_succeeds_without_activation_auxiliary():
    name = "text/layers/0/moe/experts/0/gate"
    input_name = "text/layers/0/ffn_input"

    with _ModelOptStore(input_scale=False) as store:
        full = build_model(store)
        model = _single_parameter_model(full, name)
        source = model.parameters[name].source

        assert source.input_divisor is None

        recipe = Recipe(model)
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=source,
            activation_policy="A16Only",
        )

        prepared = recipe.prepare(device="cpu", rows_per_chunk=128)

        uses = [
            use
            for use in prepared.uses
            if use["parameter"] == name and use["input"] == input_name
        ]
        assert uses == [
            {
                "parameter": name,
                "input": input_name,
                "activation_policy": "A16Only",
            }
        ]

        assert prepared.auxiliaries == ()


def test_batch3b2_forced_allowa4_without_divisor_fails():
    name = "text/layers/0/moe/experts/0/gate"

    with _ModelOptStore(input_scale=False) as store:
        full = build_model(store)
        model = _single_parameter_model(full, name)
        source = model.parameters[name].source

        recipe = Recipe(model)
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=source,
            activation_policy="AllowA4",
        )

        with pytest.raises(
            ValueError,
            match="supply an activation divisor for AllowA4",
        ):
            recipe.prepare(device="cpu", rows_per_chunk=128)
