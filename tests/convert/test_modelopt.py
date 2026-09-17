from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools.convert.sources.modelopt import (
    modelopt_matrix_source,
    modelopt_source,
)
from tools.convert.sources.safetensors import SafetensorsSource

_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# Reference matrix: N=4 rows, K=16 logical weights (one packed byte / 2 weights,
# one E4M3 group scale / 16 weights).  low nibble = earlier (even) value.
_LOW = [1, 2, 5, 11]
_HIGH = [0, 3, 1, 4]


def _e4m3f(word: int) -> float:
    """E4M3FN decode of one byte, identical to the source's torch path."""
    return torch.tensor([word], dtype=torch.uint8).view(
        torch.float8_e4m3fn
    ).float().item()


def _flat_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.view(-1)
    if tensor.numel() == 0:
        return b""
    return tensor.view(torch.uint8).numpy().tobytes()


def _build_packed(n: int, k: int, low: list, high: list) -> bytes:
    """Two E2M1 nibbles per byte: low nibble = earlier logical value."""
    out = bytearray()
    for _ in range(n):
        for col in range(k // 2):
            out.append((high[col % len(high)] << 4) | low[col % len(low)])
    return bytes(out)


def _good_store(
    root: Path,
    *,
    drop: tuple = (),
    extra: dict | None = None,
    scale2: float = 2.0,
    input_scale: float | None = None,
    scale_words: list | None = None,
) -> SafetensorsSource:
    """Write a minimal valid ModelOpt NVFP4 checkpoint and open it."""
    n, k = 4, 16
    packed = _build_packed(n, k, _LOW, _HIGH)
    scale_words = [0x00] * n if scale_words is None else scale_words
    scales = torch.tensor(scale_words, dtype=torch.uint8).view(torch.float8_e4m3fn)
    tensors: dict = {
        "proj.weight": torch.frombuffer(packed, dtype=torch.uint8).clone().reshape(n, k // 2),
        "proj.weight_scale": scales.clone().reshape(n, k // 16),
        "proj.weight_scale_2": torch.tensor(scale2, dtype=torch.float32),
    }
    if input_scale is not None:
        tensors["proj.input_scale"] = torch.tensor(input_scale, dtype=torch.float32)
    for name in drop:
        tensors.pop(name, None)
    if extra:
        tensors.update(extra)
    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(root / "model.safetensors"))
    return SafetensorsSource(root)


# --- 1. exact packed / FP8 byte preservation -----------------------------

def test_packed_and_fp8_bytes_preserved_exactly(tmp_path):
    store = _good_store(tmp_path / "model")
    with store:
        source = modelopt_source(store, "proj", (4, 16))
        expect_packed = _build_packed(4, 16, _LOW, _HIGH)
        expect_scale = bytes([0x00] * 4)
        words = source.read_encoded(0, 4)
        # encoded bytes returned by the adapter are byte-identical to the file.
        assert _flat_bytes(words.codes) == expect_packed
        assert _flat_bytes(words.scales) == expect_scale
        assert words.format == "nvfp4"
        # on-disk bytes recovered from the store match the encoded payload too.
        assert _tensor_bytes(store, "proj.weight") == expect_packed
        assert _tensor_bytes(store, "proj.weight_scale") == expect_scale


def _tensor_bytes(store: SafetensorsSource, name: str) -> bytes:
    info = store.describe(name)
    with open(info.file, "rb") as stream:
        stream.seek(info.offset)
        return stream.read(info.bytes)


# --- 2. low-nibble-first interpretation (even = low, odd = high) ---------

def test_low_nibble_is_earlier_logical_value(tmp_path):
    # scale bytes 0x00 -> E4M3 +0.0, scale2=2.0, so value == pure E2M1 nibble.
    store = _good_store(tmp_path / "model", scale_words=[0x30] * 4)
    with store:
        source = modelopt_source(store, "proj", (4, 16))
        values = source.values().reshape(4, 16)
        for row in range(4):
            for col in range(0, 16, 2):
                assert values[row, col] == pytest.approx(
                    _E2M1_VALUES[_LOW[col // 2 % len(_LOW)]], abs=1e-3
                )
                assert values[row, col + 1] == pytest.approx(
                    _E2M1_VALUES[_HIGH[col // 2 % len(_HIGH)]], abs=1e-3
                )


# --- 3. decoded values: E2M1 x E4M3 x scale2 -----------------------------

def test_decoded_values_match_formula(tmp_path):
    scale2 = 2.0
    store = _good_store(
        tmp_path / "model", scale2=scale2, scale_words=[0x30] * 4
    )  # 0x30 -> +1.0
    with store:
        source = modelopt_source(store, "proj", (4, 16))
        expected = torch.empty(4, 16)
        for row in range(4):
            for col in range(16):
                code = _E2M1_VALUES[
                    _LOW[col // 2 % len(_LOW)] if col % 2 == 0
                    else _HIGH[col // 2 % len(_HIGH)]
                ]
                expected[row, col] = code * _e4m3f(0x30) * scale2
        assert torch.allclose(source.values().reshape(4, 16), expected,
                              atol=1e-5, rtol=1e-5)
        # one explicit spot check: first weight of row 0 (low nibble 1 -> 0.5).
        assert float(source.values()[0]) == pytest.approx(
            _E2M1_VALUES[_LOW[0]] * _e4m3f(0x30) * scale2, abs=1e-3
        )


# --- 4. reciprocal divisors (invariant to the *// decode line) -----------

def test_weight_divisor_is_reciprocal(tmp_path):
    store = _good_store(tmp_path / "model", scale2=2.0)
    with store:
        source = modelopt_source(store, "proj", (4, 16))
        # FP32(1 / 2.0) == 0.5; assert on the divisor only, not the decode line.
        assert source.weight_divisor() == struct.pack("<f", 0.5)
        assert source.read_encoded(0, 4).weight_divisor == struct.pack("<f", 0.5)


def test_input_divisor_present_and_reciprocal(tmp_path):
    store = _good_store(tmp_path / "model", input_scale=4.0)
    with store:
        source = modelopt_source(store, "proj", (4, 16))
        assert source.input_divisor is not None
        # FP32(1 / 4.0) == 0.25
        assert source.input_divisor() == struct.pack("<f", 0.25)


def test_absent_input_scale_gives_no_input_divisor(tmp_path):
    store = _good_store(tmp_path / "model")  # no input_scale
    with store:
        source = modelopt_source(store, "proj", (4, 16))
        assert source.input_divisor is None


# --- 5. pre_quant_scale rejection -----------------------------------------

def test_pre_quant_scale_rejected(tmp_path):
    store = _good_store(
        tmp_path / "model",
        extra={"proj.pre_quant_scale": torch.tensor([[1.0]], dtype=torch.float32)},
    )
    with store:
        with pytest.raises(ValueError, match="pre_quant_scale"):
            modelopt_source(store, "proj", (4, 16))


# --- 6. packed weight signature / dtype / shape rejections ----------------

@pytest.mark.parametrize(
    "wrong",
    [
        # weight has the right shape but a non-U8 dtype (F32).
        dict(dtype=torch.float32, scale_words=[0x00] * 4),
        # weight has the right dtype (U8) but a too-small N dimension.
        dict(dtype=torch.uint8, scale_words=[0x00] * 3),
    ],
)
def test_weight_signature_rejection(tmp_path, wrong):
    n, k = 4, 16
    n_rows = 3 if wrong["dtype"] is torch.uint8 else 4
    packed = _build_packed(n_rows, k, _LOW, _HIGH)
    scales = torch.tensor(wrong["scale_words"], dtype=torch.uint8).view(torch.float8_e4m3fn)
    tensors = {
        "proj.weight": (torch.frombuffer(packed, dtype=wrong["dtype"]).clone().reshape(n_rows, k // 2)
                        if wrong["dtype"] is torch.uint8
                        else torch.zeros(n_rows, k // 2, dtype=wrong["dtype"])),
        "proj.weight_scale": scales.clone(),
        "proj.weight_scale_2": torch.tensor(2.0, dtype=torch.float32),
    }
    (tmp_path / "bad").mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tmp_path / "bad" / "model.safetensors"))
    with SafetensorsSource(tmp_path / "bad") as store:
        with pytest.raises(ValueError):
            modelopt_source(store, "proj", (n, k))


def test_weight_scale_signature_rejection(tmp_path):
    n, k = 4, 16
    packed = _build_packed(n, k, _LOW, _HIGH)
    # weight_scale has the wrong dtype (F32) and wrong K dimension (2 instead of 1).
    tensors = {
        "proj.weight": torch.frombuffer(packed, dtype=torch.uint8).clone().reshape(n, k // 2),
        "proj.weight_scale": torch.zeros(n, 2, dtype=torch.float32),
        "proj.weight_scale_2": torch.tensor(2.0, dtype=torch.float32),
    }
    (tmp_path / "bad").mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tmp_path / "bad" / "model.safetensors"))
    with SafetensorsSource(tmp_path / "bad") as store:
        with pytest.raises(ValueError):
            modelopt_source(store, "proj", (n, k))


# --- 7. weight_scale_2 / input_scale scalar rejections --------------------

@pytest.mark.parametrize("value", [float("inf"), -1.0, 0.0])
def test_weight_scale_2_value_rejection(tmp_path, value):
    n, k = 4, 16
    packed = _build_packed(n, k, _LOW, _HIGH)
    scales = torch.tensor([0x00] * n, dtype=torch.uint8).view(torch.float8_e4m3fn)
    tensors = {
        "proj.weight": torch.frombuffer(packed, dtype=torch.uint8).clone().reshape(n, k // 2),
        "proj.weight_scale": scales.clone(),
        "proj.weight_scale_2": torch.tensor(value, dtype=torch.float32),
    }
    (tmp_path / "bad").mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tmp_path / "bad" / "model.safetensors"))
    with SafetensorsSource(tmp_path / "bad") as store:
        with pytest.raises(ValueError):
            modelopt_source(store, "proj", (n, k))


@pytest.mark.parametrize("value", [float("inf"), -2.0, 0.0])
def test_input_scale_value_rejection(tmp_path, value):
    n, k = 4, 16
    packed = _build_packed(n, k, _LOW, _HIGH)
    scales = torch.tensor([0x00] * n, dtype=torch.uint8).view(torch.float8_e4m3fn)
    tensors = {
        "proj.weight": torch.frombuffer(packed, dtype=torch.uint8).clone().reshape(n, k // 2),
        "proj.weight_scale": scales.clone().reshape(n, k // 16),
        "proj.weight_scale_2": torch.tensor(2.0, dtype=torch.float32),
        "proj.input_scale": torch.tensor([value], dtype=torch.float32),
    }
    (tmp_path / "bad").mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tmp_path / "bad" / "model.safetensors"))
    with SafetensorsSource(tmp_path / "bad") as store:
        with pytest.raises(ValueError):
            modelopt_source(store, "proj", (n, k))


def test_weight_scale_2_non_scalar_rejected(tmp_path):
    n, k = 4, 16
    packed = _build_packed(n, k, _LOW, _HIGH)
    scales = torch.tensor([0x00] * n, dtype=torch.uint8).view(torch.float8_e4m3fn)
    tensors = {
        "proj.weight": torch.frombuffer(packed, dtype=torch.uint8).clone().reshape(n, k // 2),
        "proj.weight_scale": scales.clone(),
        "proj.weight_scale_2": torch.tensor([2.0, 4.0], dtype=torch.float32),
    }
    (tmp_path / "bad").mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tmp_path / "bad" / "model.safetensors"))
    with SafetensorsSource(tmp_path / "bad") as store:
        with pytest.raises(ValueError):
            modelopt_source(store, "proj", (n, k))


def test_weight_scale_2_wrong_dtype_rejected(tmp_path):
    n, k = 4, 16
    packed = _build_packed(n, k, _LOW, _HIGH)
    scales = torch.tensor([0x00] * n, dtype=torch.uint8).view(torch.float8_e4m3fn)
    tensors = {
        "proj.weight": torch.frombuffer(packed, dtype=torch.uint8).clone().reshape(n, k // 2),
        "proj.weight_scale": scales.clone(),
        "proj.weight_scale_2": torch.tensor(2.0, dtype=torch.bfloat16),
    }
    (tmp_path / "bad").mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tmp_path / "bad" / "model.safetensors"))
    with SafetensorsSource(tmp_path / "bad") as store:
        with pytest.raises(ValueError):
            modelopt_source(store, "proj", (n, k))


# --- 8. invalid E4M3 group scale (rejected at encode time) ---------------

def test_invalid_e4m3_group_scale_rejected(tmp_path):
    # 0x7F is the reserved E4M3FN NaN word -> must be rejected on read.
    store = _good_store(tmp_path / "model", scale_words=[0x30, 0x30, 0x30, 0x7F])
    with store:
        source = modelopt_source(store, "proj", (4, 16))
        with pytest.raises(ValueError):
            source.read_encoded(0, 4)


def test_signed_e4m3_group_scale_rejected(tmp_path):
    # 0xB0 carries the sign bit -> must be rejected.
    store = _good_store(tmp_path / "model", scale_words=[0x30, 0x30, 0x30, 0xB0])
    with store:
        source = modelopt_source(store, "proj", (4, 16))
        with pytest.raises(ValueError):
            source.read_encoded(0, 4)


# --- 9. K not divisible by 16 ---------------------------------------------

def test_k_not_divisible_by_16_rejected(tmp_path):
    n, k = 4, 18
    packed = _build_packed(n, k, _LOW, _HIGH)
    scales = torch.tensor([0x00] * n, dtype=torch.uint8).view(torch.float8_e4m3fn)
    tensors = {
        "proj.weight": torch.frombuffer(packed, dtype=torch.uint8).clone().reshape(n, k // 2),
        "proj.weight_scale": scales.clone(),
        "proj.weight_scale_2": torch.tensor(2.0, dtype=torch.float32),
    }
    (tmp_path / "bad").mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tmp_path / "bad" / "model.safetensors"))
    with SafetensorsSource(tmp_path / "bad") as store:
        with pytest.raises(ValueError, match="16"):
            modelopt_source(store, "proj", (n, k))


# --- 10. modelopt_matrix_source lazy dispatch -----------------------------

def test_matrix_source_dispatches_modelopt_when_scaled(tmp_path):
    store = _good_store(tmp_path / "model")
    with store:
        source = modelopt_matrix_source(store, "proj.weight", (4, 16))
        words = source.read_encoded(0, 4)
        assert words.format == "nvfp4"
        assert _flat_bytes(words.codes) == _build_packed(4, 16, _LOW, _HIGH)
        assert _flat_bytes(words.scales) == bytes([0x00] * 4)


def test_matrix_source_dispatches_plain_when_unscaled(tmp_path):
    store = _good_store(
        tmp_path / "model", drop=("proj.weight_scale", "proj.weight_scale_2")
    )
    with store:
        source = modelopt_matrix_source(store, "proj.weight", (4, 16))
        # no scale companions -> plain tensor path; wrapper rejects encoded reads.
        with pytest.raises(ValueError, match="encoded rows"):
            source.read_encoded(0, 1)


# --- 11. E2M1 low-nibble logical-value coverage ---------------------------

def test_e2m1_nibble_values_canonical(tmp_path):
    # scale == 1.0 (scale2) with group scale 0x30 (E4M3 +1.0): value == E2M1.
    store = _good_store(tmp_path / "model", scale2=1.0, scale_words=[0x30] * 4)
    with store:
        source = modelopt_source(store, "proj", (4, 16))
        values = source.values().reshape(4, 16)
        for row in range(4):
            for col in range(16):
                code = (_LOW[col // 2 % len(_LOW)] if col % 2 == 0
                        else _HIGH[col // 2 % len(_HIGH)])
                # scale2=1.0 and group scale 0x30 (E4M3 +0.5) => value is scaled E2M1.
                assert values[row, col] == pytest.approx(_E2M1_VALUES[code] * 0.5, abs=1e-3)
