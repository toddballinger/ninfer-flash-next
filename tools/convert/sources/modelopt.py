"""Interpret a ModelOpt NVFP4 weight checkpoint into logical source rows.

ModelOpt stores a routed-expert (or any) quantized matrix as three tensors:

    <prefix>.weight          U8         (N, K//2)     two E2M1 nibbles / byte
    <prefix>.weight_scale    F8_E4M3    (N, K//16)    one block scale / 16
    <prefix>.weight_scale_2  F32         scalar        global scale

plus an optional

    <prefix>.input_scale     F32         scalar        activation scale

The low nibble of each packed byte decodes first.  NInfer stores weight
divisors, so the weight divisor is FP32(1 / weight_scale_2) and the optional
input divisor is FP32(1 / input_scale).
"""

from __future__ import annotations

import math
import struct

import torch

from tools.artifact.formats import valid_positive_fp32_word
from math import prod
from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource, tensor_source

_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _reciprocal_word(value: float) -> bytes:
    """FP32(1 / value), finite and strictly positive."""
    reciprocal = torch.tensor(1.0 / value, dtype=torch.float32).item()
    raw = struct.pack("<f", reciprocal)
    if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
        raise ValueError(f"reciprocal of {value!r} is not finite and positive")
    return raw


def _scale_valid(scales: torch.Tensor) -> bool:
    """E4M3FN group scale: no sign bit set and not the 0x7F NaN word."""
    return not bool((scales > 0x7E).any())


def _signature(store: SafetensorsSource, name: str,
               expected: tuple[int, ...], dtype: str) -> None:
    info = store.describe(name)
    if info.shape != expected or info.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}{list(expected)}, "
                         f"got {info.dtype}{list(info.shape)}")


def _scalar_words(store: SafetensorsSource, name: str) -> tuple[float, bytes]:
    """Read a single-FP32 tensor and validate finite/positive; return (value, bytes)."""
    info = store.describe(name)
    if info.dtype != "F32":
        raise ValueError(f"{name}: expected F32, got {info.dtype}")
    if info.shape != ():
        raise ValueError(f"{name}: expected a scalar, got {list(info.shape)}")
    tensor = store.read_flat(name)
    value = tensor[0].item()
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name}: scale must be finite and positive")
    return value, _reciprocal_word(value)


def modelopt_source(store: SafetensorsSource, prefix: str,
                    shape: tuple[int, int]) -> LogicalSource:
    """Resolve one ModelOpt NVFP4 matrix into an encoded logical source."""
    n, k = shape
    if k % 16:
        raise ValueError(f"{prefix}: NVFP4 source K must be divisible by 16")

    weight, scale, scale2 = (
        f"{prefix}.weight", f"{prefix}.weight_scale", f"{prefix}.weight_scale_2")
    for name, expected, dtype in (
        (weight, (n, k // 2), "U8"),
        (scale, (n, k // 16), "F8_E4M3"),
    ):
        _signature(store, name, expected, dtype)
    _signature(store, scale2, (), "F32")

    if store.has(f"{prefix}.pre_quant_scale"):
        raise ValueError(f"{prefix}.pre_quant_scale: unsupported; use weight_scale")

    scale2_value, weight_divisor = _scalar_words(store, scale2)
    if store.has(f"{prefix}.input_scale"):
        input_value, input_divisor = _scalar_words(store, f"{prefix}.input_scale")
        has_input = True
    else:
        input_value, input_divisor = 0.0, b""
        has_input = False

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        codes = store.read_flat(weight, begin * (k // 2), end * (k // 2)).reshape(
            end - begin, k // 2)
        scales = store.read_flat(scale, begin * (k // 16), end * (k // 16)).view(
            torch.uint8).reshape(end - begin, k // 16)
        if not _scale_valid(scales):
            raise ValueError(f"{scale}: expected nonnegative finite E4M3FN scales")
        return EncodedRows("nvfp4", codes, scales, weight_divisor)

    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        codes = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(
            last - first, k)
        values = table[codes.long()]
        scales = (words.scales.view(torch.float8_e4m3fn)
                  .float().repeat_interleave(16, dim=1))
        values = values * scales * scale2_value
        return values.reshape(-1)[begin - first * k: end - first * k]

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} (modelopt nvfp4)",
        read,
        encoded,
        lambda: weight_divisor,
        (lambda: input_divisor) if has_input else None,
    )


def modelopt_matrix_source(
    store: SafetensorsSource, name: str, shape: tuple[int, int],
) -> LogicalSource:
    """Resolve a matrix lazily: ModelOpt NVFP4 or the existing decoded view.

    The fused / decoded path is chosen when the tensor has no ``weight_scale_2``
    companion (the current BF16 and compressed-tensors layouts).
    """
    prefix = name.removesuffix(".weight")
    resolved: LogicalSource | None = None

    def resolve() -> LogicalSource:
        nonlocal resolved
        if resolved is None:
            if store.has(prefix + ".weight_scale_2") and store.has(prefix + ".weight_scale"):
                resolved = modelopt_source(store, prefix, shape)
            else:
                resolved = tensor_source(store, name, shape)
        return resolved

    def encoded(begin: int, end: int) -> EncodedRows:
        reader = resolve().read_encoded
        if reader is None:
            raise ValueError(f"{name}: selected source does not provide encoded rows")
        return reader(begin, end)

    def divisor(which: str) -> bytes:
        read = getattr(resolve(), which)
        if read is None:
            raise ValueError(f"{name}: selected source does not provide {which}")
        return read()

    return LogicalSource(
        shape,
        f"{store.path}:{name}",
        lambda begin, end: resolve().values(begin, end),
        encoded,
        lambda: divisor("weight_divisor"),
        lambda: divisor("input_divisor"),
    )