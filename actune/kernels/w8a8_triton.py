"""Signed-INT8 weight / dynamic-INT8 activation Triton Linear.

This is the integer 8-bit member of the QVLA-aligned successor ladder.  The
weight matrix is stored K-major as INT8, activations use fused per-row dynamic
INT8 quantization, tensor-core dot products accumulate in INT32, and one FP32
scale per output channel is applied in the epilogue.

The module is separate from ``oft.py`` so it cannot change the currently
running W4A8/FP8/BF16 controller matrix.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from actune.kernels.w4a8_triton import _quantize_rows_shared


@triton.jit
def _w8a8_gemm_kernel(
    activation_ptr,
    weight_ptr,
    activation_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    output_ptr,
    rows,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offsets_m < rows
    mask_n = offsets_n < out_features
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for start_k in range(0, in_features, BLOCK_K):
        offsets_k = start_k + tl.arange(0, BLOCK_K)
        mask_k = offsets_k < in_features
        activation = tl.load(
            activation_ptr
            + offsets_m[:, None] * in_features
            + offsets_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        )
        weight = tl.load(
            weight_ptr
            + offsets_k[:, None] * out_features
            + offsets_n[None, :],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        )
        accumulator += tl.dot(activation, weight, out_dtype=tl.int32)

    output = accumulator.to(tl.float32)
    output *= tl.load(
        activation_scale_ptr + offsets_m, mask=mask_m, other=0.0
    )[:, None]
    output *= tl.load(
        weight_scale_ptr + offsets_n, mask=mask_n, other=0.0
    )[None, :]
    if has_bias:
        output += tl.load(bias_ptr + offsets_n, mask=mask_n, other=0.0)[None, :]
    tl.store(
        output_ptr + offsets_m[:, None] * out_features + offsets_n[None, :],
        output,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _config_for(rows: int) -> tuple[int, int, int, int]:
    """Return ``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps)``."""

    if rows <= 16:
        return 16, 128, 64, 4
    if rows <= 64:
        return 32, 128, 64, 4
    return 64, 128, 64, 8


class W8A8TritonLinear(torch.nn.Module):
    """Inference-only Linear with K-major signed-INT8 weights."""

    def __init__(self, linear: torch.nn.Linear, fused_quantization: bool = True) -> None:
        super().__init__()
        if linear.in_features % 128:
            raise ValueError("in_features must be divisible by 128")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.fused_quantization = fused_quantization
        with torch.no_grad():
            weight = linear.weight.detach().float()
            scale = weight.abs().amax(dim=1).clamp_min(1e-8) / 127.0
            quantized = (
                (weight / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
            )
            weight_k_major = quantized.t().contiguous()
        self.register_buffer("weight_int8", weight_k_major)
        self.register_buffer("weight_scale", scale.contiguous())
        if linear.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", linear.bias.detach().clone())

    def _quantize(
        self, flat: torch.Tensor, *, source: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fused_quantization:
            return _quantize_rows_shared(
                flat if source is None else source,
                flat,
                in_features=self.in_features,
            )
        scale = flat.float().abs().amax(dim=1).clamp_min(1e-8) / 127.0
        activation = (
            (flat.float() / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
        )
        return activation, scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original_shape = value.shape[:-1]
        flat = value.reshape(-1, self.in_features)
        activation, scale = self._quantize(flat, source=value)
        rows = flat.shape[0]
        output = torch.empty(
            (rows, self.out_features), device=value.device, dtype=value.dtype
        )
        block_m, block_n, block_k, warps = _config_for(rows)
        grid = (triton.cdiv(rows, block_m), triton.cdiv(self.out_features, block_n))
        bias = self.bias if self.bias is not None else self.weight_scale
        _w8a8_gemm_kernel[grid](
            activation,
            self.weight_int8,
            scale,
            self.weight_scale,
            bias,
            output,
            rows,
            out_features=self.out_features,
            in_features=self.in_features,
            has_bias=self.bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=3,
        )
        return output.reshape(*original_shape, self.out_features)


def replace_with_w8a8_triton_(module: torch.nn.Module, filter_fn) -> int:
    """Recursively replace eligible Linear children with W8A8 backends."""

    replaced = 0
    for name, child in list(module.named_children()):
        if filter_fn(child, name):
            setattr(module, name, W8A8TritonLinear(child))
            replaced += 1
        else:
            replaced += replace_with_w8a8_triton_(child, filter_fn)
    return replaced
