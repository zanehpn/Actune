"""Packed signed-INT2 weight / dynamic-INT8 activation Triton Linear.

This is a custom experimental SM120 path, not an NVIDIA-official 2-bit MMA
mode.  Four signed two-bit weights are packed into each byte.  The kernel
unpacks them directly into INT8 tiles, uses INT8 tensor-core dot products with
INT32 accumulation, and applies per-output-channel weight scales in the
epilogue.  Activations use the same fused per-row dynamic INT8 quantizer as the
frozen W4A8 baseline.

The module is deliberately standalone so adding it cannot change the currently
running W4A8/FP8/BF16 controller matrix's code path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from actune.kernels.w4a8_triton import _quantize_rows_shared
from actune.kernels.quantizers import symmetric_int2_quantize


@triton.jit
def _w2a8_gemm_kernel(
    activation_ptr,
    packed_weight_ptr,
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
    BLOCK_J: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offsets_m < rows
    mask_n = offsets_n < out_features
    quarter = in_features // 4
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for start_j in range(0, quarter, BLOCK_J):
        offsets_j = start_j + tl.arange(0, BLOCK_J)
        mask_j = offsets_j < quarter
        packed = tl.load(
            packed_weight_ptr + offsets_j[:, None] * out_features + offsets_n[None, :],
            mask=mask_j[:, None] & mask_n[None, :],
            other=0,
        )

        # (v ^ 2) - 2 sign-extends a two-bit two's-complement value:
        # 00 -> 0, 01 -> 1, 10 -> -2, 11 -> -1.
        weight_0 = (((packed & 3).to(tl.int8)) ^ 2) - 2
        weight_1 = ((((packed >> 2) & 3).to(tl.int8)) ^ 2) - 2
        weight_2 = ((((packed >> 4) & 3).to(tl.int8)) ^ 2) - 2
        weight_3 = ((((packed >> 6) & 3).to(tl.int8)) ^ 2) - 2

        activation_0 = tl.load(
            activation_ptr + offsets_m[:, None] * in_features + offsets_j[None, :],
            mask=mask_m[:, None] & mask_j[None, :],
            other=0,
        )
        activation_1 = tl.load(
            activation_ptr
            + offsets_m[:, None] * in_features
            + offsets_j[None, :]
            + quarter,
            mask=mask_m[:, None] & mask_j[None, :],
            other=0,
        )
        activation_2 = tl.load(
            activation_ptr
            + offsets_m[:, None] * in_features
            + offsets_j[None, :]
            + 2 * quarter,
            mask=mask_m[:, None] & mask_j[None, :],
            other=0,
        )
        activation_3 = tl.load(
            activation_ptr
            + offsets_m[:, None] * in_features
            + offsets_j[None, :]
            + 3 * quarter,
            mask=mask_m[:, None] & mask_j[None, :],
            other=0,
        )
        accumulator += tl.dot(activation_0, weight_0, out_dtype=tl.int32)
        accumulator += tl.dot(activation_1, weight_1, out_dtype=tl.int32)
        accumulator += tl.dot(activation_2, weight_2, out_dtype=tl.int32)
        accumulator += tl.dot(activation_3, weight_3, out_dtype=tl.int32)

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
    """Return ``(BLOCK_M, BLOCK_N, BLOCK_J, num_warps)``."""

    if rows <= 16:
        return 16, 128, 64, 4
    if rows <= 64:
        return 32, 128, 64, 4
    return 64, 128, 64, 8


class W2A8TritonLinear(torch.nn.Module):
    """Inference-only Linear with K-major packed signed-INT2 weights."""

    def __init__(
        self,
        linear: torch.nn.Linear,
        fused_quantization: bool = True,
        *,
        weight_clip_ratios: torch.Tensor | float | None = None,
    ) -> None:
        super().__init__()
        if linear.in_features % 128:
            raise ValueError("in_features must be divisible by 128")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.fused_quantization = fused_quantization
        with torch.no_grad():
            # This matches QVLA's signed two-bit fake-quant convention.  The
            # representable integer set is {-2,-1,0,1}; max-abs scaling avoids
            # clipping and commonly produces a sparse ternary-like codebook.
            quantized, scale = symmetric_int2_quantize(
                linear.weight,
                clip_ratios=weight_clip_ratios,
            )
            quarter = self.in_features // 4
            fields = [
                quantized[:, i * quarter : (i + 1) * quarter].to(torch.int16) & 3
                for i in range(4)
            ]
            packed = (
                fields[0]
                | (fields[1] << 2)
                | (fields[2] << 4)
                | (fields[3] << 6)
            ).to(torch.uint8).t().contiguous()
        self.register_buffer("packed_weight", packed)
        self.register_buffer("weight_scale", scale.contiguous())
        self.weight_quantizer = (
            "rtn_per_output_row_w2_v1"
            if weight_clip_ratios is None
            else "awq_style_joint_precision_lattice_clip_v1"
        )
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
        block_m, block_n, block_j, warps = _config_for(rows)
        grid = (triton.cdiv(rows, block_m), triton.cdiv(self.out_features, block_n))
        bias = self.bias if self.bias is not None else self.weight_scale
        _w2a8_gemm_kernel[grid](
            activation,
            self.packed_weight,
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
            BLOCK_J=block_j,
            num_warps=warps,
            num_stages=3,
        )
        return output.reshape(*original_shape, self.out_features)


def replace_with_w2a8_triton_(module: torch.nn.Module, filter_fn) -> int:
    """Recursively replace eligible Linear children with W2A8 backends."""

    replaced = 0
    for name, child in list(module.named_children()):
        if filter_fn(child, name):
            setattr(module, name, W2A8TritonLinear(child))
            replaced += 1
        else:
            replaced += replace_with_w2a8_triton_(child, filter_fn)
    return replaced
