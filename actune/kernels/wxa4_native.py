"""Native Ampere integer Tensor-Core backends for W2A4, W4A4, and W8A4.

W4A4 consumes packed S4 weights directly.  W2A4 retains two-bit weight
storage, sign-extends each code to an S4 operand in the CUDA kernel, and then
executes the same native ``mma.sync.m16n8k64.s4.s4.s32`` instruction.  Thus
W2A4 means two-bit storage plus four-bit compute, not nonexistent INT2 MMA.
W8A4 sign-extends each A4 code to an INT8 lane and uses one S8xS8 MMA. This
preserves A4 quantization semantics without paying for a two-S4 decomposition;
Ampere has no native mixed S8xS4 Tensor-Core instruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from actune.kernels.quantizers import symmetric_int4_quantize
from actune.kernels.quantizers import signed_r4_refine
from actune.kernels.quantizers import symmetric_int2_quantize
from actune.kernels.w8a8_triton import _config_for, _w8a8_gemm_kernel


@triton.jit
def _quantize_rows_a4_kernel(
    source_ptr,
    packed_ptr,
    scale_ptr,
    in_features: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    base = source_ptr + row * in_features
    amax = tl.zeros((), dtype=tl.float32)
    for start in range(0, in_features, BLOCK_K):
        mask = start + offsets < in_features
        values = tl.load(base + start + offsets, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(values)))
    scale = tl.maximum(amax, tl.full((), 1e-8, tl.float32)) * tl.full(
        (), 1.0 / 7.0, tl.float32
    )
    tl.store(scale_ptr + row, scale)

    pair_offsets = tl.arange(0, BLOCK_K // 2)
    for start in range(0, in_features, BLOCK_K):
        low_index = start + 2 * pair_offsets
        high_index = low_index + 1
        low_mask = low_index < in_features
        high_mask = high_index < in_features
        low = tl.load(base + low_index, mask=low_mask, other=0.0).to(tl.float32)
        high = tl.load(base + high_index, mask=high_mask, other=0.0).to(tl.float32)
        low = tl.extra.libdevice.rint(tl.extra.libdevice.div_rn(low, scale))
        high = tl.extra.libdevice.rint(tl.extra.libdevice.div_rn(high, scale))
        low = tl.minimum(tl.maximum(low, -8.0), 7.0).to(tl.int16) & 15
        high = tl.minimum(tl.maximum(high, -8.0), 7.0).to(tl.int16) & 15
        packed = (low | (high << 4)).to(tl.uint8)
        tl.store(
            packed_ptr + row * (in_features // 2) + start // 2 + pair_offsets,
            packed,
            mask=low_mask,
        )


@triton.jit
def _unpack_a4_to_int8_kernel(
    packed_ptr,
    activation_ptr,
    packed_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < packed_elements
    packed = tl.load(packed_ptr + offsets, mask=mask, other=0).to(tl.int16)
    low = ((packed & 15) ^ 8) - 8
    high = (((packed >> 4) & 15) ^ 8) - 8
    tl.store(activation_ptr + 2 * offsets, low.to(tl.int8), mask=mask)
    tl.store(activation_ptr + 2 * offsets + 1, high.to(tl.int8), mask=mask)


class _A4Entry:
    __slots__ = ("source", "in_features", "packed", "scale", "activation_int8")

    def __init__(
        self,
        source: torch.Tensor,
        in_features: int,
        packed: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        self.source = source
        self.in_features = in_features
        self.packed = packed
        self.scale = scale
        self.activation_int8: torch.Tensor | None = None


_EXTENSION: Any | None = None


def _native_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parent
        _EXTENSION = load(
            name="vla_native_s4s4_sm80_v1",
            sources=[
                str(root / "native_s4_extension.cpp"),
                str(root / "native_s4_extension_kernel.cu"),
            ],
            extra_cuda_cflags=("-O3", "--use_fast_math"),
            extra_cflags=("-O3",),
            verbose=False,
        )
    return _EXTENSION


def _quantize_a4_shared(
    source: torch.Tensor,
    flat: torch.Tensor,
    *,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not source.is_cuda or flat.shape[-1] != in_features:
        raise ValueError("native A4 quantization requires a CUDA input with matching K")
    # Reuse the same explicitly bounded policy-forward scope as dynamic A8.
    # Outside that scope caching is disabled so an in-place-reused tensor can
    # never observe stale quantization from an earlier request.
    from actune.kernels.w4a8_triton import _SHARED_A8_STATE

    state = _SHARED_A8_STATE.get()
    entry = None if state is None else state.a4_entry
    if (
        entry is not None
        and entry.source is source
        and entry.in_features == in_features
        and entry.packed.shape[0] == flat.shape[0]
    ):
        return entry.packed, entry.scale
    packed = torch.empty(
        (flat.shape[0], in_features // 2), device=flat.device, dtype=torch.uint8
    )
    scale = torch.empty(flat.shape[0], device=flat.device, dtype=torch.float32)
    block = 1024 if in_features >= 1024 else 256
    _quantize_rows_a4_kernel[(flat.shape[0],)](
        flat,
        packed,
        scale,
        in_features=in_features,
        BLOCK_K=block,
        num_warps=4,
    )
    if state is not None and state.enabled:
        state.a4_entry = _A4Entry(source, in_features, packed, scale)
    return packed, scale


def _quantize_a4_int8_shared(
    source: torch.Tensor,
    flat: torch.Tensor,
    *,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return A4 codes sign-extended to INT8, reusing the bounded A4 cache."""

    from actune.kernels.w4a8_triton import _SHARED_A8_STATE

    state = _SHARED_A8_STATE.get()
    entry = None if state is None else state.a4_entry
    if (
        entry is not None
        and entry.source is source
        and entry.in_features == in_features
        and entry.packed.shape[0] == flat.shape[0]
        and entry.activation_int8 is not None
    ):
        return entry.activation_int8, entry.scale
    packed, scale = _quantize_a4_shared(source, flat, in_features=in_features)
    activation = torch.empty(flat.shape, device=flat.device, dtype=torch.int8)
    block = 256
    packed_elements = packed.numel()
    _unpack_a4_to_int8_kernel[(triton.cdiv(packed_elements, block),)](
        packed,
        activation,
        packed_elements,
        BLOCK=block,
        num_warps=4,
    )
    state = _SHARED_A8_STATE.get()
    entry = None if state is None else state.a4_entry
    if (
        entry is not None
        and entry.source is source
        and entry.in_features == in_features
        and entry.packed.shape[0] == flat.shape[0]
    ):
        entry.activation_int8 = activation
    return activation, scale


def _pack_signed(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack consecutive signed codes along K, retaining row-major weights."""

    if bits == 4:
        fields = values.to(torch.int16).reshape(values.shape[0], -1, 2) & 15
        return (fields[:, :, 0] | (fields[:, :, 1] << 4)).to(torch.uint8).contiguous()
    if bits == 2:
        fields = values.to(torch.int16).reshape(values.shape[0], -1, 4) & 3
        return (
            fields[:, :, 0]
            | (fields[:, :, 1] << 2)
            | (fields[:, :, 2] << 4)
            | (fields[:, :, 3] << 6)
        ).to(torch.uint8).contiguous()
    raise ValueError("bits must be 2 or 4")


class _WxA4NativeLinear(torch.nn.Module):
    weight_bits: int

    def __init__(
        self,
        linear: torch.nn.Linear,
        *,
        weight_bits: int,
        weight_clip_ratios: torch.Tensor | float | None = None,
    ) -> None:
        super().__init__()
        if linear.in_features % 64:
            raise ValueError("native S4 MMA requires in_features divisible by 64")
        if weight_bits not in (2, 4):
            raise ValueError("weight_bits must be 2 or 4")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight_bits = weight_bits
        with torch.no_grad():
            if weight_bits == 2:
                quantized, scale = symmetric_int2_quantize(
                    linear.weight, clip_ratios=weight_clip_ratios
                )
            else:
                quantized, scale = symmetric_int4_quantize(
                    linear.weight, clip_ratios=weight_clip_ratios
                )
            packed = _pack_signed(quantized, weight_bits)
        self.register_buffer("packed_weight", packed)
        self.register_buffer("weight_scale", scale.contiguous())
        self.register_buffer(
            "bias",
            torch.empty(0, dtype=torch.float32, device=linear.weight.device)
            if linear.bias is None
            else linear.bias.detach().float().contiguous(),
        )
        self.weight_quantizer = (
            f"rtn_per_output_row_w{weight_bits}_v1"
            if weight_clip_ratios is None
            else "awq_style_joint_precision_lattice_clip_v1"
        )
        self.activation_quantizer = "dynamic_per_row_s4_v1"
        self.compute_instruction = "mma.sync.m16n8k64.s4.s4.s32"
        self.native_int4_tensor_core = True

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not value.is_cuda:
            raise ValueError("native S4 MMA requires a CUDA input")
        major, _minor = torch.cuda.get_device_capability(value.device)
        if major < 8:
            raise RuntimeError("native S4 MMA requires compute capability 8.0+")
        original_shape = value.shape[:-1]
        flat = value.reshape(-1, self.in_features)
        packed, activation_scale = _quantize_a4_shared(
            value, flat, in_features=self.in_features
        )
        output = _native_extension().linear(
            packed,
            activation_scale,
            self.packed_weight,
            self.weight_scale,
            self.bias,
            flat,
            self.weight_bits,
        )
        return output.reshape(*original_shape, self.out_features)


class W2A4NativeLinear(_WxA4NativeLinear):
    def __init__(
        self,
        linear: torch.nn.Linear,
        *,
        weight_clip_ratios: torch.Tensor | float | None = None,
    ) -> None:
        super().__init__(
            linear, weight_bits=2, weight_clip_ratios=weight_clip_ratios
        )


class W4A4NativeLinear(_WxA4NativeLinear):
    def __init__(
        self,
        linear: torch.nn.Linear,
        *,
        weight_clip_ratios: torch.Tensor | float | None = None,
    ) -> None:
        super().__init__(
            linear, weight_bits=4, weight_clip_ratios=weight_clip_ratios
        )


class W8A4NativeLinear(torch.nn.Module):
    """INT8-weight/A4 Linear implemented as one signed-S8 Tensor-Core GEMM."""

    def __init__(
        self,
        linear: torch.nn.Linear,
        *,
        weight_clip_ratios: torch.Tensor | float | None = None,
    ) -> None:
        super().__init__()
        if linear.in_features % 128:
            raise ValueError("W8A4 S8 MMA requires in_features divisible by 128")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        with torch.no_grad():
            q4, scale4 = symmetric_int4_quantize(
                linear.weight, clip_ratios=weight_clip_ratios
            )
            _residual, q8 = signed_r4_refine(linear.weight, q4, scale4)
            weight_int8 = q8.t().contiguous()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("weight_scale", (scale4 / 16.0).contiguous())
        self.register_buffer(
            "bias",
            torch.empty(0, dtype=torch.float32, device=linear.weight.device)
            if linear.bias is None
            else linear.bias.detach().float().contiguous(),
        )
        self.weight_bits = 8
        self.weight_quantizer = (
            "nested_signed_w8_rtn_v1"
            if weight_clip_ratios is None
            else "awq_style_joint_precision_lattice_clip_v1"
        )
        self.activation_quantizer = "dynamic_per_row_s4_v1"
        self.compute_instruction = "mma.sync.s8.s8.s32 (A4 sign-extended to S8)"
        self.native_int4_tensor_core = False
        self.native_int8_tensor_core = True
        self.a4_sign_extended_to_int8 = True
        self.s4_mma_count_per_k_tile = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not value.is_cuda:
            raise ValueError("W8A4 integer MMA requires a CUDA input")
        major, _minor = torch.cuda.get_device_capability(value.device)
        if major < 8:
            raise RuntimeError("W8A4 integer MMA requires compute capability 8.0+")
        original_shape = value.shape[:-1]
        flat = value.reshape(-1, self.in_features)
        activation, activation_scale = _quantize_a4_int8_shared(
            value, flat, in_features=self.in_features
        )
        rows = flat.shape[0]
        output = torch.empty(
            (rows, self.out_features), device=value.device, dtype=value.dtype
        )
        block_m, block_n, block_k, warps = _config_for(rows)
        grid = (triton.cdiv(rows, block_m), triton.cdiv(self.out_features, block_n))
        bias = self.bias if self.bias.numel() else self.weight_scale
        _w8a8_gemm_kernel[grid](
            activation,
            self.weight_int8,
            activation_scale,
            self.weight_scale,
            bias,
            output,
            rows,
            out_features=self.out_features,
            in_features=self.in_features,
            has_bias=bool(self.bias.numel()),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=3,
        )
        return output.reshape(*original_shape, self.out_features)
