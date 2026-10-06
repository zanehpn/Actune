"""Calibrated integer codebooks, extracted unchanged from the VLA runtime."""
import torch

def _validate_clip_ratios(
    clip_ratios: torch.Tensor | float,
    *,
    out_features: int,
    device: torch.device,
) -> torch.Tensor:
    ratio = torch.as_tensor(clip_ratios, dtype=torch.float32, device=device).flatten()
    if ratio.numel() == 1:
        ratio = ratio.expand(out_features)
    if ratio.numel() != out_features:
        raise ValueError(
            f"expected {out_features} AWQ-style clip ratios, got {ratio.numel()}"
        )
    if not torch.isfinite(ratio).all():
        raise ValueError("AWQ-style clip ratios must be finite")
    if bool(((ratio <= 0.0) | (ratio > 1.0)).any()):
        raise ValueError("AWQ-style clip ratios must lie in (0, 1]")
    return ratio.contiguous()

def symmetric_int4_quantize(
    weight: torch.Tensor,
    *,
    clip_ratios: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to the frozen signed INT4 layout, optionally clipping rows."""

    value = weight.detach().float()
    row_absmax = value.abs().amax(dim=1).clamp_min(1e-8)
    if clip_ratios is not None:
        row_absmax = row_absmax * _validate_clip_ratios(
            clip_ratios,
            out_features=value.shape[0],
            device=value.device,
        )
    scale = row_absmax / 7.0
    q4 = (value / scale[:, None]).round().clamp(-8, 7).to(torch.int8)
    return q4, scale.contiguous()

def signed_r4_refine(
    weight: torch.Tensor,
    q4: torch.Tensor,
    scale4: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed R4 and the exact nested INT8 code used by the kernel."""

    value = weight.detach().float()
    residual_float = 16.0 * (value / scale4[:, None] - q4.float())
    q4_i16 = q4.to(torch.int16)
    lower = torch.maximum(torch.full_like(q4_i16, -8), -128 - 16 * q4_i16)
    upper = torch.minimum(torch.full_like(q4_i16, 7), 127 - 16 * q4_i16)
    residual = residual_float.round().clamp(-8, 7).to(torch.int16)
    residual = torch.minimum(torch.maximum(residual, lower), upper).to(torch.int8)
    q8 = (16 * q4_i16 + residual.to(torch.int16)).to(torch.int8)
    return residual, q8

def symmetric_int2_quantize(
    weight: torch.Tensor,
    *,
    clip_ratios: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to the signed W2 layout, optionally clipping output rows.

    Ratio one is bit-exact with the pre-existing ``W2A8TritonLinear`` RTN
    formula.  Its integer codebook is ``{-2, -1, 0, 1}`` and its row scale is
    the (possibly clipped) absolute maximum.
    """

    value = weight.detach().float()
    row_absmax = value.abs().amax(dim=1).clamp_min(1e-8)
    if clip_ratios is not None:
        row_absmax = row_absmax * _validate_clip_ratios(
            clip_ratios,
            out_features=value.shape[0],
            device=value.device,
        )
    scale = row_absmax
    q2 = (value / scale[:, None]).round().clamp(-2, 1).to(torch.int8)
    return q2, scale.contiguous()

def weight_codes(weight, w, a, *, clip_ratios=None):
    if w == 2:
        return symmetric_int2_quantize(weight, clip_ratios=clip_ratios)
    if w == 4:
        return symmetric_int4_quantize(weight, clip_ratios=clip_ratios)
    if a == 4:
        q4, scale4 = symmetric_int4_quantize(weight, clip_ratios=clip_ratios)
        _, q8 = signed_r4_refine(weight, q4, scale4)
        return q8, scale4 / 16.0
    value = weight.detach().float()
    scale = value.abs().amax(1).clamp_min(1e-8) / 127.0
    return (value / scale[:, None]).round().clamp(-128, 127).to(torch.int8), scale
