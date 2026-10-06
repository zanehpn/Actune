"""Layer reconstruction calibration; reference math is offline only."""
import torch

from .kernels.quantizers import weight_codes


def activation_reference(value, bits, group_size=64):
    """The exact symmetric codebook used by the grouped inference path."""
    if bits not in (2, 4, 8) or value.shape[-1] % group_size:
        raise ValueError("Unsupported activation width or unaligned input")
    shape = value.shape
    value = value.float().reshape(*shape[:-1], -1, group_size)
    positive = (1 << (bits - 1)) - 1
    scale = value.abs().amax(-1, keepdim=True).clamp_min(1e-8) / positive
    codes = (value / scale).round().clamp(-positive - 1, positive)
    return (codes * scale).reshape(shape)


def jacobian_row_weights(actions, layer_outputs, *, probes=3, seed=7):
    """Hutchinson estimate of action-Jacobian squared norm per output channel.

    The caller supplies a differentiable BF16 action computation, including all
    denoising iterations and executed action coordinates. Repeated invocations
    of a matrix can be supplied as a list of captured outputs.
    """
    if probes < 1 or not actions.requires_grad:
        raise ValueError("Need differentiable reference actions and positive probe count")
    pairs = [(name, value) for name, values in layer_outputs.items()
             for value in (values if isinstance(values, (list, tuple)) else [values])]
    generator = torch.Generator(device=actions.device).manual_seed(seed)
    result = {name: torch.zeros(value.shape[-1], device=value.device) for name, value in pairs}
    for _ in range(probes):
        signs = torch.randint(0, 2, actions.shape, device=actions.device, generator=generator) * 2 - 1
        scalar = (actions.float() * signs).sum() / actions.numel()**0.5
        gradients = torch.autograd.grad(scalar, [v for _, v in pairs], retain_graph=True, allow_unused=True)
        for (name, value), gradient in zip(pairs, gradients):
            if gradient is not None:
                result[name] += gradient.detach().float().reshape(-1, value.shape[-1]).square().sum(0) / probes
    return result


@torch.no_grad()
def calibrate_linear(linear, inputs, row_weights=None, *, budget="w4a4", group_size=64,
                     clip_grid=(1., .975, .95, .925, .9, .875, .85, .8, .75)):
    """Fit activation-guided scales, row clips, and weighted errors per pair.

    inputs are BF16 fitting inputs only. Input scaling is performed in the
    source dtype, then activations use the same group-64 quantizer as inference.
    Return values are calibration tensors; this repository ships none of them.
    """
    if budget not in ("w4a4", "w4a8"):
        raise ValueError("Unknown precision budget")
    x = inputs.detach().reshape(-1, linear.in_features).to(linear.weight.device, linear.weight.dtype)
    if not len(x) or not torch.isfinite(x).all() or linear.in_features % group_size:
        raise ValueError("Nonempty finite aligned fitting inputs required")
    gains = torch.ones(linear.out_features, device=x.device) if row_weights is None else row_weights.to(x.device).float()
    if gains.shape != (linear.out_features,) or not torch.isfinite(gains).all() or (gains < 0).any():
        raise ValueError("Invalid action-Jacobian weights")
    if not bool(gains.sum()):
        gains = torch.ones_like(gains)
    reference = torch.nn.functional.linear(x.float(), linear.weight.float())
    base_a = 4 if budget == "w4a4" else 8
    moment = x.float().abs().mean(0).clamp_min(1e-4)
    best = None
    for alpha in (0., .25, .5, .75, 1.):
        scale = moment.pow(alpha)
        scale = (scale / (scale.max() * scale.min()).sqrt()).clamp(1/16, 16).to(x.dtype)
        weight = (linear.weight * scale[None, :]).to(x.dtype)
        q, s = weight_codes(weight, 4, base_a)
        y = torch.nn.functional.linear(activation_reference(x / scale, base_a, group_size), q.float() * s[:, None])
        score = float(((y - reference).square().mean(0) * gains).sum())
        if best is None or score < best[0]:
            best = score, scale
    scale = best[1]
    weight = (linear.weight * scale[None, :]).to(x.dtype)
    errors, clips = {}, {}
    for w in (2, 4, 8):
        for a in ((8,) if budget == "w4a8" else (2, 4, 8)):
            quantized_x = activation_reference(x / scale, a, group_size)
            row_error = torch.full((linear.out_features,), float("inf"), device=x.device)
            selected = torch.ones_like(row_error)
            for ratio in clip_grid:
                if not 0 < ratio <= 1:
                    raise ValueError("Clip ratios must lie in (0, 1]")
                q, s = weight_codes(weight, w, a, clip_ratios=ratio)
                output = torch.nn.functional.linear(quantized_x, q.float() * s[:, None])
                loss = (output - reference).square().mean(0)
                improve = loss < row_error
                selected[improve], row_error[improve] = ratio, loss[improve]
            mode = f"w{w}a{a}"
            errors[mode] = float((row_error * gains).sum())
            clips[mode] = selected
    return dict(channel_scale=scale, clips=clips, errors=errors, group_size=group_size)
