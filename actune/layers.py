"""Install calibrated resident backends into an existing PyTorch backbone."""
from types import MethodType
import torch

from .kernels.grouped import GroupedPaddedLinear
from .precision import active_precision_profile


def tensor_storage_bytes(module):
    seen, total = set(), 0
    for value in list(module.parameters()) + list(module.buffers()):
        storage = value.untyped_storage()
        key = (value.device, storage.data_ptr())
        if key not in seen:
            total += storage.nbytes()
            seen.add(key)
    return total


class ResidentLinear(GroupedPaddedLinear):
    def __init__(self, linear, assignments, clips, input_scale=None):
        # Scales can instead be folded into adjacent norm/linear modules before
        # installation. In that case pass input_scale=None and calibrated weights.
        if input_scale is not None:
            scale = torch.as_tensor(input_scale, device=linear.weight.device, dtype=linear.weight.dtype)
            if scale.shape != (linear.in_features,) or not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError("Invalid calibrated channel scale")
            scaled = torch.nn.Linear(linear.in_features, linear.out_features,
                bias=linear.bias is not None, device=linear.weight.device, dtype=linear.weight.dtype)
            with torch.no_grad():
                scaled.weight.copy_(linear.weight * scale[None, :])
                if linear.bias is not None:
                    scaled.bias.copy_(linear.bias)
            linear = scaled
        else:
            scale = None
        super().__init__(linear, assignments, "q11", clips)
        self.register_buffer("input_scale", scale)

    def forward(self, value):
        if self.input_scale is not None:
            value = value / self.input_scale
        return super().forward(value)


class BankRuntime:
    def __init__(self, bank, modules):
        self.bank, self.modules = bank, modules

    def before(self):
        for module in self.modules.values():
            module.calls = module.input_elements = 0

    def audit(self, configuration):
        counts = {name: module.input_elements for name, module in self.modules.items()}
        return self.bank.check_request(configuration, counts)


def install_bank(model, bank, clips, *, channel_scales=None, tune=False):
    """Group-64 path covering every W{2,4,8}/A{2,4,8} pair.

    Bank errors/clips MUST have been scored with this same group-64 quantizer.
    Other (e.g. tokenwise A8) backends are supplied under actune.kernels and
    require their own calibration. This function does not silently mix them.
    """
    bank.validate()
    scales = {} if channel_scales is None else channel_scales
    # Validate coverage before modifying any module.
    originals = {name: model.get_submodule(name) for name in bank.shapes}
    for name, module in originals.items():
        if not isinstance(module, torch.nn.Linear) or [module.out_features, module.in_features] != list(bank.shapes[name]):
            raise ValueError(f"Target does not match the calibrated matrix: {name}")
    replacements = {}
    for name, linear in originals.items():
        assignments = {q: c[name] for q, c in bank.configurations.items()}
        replacement = ResidentLinear(linear, assignments, clips[name], scales.get(name))
        for mode, backend in replacement.backends.items():
            if tensor_storage_bytes(backend) != bank.version_bytes[name][mode]:
                raise ValueError(f"Loaded backend storage differs from calibration: {name}/{mode}")
            if tune:
                from .kernels.tuning import grouped_forward
                backend._round2_original = backend.forward
                backend.forward = MethodType(grouped_forward, backend)
        replacements[name] = replacement
    actual = sum(tensor_storage_bytes(m) for m in replacements.values())
    if actual > bank.storage_limit:
        raise ValueError("Loaded bank, including channel scales, exceeds storage budget")
    for name, module in replacements.items():
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, module)
    return BankRuntime(bank, replacements)
