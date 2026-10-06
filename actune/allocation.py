"""Sensitivity-guided four-configuration bank construction."""
from copy import deepcopy
import math

from ._allocation.oft_accuracy import conservative_profiles
from ._allocation.oft_budget import OFTBudgetProfiles, QUANTIZER
from ._allocation.pi05_budget import pair
from .precision import SharedBank
from .tree import CONFIGURATIONS

LOW_FRACTIONS = ((0.0025, 0.025), (0.01, 0.05), (0.025, 0.10))


def build_banks(shapes, activation_elements, shape_groups, errors, version_bytes, *,
                budget, storage_limit, fixed_layers=(), compact_storage=False):
    """All statistics and sensitivities must come from the fitting split.

    version_bytes includes each backend's packed codes, scales, biases and any
    alignment padding. fixed_layers cannot switch (e.g. pi0.5 vision graphs).
    compact_storage permits the OFT W4->W2 storage refinement after W8 repair.
    Rank candidate banks with BF16 action agreement on fitting data only.
    """
    if budget not in ("w4a4", "w4a8") or storage_limit <= 0:
        raise ValueError("Invalid precision or resident-storage budget")
    modes = [f"w{w}a{a}" for w in (2, 4, 8) for a in ((8,) if budget == "w4a8" else (2, 4, 8))]
    for name in shapes:
        for mode in modes:
            if not math.isfinite(errors[name][mode]) or errors[name][mode] < 0:
                raise ValueError("Reconstruction errors must be finite and nonnegative")
            if type(version_bytes[name][mode]) is not int or version_bytes[name][mode] <= 0:
                raise ValueError("Missing backend storage accounting")
    meta = dict(layer_shapes=shapes, activation_elements=activation_elements,
        activation_budget_groups=shape_groups, variant=budget, model_family="oft",
        quantizer=QUANTIZER, chunk_size=8)
    # The model tag above is an internal allocator interface; no model is loaded.
    source = OFTBudgetProfiles((), meta)
    result = {}
    for wf, af in LOW_FRACTIONS:
        allocated = conservative_profiles(source, errors, weight_low_fraction=wf,
            activation_low_fraction=af if budget == "w4a4" else 0.)
        high = dict(allocated.profile("contact").assignments)
        alternatives = []
        for name, mode in high.items():
            if name in fixed_layers:
                continue
            w, a = pair(mode)
            for lower in modes:
                lw, la = pair(lower)
                if lower != mode and lw <= w and la <= a and version_bytes[name][lower] <= 2 * 1024**2:
                    alternatives.append((errors[name][lower] - errors[name][mode], name, lower))
        switches = {}
        for _, name, lower in sorted(alternatives):
            switches.setdefault(name, lower)
            if len(switches) == 2:
                break
        if len(switches) != 2:
            continue
        first, second = list(switches)
        configurations = {q: dict(high) for q in CONFIGURATIONS}
        for q in ("q00", "q01"):
            configurations[q][first] = switches[first]
        for q in ("q00", "q10"):
            configurations[q][second] = switches[second]
        bank = SharedBank(configurations, deepcopy(shapes), deepcopy(activation_elements),
            deepcopy(shape_groups), deepcopy(version_bytes), budget, storage_limit)
        while bank.resident_bytes() > storage_limit:
            choices = []
            for name, mode in high.items():
                if name in switches:
                    continue
                w, a = pair(mode)
                if w != 8:
                    continue
                lower = f"w4a{a}"
                saved = version_bytes[name][mode] - version_bytes[name][lower]
                if saved > 0:
                    choices.append(((errors[name][lower] - errors[name][mode]) / saved, name, lower))
            if not choices and compact_storage:
                for name, mode in high.items():
                    if name in switches or pair(mode)[0] != 4:
                        continue
                    lower = f"w2a{pair(mode)[1]}"
                    saved = version_bytes[name][mode] - version_bytes[name][lower]
                    if saved > 0:
                        choices.append(((errors[name][lower] - errors[name][mode]) / saved, name, lower))
            if not choices:
                break
            _, name, lower = min(choices)
            high[name] = lower
            for allocation in configurations.values():
                allocation[name] = lower
        if bank.resident_bytes() <= storage_limit:
            result[f"w2_{wf:g}_a2_{af if budget == 'w4a4' else 0:g}"] = bank.validate()
    if not result:
        raise ValueError("No candidate satisfies active precision and full resident-storage budgets")
    return result


def select_bank(banks, fitting_action_losses):
    """Mean action loss over q00/q01/q11; q10 reuses their retained versions."""
    import numpy as np
    scores = {}
    for key in banks:
        loss = np.asarray(fitting_action_losses[key], float)
        if loss.ndim != 2 or loss.shape[1] != 4 or not len(loss) or not np.isfinite(loss).all() or (loss < 0).any():
            raise ValueError("Expected fitting-context losses for all four configurations")
        scores[key] = float(loss[:, :3].mean())
    key = min(scores, key=lambda k: (scores[k], k))
    return key, banks[key]
