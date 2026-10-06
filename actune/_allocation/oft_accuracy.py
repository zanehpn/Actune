"""Conservative OFT allocation anchored at uniform W4 instead of uniform W2.

Only a capped set of low-penalty matrices may use W2/A2. Their saved bits
fund upgrades, while every action-routed template independently satisfies W4
and A8/A4. This changes allocation, not the frozen quantization formulas.
"""
from collections import defaultdict
from copy import deepcopy
import math

from .oft_budget import OFTBudgetProfiles
from .pi05_budget import BudgetProfile, pair


def has_mixed_activations(profiles):
    """All risk templates must actually assign each permitted activation width."""
    return bool(profiles.profiles) and all({pair(m)[1] for m in p.assignments.values()} == {2, 4, 8}
                                          for p in profiles.profiles)


def conservative_profiles(source, errors, *, weight_low_fraction, activation_low_fraction=0.0):
    if not 0 <= weight_low_fraction <= 1 or not 0 <= activation_low_fraction <= 1:
        raise ValueError("low-precision fractions must lie in [0, 1]")
    metadata = deepcopy(source.metadata)
    shapes = metadata["layer_shapes"]
    elements = metadata["activation_elements"]
    groups = metadata["activation_budget_groups"]
    variant = metadata["variant"]
    a_base = 8 if variant == "w4a8" else 4
    if a_base == 8 and activation_low_fraction:
        raise ValueError("fixed A8 does not permit A2")
    sizes = {n: math.prod(s) for n, s in shapes.items()}
    total_w = sum(sizes.values())
    total_a = sum(elements.values())
    base = f"w4a{a_base}"
    modes = [f"w{w}a{a}" for w in (2, 4, 8)
             for a in ((8,) if a_base == 8 else (2, 4, 8))]
    for n in shapes:
        if any(not math.isfinite(errors[n][m]) or errors[n][m] < 0 for m in modes):
            raise ValueError("sensitivity must be finite and nonnegative")

    def capped_low(names, costs, cap, low_mode):
        selected, used = set(), 0
        ordered = sorted(names, key=lambda n: ((errors[n][low_mode] - errors[n][base]) / costs[n], n))
        for n in ordered:
            if used + costs[n] <= cap:
                selected.add(n)
                used += costs[n]
        return selected

    w2 = capped_low(shapes, sizes, weight_low_fraction * total_w, f"w2a{a_base}")
    a2 = set()
    if a_base == 4:
        for group in groups:
            a2 |= capped_low(group, elements, activation_low_fraction * sum(elements[n] for n in group), "w4a2")
    group_of = {n: i for i, group in enumerate(groups) for n in group}
    group_totals = [sum(elements[n] for n in group) for group in groups]
    assignment = {n: f"w{2 if n in w2 else 4}a{2 if n in a2 else a_base}" for n in shapes}
    used_w = sum(sizes[n] * pair(m)[0] for n, m in assignment.items())
    used_a = [sum(elements[n] * pair(assignment[n])[1] for n in group) for group in groups]
    states = [dict(assignment)]
    while True:
        best = None
        for n in sorted(shapes):
            old = assignment[n]
            w0, a0 = pair(old)
            for mode in modes:
                w, a = pair(mode)
                if mode == old or w < w0 or a < a0:
                    continue
                gain = errors[n][old] - errors[n][mode]
                dw, da = (w - w0) * sizes[n], (a - a0) * elements[n]
                g = group_of[n]
                if gain <= 0 or used_w + dw > 4 * total_w or used_a[g] + da > a_base * group_totals[g]:
                    continue
                score = gain / (dw / total_w + da / total_a)
                if best is None or score > best[0]:
                    best = (score, n, mode, dw, da)
        if best is None:
            break
        _, n, mode, dw, da = best
        assignment[n] = mode
        used_w += dw
        used_a[group_of[n]] += da
        states.append(dict(assignment))

    metadata.update(allocator="uniform_w4_anchored_capped_low_precision_v1",
                    maximum_w2_parameter_fraction=weight_low_fraction,
                    maximum_a2_input_fraction_per_shape_group=activation_low_fraction,
                    profile_budget_policy="all_templates_at_most_W4; risk tiers retain progressively more feasible upgrades",
                    allocation_selection_data="existing training-observation sensitivity only")
    result = OFTBudgetProfiles((), metadata)
    profiles = []
    for name, fraction in (("cruise", .5), ("alignment", .75), ("contact", 1.0)):
        assignments = states[round((len(states) - 1) * fraction)]
        metrics = result.metrics(assignments)
        profiles.append(BudgetProfile(name, assignments, 4.0, metrics["weight_bits"],
                                      float(a_base), metrics["activation_bits"],
                                      sum(errors[n][m] for n, m in assignments.items())))
    result.profiles = tuple(profiles)
    return result.validate()
