"""Strict, separately weighted W/A budgets for whole pi05 Linear matrices.

W counts logical parameters once. A counts input elements at every invocation,
including image encodings and denoising. Neither metric is a latency estimate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


SCHEMA = "pi05_joint_budget_v1"
PROFILE_BUDGETS = (("cruise", 3.5), ("alignment", 3.75), ("contact", 4.0))


def pair(mode):
    if not isinstance(mode, str) or mode not in {f"w{w}a{a}" for w in (2, 4, 8) for a in (2, 4, 8)}:
        raise ValueError(f"invalid pi05 precision: {mode}")
    return int(mode[1]), int(mode[3])


@dataclass
class BudgetProfile:
    name: str
    assignments: dict[str, str]
    target_average_cost: float
    realized_average_cost: float
    target_activation_bits: float
    realized_activation_bits: float
    predicted_error: float


@dataclass
class Pi05BudgetProfiles:
    profiles: tuple[BudgetProfile, ...]
    metadata: dict

    def profile(self, name):
        return next(p for p in self.profiles if p.name == name)

    def metrics(self, assignments, activation_elements=None):
        shapes = self.metadata["layer_shapes"]
        elements = self.metadata["activation_elements"] if activation_elements is None else activation_elements
        if set(assignments) != set(shapes) or set(elements) != set(shapes):
            raise ValueError("budget coverage differs from the target layers")
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in elements.values()):
            raise ValueError("activation element counts must be positive integers")
        total_w = sum(n * k for n, k in shapes.values())
        total_a = sum(elements.values())
        w = sum(math.prod(shapes[n]) * pair(m)[0] for n, m in assignments.items())
        a = sum(elements[n] * pair(m)[1] for n, m in assignments.items())
        return {"weight_bits": w / total_w, "activation_bits": a / total_a,
                "weight_bit_count": w, "weight_elements": total_w,
                "activation_bit_count": a, "activation_elements": total_a,
                "layers_by_precision": dict(sorted(Counter(assignments.values()).items()))}

    def validate(self, *, profile_names=None, require_nested=True):
        if self.metadata.get("variant") not in {"w4a8", "w4a4"}:
            raise ValueError("unknown budget variant")
        shapes = self.metadata["layer_shapes"]
        if not shapes or any(len(s) != 2 or any(type(x) is not int or x <= 0 for x in s) for s in shapes.values()):
            raise ValueError("invalid logical layer shapes")
        if [p.name for p in self.profiles] != (profile_names if profile_names is not None else [n for n, _ in PROFILE_BUDGETS]):
            raise ValueError("budget profiles must be cruise/alignment/contact")
        previous = None
        for profile in self.profiles:
            metrics = self.metrics(profile.assignments)
            a8 = self.metadata["variant"] == "w4a8"
            if a8 and any(pair(m)[1] != 8 for m in profile.assignments.values()):
                raise ValueError("w4a8 requires fixed A8 on every target layer")
            if not (2 <= profile.target_average_cost <= 4 and
                    2 <= profile.target_activation_bits <= (8 if a8 else 4)):
                raise ValueError("requested budgets exceed variant limits")
            if metrics["weight_bit_count"] > profile.target_average_cost * metrics["weight_elements"]:
                raise ValueError(f"{profile.name}: weight budget exceeded")
            if metrics["activation_bit_count"] > profile.target_activation_bits * metrics["activation_elements"]:
                raise ValueError(f"{profile.name}: activation budget exceeded")
            if not math.isclose(metrics["weight_bits"], profile.realized_average_cost, abs_tol=1e-12, rel_tol=0):
                raise ValueError("reported weight average differs from assignments")
            if not math.isclose(metrics["activation_bits"], profile.realized_activation_bits, abs_tol=1e-12, rel_tol=0):
                raise ValueError("reported activation average differs from assignments")
            if not math.isfinite(profile.predicted_error) or profile.predicted_error < 0:
                raise ValueError("invalid sensitivity score")
            if previous and require_nested:
                for name, mode in profile.assignments.items():
                    if any(x < y for x, y in zip(pair(mode), pair(previous.assignments[name]))):
                        raise ValueError("budget profiles must be componentwise nested")
            previous = profile
        return self

    def save(self, path):
        self.validate()
        Path(path).write_text(json.dumps({"schema": SCHEMA, "metadata": self.metadata,
            "profiles": [asdict(p) for p in self.profiles]}, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text())
        if raw.get("schema") != SCHEMA:
            raise ValueError("not a pi05 budget artifact")
        return cls(tuple(BudgetProfile(**p) for p in raw["profiles"]), raw["metadata"]).validate()


def make_budget_profiles(shapes, activation_elements, errors, metadata, *, variant):
    """Greedy whole-layer upgrades under two exact integer bit-count caps.

This is a feasible allocator, not an exact multidimensional knapsack solver.
Sensitivity is measured for all nine pairs; coupled W/A upgrades are allowed.
"""
    if variant not in {"w4a8", "w4a4"}:
        raise ValueError("variant must be w4a8 or w4a4")
    modes = [f"w{w}a{a}" for w in (2, 4, 8) for a in ((8,) if variant == "w4a8" else (2, 4, 8))]
    for name in shapes:
        if any(not math.isfinite(errors[name][m]) or errors[name][m] < 0 for m in modes):
            raise ValueError("sensitivity must be finite and nonnegative")
    result = Pi05BudgetProfiles((), {**metadata, "variant": variant, "layer_shapes": shapes,
        "activation_elements": activation_elements, "profiles_are_nested": True,
        "budget_scope": "each_profile_and_each_request_on_target_matrices",
        "weight_cost": "logical_parameters_once",
        "activation_cost": "logical_input_elements_at_all_invocations",
        "quantizer": "nested_q2_r2_r4_affine_w2_v1",
        "sensitivity_source": "BF16_training_input_local_output_relative_MSE_proxy",
        "allocator": "greedy_componentwise_upgrades_under_separate_W_A_caps"})
    assignments = {n: ("w2a8" if variant == "w4a8" else "w2a2") for n in shapes}
    total_w, total_a = sum(math.prod(s) for s in shapes.values()), sum(activation_elements.values())
    used_w, used_a = 2 * total_w, (8 if variant == "w4a8" else 2) * total_a
    profiles = []
    for profile_name, budget in PROFILE_BUDGETS:
        a_budget = 8.0 if variant == "w4a8" else budget
        while True:
            best = None
            for name in sorted(shapes):
                old = assignments[name]
                w0, a0 = pair(old)
                for mode in modes:
                    w, a = pair(mode)
                    dw, da = (w - w0) * math.prod(shapes[name]), (a - a0) * activation_elements[name]
                    if w < w0 or a < a0 or mode == old:
                        continue
                    gain = errors[name][old] - errors[name][mode]
                    if gain <= 0 or used_w + dw > budget * total_w or used_a + da > a_budget * total_a:
                        continue
                    score = gain / (dw / total_w + da / total_a)
                    if best is None or score > best[0]:
                        best = (score, name, mode, dw, da)
            if best is None:
                break
            _, name, mode, dw, da = best
            assignments[name] = mode
            used_w, used_a = used_w + dw, used_a + da
        profiles.append(BudgetProfile(profile_name, dict(assignments), budget, used_w / total_w,
            a_budget, used_a / total_a, sum(errors[n][m] for n, m in assignments.items())))
    result.profiles = tuple(profiles)
    return result.validate()
