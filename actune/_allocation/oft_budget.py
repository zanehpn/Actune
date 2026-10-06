"""OFT whole-layer templates with separate logical weight/activation caps.

Reuse the existing two-budget greedy allocator and causal router. The new
artifact is separate from pi05 and from frozen OFT accuracy baselines.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path

from .pi05_budget import BudgetProfile, Pi05BudgetProfiles, make_budget_profiles

SCHEMA = "oft_joint_budget_v1"
QUANTIZER = "oft_existing_six_pairs_plus_rtn_a2_v1"
GROUPED_QUANTIZER = "oft_group_affine_packed_v1"


class OFTBudgetProfiles(Pi05BudgetProfiles):
    def validate(self):
        if self.metadata.get('profile_layout') == 'phase6_shared_bank_v1':
            super().validate(profile_names=['cruise','alignment','contact','complement'],require_nested=False)
            # Keep all original per-profile bit-count and nested-parent checks.
            Pi05BudgetProfiles(self.profiles[:3],self.metadata).validate()
            for name, mode in self.profile('complement').assignments.items():
                if mode not in {p.assignments[name] for p in self.profiles[:3]}:
                    raise ValueError('Complement allocation requires a new resident backend')
            if len({json.dumps(p.assignments,sort_keys=True) for p in self.profiles})!=4:
                raise ValueError('Shared-bank allocations must be distinct')
        else:
            super().validate()
        if self.metadata.get("model_family") != "oft" or self.metadata.get("quantizer") not in (QUANTIZER, GROUPED_QUANTIZER):
            raise ValueError("OFT artifact model/quantizer mismatch")
        if self.metadata.get("quantizer") == GROUPED_QUANTIZER:
            config = self.metadata.get("grouped_quantization", {})
            if config.get("group_size") not in (32, 64, 128) or not 0 < config.get("activation_clip", 0) <= 1:
                raise ValueError("invalid grouped OFT quantization configuration")
            if any(s[1] % config["group_size"] for s in self.metadata["layer_shapes"].values()):
                raise ValueError("grouped OFT quantization requires aligned logical widths")
        if self.metadata.get("chunk_size") != 8:
            raise ValueError("OFT budget artifact requires action chunk size eight")
        groups = self.metadata["activation_budget_groups"]
        if sorted(n for group in groups for n in group) != sorted(self.metadata["layer_shapes"]):
            raise ValueError("activation groups must cover every target exactly once")
        for p in self.profiles:
            if self.metadata.get("require_mixed_activations"):
                if self.metadata.get("variant") != "w4a4" or {int(m[3]) for m in p.assignments.values()} != {2, 4, 8}:
                    raise ValueError("mixed-activation OFT requires actual A2/A4/A8 in every template")
            for group in groups:
                elements = self.metadata["activation_elements"]
                bits = sum(elements[n] * int(p.assignments[n][3]) for n in group)
                if bits > p.target_activation_bits * sum(elements[n] for n in group):
                    raise ValueError("activation shape-group budget exceeded")
        return self

    def save(self, path):
        self.validate()
        Path(path).write_text(json.dumps({"schema": SCHEMA, "metadata": self.metadata,
            "profiles": [asdict(p) for p in self.profiles]}, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text())
        if raw.get("schema") != SCHEMA:
            raise ValueError("not an OFT budget artifact")
        return cls(tuple(BudgetProfile(**p) for p in raw["profiles"]), raw["metadata"]).validate()


def make_oft_budget_profiles(shapes, activation_elements, errors, metadata, *, variant):
    from .pi05_budget import PROFILE_BUDGETS, pair
    # The same greedy marginal-error allocator, with an additional A constraint
    # per shared row-count pattern. A longer language prompt cannot borrow A
    # budget from a fixed-size vision branch.
    allocated = make_budget_profiles(shapes, activation_elements, errors, metadata, variant=variant)
    groups = metadata["activation_budget_groups"]
    group_of = {n: i for i, group in enumerate(groups) for n in group}
    totals = [sum(activation_elements[n] for n in group) for group in groups]
    modes = [f"w{w}a{a}" for w in (2, 4, 8) for a in ((8,) if variant == "w4a8" else (2, 4, 8))]
    assignments = {n: modes[0] for n in shapes}
    total_w = sum(math.prod(s) for s in shapes.values())
    total_a = sum(activation_elements.values())
    used_w = 2 * total_w
    used_a = [(8 if variant == "w4a8" else 2) * t for t in totals]
    profiles = []
    for name, budget in PROFILE_BUDGETS:
        a_budget = 8 if variant == "w4a8" else budget
        while True:
            best = None
            for layer in sorted(shapes):
                w0, a0 = pair(assignments[layer])
                g = group_of[layer]
                for mode in modes:
                    w, a = pair(mode)
                    dw, da = (w-w0)*math.prod(shapes[layer]), (a-a0)*activation_elements[layer]
                    if w < w0 or a < a0 or mode == assignments[layer]:
                        continue
                    gain = errors[layer][assignments[layer]] - errors[layer][mode]
                    if gain <= 0 or used_w+dw > budget*total_w or used_a[g]+da > a_budget*totals[g]:
                        continue
                    score = gain / (dw/total_w + da/total_a)
                    if best is None or score > best[0]:
                        best = (score, layer, mode, dw, da)
            if best is None:
                break
            _, layer, mode, dw, da = best
            assignments[layer] = mode
            used_w += dw
            used_a[group_of[layer]] += da
        profiles.append(BudgetProfile(name, dict(assignments), budget, used_w/total_w,
            a_budget, sum(used_a)/total_a, sum(errors[n][m] for n,m in assignments.items())))
    allocated.profiles = tuple(profiles)
    allocated.metadata.update(model_family="oft", quantizer=QUANTIZER, chunk_size=8,
        budget_scope="each_template_and_actual_request_on_original_OFT_eligible_Linear_matrices",
        storage_scope="logical_selected_precision; resident alternative weight copies and scales reported separately",
        unquantized_scope="components outside target_components, ineligible/unexecuted linears, embeddings, norms and attention products remain unchanged")
    return OFTBudgetProfiles(allocated.profiles, allocated.metadata).validate()
