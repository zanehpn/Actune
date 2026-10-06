"""Portable calibration entry points; all data paths are supplied by the caller."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .tree import fit_candidates, PrecisionTree, routing_cost
from .hardware import select_hardware_policy
from .allocation import build_banks


def main():
    parser = argparse.ArgumentParser(prog="actune")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-tree", help="Grow and prune using disjoint calibration trajectories")
    fit.add_argument("--fit", type=Path, required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--leaves", type=int, help="Explicit operating point chosen after comparing validation curves")
    bank = sub.add_parser("build-banks", help="Construct candidate banks from training sensitivities")
    bank.add_argument("--input", type=Path, required=True)
    bank.add_argument("--output", type=Path, required=True)
    hw = sub.add_parser("select-hardware", help="Search paired training frequency/power measurements")
    hw.add_argument("--input", type=Path, required=True)
    hw.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "fit-tree":
        with np.load(args.fit, allow_pickle=False) as f, np.load(args.validation, allow_pickle=False) as v:
            fitting, validation = dict(f), dict(v)
        curve = fit_candidates(fitting, validation)
        rows = validation["x"][validation["has_history"].astype(bool)]
        rows = rows[np.isfinite(rows).all(axis=1)]
        for c in curve:
            c["routing_us"] = routing_cost(c["tree"], rows) if len(rows) else None
        (args.output / "candidates.json").write_text(json.dumps(curve, indent=2, allow_nan=False) + "\n")
        if args.leaves is not None:
            choices = [c for c in curve if c["leaves"] == args.leaves]
            if not choices:
                parser.error("Requested leaf count is not feasible in the grown tree")
            chosen = choices[0]
            PrecisionTree(chosen["tree"], chosen["state_dim"]).save(args.output / "tree.json")
        print("Feasible leaf counts:", [c["leaves"] for c in curve])
    elif args.command == "build-banks":
        specification = json.loads(args.input.read_text())
        banks = build_banks(**specification)
        for key, value in banks.items():
            (args.output / f"{key}.json").write_text(json.dumps(asdict(value), indent=2, allow_nan=False) + "\n")
        print("Feasible banks:", list(banks))
    else:
        rows = json.loads(args.input.read_text())
        policy, diagnostics = select_hardware_policy(rows)
        policy.save(args.output / "candidate_policy.json")
        print(json.dumps(diagnostics))


if __name__ == "__main__":
    main()
