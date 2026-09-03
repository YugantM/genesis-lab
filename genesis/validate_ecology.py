"""Held-out validation of Genesis 001's ecological-persistence lead."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .coupled_ecology import run_experiment, web_summary


def exact_sign_test(genesis_only: int, parent_only: int) -> float:
    """Two-sided exact binomial test over discordant paired outcomes."""
    discordant = genesis_only + parent_only
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(genesis_only, parent_only) + 1)
    ) / 2**discordant
    return min(1.0, 2 * tail)


def paired_advantage(record: dict) -> dict:
    """Compare Genesis and its Orbium parent in identical opponent contexts."""
    lookup: dict[tuple, bool] = {}
    for row in record["results"]:
        codes = {row["left_code"], row["right_code"]}
        if row["outcome"] == "invalid-control":
            continue
        for focal in ("GEN001", "O2u"):
            if focal not in codes:
                continue
            opponent = next(code for code in codes if code != focal)
            if opponent in {"GEN001", "O2u"}:
                continue
            key = (
                focal,
                opponent,
                row["competition"],
                row["geometry"],
                row["orientation_degrees"],
                row["noise_seed"],
            )
            lookup[key] = row["outcome"] == "coexistence"

    paired: list[tuple[bool, bool]] = []
    for key, genesis_coexists in lookup.items():
        focal, opponent, *condition = key
        if focal != "GEN001":
            continue
        parent_key = ("O2u", opponent, *condition)
        if parent_key not in lookup:
            raise RuntimeError(f"missing matched parental trial: {parent_key}")
        paired.append((genesis_coexists, lookup[parent_key]))

    genesis_only = sum(genesis and not parent for genesis, parent in paired)
    parent_only = sum(parent and not genesis for genesis, parent in paired)
    both = sum(genesis and parent for genesis, parent in paired)
    neither = sum(not genesis and not parent for genesis, parent in paired)
    count = len(paired)
    genesis_rate = sum(genesis for genesis, _ in paired) / max(count, 1)
    parent_rate = sum(parent for _, parent in paired) / max(count, 1)
    p_value = exact_sign_test(genesis_only, parent_only)
    gate = bool(
        count > 0
        and genesis_rate - parent_rate >= 0.05
        and genesis_only >= 2 * max(parent_only, 1)
        and p_value <= 0.05
    )
    return {
        "matched_contexts": count,
        "genesis_coexistence_rate": genesis_rate,
        "parent_coexistence_rate": parent_rate,
        "difference": genesis_rate - parent_rate,
        "both_coexist": both,
        "genesis_only_coexists": genesis_only,
        "parent_only_coexists": parent_only,
        "neither_coexists": neither,
        "two_sided_exact_sign_p": p_value,
        "validation_gate": "difference >= 0.05, Genesis-only >= 2x parent-only, exact paired p <= 0.05",
        "validated": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="web/specimens/species-benchmark.json")
    parser.add_argument("--genesis", default="web/specimens/genesis-001.json")
    parser.add_argument("--out", default="runs/coupled-ecology-validation.json")
    parser.add_argument("--web-summary", default="web/data/coupled-ecology-validation.json")
    args = parser.parse_args()

    experiment = run_experiment(
        manifest=Path(args.manifest),
        genesis_path=Path(args.genesis),
        size=128,
        phase=300,
        post_steps=240,
        strengths=(0.25, 0.5, 1.0),
        orientations=(180, 270),
        noise_seeds=(2, 3),
        seed_base=20260905,
    )
    comparison = paired_advantage(experiment)
    record = {
        "schema": "genesis.coupled-ecology-validation/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "Genesis 001 ecological persistence advantage over Orbium parent",
        "separation_from_discovery": {
            "development_phase": 300,
            "orientations_degrees": [180, 270],
            "noise_seeds": [2, 3],
            "seed_base": 20260905,
        },
        "comparison": comparison,
        "experiment": experiment,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    summary = {
        key: record[key]
        for key in ("schema", "created_at", "candidate", "separation_from_discovery", "comparison")
    }
    summary["experiment"] = web_summary(experiment)
    web_destination = Path(args.web_summary)
    web_destination.parent.mkdir(parents=True, exist_ok=True)
    web_destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
