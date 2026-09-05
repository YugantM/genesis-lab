"""Select magnitude-robust mutants across independent noisy initial conditions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .core import GenesisWorld
from .genome import load_specimen
from .metrics import (
    aligned_similarity,
    center_of_mass,
    mass,
    occupied_fraction,
    toroidal_displacement,
    functional_recovery,
    functional_score,
)
from .robust_search import TRAIN_DAMAGE_LEVELS, calibrated_damage, make_world


PHASES = (180, 300, 420)
INITIAL_CONDITION_SEEDS = (101, 211, 307, 401)
NOISE_SIGMA = 0.012


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", default="web/specimens/orbium-unicaudatus.json")
    parser.add_argument("--search", default="runs/robust-search.json")
    parser.add_argument("--out", default="runs/robust-selection.json")
    parser.add_argument("--post-steps", type=int, default=300)
    parser.add_argument("--candidate-limit", type=int, default=100)
    args = parser.parse_args()

    specimen = load_specimen(args.specimen)
    search = json.loads(Path(args.search).read_text())
    candidates = search["top_candidates"][: args.candidate_limit]
    if not candidates:
        raise RuntimeError("robust search produced no viable candidates")
    initial = GenesisWorld.from_specimen(specimen).numpy()[0]
    active = initial > 0

    starting_states = []
    labels = []
    centers = []
    widths = []
    for candidate_index, candidate in enumerate(candidates):
        for seed in INITIAL_CONDITION_SEEDS:
            rng = np.random.default_rng(88000 + seed)
            perturbed = initial.copy()
            perturbed[active] = np.clip(
                perturbed[active] + rng.normal(0, NOISE_SIGMA, int(active.sum())),
                0,
                1,
            )
            starting_states.append(perturbed)
            labels.append((candidate_index, seed))
            centers.append(candidate["growth_center"])
            widths.append(candidate["growth_width"])

    template = GenesisWorld.from_specimen(specimen)
    calibration = make_world(
        template,
        np.stack(starting_states),
        np.asarray(centers, np.float32),
        np.asarray(widths, np.float32),
    )
    requested = set(PHASES) | {phase - 50 for phase in PHASES}
    snapshots = {}
    for step in range(1, max(PHASES) + 1):
        calibration.step()
        if step in requested:
            snapshots[step] = calibration.numpy().copy()

    base_cases = []
    for batch_index, (candidate_index, seed) in enumerate(labels):
        candidate = candidates[candidate_index]
        for phase in PHASES:
            state = snapshots[phase][batch_index]
            centre = center_of_mass(state[None])[0]
            early = center_of_mass(
                snapshots[phase - 50][batch_index : batch_index + 1]
            )[0]
            base_cases.append(
                {
                    "candidate_index": candidate_index,
                    "initial_condition_seed": seed,
                    "phase": phase,
                    "state": state,
                    "center": centre,
                    "pre_mass": float(state.sum()),
                    "pre_motion": float(
                        np.linalg.norm(
                            toroidal_displacement(early, centre, template.config.size)
                        )
                    ),
                    "growth_center": candidate["growth_center"],
                    "growth_width": candidate["growth_width"],
                }
            )

    base_centers = np.asarray(
        [base["growth_center"] for base in base_cases], np.float32
    )
    base_widths = np.asarray(
        [base["growth_width"] for base in base_cases], np.float32
    )
    controls = make_world(
        template,
        np.stack([base["state"] for base in base_cases]),
        base_centers,
        base_widths,
    )
    controls.step(args.post_steps - 50)
    control_early = center_of_mass(controls.numpy())
    controls.step(50)
    control_final = controls.numpy()
    control_mass = mass(control_final)
    control_occupied = occupied_fraction(control_final)
    control_motion = np.linalg.norm(
        toroidal_displacement(
            control_early, center_of_mass(control_final), template.config.size
        ),
        axis=1,
    )

    injured_states = []
    trial_base = []
    trial_damage = []
    for base_index, base in enumerate(base_cases):
        for damage in TRAIN_DAMAGE_LEVELS:
            injured, _, actual = calibrated_damage(
                base["state"], base["center"], damage
            )
            injured_states.append(injured)
            trial_base.append(base_index)
            trial_damage.append((damage, actual))
    trial_base_array = np.asarray(trial_base)
    injured = make_world(
        template,
        np.stack(injured_states),
        base_centers[trial_base_array],
        base_widths[trial_base_array],
    )
    injured.step(args.post_steps - 50)
    injured_early = center_of_mass(injured.numpy())
    injured.step(50)
    injured_final = injured.numpy()
    injured_mass = mass(injured_final)
    injured_occupied = occupied_fraction(injured_final)
    injured_motion = np.linalg.norm(
        toroidal_displacement(
            injured_early, center_of_mass(injured_final), template.config.size
        ),
        axis=1,
    )

    rows = []
    for index, base_index in enumerate(trial_base):
        base = base_cases[base_index]
        valid_control = bool(
            40 <= control_mass[base_index] <= 130
            and 40 <= base["pre_mass"] <= 130
            and base["pre_motion"] >= 1
            and control_occupied[base_index] <= 0.03
            and control_motion[base_index] >= 1
        )
        mass_ratio = float(injured_mass[index] / max(control_mass[base_index], 1e-12))
        motion_ratio = float(
            injured_motion[index] / max(control_motion[base_index], 1e-12)
        )
        similarity = aligned_similarity(injured_final[index], control_final[base_index])
        recovered = functional_recovery(
            control_valid=valid_control, mass_ratio=mass_ratio,
            motion_ratio=motion_ratio, occupied=float(injured_occupied[index]),
        )
        rows.append(
            {
                "candidate_index": base["candidate_index"],
                "candidate": candidates[base["candidate_index"]]["candidate"],
                "initial_condition_seed": base["initial_condition_seed"],
                "phase": base["phase"],
                "target_removed_fraction": trial_damage[index][0],
                "actual_removed_fraction": trial_damage[index][1],
                "control_valid": valid_control,
                "mass_ratio_to_control": mass_ratio,
                "motion_ratio_to_control": motion_ratio,
                "aligned_similarity_to_control": similarity,
                "functionally_recovered": recovered,
                "recovery_score": functional_score(mass_ratio, motion_ratio, valid_control and injured_occupied[index] <= 0.03),
            }
        )

    by_candidate: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_candidate[row["candidate_index"]].append(row)
    summaries = []
    for index, candidate in enumerate(candidates):
        group = by_candidate[index]
        seed_breakdown = []
        for seed in INITIAL_CONDITION_SEEDS:
            seed_rows = [row for row in group if row["initial_condition_seed"] == seed]
            seed_breakdown.append(
                {
                    "initial_condition_seed": seed,
                    "valid_trials": sum(row["control_valid"] for row in seed_rows),
                    "trials": len(seed_rows),
                    "recovered": sum(row["functionally_recovered"] for row in seed_rows),
                    "recovery_rate": sum(
                        row["functionally_recovered"] for row in seed_rows
                    )
                    / len(seed_rows),
                    "mean_recovery_score": float(
                        np.mean([row["recovery_score"] for row in seed_rows])
                    ),
                }
            )
        summaries.append(
            {
                **candidate,
                "independent_initial_conditions": len(INITIAL_CONDITION_SEEDS),
                "trials": len(group),
                "valid_trials": sum(row["control_valid"] for row in group),
                "recovered": sum(row["functionally_recovered"] for row in group),
                "mean_seed_recovery_rate": float(
                    np.mean([row["recovery_rate"] for row in seed_breakdown])
                ),
                "mean_seed_recovery_score": float(
                    np.mean([row["mean_recovery_score"] for row in seed_breakdown])
                ),
                "seed_breakdown": seed_breakdown,
                "damage_breakdown": [
                    {
                        "target_removed_fraction": damage,
                        "recovered": sum(
                            row["functionally_recovered"]
                            for row in group
                            if row["target_removed_fraction"] == damage
                        ),
                        "trials": sum(
                            row["target_removed_fraction"] == damage for row in group
                        ),
                    }
                    for damage in TRAIN_DAMAGE_LEVELS
                ],
            }
        )
    summaries.sort(
        key=lambda row: (
            row["valid_trials"] == row["trials"],
            row["mean_seed_recovery_score"],
            row["mean_seed_recovery_rate"],
            -row["parameter_distance"],
        ),
        reverse=True,
    )
    selected = summaries[0]
    record = {
        "schema": "genesis.robust-selection/v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_search": args.search,
        "random_seed": search["random_seed"],
        "protocol": {
            "candidate_count": len(candidates),
            "phases": list(PHASES),
            "initial_condition_seeds": list(INITIAL_CONDITION_SEEDS),
            "initial_state_noise_sigma": NOISE_SIGMA,
            "rotations_as_replicates": False,
            "training_damage_levels": list(TRAIN_DAMAGE_LEVELS),
            "post_steps": args.post_steps,
            "independent_unit": "initial_condition_seed",
            "trials_per_candidate": len(PHASES)
            * len(INITIAL_CONDITION_SEEDS)
            * len(TRAIN_DAMAGE_LEVELS),
        },
        "selected_candidate": selected,
        "top_candidates": summaries[:25],
        "selected_condition_results": [
            {key: value for key, value in row.items() if key != "candidate_index"}
            for row in rows
            if row["candidate_index"] == candidates.index(
                next(item for item in candidates if item["candidate"] == selected["candidate"])
            )
        ],
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
