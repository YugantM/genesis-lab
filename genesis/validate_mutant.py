"""Held-out validation and promotion of the best robust-search candidate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from .core import GenesisWorld
from .genome import load_specimen
from .metrics import center_of_mass, mass, occupied_fraction, toroidal_displacement
from .robust_search import calibrated_damage


def make_batch(template: GenesisWorld, states: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> GenesisWorld:
    config = type(template.config)(**{**template.config.__dict__, "batch": len(states)})
    world = GenesisWorld(config)
    world.state = mx.array(states)
    world.set_growth_parameters(centers, widths)
    mx.eval(world.state)
    return world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", default="web/specimens/orbium-unicaudatus.json")
    parser.add_argument("--search", default="runs/robust-selection.json")
    parser.add_argument("--out", default="runs/genesis-001-validation.json")
    parser.add_argument("--promote", default="web/specimens/genesis-001.json")
    parser.add_argument("--post-steps", type=int, default=300)
    args = parser.parse_args()

    source_path = Path(args.specimen)
    source_record = json.loads(source_path.read_text())
    specimen = load_specimen(source_path)
    search = json.loads(Path(args.search).read_text())
    if not search.get("top_candidates"):
        raise RuntimeError("robust search produced no candidate")
    candidate = search.get("selected_candidate", search["top_candidates"][0])
    genotypes = {
        "canonical": (0.15, 0.015),
        "genesis-001": (candidate["growth_center"], candidate["growth_width"]),
    }
    phases = (220, 260, 340, 380, 420)  # excludes search phase 300
    orientations = (0, 90, 180, 270)
    noise_seeds = tuple(range(5))
    damage_levels = (0.05, 0.075, 0.10)

    initial = GenesisWorld.from_specimen(specimen).numpy()[0]
    starting_states, labels, centers, widths = [], [], [], []
    for genotype, (mu, sigma) in genotypes.items():
        for angle in orientations:
            rotated = np.rot90(initial, k=angle // 90).copy()
            active = rotated > 0
            for noise_seed in noise_seeds:
                rng = np.random.default_rng(91000 + noise_seed)
                perturbed = rotated.copy()
                perturbed[active] = np.clip(
                    perturbed[active] + rng.normal(0, 0.002, active.sum()), 0, 1
                )
                starting_states.append(perturbed)
                labels.append((genotype, angle, noise_seed))
                centers.append(mu)
                widths.append(sigma)
    calibration = make_batch(
        GenesisWorld.from_specimen(specimen),
        np.stack(starting_states),
        np.asarray(centers, np.float32),
        np.asarray(widths, np.float32),
    )
    requested = set(phases) | {phase - 50 for phase in phases}
    snapshots: dict[int, np.ndarray] = {}
    for step in range(1, max(phases) + 1):
        calibration.step()
        if step in requested:
            snapshots[step] = calibration.numpy().copy()

    base_cases = []
    for batch_index, (genotype, angle, noise_seed) in enumerate(labels):
        for phase in phases:
            state = snapshots[phase][batch_index]
            early = snapshots[phase - 50][batch_index : batch_index + 1]
            centre = center_of_mass(state[None, :, :])[0]
            movement = float(
                np.linalg.norm(
                    toroidal_displacement(
                        center_of_mass(early)[0], centre, calibration.config.size
                    )
                )
            )
            base_cases.append(
                {
                    "genotype": genotype,
                    "orientation": angle,
                    "noise_seed": noise_seed,
                    "phase": phase,
                    "state": state,
                    "center": centre,
                    "pre_mass": float(state.sum()),
                    "pre_motion": movement,
                    "mu": genotypes[genotype][0],
                    "sigma": genotypes[genotype][1],
                }
            )

    damaged_states, trial_meta, trial_mu, trial_sigma = [], [], [], []
    for base_index, base in enumerate(base_cases):
        for damage in damage_levels:
            injured, radius, actual = calibrated_damage(base["state"], base["center"], damage)
            damaged_states.append(injured)
            trial_mu.append(base["mu"])
            trial_sigma.append(base["sigma"])
            trial_meta.append(
                {
                    "base_index": base_index,
                    "genotype": base["genotype"],
                    "orientation": base["orientation"],
                    "noise_seed": base["noise_seed"],
                    "phase": base["phase"],
                    "target_removed_fraction": damage,
                    "actual_removed_fraction": actual,
                    "radius": radius,
                }
            )
    injured = make_batch(
        GenesisWorld.from_specimen(specimen),
        np.stack(damaged_states),
        np.asarray(trial_mu, np.float32),
        np.asarray(trial_sigma, np.float32),
    )
    controls = make_batch(
        GenesisWorld.from_specimen(specimen),
        np.stack([base["state"] for base in base_cases]),
        np.asarray([base["mu"] for base in base_cases], np.float32),
        np.asarray([base["sigma"] for base in base_cases], np.float32),
    )

    injured.step(args.post_steps - 50)
    post_early_center = center_of_mass(injured.numpy())
    injured.step(50)
    controls.step(args.post_steps)
    final_state = injured.numpy()
    final_mass = mass(final_state)
    final_occupied = occupied_fraction(final_state)
    final_center = center_of_mass(final_state)
    post_motion = np.linalg.norm(
        toroidal_displacement(post_early_center, final_center, injured.config.size), axis=1
    )
    control_final_mass = mass(controls.numpy())

    rows = []
    for index, meta in enumerate(trial_meta):
        base = base_cases[meta["base_index"]]
        mass_ratio = float(final_mass[index] / base["pre_mass"])
        motion_ratio = float(post_motion[index] / max(base["pre_motion"], 1e-12))
        functional = bool(
            40 <= final_mass[index] <= 130
            and final_occupied[index] <= 0.03
            and 0.75 <= mass_ratio <= 1.25
            and post_motion[index] >= 1.0
            and motion_ratio >= 0.25
        )
        rows.append(
            {
                **{key: value for key, value in meta.items() if key != "base_index"},
                "pre_mass": base["pre_mass"],
                "final_mass": float(final_mass[index]),
                "mass_ratio": mass_ratio,
                "pre_motion_50_steps": base["pre_motion"],
                "post_motion_50_steps": float(post_motion[index]),
                "motion_ratio": motion_ratio,
                "functionally_recovered": functional,
            }
        )

    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["genotype"], row["target_removed_fraction"])].append(row)
    aggregate = []
    for genotype in genotypes:
        for damage in damage_levels:
            group = grouped[(genotype, damage)]
            recovered = sum(row["functionally_recovered"] for row in group)
            aggregate.append(
                {
                    "genotype": genotype,
                    "target_removed_fraction": damage,
                    "trials": len(group),
                    "recovered": recovered,
                    "recovery_rate": recovered / len(group),
                    "median_mass_ratio": float(np.median([row["mass_ratio"] for row in group])),
                    "median_motion_ratio": float(np.median([row["motion_ratio"] for row in group])),
                }
            )

    control_ratios = control_final_mass / np.asarray([base["pre_mass"] for base in base_cases])
    mutant_5 = next(
        item for item in aggregate
        if item["genotype"] == "genesis-001" and item["target_removed_fraction"] == 0.05
    )
    canonical_5 = next(
        item for item in aggregate
        if item["genotype"] == "canonical" and item["target_removed_fraction"] == 0.05
    )
    accepted = bool(
        mutant_5["recovery_rate"] >= 0.9
        and canonical_5["recovery_rate"] <= 0.1
        and float(control_ratios.min()) >= 0.95
    )
    record = {
        "schema": "genesis.mutant-validation/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate,
        "protocol": {
            "held_out_phases": list(phases),
            "orientations_degrees": list(orientations),
            "initial_state_noise_seeds": list(noise_seeds),
            "initial_state_noise_sigma": 0.002,
            "damage_levels": list(damage_levels),
            "post_steps": args.post_steps,
            "trials_per_genotype_damage": len(phases) * len(orientations) * len(noise_seeds),
        },
        "control": {
            "minimum_mass_retention": float(control_ratios.min()),
            "mean_mass_retention": float(control_ratios.mean()),
            "maximum_mass_retention": float(control_ratios.max()),
        },
        "acceptance": {
            "accepted": accepted,
            "criterion": "mutant >=90% and canonical <=10% functional recovery at 5% centered damage; all controls >=95% mass retention",
        },
        "aggregate": aggregate,
        "results": rows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")

    if accepted:
        promoted = source_record.copy()
        promoted.update(
            {
                "id": "specimen-001-genesis",
                "name": "Genesis 001",
                "parent": specimen.id,
                "parameters": {
                    **source_record["parameters"],
                    "growth_center": candidate["growth_center"],
                    "growth_width": candidate["growth_width"],
                },
                "discovery": {
                    "method": "seeded Gaussian parameter search",
                    "search_seed": search["random_seed"],
                    "candidate_index": candidate["candidate"],
                    "validation_record": args.out,
                    "accepted_at": record["created_at"],
                },
            }
        )
        promoted_path = Path(args.promote)
        promoted_path.parent.mkdir(parents=True, exist_ok=True)
        promoted_path.write_text(json.dumps(promoted, indent=2) + "\n")
        print(f"Promoted candidate to {promoted_path}")
    print(json.dumps({"accepted": accepted, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
