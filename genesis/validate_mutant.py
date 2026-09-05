"""Held-out validation for the magnitude-robust search candidate."""

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
from .metrics import (
    aligned_similarity,
    center_of_mass,
    mass,
    occupied_fraction,
    recovery_time,
    threshold_recovery_time,
    toroidal_displacement,
    functional_recovery,
    functional_recovery_time,
)
from .robust_search import calibrated_damage, make_world


HELD_OUT_PHASES = (220, 360, 500, 640)
HELD_OUT_SEEDS = (503, 607, 709, 811, 907, 1009, 1103, 1201)
HELD_OUT_DAMAGE = (0.05, 0.07, 0.10, 0.12)
NOISE_SIGMA = 0.015


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", default="web/specimens/orbium-unicaudatus.json")
    parser.add_argument("--search", default="runs/robust-selection.json")
    parser.add_argument("--out", default="runs/genesis-magnitude-validation.json")
    parser.add_argument("--post-steps", type=int, default=300)
    parser.add_argument("--sample-every", type=int, default=10)
    args = parser.parse_args()
    if args.post_steps < 50 or args.sample_every <= 0 or 50 % args.sample_every:
        parser.error("post-steps must be >=50; sample-every must be a positive divisor of50")

    specimen = load_specimen(args.specimen)
    selection = json.loads(Path(args.search).read_text())
    candidate = selection["selected_candidate"]
    genotypes = {
        "canonical": (0.15, 0.015),
        "magnitude-selected": (
            candidate["growth_center"],
            candidate["growth_width"],
        ),
    }
    initial = GenesisWorld.from_specimen(specimen).numpy()[0]
    active = initial > 0

    starting_states = []
    labels = []
    centers = []
    widths = []
    for genotype, (mu, sigma) in genotypes.items():
        for seed in HELD_OUT_SEEDS:
            rng = np.random.default_rng(99000 + seed)
            perturbed = initial.copy()
            perturbed[active] = np.clip(
                perturbed[active] + rng.normal(0, NOISE_SIGMA, int(active.sum())),
                0,
                1,
            )
            starting_states.append(perturbed)
            labels.append((genotype, seed))
            centers.append(mu)
            widths.append(sigma)

    template = GenesisWorld.from_specimen(specimen)
    calibration = make_world(
        template,
        np.stack(starting_states),
        np.asarray(centers, np.float32),
        np.asarray(widths, np.float32),
    )
    requested = set(HELD_OUT_PHASES) | {phase - 50 for phase in HELD_OUT_PHASES}
    snapshots = {}
    for step in range(1, max(HELD_OUT_PHASES) + 1):
        calibration.step()
        if step in requested:
            snapshots[step] = calibration.numpy().copy()

    base_cases = []
    for batch_index, (genotype, seed) in enumerate(labels):
        for phase in HELD_OUT_PHASES:
            state = snapshots[phase][batch_index]
            centre = center_of_mass(state[None])[0]
            prior = center_of_mass(
                snapshots[phase - 50][batch_index : batch_index + 1]
            )[0]
            base_cases.append(
                {
                    "genotype": genotype,
                    "initial_condition_seed": seed,
                    "phase": phase,
                    "state": state,
                    "center": centre,
                    "pre_mass": float(state.sum()),
                    "pre_motion": float(
                        np.linalg.norm(
                            toroidal_displacement(prior, centre, template.config.size)
                        )
                    ),
                    "growth_center": genotypes[genotype][0],
                    "growth_width": genotypes[genotype][1],
                }
            )
    base_centers = np.asarray(
        [base["growth_center"] for base in base_cases], np.float32
    )
    base_widths = np.asarray(
        [base["growth_width"] for base in base_cases], np.float32
    )

    injured_states = []
    trial_base = []
    trial_damage = []
    trial_actual = []
    for base_index, base in enumerate(base_cases):
        for damage in HELD_OUT_DAMAGE:
            injured, _, actual = calibrated_damage(
                base["state"], base["center"], damage
            )
            injured_states.append(injured)
            trial_base.append(base_index)
            trial_damage.append(damage)
            trial_actual.append(actual)
    trial_base_array = np.asarray(trial_base)
    injured = make_world(
        template,
        np.stack(injured_states),
        base_centers[trial_base_array],
        base_widths[trial_base_array],
    )
    controls = make_world(
        template,
        np.stack([base["state"] for base in base_cases]),
        base_centers,
        base_widths,
    )

    sample_steps = []
    mass_samples = [[] for _ in trial_base]
    control_mass_samples = [[] for _ in trial_base]
    shape_samples = [[] for _ in trial_base]
    occupied_samples = []
    injured_centers = []
    matched_centers = []
    injured_early = None
    control_early = None
    for step in range(args.post_steps + 1):
        if step == args.post_steps - 50:
            injured_early = center_of_mass(injured.numpy())
            control_early = center_of_mass(controls.numpy())
        if step % args.sample_every == 0 or step == args.post_steps:
            sample_steps.append(step)
            injured_snapshot = injured.numpy()
            control_snapshot = controls.numpy()
            injured_masses = mass(injured_snapshot)
            control_masses = mass(control_snapshot)
            occupied_samples.append(occupied_fraction(injured_snapshot))
            injured_centers.append(center_of_mass(injured_snapshot))
            matched_centers.append(center_of_mass(control_snapshot))
            for trial, base_index in enumerate(trial_base):
                mass_samples[trial].append(float(injured_masses[trial]))
                control_mass_samples[trial].append(float(control_masses[base_index]))
                shape_samples[trial].append(
                    aligned_similarity(
                        injured_snapshot[trial], control_snapshot[base_index]
                    )
                )
        if step < args.post_steps:
            injured.step()
            controls.step()

    assert injured_early is not None and control_early is not None
    injured_final = injured.numpy()
    control_final = controls.numpy()
    injured_mass = mass(injured_final)
    injured_occupied = occupied_fraction(injured_final)
    control_mass = mass(control_final)
    control_occupied = occupied_fraction(control_final)
    injured_motion = np.linalg.norm(
        toroidal_displacement(
            injured_early, center_of_mass(injured_final), template.config.size
        ),
        axis=1,
    )
    control_motion = np.linalg.norm(
        toroidal_displacement(
            control_early, center_of_mass(control_final), template.config.size
        ),
        axis=1,
    )

    rows = []
    for trial, base_index in enumerate(trial_base):
        base = base_cases[base_index]
        valid_control = bool(
            40 <= control_mass[base_index] <= 130
            and 40 <= base["pre_mass"] <= 130
            and base["pre_motion"] >= 1
            and control_occupied[base_index] <= 0.03
            and control_motion[base_index] >= 1
        )
        mass_ratio = float(injured_mass[trial] / max(control_mass[base_index], 1e-12))
        motion_ratio = float(
            injured_motion[trial] / max(control_motion[base_index], 1e-12)
        )
        similarity = shape_samples[trial][-1]
        mass_time = recovery_time(
            sample_steps,
            mass_samples[trial],
            control_mass_samples[trial],
            relative_tolerance=0.02,
            consecutive_samples=3,
        )
        shape_time = threshold_recovery_time(
            sample_steps,
            shape_samples[trial],
            threshold=0.9,
            consecutive_samples=3,
        )
        if not valid_control:
            mass_time = shape_time = None
        functional_steps, mass_ratios, motion_ratios, areas, validities = [], [], [], [], []
        step_index = {step: i for i, step in enumerate(sample_steps)}
        for i, step in enumerate(sample_steps):
            if step - 50 not in step_index:
                continue
            earlier = step_index[step - 50]
            injured_speed = float(np.linalg.norm(toroidal_displacement(
                injured_centers[earlier][trial], injured_centers[i][trial], template.config.size)))
            control_speed = float(np.linalg.norm(toroidal_displacement(
                matched_centers[earlier][base_index], matched_centers[i][base_index], template.config.size)))
            functional_steps.append(step)
            mass_ratios.append(mass_samples[trial][i] / max(control_mass_samples[trial][i], 1e-12))
            motion_ratios.append(injured_speed / max(control_speed, 1e-12))
            areas.append(occupied_samples[i][trial])
            validities.append(valid_control and 40 <= control_mass_samples[trial][i] <=130 and control_speed >=1)
        function_time = functional_recovery_time(functional_steps, mass_ratios, motion_ratios, areas, validities)
        recovered = functional_recovery(
            control_valid=valid_control, mass_ratio=mass_ratio,
            motion_ratio=motion_ratio, occupied=float(injured_occupied[trial]),
        )
        rows.append(
            {
                "genotype": base["genotype"],
                "initial_condition_seed": base["initial_condition_seed"],
                "phase": base["phase"],
                "target_removed_fraction": trial_damage[trial],
                "actual_removed_fraction": trial_actual[trial],
                "control_valid": valid_control,
                "mass_ratio_to_control": mass_ratio,
                "motion_ratio_to_control": motion_ratio,
                "aligned_similarity_to_control": similarity,
                "mass_recovery_time_steps": mass_time,
                "shape_recovery_time_steps": shape_time,
                "functional_recovery_time_steps": function_time,
                "pre_injury_mass": base["pre_mass"],
                "pre_injury_motion_50_steps": base["pre_motion"],
                "motion_ratio_to_pre_injury": float(injured_motion[trial] / max(base["pre_motion"], 1e-12)),
                "functionally_recovered": recovered,
            }
        )

    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["genotype"], row["target_removed_fraction"])].append(row)
    aggregate = []
    for genotype in genotypes:
        for damage in HELD_OUT_DAMAGE:
            group = grouped[(genotype, damage)]
            seed_rows = []
            for seed in HELD_OUT_SEEDS:
                subset = [row for row in group if row["initial_condition_seed"] == seed]
                seed_rows.append(
                    {
                        "initial_condition_seed": seed,
                        "recovery_rate_across_phases": sum(
                            row["functionally_recovered"] for row in subset
                        )
                        / len(subset),
                        "median_mass_recovery_time_steps": float(
                            np.median(
                                [
                                    row["mass_recovery_time_steps"]
                                    for row in subset
                                    if row["mass_recovery_time_steps"] is not None
                                ]
                            )
                        )
                        if any(row["mass_recovery_time_steps"] is not None for row in subset)
                        else None,
                    }
                )
            aggregate.append(
                {
                    "genotype": genotype,
                    "target_removed_fraction": damage,
                    "independent_initial_conditions": len(seed_rows),
                    "phase_trials_per_initial_condition": len(HELD_OUT_PHASES),
                    "valid_trials": sum(row["control_valid"] for row in group),
                    "trials": len(group),
                    "mean_seed_recovery_rate": float(
                        np.mean(
                            [row["recovery_rate_across_phases"] for row in seed_rows]
                        )
                    ),
                    "median_final_mass_ratio_to_control": float(
                        np.median([row["mass_ratio_to_control"] for row in group])
                    ),
                    "median_final_aligned_similarity_to_control": float(
                        np.median(
                            [row["aligned_similarity_to_control"] for row in group]
                        )
                    ),
                    "seed_breakdown": seed_rows,
                }
            )

    mutant = [row for row in aggregate if row["genotype"] == "magnitude-selected"]
    canonical = [row for row in aggregate if row["genotype"] == "canonical"]
    mutant_mean = float(np.mean([row["mean_seed_recovery_rate"] for row in mutant]))
    canonical_mean = float(
        np.mean([row["mean_seed_recovery_rate"] for row in canonical])
    )
    minimum_control_validity = min(row["valid_trials"] / row["trials"] for row in aggregate)
    accepted = bool(
        mutant_mean >= 0.65
        and mutant_mean - canonical_mean >= 0.25
        and minimum_control_validity >= 0.95
    )
    record = {
        "schema": "genesis.magnitude-validation/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate,
        "protocol": {
            "held_out_phases": list(HELD_OUT_PHASES),
            "held_out_initial_condition_seeds": list(HELD_OUT_SEEDS),
            "initial_state_noise_sigma": NOISE_SIGMA,
            "rotations_as_replicates": False,
            "held_out_damage_levels": list(HELD_OUT_DAMAGE),
            "post_steps": args.post_steps,
            "sample_every": args.sample_every,
            "mass_recovery_tolerance": 0.02,
            "shape_recovery_similarity": 0.9,
            "recovery_sustain_samples": 3,
            "independent_unit": "initial_condition_seed",
            "functional_endpoint": "valid controls; mass ratio0.75–1.25; motion ratio>=0.25; occupied<=0.03; shape secondary",
            "functional_time": "first3 consecutive completed rolling50-step motion windows meeting the functional endpoint",
            "heldout_scope": "excluded from this training objective; damage levels have historical exploratory exposure",
        },
        "acceptance": {
            "accepted": accepted,
            "criterion": "mean held-out recovery >=65%, >=25 points above canonical, and >=95% valid matched controls",
            "candidate_mean_seed_recovery_rate": mutant_mean,
            "canonical_mean_seed_recovery_rate": canonical_mean,
            "difference": mutant_mean - canonical_mean,
            "minimum_control_validity": minimum_control_validity,
        },
        "aggregate": aggregate,
        "results": rows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"acceptance": record["acceptance"], "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
