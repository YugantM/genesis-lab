"""Damage recovery in twenty independent, perturbed Orbium initial states.

Each seed receives one phase. Damage magnitude and wound location are repeated
conditions within that initial state; they never increase the independent n.
"""

from __future__ import annotations

import argparse
import csv
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
    functional_recovery,
    functional_recovery_time,
    mass,
    occupied_fraction,
    recovery_time,
    threshold_recovery_time,
    toroidal_displacement,
)
from .robust_search import calibrated_damage


PHASES = (180, 260, 340, 420, 500)
INITIAL_CONDITION_SEEDS = tuple(range(10001, 10021))
NOISE_SIGMA = 0.012
TARGETS = (0.05, 0.10, 0.15, 0.20, 0.25)
LOCATIONS = ("center", "leading", "trailing")


def initial_condition_plan() -> list[dict]:
    """Assign exactly one balanced phase to each independent seed."""
    return [
        {"initial_condition_seed": seed, "phase": PHASES[index % len(PHASES)]}
        for index, seed in enumerate(INITIAL_CONDITION_SEEDS)
    ]


def perturbed_initial_state(initial: np.ndarray, seed: int) -> np.ndarray:
    active = initial > 0
    state = initial.copy()
    state[active] = np.clip(
        state[active]
        + np.random.default_rng(seed).normal(0, NOISE_SIGMA, int(active.sum())),
        0,
        1,
    )
    return state


def batched_world(template: GenesisWorld, states: np.ndarray) -> GenesisWorld:
    config = type(template.config)(**{**template.config.__dict__, "batch": len(states)})
    world = GenesisWorld(config)
    world.state = mx.array(states)
    mx.eval(world.state)
    return world


def condition_summary(rows: list[dict]) -> dict:
    """Summarize one condition, refusing to count repeated seeds as independent."""
    seeds = {row["initial_condition_seed"] for row in rows}
    if not rows or len(seeds) != len(rows):
        raise ValueError("a condition must contain exactly one trial per independent seed")
    recovered = sum(row["functionally_recovered"] for row in rows)
    survivors = sum(row["survived"] for row in rows)
    valid = sum(row["control_valid"] for row in rows)
    recovery_times = [
        row["mass_recovery_time_steps"]
        for row in rows
        if row["mass_recovery_time_steps"] is not None
    ]
    extinct_times = [row["extinction_step"] for row in rows if row["extinction_step"] is not None]
    functional_times = [row["functional_recovery_time_steps"] for row in rows if row["functional_recovery_time_steps"] is not None]
    return {
        "independent_initial_conditions": len(seeds),
        "trials": len(rows),
        "valid_controls": valid,
        "survivors": survivors,
        "survival_rate": survivors / len(rows),
        "functionally_recovered": recovered,
        "functional_recovery_rate": recovered / len(rows),
        "functional_recovery_rate_valid_controls": recovered / valid if valid else None,
        "mean_actual_removed_fraction": float(np.mean([row["actual_removed_fraction"] for row in rows])),
        "mass_recovery_events": len(recovery_times),
        "median_mass_recovery_time_steps_among_events": float(np.median(recovery_times)) if recovery_times else None,
        "functional_recovery_events": len(functional_times),
        "median_functional_recovery_time_steps_among_events": float(np.median(functional_times)) if functional_times else None,
        "median_extinction_step": float(np.median(extinct_times)) if extinct_times else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", default="web/specimens/orbium-unicaudatus.json")
    parser.add_argument("--out", default="runs/replicated-damage-sweep.json")
    parser.add_argument("--csv", default="runs/replicated-damage-sweep.csv")
    parser.add_argument("--post-steps", type=int, default=300)
    parser.add_argument("--sample-every", type=int, default=10)
    args = parser.parse_args()
    if args.post_steps < 50 or args.sample_every <= 0:
        parser.error("post-steps must be at least 50 and sample-every must be positive")

    specimen = load_specimen(args.specimen)
    template = GenesisWorld.from_specimen(specimen)
    plan = initial_condition_plan()
    initial = template.numpy()[0]
    calibration = batched_world(
        template,
        np.stack([perturbed_initial_state(initial, item["initial_condition_seed"]) for item in plan]),
    )
    requested = set(PHASES) | {phase - 50 for phase in PHASES}
    snapshots = {}
    for step in range(1, max(PHASES) + 1):
        calibration.step()
        if step in requested:
            snapshots[step] = calibration.numpy().copy()

    base_cases = []
    for index, item in enumerate(plan):
        phase = item["phase"]
        state = snapshots[phase][index]
        prior = center_of_mass(snapshots[phase - 50][index : index + 1])[0]
        centre = center_of_mass(state[None])[0]
        velocity = toroidal_displacement(prior, centre, template.config.size)
        pre_motion = float(np.linalg.norm(velocity))
        direction = velocity / max(pre_motion, 1e-12)
        base_cases.append({
            **item,
            "state": state,
            "pre_mass": float(state.sum()),
            "pre_motion": pre_motion,
            "pre_occupied_fraction": float(occupied_fraction(state[None])[0]),
            "locations": {
                "center": centre,
                "leading": (centre + direction * 4.0) % template.config.size,
                "trailing": (centre - direction * 4.0) % template.config.size,
            },
        })

    conditions = []
    injured_states = []
    for base_index, base in enumerate(base_cases):
        for target in TARGETS:
            for location in LOCATIONS:
                injured, radius, actual = calibrated_damage(base["state"], base["locations"][location], target)
                injured_states.append(injured)
                conditions.append({
                    "base_index": base_index,
                    "initial_condition_seed": base["initial_condition_seed"],
                    "phase": base["phase"],
                    "target_removed_fraction": target,
                    "actual_removed_fraction": actual,
                    "location": location,
                    "radius": radius,
                    "pre_mass": base["pre_mass"],
                })

    injured_worlds = batched_world(template, np.stack(injured_states))
    controls = batched_world(template, np.stack([base["state"] for base in base_cases]))
    extinction = np.full(len(conditions), -1, dtype=np.int32)
    sample_steps, mass_samples, control_mass_samples, shape_samples = [], [], [], []
    planned_samples = sorted(set(range(0, args.post_steps + 1, args.sample_every)) | {args.post_steps})
    center_steps = set(planned_samples) | {step - 50 for step in planned_samples if step >= 50}
    center_history, control_center_history, occupied_samples = {}, {}, []
    injured_early = control_early = None
    for step in range(args.post_steps + 1):
        if step in center_steps:
            center_history[step] = center_of_mass(injured_worlds.numpy())
            control_center_history[step] = center_of_mass(controls.numpy())
        if step == args.post_steps - 50:
            injured_early = center_of_mass(injured_worlds.numpy())
            control_early = center_of_mass(controls.numpy())
        if step % args.sample_every == 0 or step == args.post_steps:
            injured_snapshot = injured_worlds.numpy()
            control_snapshot = controls.numpy()
            measured = mass(injured_snapshot)
            control_measured = mass(control_snapshot)
            sample_steps.append(step)
            mass_samples.append({"step": step, "mass": measured.tolist()})
            control_mass_samples.append({"step": step, "mass": control_measured.tolist()})
            occupied_samples.append(occupied_fraction(injured_snapshot))
            shape_samples.append({"step": step, "aligned_similarity": [
                aligned_similarity(injured_snapshot[index], control_snapshot[condition["base_index"]])
                for index, condition in enumerate(conditions)
            ]})
            newly_extinct = (extinction < 0) & (measured < 1.0)
            extinction[newly_extinct] = step
        if step < args.post_steps:
            injured_worlds.step()
            controls.step()

    injured_final = injured_worlds.numpy()
    control_final = controls.numpy()
    final_mass = mass(injured_final)
    control_final_mass = mass(control_final)
    final_occupied = occupied_fraction(injured_final)
    control_occupied = occupied_fraction(control_final)
    injured_motion = np.linalg.norm(toroidal_displacement(injured_early, center_of_mass(injured_final), template.config.size), axis=1)
    control_motion = np.linalg.norm(toroidal_displacement(control_early, center_of_mass(control_final), template.config.size), axis=1)
    control_rows = []
    for index, base in enumerate(base_cases):
        control_rows.append({
            **{key: base[key] for key in ("initial_condition_seed", "phase", "pre_mass", "pre_motion", "pre_occupied_fraction")},
            "final_mass": float(control_final_mass[index]),
            "final_occupied_fraction": float(control_occupied[index]),
            "final_motion": float(control_motion[index]),
            "control_valid": bool(
                40 <= base["pre_mass"] <= 130
                and base["pre_occupied_fraction"] <= 0.03
                and base["pre_motion"] >= 1
                and 40 <= control_final_mass[index] <= 130
                and control_occupied[index] <= 0.03
                and control_motion[index] >= 1
            ),
        })

    rows = []
    for index, condition in enumerate(conditions):
        base_index = condition["base_index"]
        valid = control_rows[base_index]["control_valid"]
        value = float(final_mass[index])
        mass_ratio = value / max(float(control_final_mass[base_index]), 1e-12)
        motion_ratio = float(injured_motion[index] / max(control_motion[base_index], 1e-12))
        similarity = shape_samples[-1]["aligned_similarity"][index]
        mass_time = recovery_time(
            sample_steps,
            [sample["mass"][index] for sample in mass_samples],
            [sample["mass"][base_index] for sample in control_mass_samples],
            relative_tolerance=0.02,
            consecutive_samples=3,
        ) if valid else None
        shape_time = threshold_recovery_time(
            sample_steps,
            [sample["aligned_similarity"][index] for sample in shape_samples],
            threshold=0.9,
            consecutive_samples=3,
        ) if valid else None
        functional_steps, functional_mass, functional_motion, functional_occupied, functional_valid = [], [], [], [], []
        for sample_index, sample_step in enumerate(sample_steps):
            if sample_step < 50:
                continue
            moved = float(np.linalg.norm(toroidal_displacement(
                center_history[sample_step - 50][index], center_history[sample_step][index], template.config.size)))
            control_moved = float(np.linalg.norm(toroidal_displacement(
                control_center_history[sample_step - 50][base_index], control_center_history[sample_step][base_index], template.config.size)))
            ratio = moved / max(control_moved, 1e-12)
            functional_steps.append(sample_step)
            functional_mass.append(mass_samples[sample_index]["mass"][index] / max(control_mass_samples[sample_index]["mass"][base_index], 1e-12))
            functional_motion.append(ratio)
            functional_occupied.append(float(occupied_samples[sample_index][index]))
            functional_valid.append(valid and control_moved >= 1)
        functional_time = functional_recovery_time(functional_steps, functional_mass, functional_motion, functional_occupied, functional_valid)
        rows.append({
            **{key: value_ for key, value_ in condition.items() if key != "base_index"},
            "control_valid": valid,
            "final_mass": value,
            "final_mass_ratio": value / max(condition["pre_mass"], 1e-12),
            "mass_ratio_to_control": mass_ratio,
            "motion_ratio_to_control": motion_ratio,
            "aligned_similarity_to_control": similarity,
            "final_occupied_fraction": float(final_occupied[index]),
            "mass_recovery_time_steps": mass_time,
            "shape_recovery_time_steps": shape_time,
            "functional_recovery_time_steps": functional_time,
            "functionally_recovered": functional_recovery(control_valid=valid, mass_ratio=mass_ratio, motion_ratio=motion_ratio, occupied=float(final_occupied[index])),
            "shape_recovered": bool(valid and similarity >= 0.9),
            "survived": value >= 1.0,
            "extinction_step": None if extinction[index] < 0 else int(extinction[index]),
        })

    groups = defaultdict(list)
    phase_groups = defaultdict(list)
    for row in rows:
        groups[(row["target_removed_fraction"], row["location"])].append(row)
        phase_groups[(row["phase"], row["target_removed_fraction"], row["location"])].append(row)
    aggregate = [{"target_removed_fraction": target, "location": location, **condition_summary(groups[(target, location)])}
                 for target in TARGETS for location in LOCATIONS]
    by_phase = [{"phase": phase, "target_removed_fraction": target, "location": location,
                 **condition_summary(phase_groups[(phase, target, location)])}
                for phase in PHASES for target in TARGETS for location in LOCATIONS]
    by_seed = []
    for item in plan:
        subset = [row for row in rows if row["initial_condition_seed"] == item["initial_condition_seed"]]
        by_seed.append({
            **item,
            "repeated_conditions": len(subset),
            "control_valid": subset[0]["control_valid"],
            "functional_recovery_rate_across_conditions": sum(row["functionally_recovered"] for row in subset) / len(subset),
            "survival_rate_across_conditions": sum(row["survived"] for row in subset) / len(subset),
        })
    retention = control_final_mass / np.maximum(np.asarray([base["pre_mass"] for base in base_cases]), 1e-12)
    record = {
        "schema": "genesis.replicated-damage-sweep/v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specimen_id": specimen.id,
        "simulator": specimen.simulator,
        "protocol": {
            "primary_independent_unit": "initial_condition_seed",
            "independent_initial_conditions": len(plan),
            "initial_conditions": plan,
            "initial_state_noise_sigma": NOISE_SIGMA,
            "noise_support": "cells active in the canonical starting specimen; clipped to [0,1]",
            "phases": list(PHASES),
            "phase_assignment": "one phase per seed, balanced with four independent seeds per phase",
            "rotation_used_as_replicate": False,
            "repeated_conditions_within_seed": len(TARGETS) * len(LOCATIONS),
            "target_removed_fractions": list(TARGETS),
            "damage_calibration": "exact mass removal using a disk with a fractional edge ring",
            "locations": list(LOCATIONS),
            "post_steps": args.post_steps,
            "sample_every": args.sample_every,
            "extinction_mass": 1.0,
            "location_offset_cells": 4.0,
            "trial_count": len(rows),
            "control_count": len(base_cases),
            "control_validity": "pre-injury and final mass 40–130, occupied fraction <=0.03, motion >=1 cell in a 50-step window",
            "functional_recovery": "valid matched control, final control-relative mass 0.75–1.25, motion >=0.25, occupied fraction <=0.03; aligned shape is a secondary outcome",
            "functional_recovery_time": "first of three consecutive samples satisfying the functional endpoint, using trailing 50-step displacement and matched-control displacement >=1; earliest measurable time is 50 steps",
            "recovery_mass_relative_tolerance": 0.02,
            "recovery_shape_threshold": 0.9,
            "recovery_consecutive_samples": 3,
            "recovery_time_null": "no sustained threshold crossing during observation, or invalid matched control",
            "inference_limit": "Conditions within a seed are repeated measures. Phase comparisons have n=4 per phase; all 300 injury trials do not form an independent n=300.",
        },
        "control": {
            "minimum_mass_retention": float(retention.min()),
            "mean_mass_retention": float(retention.mean()),
            "maximum_mass_retention": float(retention.max()),
            "valid_initial_conditions": sum(row["control_valid"] for row in control_rows),
            "per_initial_condition": control_rows,
        },
        "aggregate": aggregate,
        "per_phase_condition": by_phase,
        "per_initial_condition": by_seed,
        "results": rows,
        "mass_samples": mass_samples,
        "control_mass_samples": control_mass_samples,
        "shape_samples": shape_samples,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Recorded {len(plan)} independent initial conditions; {len(rows)} repeated injury trials")
    print(f"Valid controls: {record['control']['valid_initial_conditions']}/{len(plan)}")
    for item in aggregate:
        print(f"{item['target_removed_fraction']:>5.0%} {item['location']:<8} "
              f"{item['functionally_recovered']:>2}/{item['independent_initial_conditions']} recover; "
              f"{item['survivors']:>2}/{item['independent_initial_conditions']} survive")


if __name__ == "__main__":
    main()
