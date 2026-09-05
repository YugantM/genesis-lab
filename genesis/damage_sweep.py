"""Automated Orbium damage sweep with calibrated injury locations."""

from __future__ import annotations

import argparse
import csv
import json
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


def disk_mask(size: int, center: np.ndarray, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    dy = np.minimum(abs(yy - center[0]), size - abs(yy - center[0]))
    dx = np.minimum(abs(xx - center[1]), size - abs(xx - center[1]))
    return dy * dy + dx * dx <= radius * radius


def calibrate_radius(
    state: np.ndarray, center: np.ndarray, target_fraction: float
) -> tuple[float, float]:
    """Choose the 0.05-cell radius closest to a requested mass removal."""
    total = float(state.sum())
    best = (float("inf"), 0.0, 0.0)
    for radius in np.arange(0.25, 12.01, 0.05):
        removed = float(state[disk_mask(state.shape[0], center, radius)].sum()) / total
        candidate = (abs(removed - target_fraction), float(radius), removed)
        if candidate < best:
            best = candidate
    return best[1], best[2]


def clone_at_state(template: GenesisWorld, state: np.ndarray) -> GenesisWorld:
    world = GenesisWorld(template.config)
    world.state = mx.array(state[None, :, :])
    mx.eval(world.state)
    return world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", default="web/specimens/orbium-unicaudatus.json")
    parser.add_argument("--out", default="runs/damage-sweep.json")
    parser.add_argument("--csv", default="runs/damage-sweep.csv")
    parser.add_argument("--settle-steps", type=int, default=300)
    parser.add_argument("--post-steps", type=int, default=300)
    parser.add_argument("--sample-every", type=int, default=10)
    args = parser.parse_args()
    if args.settle_steps < 50 or args.post_steps < 50 or args.sample_every < 1:
        parser.error("settle-steps and post-steps must be at least 50; sample-every must be positive")

    specimen = load_specimen(args.specimen)
    template = GenesisWorld.from_specimen(specimen)
    # Estimate motion direction immediately before intervention.
    template.step(args.settle_steps - 50)
    pre_motion_start = center_of_mass(template.numpy())[0]
    template.step(30)
    earlier_center = center_of_mass(template.numpy())[0]
    template.step(20)
    state = template.numpy()[0]
    centre = center_of_mass(state[None, :, :])[0]
    velocity = toroidal_displacement(earlier_center, centre, template.config.size)
    direction = velocity / max(float(np.linalg.norm(velocity)), 1e-12)
    locations = {
        "center": centre,
        "leading": (centre + direction * 4.0) % template.config.size,
        "trailing": (centre - direction * 4.0) % template.config.size,
    }

    conditions: list[dict] = []
    for target in (0.05, 0.10, 0.15, 0.20, 0.25):
        for location, wound_center in locations.items():
            _, radius, actual = calibrated_damage(state, wound_center, target)
            conditions.append(
                {
                    "target_removed_fraction": target,
                    "location": location,
                    "center": wound_center,
                    "radius": radius,
                    "actual_removed_fraction": actual,
                }
            )

    # One world per injury, evaluated in a single MLX batch.
    batch_config = type(template.config)(**{**template.config.__dict__, "batch": len(conditions)})
    worlds = GenesisWorld(batch_config)
    batch_state = np.repeat(state[None, :, :], len(conditions), axis=0)
    for index, condition in enumerate(conditions):
        batch_state[index], _, _ = calibrated_damage(
            state, condition["center"], condition["target_removed_fraction"]
        )
    worlds.state = mx.array(batch_state)
    mx.eval(worlds.state)

    control = clone_at_state(template, state)
    pre_mass = float(state.sum())
    pre_motion = float(np.linalg.norm(toroidal_displacement(
        pre_motion_start, centre, template.config.size
    )))
    pre_valid = bool(40 <= pre_mass <= 130 and occupied_fraction(state) <= 0.03 and pre_motion >= 1)
    extinction = [None] * len(conditions)
    trajectories = [[] for _ in conditions]
    control_trajectory = []
    observed_centers = {}
    control_centers = {}
    for step in range(0, args.post_steps + 1):
        measured_states = worlds.numpy()
        control_state = control.numpy()[0]
        observed_centers[step] = center_of_mass(measured_states)
        control_centers[step] = center_of_mass(control_state[None])[0]
        if step % args.sample_every == 0 or step == args.post_steps:
            masses = mass(measured_states)
            control_mass = float(mass(control_state[None])[0])
            control_motion = pre_motion
            motion_ratios = [None] * len(conditions)
            if step >= 50:
                observed_motion = np.linalg.norm(toroidal_displacement(
                    observed_centers[step - 50], observed_centers[step], template.config.size
                ), axis=1)
                observed_motion = np.where(masses >= 1.0, observed_motion, 0.0)
                control_motion = float(np.linalg.norm(toroidal_displacement(
                    control_centers[step - 50], control_centers[step], template.config.size
                )))
                motion_ratios = (observed_motion / max(control_motion, 1e-12)).tolist()
            valid_control = bool(
                pre_valid and 40 <= control_mass <= 130
                and occupied_fraction(control_state) <= 0.03 and control_motion >= 1
            )
            occupancies = occupied_fraction(measured_states)
            for index, value in enumerate(masses):
                value = float(value)
                trajectories[index].append(
                    {
                        "step": step,
                        "mass": value,
                        "control_mass": control_mass,
                        "mass_ratio_to_control": value / max(control_mass, 1e-12),
                        "occupied_fraction": float(occupancies[index]),
                        "motion_ratio_to_control": motion_ratios[index],
                        "control_valid": valid_control,
                        "aligned_similarity_to_control": aligned_similarity(
                            measured_states[index], control_state
                        ),
                    }
                )
                if extinction[index] is None and value < 1.0:
                    extinction[index] = step
            control_trajectory.append({
                "step": step, "mass": control_mass,
                "occupied_fraction": float(occupied_fraction(control_state)),
                "motion_50_steps": control_motion if step >= 50 else None,
                "control_valid": valid_control,
            })
        if step < args.post_steps:
            worlds.step()
            control.step()

    final_masses = mass(worlds.numpy())
    rows = []
    for index, condition in enumerate(conditions):
        final_mass = float(final_masses[index])
        final = trajectories[index][-1]
        final_valid = final["control_valid"]
        steps = [sample["step"] for sample in trajectories[index]]
        mass_recovery_step = recovery_time(
            steps,
            [sample["mass"] if sample["control_valid"] else np.nan for sample in trajectories[index]],
            [sample["control_mass"] for sample in trajectories[index]],
            relative_tolerance=0.02,
            consecutive_samples=3,
        ) if final_valid else None
        shape_recovery_step = threshold_recovery_time(
            steps,
            [sample["aligned_similarity_to_control"] if sample["control_valid"] else np.nan for sample in trajectories[index]],
            threshold=0.9,
            consecutive_samples=3,
        ) if final_valid else None
        functional_samples = [sample for sample in trajectories[index] if sample["step"] >= 50]
        functional_time = functional_recovery_time(
            [sample["step"] for sample in functional_samples],
            [sample["mass_ratio_to_control"] for sample in functional_samples],
            [sample["motion_ratio_to_control"] for sample in functional_samples],
            [sample["occupied_fraction"] for sample in functional_samples],
            [sample["control_valid"] for sample in functional_samples],
        ) if final_valid else None
        rows.append(
            {
                "target_removed_fraction": condition["target_removed_fraction"],
                "actual_removed_fraction": condition["actual_removed_fraction"],
                "location": condition["location"],
                "radius": condition["radius"],
                "initial_post_damage_mass": trajectories[index][0]["mass"],
                "final_mass": final_mass,
                "final_mass_ratio": final_mass / pre_mass,
                "final_mass_ratio_to_control": trajectories[index][-1][
                    "mass_ratio_to_control"
                ],
                "final_aligned_similarity_to_control": trajectories[index][-1][
                    "aligned_similarity_to_control"
                ],
                "mass_recovered": mass_recovery_step is not None,
                "mass_recovery_time_steps": mass_recovery_step,
                "shape_recovered": shape_recovery_step is not None,
                "shape_recovery_time_steps": shape_recovery_step,
                "control_valid": final_valid,
                "final_motion_ratio_to_control": final["motion_ratio_to_control"],
                "functionally_recovered": functional_recovery(
                    control_valid=final_valid,
                    mass_ratio=final["mass_ratio_to_control"],
                    motion_ratio=final["motion_ratio_to_control"],
                    occupied=final["occupied_fraction"],
                ),
                "functional_recovery_time_steps": functional_time,
                "survived": final_mass >= 1.0,
                "extinction_step": extinction[index],
            }
        )

    control_final = float(mass(control.numpy())[0])
    record = {
        "schema": "genesis.damage-sweep/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specimen_id": specimen.id,
        "simulator": specimen.simulator,
        "protocol": {
            "settle_steps": args.settle_steps,
            "post_steps": args.post_steps,
            "sample_every": args.sample_every,
            "extinction_mass": 1.0,
            "location_offset_cells": 4.0,
            "damage_calibration": "exact fractional edge ring",
            "mass_recovery_tolerance": 0.02,
            "shape_recovery_similarity": 0.9,
            "recovery_sustain_samples": 3,
            "functional_recovery": "mass ratio 0.75–1.25, motion ratio >=0.25, occupancy <=0.03, valid matched control",
            "motion_window_steps": 50,
            "functional_time_first_eligible_step": 50,
            "recovery_time_null": "no sustained recovery during follow-up, or invalid control",
            "recovery_time_zero": "already inside the tolerance band immediately after injury",
        },
        "baseline": {
            "pre_damage_mass": pre_mass,
            "pre_motion_50_steps": pre_motion,
            "control_valid": control_trajectory[-1]["control_valid"],
            "control_final_mass": control_final,
            "control_mass_retention": control_final / pre_mass,
            "motion_direction_yx": direction.tolist(),
        },
        "results": rows,
        "trajectories": trajectories,
        "control_trajectory": control_trajectory,
    }

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Recorded {destination} and {csv_path}")
    print(f"Control mass retention: {control_final / pre_mass:.3f}")
    for target in (0.05, 0.10, 0.15, 0.20, 0.25):
        subset = [row for row in rows if row["target_removed_fraction"] == target]
        survivors = sum(row["survived"] for row in subset)
        details = ", ".join(
            f"{row['location']}={'alive' if row['survived'] else 'dead'}"
            for row in subset
        )
        print(f"{target:>5.0%}: {survivors}/3 survive ({details})")


if __name__ == "__main__":
    main()
