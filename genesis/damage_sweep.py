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
from .metrics import center_of_mass, mass, toroidal_displacement


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

    specimen = load_specimen(args.specimen)
    template = GenesisWorld.from_specimen(specimen)
    # Estimate motion direction immediately before intervention.
    template.step(args.settle_steps - 20)
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
            radius, actual = calibrate_radius(state, wound_center, target)
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
        batch_state[index][disk_mask(template.config.size, condition["center"], condition["radius"])] = 0
    worlds.state = mx.array(batch_state)
    mx.eval(worlds.state)

    control = clone_at_state(template, state)
    pre_mass = float(state.sum())
    extinction = [None] * len(conditions)
    trajectories = [[] for _ in conditions]
    control_trajectory = []
    for step in range(0, args.post_steps + 1):
        if step % args.sample_every == 0:
            masses = mass(worlds.numpy())
            for index, value in enumerate(masses):
                value = float(value)
                trajectories[index].append({"step": step, "mass": value})
                if extinction[index] is None and value < 1.0:
                    extinction[index] = step
            control_trajectory.append({"step": step, "mass": float(mass(control.numpy())[0])})
        if step < args.post_steps:
            worlds.step()
            control.step()

    final_masses = mass(worlds.numpy())
    rows = []
    for index, condition in enumerate(conditions):
        final_mass = float(final_masses[index])
        rows.append(
            {
                "target_removed_fraction": condition["target_removed_fraction"],
                "actual_removed_fraction": condition["actual_removed_fraction"],
                "location": condition["location"],
                "radius": condition["radius"],
                "initial_post_damage_mass": trajectories[index][0]["mass"],
                "final_mass": final_mass,
                "final_mass_ratio": final_mass / pre_mass,
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
        },
        "baseline": {
            "pre_damage_mass": pre_mass,
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
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
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
