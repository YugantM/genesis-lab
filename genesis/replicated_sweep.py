"""Replicate the Orbium damage sweep across phase and orientation."""

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
from .damage_sweep import calibrate_radius, disk_mask
from .genome import load_specimen
from .metrics import center_of_mass, mass, toroidal_displacement


def batched_world(template: GenesisWorld, states: np.ndarray) -> GenesisWorld:
    config = type(template.config)(**{**template.config.__dict__, "batch": len(states)})
    world = GenesisWorld(config)
    world.state = mx.array(states)
    mx.eval(world.state)
    return world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", default="web/specimens/orbium-unicaudatus.json")
    parser.add_argument("--out", default="runs/replicated-damage-sweep.json")
    parser.add_argument("--csv", default="runs/replicated-damage-sweep.csv")
    parser.add_argument("--post-steps", type=int, default=300)
    parser.add_argument("--sample-every", type=int, default=10)
    args = parser.parse_args()

    specimen = load_specimen(args.specimen)
    template = GenesisWorld.from_specimen(specimen)
    phases = (260, 280, 300, 320, 340)
    orientations = (0, 90, 180, 270)
    targets = (0.05, 0.10, 0.15, 0.20, 0.25)
    location_names = ("center", "leading", "trailing")

    initial = template.numpy()[0]
    oriented_states = np.stack(
        [np.rot90(initial, k=angle // 90).copy() for angle in orientations]
    )
    calibration = batched_world(template, oriented_states)
    requested_steps = {phase - 20 for phase in phases} | set(phases)
    snapshots: dict[int, np.ndarray] = {}
    for step in range(1, max(phases) + 1):
        calibration.step()
        if step in requested_steps:
            snapshots[step] = calibration.numpy().copy()

    base_cases = []
    for orientation_index, orientation in enumerate(orientations):
        for phase in phases:
            state = snapshots[phase][orientation_index]
            prior_center = center_of_mass(snapshots[phase - 20][orientation_index : orientation_index + 1])[0]
            centre = center_of_mass(state[None, :, :])[0]
            velocity = toroidal_displacement(prior_center, centre, template.config.size)
            direction = velocity / max(float(np.linalg.norm(velocity)), 1e-12)
            locations = {
                "center": centre,
                "leading": (centre + direction * 4.0) % template.config.size,
                "trailing": (centre - direction * 4.0) % template.config.size,
            }
            base_cases.append(
                {
                    "orientation": orientation,
                    "phase": phase,
                    "state": state,
                    "locations": locations,
                    "pre_mass": float(state.sum()),
                }
            )

    conditions = []
    injured_states = []
    for base_index, base in enumerate(base_cases):
        for target in targets:
            for location in location_names:
                wound_center = base["locations"][location]
                radius, actual = calibrate_radius(base["state"], wound_center, target)
                injured = base["state"].copy()
                injured[disk_mask(template.config.size, wound_center, radius)] = 0.0
                injured_states.append(injured)
                conditions.append(
                    {
                        "base_index": base_index,
                        "orientation": base["orientation"],
                        "phase": base["phase"],
                        "target_removed_fraction": target,
                        "actual_removed_fraction": actual,
                        "location": location,
                        "radius": radius,
                        "pre_mass": base["pre_mass"],
                    }
                )

    injured_worlds = batched_world(template, np.stack(injured_states))
    controls = batched_world(template, np.stack([case["state"] for case in base_cases]))
    extinction = np.full(len(conditions), -1, dtype=np.int32)
    mass_samples = []
    control_mass_samples = []
    for step in range(args.post_steps + 1):
        if step % args.sample_every == 0:
            measured = mass(injured_worlds.numpy())
            control_measured = mass(controls.numpy())
            mass_samples.append({"step": step, "mass": measured.tolist()})
            control_mass_samples.append({"step": step, "mass": control_measured.tolist()})
            newly_extinct = (extinction < 0) & (measured < 1.0)
            extinction[newly_extinct] = step
        if step < args.post_steps:
            injured_worlds.step()
            controls.step()

    final_mass = mass(injured_worlds.numpy())
    control_final_mass = mass(controls.numpy())
    rows = []
    for index, condition in enumerate(conditions):
        value = float(final_mass[index])
        rows.append(
            {
                **{key: value_ for key, value_ in condition.items() if key != "base_index"},
                "final_mass": value,
                "final_mass_ratio": value / condition["pre_mass"],
                "survived": value >= 1.0,
                "extinction_step": None if extinction[index] < 0 else int(extinction[index]),
            }
        )

    groups: dict[tuple[float, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["target_removed_fraction"], row["location"])].append(row)
    aggregate = []
    for target in targets:
        for location in location_names:
            group = groups[(target, location)]
            survived = sum(row["survived"] for row in group)
            extinction_steps = [row["extinction_step"] for row in group if row["extinction_step"] is not None]
            aggregate.append(
                {
                    "target_removed_fraction": target,
                    "location": location,
                    "trials": len(group),
                    "survivors": survived,
                    "survival_rate": survived / len(group),
                    "mean_actual_removed_fraction": float(np.mean([row["actual_removed_fraction"] for row in group])),
                    "median_extinction_step": float(np.median(extinction_steps)) if extinction_steps else None,
                }
            )

    control_retention = control_final_mass / np.asarray([case["pre_mass"] for case in base_cases])
    record = {
        "schema": "genesis.replicated-damage-sweep/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specimen_id": specimen.id,
        "simulator": specimen.simulator,
        "protocol": {
            "phases": list(phases),
            "orientations_degrees": list(orientations),
            "target_removed_fractions": list(targets),
            "locations": list(location_names),
            "post_steps": args.post_steps,
            "sample_every": args.sample_every,
            "extinction_mass": 1.0,
            "location_offset_cells": 4.0,
            "trial_count": len(rows),
            "control_count": len(base_cases),
        },
        "control": {
            "minimum_mass_retention": float(control_retention.min()),
            "mean_mass_retention": float(control_retention.mean()),
            "maximum_mass_retention": float(control_retention.max()),
        },
        "aggregate": aggregate,
        "results": rows,
        "mass_samples": mass_samples,
        "control_mass_samples": control_mass_samples,
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

    print(f"Recorded {len(rows)} trials and {len(base_cases)} controls")
    print(f"Control retention: {control_retention.min():.3f}–{control_retention.max():.3f}")
    for item in aggregate:
        print(
            f"{item['target_removed_fraction']:>5.0%} {item['location']:<8} "
            f"{item['survivors']:>2}/{item['trials']} survive "
            f"({item['survival_rate']:.0%})"
        )


if __name__ == "__main__":
    main()
