"""Run and record the first calibrated Genesis damage experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .core import GenesisWorld
from .genome import load_specimen
from .metrics import center_of_mass, mass, occupied_fraction, toroidal_displacement


def snapshot(world: GenesisWorld, step: int, origin: np.ndarray, phase: str = "run") -> dict:
    state = world.numpy()
    centre = center_of_mass(state)[0]
    displacement = toroidal_displacement(origin, centre, world.config.size)
    return {
        "step": step,
        "phase": phase,
        "mass": float(mass(state)[0]),
        "occupied_fraction": float(occupied_fraction(state)[0]),
        "center_y": float(centre[0]),
        "center_x": float(centre[1]),
        "displacement_y": float(displacement[0]),
        "displacement_x": float(displacement[1]),
        "displacement": float(np.linalg.norm(displacement)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", default="web/specimens/orbium-unicaudatus.json")
    parser.add_argument("--out", default="runs/specimen-zero.json")
    parser.add_argument("--pre-steps", type=int, default=300)
    parser.add_argument("--post-steps", type=int, default=300)
    parser.add_argument("--sample-every", type=int, default=10)
    # Calibrated on Orbium to remove approximately one quarter of its mass.
    parser.add_argument("--wound-radius", type=float, default=2.6)
    args = parser.parse_args()

    specimen = load_specimen(args.specimen)
    damaged = GenesisWorld.from_specimen(specimen)
    control = GenesisWorld.from_specimen(specimen)
    origin = center_of_mass(damaged.numpy())[0]
    trajectory = [snapshot(damaged, 0, origin)]
    control_trajectory = [snapshot(control, 0, origin)]
    intervention = None
    pre_damage = None
    total_steps = args.pre_steps + args.post_steps
    for step in range(1, total_steps + 1):
        damaged.step()
        control.step()
        if step == args.pre_steps:
            pre_damage = snapshot(damaged, step, origin, "pre-damage")
            centre = center_of_mass(damaged.numpy())[0]
            mass_before = float(mass(damaged.numpy())[0])
            damaged.damage_disk(tuple(centre), args.wound_radius)
            intervention = {
                "step": step,
                "type": "circular-excision",
                "center_y": float(centre[0]),
                "center_x": float(centre[1]),
                "radius": args.wound_radius,
                "mass_before": mass_before,
                "mass_after": float(mass(damaged.numpy())[0]),
            }
            trajectory.append(pre_damage)
            trajectory.append(snapshot(damaged, step, origin, "post-damage"))
        elif step % args.sample_every == 0:
            trajectory.append(snapshot(damaged, step, origin))
        if step % args.sample_every == 0:
            control_trajectory.append(snapshot(control, step, origin))

    assert intervention is not None and pre_damage is not None
    final = trajectory[-1]
    control_final = control_trajectory[-1]
    record = {
        "schema": "genesis.experiment/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specimen_id": specimen.id,
        "simulator": specimen.simulator,
        "world_size": damaged.config.size,
        "protocol": {
            "pre_steps": args.pre_steps,
            "post_steps": args.post_steps,
            "sample_every": args.sample_every,
        },
        "intervention": intervention,
        "summary": {
            "pre_damage_mass": pre_damage["mass"],
            "damaged_final_mass": final["mass"],
            "control_final_mass": control_final["mass"],
            "mass_recovery_ratio": final["mass"] / pre_damage["mass"],
            "control_mass_retention": control_final["mass"] / pre_damage["mass"],
            "damaged_final_displacement": final["displacement"],
            "control_final_displacement": control_final["displacement"],
        },
        "trajectory": trajectory,
        "control_trajectory": control_trajectory,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Recorded {destination}")
    print(json.dumps(record["summary"], indent=2))


if __name__ == "__main__":
    main()
