"""Run and record the first calibrated Genesis damage experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

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


def snapshot(
    world: GenesisWorld,
    step: int,
    origin: np.ndarray,
    phase: str = "run",
    matched_state: np.ndarray | None = None,
) -> dict:
    state = world.numpy()
    centre = center_of_mass(state)[0]
    displacement = toroidal_displacement(origin, centre, world.config.size)
    current_mass = float(mass(state)[0])
    alive = current_mass >= 1.0
    result = {
        "step": step,
        "phase": phase,
        "mass": current_mass,
        "occupied_fraction": float(occupied_fraction(state)[0]),
        "center_y": float(centre[0]),
        "center_x": float(centre[1]),
        "displacement_y": float(displacement[0]) if alive else None,
        "displacement_x": float(displacement[1]) if alive else None,
        "displacement": float(np.linalg.norm(displacement)) if alive else None,
    }
    if matched_state is not None:
        result["aligned_similarity_to_control"] = aligned_similarity(
            state[0], matched_state
        )
        result["control_mass"] = float(matched_state.sum())
        result["control_occupied_fraction"] = float(occupied_fraction(matched_state))
        result["mass_ratio_to_control"] = result["mass"] / max(
            result["control_mass"], 1e-12
        )
    return result


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
    if args.pre_steps < 50 or args.post_steps < 50 or args.sample_every < 1:
        parser.error("pre-steps and post-steps must be at least 50; sample-every must be positive")

    specimen = load_specimen(args.specimen)
    damaged = GenesisWorld.from_specimen(specimen)
    control = GenesisWorld.from_specimen(specimen)
    origin = center_of_mass(damaged.numpy())[0]
    trajectory = [snapshot(damaged, 0, origin, matched_state=control.numpy()[0])]
    control_trajectory = [snapshot(control, 0, origin)]
    intervention = None
    pre_damage = None
    observed_centers = {0: origin}
    control_centers = {0: origin}
    total_steps = args.pre_steps + args.post_steps
    for step in range(1, total_steps + 1):
        damaged.step()
        control.step()
        control_state = control.numpy()[0]
        if step == args.pre_steps:
            pre_damage = snapshot(
                damaged, step, origin, "pre-damage", control_state
            )
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
            trajectory.append(
                snapshot(damaged, step, origin, "post-damage", control_state)
            )
        elif step % args.sample_every == 0 or step == total_steps:
            trajectory.append(snapshot(damaged, step, origin, matched_state=control_state))
        if step % args.sample_every == 0 or step in (args.pre_steps, total_steps):
            control_trajectory.append(snapshot(control, step, origin))
        observed_centers[step] = center_of_mass(damaged.numpy())[0]
        control_centers[step] = center_of_mass(control_state[None])[0]

    assert intervention is not None and pre_damage is not None
    final = trajectory[-1]
    control_final = control_trajectory[-1]
    post = [
        item
        for item in trajectory
        if item["step"] >= args.pre_steps and item["phase"] != "pre-damage"
    ]
    post_steps = [item["step"] - args.pre_steps for item in post]
    pre_center = np.asarray([pre_damage["center_y"], pre_damage["center_x"]])
    pre_motion = float(np.linalg.norm(toroidal_displacement(
        observed_centers[args.pre_steps - 50], pre_center, damaged.config.size
    )))
    pre_valid = bool(
        40 <= pre_damage["mass"] <= 130
        and pre_damage["occupied_fraction"] <= 0.03 and pre_motion >= 1
    )
    for item in post:
        step = item["step"]
        after = step - args.pre_steps
        control_motion = pre_motion
        item["motion_ratio_to_control"] = None
        if after >= 50:
            observed_motion = float(np.linalg.norm(toroidal_displacement(
                observed_centers[step - 50], observed_centers[step], damaged.config.size
            )))
            if item["mass"] < 1.0:
                observed_motion = 0.0
            control_motion = float(np.linalg.norm(toroidal_displacement(
                control_centers[step - 50], control_centers[step], damaged.config.size
            )))
            item["motion_50_steps"] = observed_motion
            item["control_motion_50_steps"] = control_motion
            item["motion_ratio_to_control"] = observed_motion / max(control_motion, 1e-12)
        item["control_valid"] = bool(
            pre_valid and 40 <= item["control_mass"] <= 130
            and item["control_occupied_fraction"] <= 0.03 and control_motion >= 1
        )
    final_valid = final["control_valid"]
    mass_recovery_step = recovery_time(
        post_steps,
        [item["mass"] if item["control_valid"] else np.nan for item in post],
        [item["control_mass"] for item in post],
        relative_tolerance=0.02,
        consecutive_samples=3,
    ) if final_valid else None
    shape_recovery_step = threshold_recovery_time(
        post_steps,
        [item["aligned_similarity_to_control"] if item["control_valid"] else np.nan for item in post],
        threshold=0.9,
        consecutive_samples=3,
    ) if final_valid else None
    functional_samples = [item for item in post if item["step"] - args.pre_steps >= 50]
    functional_time = functional_recovery_time(
        [item["step"] - args.pre_steps for item in functional_samples],
        [item["mass_ratio_to_control"] for item in functional_samples],
        [item["motion_ratio_to_control"] for item in functional_samples],
        [item["occupied_fraction"] for item in functional_samples],
        [item["control_valid"] for item in functional_samples],
    ) if final_valid else None
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
            "mass_recovery_tolerance": 0.02,
            "shape_recovery_similarity": 0.9,
            "recovery_sustain_samples": 3,
            "functional_recovery": "mass ratio 0.75–1.25, motion ratio >=0.25, occupancy <=0.03, valid matched control",
            "motion_window_steps": 50,
            "functional_time_first_eligible_step": 50,
            "recovery_time_null": "no sustained recovery during follow-up, or invalid control",
            "recovery_time_zero": "already inside the tolerance band immediately after injury",
        },
        "intervention": intervention,
        "summary": {
            "pre_damage_mass": pre_damage["mass"],
            "damaged_final_mass": final["mass"],
            "control_final_mass": control_final["mass"],
            "mass_recovery_ratio": final["mass"] / pre_damage["mass"],
            "control_mass_retention": control_final["mass"] / pre_damage["mass"],
            "final_mass_ratio_to_control": final["mass_ratio_to_control"],
            "final_aligned_similarity_to_control": final[
                "aligned_similarity_to_control"
            ],
            "mass_recovery_time_steps": mass_recovery_step,
            "shape_recovery_time_steps": shape_recovery_step,
            "control_valid": final_valid,
            "pre_motion_50_steps": pre_motion,
            "final_motion_ratio_to_control": final["motion_ratio_to_control"],
            "functionally_recovered": functional_recovery(
                control_valid=final_valid,
                mass_ratio=final["mass_ratio_to_control"],
                motion_ratio=final["motion_ratio_to_control"],
                occupied=final["occupied_fraction"],
            ),
            "functional_recovery_time_steps": functional_time,
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
