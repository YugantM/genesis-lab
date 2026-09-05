"""Search Lenia growth parameters across a distribution of damage magnitudes."""

from __future__ import annotations

import argparse
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
    mass,
    occupied_fraction,
    toroidal_displacement,
    functional_recovery,
    functional_score,
)


TRAIN_DAMAGE_LEVELS = (0.04, 0.06, 0.08)


def calibrated_damage(
    state: np.ndarray, center: np.ndarray, target: float
) -> tuple[np.ndarray, float, float]:
    """Excise an exact mass fraction using a disk with a fractional edge ring."""
    if not 0.0 <= target <= 1.0:
        raise ValueError("target damage must be between zero and one")
    total_mass = float(state.sum())
    if total_mass <= 0:
        return state.copy(), 0.0, 0.0
    size = state.shape[0]
    yy, xx = np.indices(state.shape)
    dy = np.minimum(abs(yy - center[0]), size - abs(yy - center[0]))
    dx = np.minimum(abs(xx - center[1]), size - abs(xx - center[1]))
    distance = np.sqrt(dy * dy + dx * dx)
    target_mass = total_mass * target
    ring_keys = np.round(distance, decimals=6)
    damaged = state.copy()
    removed = 0.0
    radius = 0.0
    for ring_key in np.unique(ring_keys):
        ring = ring_keys == ring_key
        ring_mass = float(state[ring].sum())
        if removed + ring_mass >= target_mass and ring_mass > 0:
            fraction = (target_mass - removed) / ring_mass
            damaged[ring] *= 1.0 - fraction
            removed = target_mass
            radius = float(distance[ring].max())
            break
        damaged[ring] = 0.0
        removed += ring_mass
        radius = float(distance[ring].max())
    actual = float((total_mass - damaged.sum()) / total_mass)
    return damaged, radius, actual


def make_world(
    template: GenesisWorld,
    states: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
) -> GenesisWorld:
    config = type(template.config)(**{**template.config.__dict__, "batch": len(states)})
    world = GenesisWorld(config)
    world.state = mx.array(states)
    world.set_growth_parameters(centers, widths)
    mx.eval(world.state)
    return world


def closeness(ratio: float) -> float:
    """Symmetric [0, 1] closeness to a ratio of one."""
    return min(ratio, 1.0 / max(ratio, 1e-12), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", default="web/specimens/orbium-unicaudatus.json")
    parser.add_argument("--out", default="runs/robust-search.json")
    parser.add_argument("--candidates", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--settle-steps", type=int, default=300)
    parser.add_argument("--post-steps", type=int, default=300)
    args = parser.parse_args()

    specimen = load_specimen(args.specimen)
    rng = np.random.default_rng(args.seed)
    centers = np.clip(
        rng.normal(0.15, 0.012, args.candidates), 0.11, 0.19
    ).astype(np.float32)
    widths = np.clip(
        rng.normal(0.015, 0.004, args.candidates), 0.007, 0.030
    ).astype(np.float32)
    centers[0], widths[0] = 0.15, 0.015

    template = GenesisWorld.from_specimen(specimen)
    initial = np.repeat(template.numpy(), args.candidates, axis=0)
    calibration = make_world(template, initial, centers, widths)
    calibration.step(args.settle_steps - 50)
    center_early = center_of_mass(calibration.numpy())
    calibration.step(50)
    pre_state = calibration.numpy()
    center_pre = center_of_mass(pre_state)
    pre_mass = mass(pre_state)
    pre_occupied = occupied_fraction(pre_state)
    pre_motion = np.linalg.norm(
        toroidal_displacement(center_early, center_pre, template.config.size), axis=1
    )
    viable_before = (
        (pre_mass >= 40)
        & (pre_mass <= 130)
        & (pre_occupied <= 0.03)
        & (pre_motion >= 1)
    )

    controls = make_world(template, pre_state.copy(), centers, widths)
    controls.step(args.post_steps - 50)
    control_early_center = center_of_mass(controls.numpy())
    controls.step(50)
    control_final = controls.numpy()
    control_mass = mass(control_final)
    control_occupied = occupied_fraction(control_final)
    control_motion = np.linalg.norm(
        toroidal_displacement(
            control_early_center,
            center_of_mass(control_final),
            template.config.size,
        ),
        axis=1,
    )
    control_valid = (
        viable_before
        & (control_mass >= 40)
        & (control_mass <= 130)
        & (control_occupied <= 0.03)
        & (control_motion >= 1)
    )

    injured_states = []
    trial_candidate = []
    trial_damage = []
    trial_radius = []
    trial_actual = []
    for candidate in range(args.candidates):
        for damage in TRAIN_DAMAGE_LEVELS:
            injured, radius, actual = calibrated_damage(
                pre_state[candidate], center_pre[candidate], damage
            )
            injured_states.append(injured)
            trial_candidate.append(candidate)
            trial_damage.append(damage)
            trial_radius.append(radius)
            trial_actual.append(actual)
    trial_candidate_array = np.asarray(trial_candidate)
    injured = make_world(
        template,
        np.stack(injured_states),
        centers[trial_candidate_array],
        widths[trial_candidate_array],
    )
    injured.step(args.post_steps - 50)
    injured_early_center = center_of_mass(injured.numpy())
    injured.step(50)
    injured_final = injured.numpy()
    injured_mass = mass(injured_final)
    injured_occupied = occupied_fraction(injured_final)
    injured_motion = np.linalg.norm(
        toroidal_displacement(
            injured_early_center,
            center_of_mass(injured_final),
            template.config.size,
        ),
        axis=1,
    )

    trials = []
    for index, candidate in enumerate(trial_candidate):
        mass_ratio = float(injured_mass[index] / max(control_mass[candidate], 1e-12))
        motion_ratio = float(
            injured_motion[index] / max(control_motion[candidate], 1e-12)
        )
        similarity = aligned_similarity(injured_final[index], control_final[candidate])
        recovered = functional_recovery(
            control_valid=control_valid[candidate], mass_ratio=mass_ratio,
            motion_ratio=motion_ratio, occupied=float(injured_occupied[index]),
        )
        recovery_score = functional_score(mass_ratio, motion_ratio, bool(control_valid[candidate] and injured_occupied[index] <= 0.03))
        trials.append(
            {
                "candidate": candidate,
                "target_removed_fraction": trial_damage[index],
                "actual_removed_fraction": trial_actual[index],
                "wound_radius": trial_radius[index],
                "mass_ratio_to_control": mass_ratio,
                "motion_ratio_to_control": motion_ratio,
                "aligned_similarity_to_control": similarity,
                "functionally_recovered": recovered,
                "recovery_score": recovery_score,
            }
        )

    parameter_distance = np.sqrt(
        ((centers - 0.15) / 0.012) ** 2 + ((widths - 0.015) / 0.004) ** 2
    )
    candidates = []
    for candidate in range(args.candidates):
        rows = [row for row in trials if row["candidate"] == candidate]
        candidates.append(
            {
                "candidate": candidate,
                "growth_center": float(centers[candidate]),
                "growth_width": float(widths[candidate]),
                "viable_control": bool(control_valid[candidate]),
                "pre_mass": float(pre_mass[candidate]),
                "control_final_mass": float(control_mass[candidate]),
                "pre_motion_50_steps": float(pre_motion[candidate]),
                "control_motion_50_steps": float(control_motion[candidate]),
                "mean_recovery_score": float(
                    np.mean([row["recovery_score"] for row in rows])
                ),
                "recovered_magnitudes": sum(
                    row["functionally_recovered"] for row in rows
                ),
                "training_magnitudes": len(rows),
                "passed_all_training_magnitudes": all(
                    row["functionally_recovered"] for row in rows
                ),
                "parameter_distance": float(parameter_distance[candidate]),
                "per_damage": rows,
            }
        )
    ranked = sorted(
        [row for row in candidates if row["viable_control"]],
        key=lambda row: (
            row["mean_recovery_score"],
            row["recovered_magnitudes"],
            -row["parameter_distance"],
        ),
        reverse=True,
    )
    top = ranked[: min(300, len(ranked))]
    record = {
        "schema": "genesis.robust-search/v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specimen_id": specimen.id,
        "random_seed": args.seed,
        "protocol": {
            "candidate_count": args.candidates,
            "settle_steps": args.settle_steps,
            "post_steps": args.post_steps,
            "training_damage_levels": list(TRAIN_DAMAGE_LEVELS),
            "matched_control": True,
            "selection": "mean control-relative recovery across 4%, 6%, and 8% centered mass excision",
            "recovery_score": "mass closeness × retained motion; bounded occupancy required; aligned shape is secondary",
        },
        "summary": {
            "viable_controls": int(control_valid.sum()),
            "passed_all_training_magnitudes": sum(
                row["passed_all_training_magnitudes"] for row in candidates
            ),
            "canonical": candidates[0],
            "best": top[0] if top else None,
        },
        "top_candidates": top,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["summary"], indent=2))


if __name__ == "__main__":
    main()
