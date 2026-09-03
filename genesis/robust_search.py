"""Search Lenia growth parameters for localized, mobile, damage-tolerant mutants."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from .core import GenesisWorld
from .genome import load_specimen
from .metrics import center_of_mass, mass, occupied_fraction, toroidal_displacement


def calibrated_damage(state: np.ndarray, center: np.ndarray, target: float) -> tuple[np.ndarray, float, float]:
    """Excise an exact mass fraction using a disk with a fractional edge ring."""
    if not 0.0 <= target <= 1.0:
        raise ValueError("target damage must be between zero and one")
    size = state.shape[0]
    yy, xx = np.indices(state.shape)
    dy = np.minimum(abs(yy - center[0]), size - abs(yy - center[0]))
    dx = np.minimum(abs(xx - center[1]), size - abs(xx - center[1]))
    distance = np.sqrt(dy * dy + dx * dx)
    target_mass = state.sum() * target
    # Quantized keys make geometrically equal rings disjoint. Using isclose for
    # every raw floating-point radius can visit nearly equal rings more than
    # once when the lesion centre is non-integral, causing over-removal.
    ring_keys = np.round(distance, decimals=6)
    radii = np.unique(ring_keys)
    damaged = state.copy()
    removed = 0.0
    radius = 0.0
    for ring_key in radii:
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
    actual = float((state.sum() - damaged.sum()) / state.sum())
    return damaged, float(radius), actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", default="web/specimens/orbium-unicaudatus.json")
    parser.add_argument("--out", default="runs/robust-search.json")
    parser.add_argument("--candidates", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--settle-steps", type=int, default=300)
    parser.add_argument("--post-steps", type=int, default=300)
    parser.add_argument("--target-damage", type=float, default=0.05)
    args = parser.parse_args()

    specimen = load_specimen(args.specimen)
    rng = np.random.default_rng(args.seed)
    centers = np.clip(rng.normal(0.15, 0.012, args.candidates), 0.11, 0.19).astype(np.float32)
    widths = np.clip(rng.normal(0.015, 0.004, args.candidates), 0.007, 0.030).astype(np.float32)
    centers[0], widths[0] = 0.15, 0.015  # canonical baseline

    world = GenesisWorld.from_specimen(specimen, batch=args.candidates)
    world.set_growth_parameters(centers, widths)
    world.step(args.settle_steps - 50)
    center_early = center_of_mass(world.numpy())
    world.step(50)
    pre_state = world.numpy()
    center_pre = center_of_mass(pre_state)
    pre_mass = mass(pre_state)
    pre_occupied = occupied_fraction(pre_state)
    pre_motion = np.linalg.norm(
        toroidal_displacement(center_early, center_pre, world.config.size), axis=1
    )
    viable_before = (
        (pre_mass >= 40.0)
        & (pre_mass <= 130.0)
        & (pre_occupied <= 0.03)
        & (pre_motion >= 1.0)
    )

    damaged_state = pre_state.copy()
    radii = np.zeros(args.candidates, np.float32)
    actual_damage = np.zeros(args.candidates, np.float32)
    for index in range(args.candidates):
        if pre_mass[index] <= 0:
            continue
        damaged_state[index], radii[index], actual_damage[index] = calibrated_damage(
            pre_state[index], center_pre[index], args.target_damage
        )
    world.state = mx.array(damaged_state)
    mx.eval(world.state)
    world.step(args.post_steps - 50)
    center_post_early = center_of_mass(world.numpy())
    world.step(50)
    final_state = world.numpy()
    final_center = center_of_mass(final_state)
    final_mass = mass(final_state)
    final_occupied = occupied_fraction(final_state)
    post_motion = np.linalg.norm(
        toroidal_displacement(center_post_early, final_center, world.config.size), axis=1
    )
    mass_ratio = np.divide(final_mass, pre_mass, out=np.zeros_like(final_mass), where=pre_mass > 0)
    motion_ratio = np.divide(post_motion, pre_motion, out=np.zeros_like(post_motion), where=pre_motion > 0)
    passed = (
        viable_before
        & (final_mass >= 40.0)
        & (final_mass <= 130.0)
        & (final_occupied <= 0.03)
        & (mass_ratio >= 0.75)
        & (mass_ratio <= 1.25)
        & (post_motion >= 1.0)
        & (motion_ratio >= 0.25)
    )
    # Favour recovery, retained motion, and small parameter departure.
    parameter_distance = np.sqrt(((centers - 0.15) / 0.012) ** 2 + ((widths - 0.015) / 0.004) ** 2)
    score = np.where(
        passed,
        2.0 - abs(mass_ratio - 1.0) + np.minimum(motion_ratio, 2.0) * 0.25 - parameter_distance * 0.02,
        -1e9,
    )
    order = np.argsort(score)[::-1]
    top = []
    for rank, index in enumerate(order[: min(300, args.candidates)], start=1):
        if not passed[index]:
            break
        top.append(
            {
                "rank": rank,
                "candidate": int(index),
                "growth_center": float(centers[index]),
                "growth_width": float(widths[index]),
                "wound_radius": float(radii[index]),
                "actual_removed_fraction": float(actual_damage[index]),
                "pre_mass": float(pre_mass[index]),
                "final_mass": float(final_mass[index]),
                "mass_ratio": float(mass_ratio[index]),
                "pre_motion_50_steps": float(pre_motion[index]),
                "post_motion_50_steps": float(post_motion[index]),
                "motion_ratio": float(motion_ratio[index]),
                "score": float(score[index]),
            }
        )
    record = {
        "schema": "genesis.robust-search/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specimen_id": specimen.id,
        "random_seed": args.seed,
        "protocol": {
            "candidate_count": args.candidates,
            "settle_steps": args.settle_steps,
            "post_steps": args.post_steps,
            "target_damage": args.target_damage,
            "selection": "localized, mass-stable, and mobile before and after centered injury",
        },
        "summary": {
            "viable_before_damage": int(viable_before.sum()),
            "passed_after_damage": int(passed.sum()),
            "canonical_passed": bool(passed[0]),
        },
        "top_candidates": top,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["summary"], indent=2))
    if top:
        print("Top candidate:")
        print(json.dumps(top[0], indent=2))
    else:
        print("No candidate passed; broaden the search distribution.")


if __name__ == "__main__":
    main()
