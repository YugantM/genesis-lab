"""Select robust mutants across multiple training phases and perturbations."""

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
from .robust_search import calibrated_damage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", default="web/specimens/orbium-unicaudatus.json")
    parser.add_argument("--search", default="runs/robust-search.json")
    parser.add_argument("--out", default="runs/robust-selection.json")
    parser.add_argument("--post-steps", type=int, default=300)
    args = parser.parse_args()

    specimen = load_specimen(args.specimen)
    search = json.loads(Path(args.search).read_text())
    candidates = search["top_candidates"]
    phases = (260, 300, 340)
    orientations = (0, 90)
    noise_seeds = (31, 47)
    base_initial = GenesisWorld.from_specimen(specimen).numpy()[0]

    states, mus, sigmas, labels = [], [], [], []
    for candidate_index, candidate in enumerate(candidates):
        for angle in orientations:
            rotated = np.rot90(base_initial, angle // 90).copy()
            active = rotated > 0
            for noise_seed in noise_seeds:
                rng = np.random.default_rng(noise_seed)
                perturbed = rotated.copy()
                perturbed[active] = np.clip(
                    perturbed[active] + rng.normal(0, 0.002, active.sum()), 0, 1
                )
                states.append(perturbed)
                mus.append(candidate["growth_center"])
                sigmas.append(candidate["growth_width"])
                labels.append((candidate_index, angle, noise_seed))

    template = GenesisWorld.from_specimen(specimen)
    config = type(template.config)(**{**template.config.__dict__, "batch": len(states)})
    calibration = GenesisWorld(config)
    calibration.state = mx.array(np.stack(states))
    calibration.set_growth_parameters(mus, sigmas)
    mx.eval(calibration.state)
    requested = set(phases) | {phase - 50 for phase in phases}
    snapshots = {}
    for step in range(1, max(phases) + 1):
        calibration.step()
        if step in requested:
            snapshots[step] = calibration.numpy().copy()

    injured_states, trial_candidate, trial_pre_mass, trial_pre_motion = [], [], [], []
    trial_mu, trial_sigma = [], []
    for batch_index, (candidate_index, _, _) in enumerate(labels):
        candidate = candidates[candidate_index]
        for phase in phases:
            state = snapshots[phase][batch_index]
            centre = center_of_mass(state[None, :, :])[0]
            early_centre = center_of_mass(snapshots[phase - 50][batch_index : batch_index + 1])[0]
            pre_motion = float(np.linalg.norm(toroidal_displacement(early_centre, centre, config.size)))
            damaged, _, _ = calibrated_damage(state, centre, 0.05)
            injured_states.append(damaged)
            trial_candidate.append(candidate_index)
            trial_pre_mass.append(float(state.sum()))
            trial_pre_motion.append(pre_motion)
            trial_mu.append(candidate["growth_center"])
            trial_sigma.append(candidate["growth_width"])

    injured_config = type(template.config)(
        **{**template.config.__dict__, "batch": len(injured_states)}
    )
    worlds = GenesisWorld(injured_config)
    worlds.state = mx.array(np.stack(injured_states))
    worlds.set_growth_parameters(trial_mu, trial_sigma)
    mx.eval(worlds.state)
    worlds.step(args.post_steps - 50)
    post_early = center_of_mass(worlds.numpy())
    worlds.step(50)
    final = worlds.numpy()
    final_mass = mass(final)
    final_occupied = occupied_fraction(final)
    post_motion = np.linalg.norm(
        toroidal_displacement(post_early, center_of_mass(final), worlds.config.size), axis=1
    )
    pre_mass_array = np.asarray(trial_pre_mass)
    pre_motion_array = np.asarray(trial_pre_motion)
    mass_ratio = np.divide(final_mass, pre_mass_array, out=np.zeros_like(final_mass), where=pre_mass_array > 0)
    motion_ratio = np.divide(post_motion, pre_motion_array, out=np.zeros_like(post_motion), where=pre_motion_array > 0)
    recovered = (
        (pre_mass_array >= 40)
        & (pre_mass_array <= 130)
        & (pre_motion_array >= 1)
        & (final_mass >= 40)
        & (final_mass <= 130)
        & (final_occupied <= 0.03)
        & (mass_ratio >= 0.75)
        & (mass_ratio <= 1.25)
        & (post_motion >= 1)
        & (motion_ratio >= 0.25)
    )

    summaries = []
    trial_candidate_array = np.asarray(trial_candidate)
    for index, candidate in enumerate(candidates):
        selector = trial_candidate_array == index
        passed = int(recovered[selector].sum())
        total = int(selector.sum())
        summaries.append(
            {
                **candidate,
                "multi_condition_passed": passed,
                "multi_condition_trials": total,
                "multi_condition_recovery_rate": passed / total,
                "median_mass_ratio": float(np.median(mass_ratio[selector])),
                "median_motion_ratio": float(np.median(motion_ratio[selector])),
            }
        )
    summaries.sort(
        key=lambda row: (
            row["multi_condition_recovery_rate"],
            -abs(row["median_mass_ratio"] - 1),
            -abs(row["median_motion_ratio"] - 1),
            row["score"],
        ),
        reverse=True,
    )
    selected = summaries[0]
    record = {
        "schema": "genesis.robust-selection/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_search": args.search,
        "random_seed": search["random_seed"],
        "protocol": {
            "candidate_count": len(candidates),
            "phases": list(phases),
            "orientations": list(orientations),
            "noise_seeds": list(noise_seeds),
            "trials_per_candidate": len(phases) * len(orientations) * len(noise_seeds),
            "target_damage": 0.05,
        },
        "selected_candidate": selected,
        "top_candidates": summaries[:25],
    }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(f"Evaluated {len(injured_states)} perturbed damage trials")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
