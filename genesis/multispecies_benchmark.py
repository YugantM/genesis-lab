"""Paired multi-species injury benchmark for the Genesis parameter shift."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from .core import GenesisConfig, GenesisWorld
from .genome import Specimen, load_specimen
from .metrics import center_of_mass, mass, occupied_fraction, toroidal_displacement
from .robust_search import calibrated_damage


GENESIS_PARENT = (0.15, 0.015)
GENESIS_001 = (0.1463322937488556, 0.01689031347632408)
GENESIS_DELTA_MU = GENESIS_001[0] - GENESIS_PARENT[0]
GENESIS_SIGMA_SCALE = GENESIS_001[1] / GENESIS_PARENT[1]
LESIONS = ("core", "leading", "trailing")


def transferred_parameters(specimen: Specimen) -> tuple[float, float]:
    """Apply Genesis 001's relative parameter change to another species."""
    params = specimen.parameters
    return (
        float(params["growth_center"]) + GENESIS_DELTA_MU,
        float(params["growth_width"]) * GENESIS_SIGMA_SCALE,
    )


def lesion_anchor(
    state: np.ndarray,
    centre: np.ndarray,
    motion: np.ndarray,
    kind: str,
    *,
    activity_threshold: float = 0.1,
) -> np.ndarray:
    """Choose a mass-relative anatomical anchor for a standardized lesion."""
    if kind == "core":
        return np.asarray(centre, dtype=float)
    if kind not in {"leading", "trailing"}:
        raise ValueError(f"unsupported lesion kind: {kind}")
    cells = np.argwhere(state >= activity_threshold)
    if not len(cells):
        return np.asarray(centre, dtype=float)
    direction = np.asarray(motion, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 0.25:
        relative = toroidal_displacement(centre, cells, state.shape[0])
        weights = state[tuple(cells.T)]
        covariance = (relative * weights[:, None]).T @ relative / max(weights.sum(), 1e-12)
        _, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, -1]
        first_nonzero = np.flatnonzero(abs(direction) > 1e-9)
        if len(first_nonzero) and direction[first_nonzero[0]] < 0:
            direction = -direction
    else:
        direction /= norm
    relative = toroidal_displacement(centre, cells, state.shape[0])
    projection = relative @ direction
    index = int(np.argmax(projection) if kind == "leading" else np.argmin(projection))
    return cells[index].astype(float)


def make_world(states: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> GenesisWorld:
    world = GenesisWorld(
        GenesisConfig(size=states.shape[-1], batch=len(states), radius=13, dt=0.1)
    )
    world.state = mx.array(states)
    world.set_growth_parameters(centers, widths)
    mx.eval(world.state)
    return world


def centered_state(specimen: Specimen, size: int) -> np.ndarray:
    cells = specimen.cells
    if max(cells.shape) > size:
        raise ValueError(f"{specimen.name} does not fit a {size}×{size} world")
    state = np.zeros((size, size), np.float32)
    top, left = (size - cells.shape[0]) // 2, (size - cells.shape[1]) // 2
    state[top : top + cells.shape[0], left : left + cells.shape[1]] = cells
    return state


def load_manifest(path: Path) -> list[tuple[dict, Specimen]]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "genesis.species-manifest/v1":
        raise ValueError("unsupported species manifest")
    root = path.parent
    return [(entry, load_specimen(root / entry["path"])) for entry in manifest["entries"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="web/specimens/species-benchmark.json")
    parser.add_argument("--out", default="runs/multispecies-transfer.json")
    parser.add_argument("--web-summary", default="web/data/multispecies-transfer.json")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--post-steps", type=int, default=300)
    parser.add_argument("--damage", type=float, default=0.05)
    args = parser.parse_args()

    catalogue = load_manifest(Path(args.manifest))
    phases = (240, 300, 360)
    orientations = (0, 90, 180, 270)
    noise_seeds = (0, 1, 2)
    initial_states, labels, centers, widths = [], [], [], []
    for species_index, (entry, specimen) in enumerate(catalogue):
        original = centered_state(specimen, args.size)
        parameter_sets = {
            "parent": (
                float(specimen.parameters["growth_center"]),
                float(specimen.parameters["growth_width"]),
            ),
            "genesis-transfer": transferred_parameters(specimen),
        }
        for orientation in orientations:
            rotated = np.rot90(original, k=orientation // 90).copy()
            active = rotated > 0
            for noise_seed in noise_seeds:
                rng = np.random.default_rng(20260903 + species_index * 100 + orientation + noise_seed)
                perturbed = rotated.copy()
                perturbation = rng.normal(0, 0.002, active.sum())
                perturbed[active] = np.clip(perturbed[active] + perturbation, 0, 1)
                for genotype, (mu, sigma) in parameter_sets.items():
                    initial_states.append(perturbed.copy())
                    labels.append(
                        {
                            "species_code": entry["code"],
                            "species_name": entry["name"],
                            "specimen_id": entry["specimen_id"],
                            "genotype": genotype,
                            "orientation": orientation,
                            "noise_seed": noise_seed,
                        }
                    )
                    centers.append(mu)
                    widths.append(sigma)

    developing = make_world(
        np.stack(initial_states), np.asarray(centers, np.float32), np.asarray(widths, np.float32)
    )
    requested = set(phases) | {phase - 50 for phase in phases}
    snapshots: dict[int, np.ndarray] = {}
    for step in range(1, max(phases) + 1):
        developing.step()
        if step in requested:
            snapshots[step] = developing.numpy().copy()

    base_cases = []
    for index, label in enumerate(labels):
        for phase in phases:
            state = snapshots[phase][index]
            centre = center_of_mass(state[None])[0]
            earlier_centre = center_of_mass(snapshots[phase - 50][index : index + 1])[0]
            motion = toroidal_displacement(earlier_centre, centre, args.size)
            base_cases.append(
                {
                    **label,
                    "phase": phase,
                    "state": state,
                    "centre": centre,
                    "pre_mass": float(state.sum()),
                    "pre_occupied": float(occupied_fraction(state[None])[0]),
                    "pre_motion": float(np.linalg.norm(motion)),
                    "motion_vector": motion,
                    "mu": float(centers[index]),
                    "sigma": float(widths[index]),
                }
            )

    control_states = np.stack([case["state"] for case in base_cases])
    case_centers = np.asarray([case["mu"] for case in base_cases], np.float32)
    case_widths = np.asarray([case["sigma"] for case in base_cases], np.float32)
    controls = make_world(control_states, case_centers, case_widths)

    injured_states, trial_meta = [], []
    for base_index, case in enumerate(base_cases):
        for lesion in LESIONS:
            anchor = lesion_anchor(
                case["state"], case["centre"], case["motion_vector"], lesion
            )
            damaged, radius, actual = calibrated_damage(
                case["state"], anchor, args.damage
            )
            injured_states.append(damaged)
            trial_meta.append(
                {
                    "base_index": base_index,
                    "lesion": lesion,
                    "anchor_yx": anchor.tolist(),
                    "radius": radius,
                    "actual_removed_fraction": actual,
                }
            )
    injured = make_world(
        np.stack(injured_states),
        np.repeat(case_centers, len(LESIONS)),
        np.repeat(case_widths, len(LESIONS)),
    )

    controls.step(args.post_steps - 50)
    control_early_center = center_of_mass(controls.numpy())
    controls.step(50)
    injured.step(args.post_steps - 50)
    injured_early_center = center_of_mass(injured.numpy())
    injured.step(50)

    control_final = controls.numpy()
    injured_final = injured.numpy()
    control_mass = mass(control_final)
    control_occupied = occupied_fraction(control_final)
    control_center = center_of_mass(control_final)
    control_motion = np.linalg.norm(
        toroidal_displacement(control_early_center, control_center, args.size), axis=1
    )
    injured_mass = mass(injured_final)
    injured_occupied = occupied_fraction(injured_final)
    injured_center = center_of_mass(injured_final)
    injured_motion = np.linalg.norm(
        toroidal_displacement(injured_early_center, injured_center, args.size), axis=1
    )

    rows = []
    for trial_index, meta in enumerate(trial_meta):
        case = base_cases[meta["base_index"]]
        control_index = meta["base_index"]
        control_mass_ratio = float(control_mass[control_index] / max(case["pre_mass"], 1e-12))
        mass_ratio = float(injured_mass[trial_index] / max(control_mass[control_index], 1e-12))
        occupied_ratio = float(
            injured_occupied[trial_index] / max(control_occupied[control_index], 1e-12)
        )
        motion_ratio = float(
            injured_motion[trial_index] / max(control_motion[control_index], 1e-12)
        )
        control_valid = bool(
            control_mass[control_index] >= 10
            and 0.5 <= control_mass_ratio <= 2.0
            and control_occupied[control_index] <= 0.15
        )
        movement_recovered = bool(
            control_motion[control_index] < 1.0
            or injured_motion[trial_index] >= max(0.25, 0.25 * control_motion[control_index])
        )
        recovered = bool(
            control_valid
            and 0.75 <= mass_ratio <= 1.25
            and 0.5 <= occupied_ratio <= 2.0
            and injured_occupied[trial_index] <= 0.15
            and movement_recovered
        )
        rows.append(
            {
                **{key: value for key, value in case.items() if key not in {"state", "centre", "motion_vector"}},
                **{key: value for key, value in meta.items() if key != "base_index"},
                "control_valid": control_valid,
                "control_final_mass": float(control_mass[control_index]),
                "control_mass_ratio": control_mass_ratio,
                "injured_final_mass": float(injured_mass[trial_index]),
                "injured_to_control_mass_ratio": mass_ratio,
                "injured_to_control_occupied_ratio": occupied_ratio,
                "control_motion_50_steps": float(control_motion[control_index]),
                "injured_motion_50_steps": float(injured_motion[trial_index]),
                "injured_to_control_motion_ratio": motion_ratio,
                "functionally_recovered": recovered,
            }
        )

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["species_code"], row["genotype"], row["lesion"])].append(row)
    aggregate = []
    for entry, _ in catalogue:
        for genotype in ("parent", "genesis-transfer"):
            for lesion in LESIONS:
                group = grouped[(entry["code"], genotype, lesion)]
                recovered = sum(row["functionally_recovered"] for row in group)
                aggregate.append(
                    {
                        "species_code": entry["code"],
                        "species_name": entry["name"],
                        "genotype": genotype,
                        "lesion": lesion,
                        "trials": len(group),
                        "valid_controls": sum(row["control_valid"] for row in group),
                        "recovered": recovered,
                        "recovery_rate": recovered / len(group),
                    }
                )

    species_comparison = []
    for entry, _ in catalogue:
        code = entry["code"]
        parent_rows = [row for row in rows if row["species_code"] == code and row["genotype"] == "parent"]
        transfer_rows = [row for row in rows if row["species_code"] == code and row["genotype"] == "genesis-transfer"]
        parent_rate = np.mean([row["functionally_recovered"] for row in parent_rows])
        transfer_rate = np.mean([row["functionally_recovered"] for row in transfer_rows])
        parent_control_rate = np.mean([row["control_valid"] for row in parent_rows])
        transfer_control_rate = np.mean([row["control_valid"] for row in transfer_rows])
        delta = float(transfer_rate - parent_rate)
        if parent_control_rate < 0.9:
            classification = "excluded-parent-unstable"
        elif transfer_control_rate < 0.9:
            classification = "transfer-nonviable"
        else:
            classification = "beneficial" if delta >= 0.1 else "harmful" if delta <= -0.1 else "neutral"
        species_comparison.append(
            {
                "species_code": code,
                "species_name": entry["name"],
                "parent_recovery_rate": float(parent_rate),
                "genesis_transfer_recovery_rate": float(transfer_rate),
                "parent_control_valid_rate": float(parent_control_rate),
                "genesis_transfer_control_valid_rate": float(transfer_control_rate),
                "difference": delta,
                "classification": classification,
            }
        )
    beneficial = sum(item["classification"] == "beneficial" for item in species_comparison)
    harmful = sum(item["classification"] == "harmful" for item in species_comparison)
    nonviable = sum(item["classification"] == "transfer-nonviable" for item in species_comparison)
    excluded = sum(item["classification"] == "excluded-parent-unstable" for item in species_comparison)
    eligible_species = len(catalogue) - excluded
    neutral = eligible_species - beneficial - harmful - nonviable
    transferable = beneficial >= eligible_species / 2 and harmful + nonviable <= 1

    record = {
        "schema": "genesis.multispecies-transfer/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": "Does the relative Genesis 001 parameter shift improve injury recovery across distinct classic Lenia species?",
        "protocol": {
            "species_count": len(catalogue),
            "phases": list(phases),
            "orientations_degrees": list(orientations),
            "noise_seeds": list(noise_seeds),
            "initial_state_noise_sigma": 0.002,
            "lesions": list(LESIONS),
            "target_removed_fraction": args.damage,
            "post_steps": args.post_steps,
            "trials_per_species_genotype": len(phases) * len(orientations) * len(noise_seeds) * len(LESIONS),
            "transfer": {
                "growth_center_delta": GENESIS_DELTA_MU,
                "growth_width_scale": GENESIS_SIGMA_SCALE,
            },
            "functional_recovery": "paired-control mass and occupied-area retention; locomotion retention required only for moving controls",
            "transferability_gate": "beneficial by >=10 percentage points in at least half of species and harmful in at most one",
        },
        "summary": {
            "trial_count": len(rows),
            "beneficial_species": beneficial,
            "neutral_species": neutral,
            "harmful_species": harmful,
            "transfer_nonviable_species": nonviable,
            "excluded_parent_unstable_species": excluded,
            "transferable": bool(transferable),
        },
        "species_comparison": species_comparison,
        "aggregate": aggregate,
        "results": rows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    web_record = {
        key: record[key]
        for key in ("schema", "created_at", "question", "protocol", "summary", "species_comparison", "aggregate")
    }
    web_destination = Path(args.web_summary)
    web_destination.parent.mkdir(parents=True, exist_ok=True)
    web_destination.write_text(json.dumps(web_record, indent=2) + "\n")
    print(json.dumps({"summary": record["summary"], "species": species_comparison}, indent=2))


if __name__ == "__main__":
    main()
