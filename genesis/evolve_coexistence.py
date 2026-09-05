"""Search paired Lenia growth rules for robust coexistence.

The search target is fixed to the reproducibly difficult O4i × P4cl pairing.
Candidates may alter each organism's Gaussian growth centre and width, but must
remain viable and morphologically recognizable in isolation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from .core import GenesisConfig, GenesisWorld
from .coupled_ecology import (
    CoupledConfig,
    CoupledWorld,
    channel_metrics,
    shift_center,
)
from .genome import Specimen, load_specimen
from .metrics import aligned_similarity
from .multispecies_benchmark import centered_state


TARGETS = (
    ("O4i", Path("web/specimens/catalog/synorbium-ignis.json")),
    ("P4cl", Path("web/specimens/catalog/paraptera-cavus-labens.json")),
)
TRAIN_CONTEXTS = (
    ("contact", 0, 10),
    ("contact", 90, 11),
    ("overlap", 0, 10),
    ("overlap", 90, 11),
)
HELD_OUT_CONTEXTS = tuple(
    (geometry, orientation, noise_seed)
    for geometry in ("contact", "overlap")
    for orientation in (180, 270)
    for noise_seed in (20, 21)
)


def candidate_parameters(
    specimens: tuple[Specimen, Specimen], count: int, seed: int
) -> np.ndarray:
    """Generate deterministic paired mutations, with parents at index zero."""
    rng = np.random.default_rng(seed)
    parent = np.asarray(
        [
            [specimen.parameters["growth_center"], specimen.parameters["growth_width"]]
            for specimen in specimens
        ],
        np.float32,
    )
    result = np.repeat(parent[None], count, axis=0)
    result[:, 0, 0] = np.clip(rng.normal(parent[0, 0], 0.012, count), 0.12, 0.19)
    result[:, 0, 1] = np.clip(
        parent[0, 1] * np.exp(rng.normal(0, 0.28, count)), 0.008, 0.032
    )
    result[:, 1, 0] = np.clip(rng.normal(parent[1, 0], 0.028, count), 0.25, 0.41)
    result[:, 1, 1] = np.clip(
        parent[1, 1] * np.exp(rng.normal(0, 0.25, count)), 0.024, 0.085
    )
    result[0] = parent
    return result


def develop(
    specimens: tuple[Specimen, Specimen], parameters: np.ndarray, phase: int, size: int
) -> np.ndarray:
    count = len(parameters)
    base = np.stack([centered_state(specimen, size) for specimen in specimens])
    states = np.repeat(base[None], count, axis=0).reshape(count * 2, size, size)
    world = GenesisWorld(GenesisConfig(size=size, batch=count * 2, radius=13, dt=0.1))
    world.state = mx.array(states)
    world.set_growth_parameters(
        parameters[:, :, 0].reshape(-1), parameters[:, :, 1].reshape(-1)
    )
    world.step(phase)
    return world.numpy().reshape(count, 2, size, size)


def solo_metrics(developed: np.ndarray) -> list[dict]:
    references = developed[0]
    reference_mass, reference_occupied = channel_metrics(references[None])
    masses, occupied = channel_metrics(developed)
    rows = []
    for candidate in range(len(developed)):
        channels = []
        for channel in range(2):
            mass_ratio = float(masses[candidate, channel] / reference_mass[0, channel])
            occupied_ratio = float(
                occupied[candidate, channel] / max(reference_occupied[0, channel], 1e-12)
            )
            similarity = aligned_similarity(developed[candidate, channel], references[channel])
            viable = bool(
                masses[candidate, channel] >= 10
                and occupied[candidate, channel] <= 0.15
                and 0.75 <= mass_ratio <= 1.25
                and 0.75 <= occupied_ratio <= 1.25
                and similarity >= 0.5
            )
            channels.append(
                {
                    "mass": float(masses[candidate, channel]),
                    "mass_ratio_to_parent": mass_ratio,
                    "occupied_ratio_to_parent": occupied_ratio,
                    "parent_similarity": similarity,
                    "viable": viable,
                }
            )
        rows.append({"channels": channels, "both_viable": all(c["viable"] for c in channels)})
    return rows


def encounter_states(
    developed: np.ndarray,
    context: tuple[str, int, int],
    *,
    size: int,
    seed_base: int,
) -> np.ndarray:
    geometry, orientation, noise_seed = context
    separation = 24 if geometry == "contact" else 12
    targets = ((size / 2, size / 2 - separation / 2), (size / 2, size / 2 + separation / 2))
    result = np.empty_like(developed)
    for candidate in range(len(developed)):
        rng = np.random.default_rng(seed_base + noise_seed * 100000 + candidate)
        for channel in range(2):
            k = orientation // 90 + (2 if channel else 0)
            state = np.rot90(developed[candidate, channel], k=k).copy()
            active = state > 0
            state[active] = np.clip(
                state[active] + rng.normal(0, 0.0015, int(active.sum())), 0, 1
            )
            result[candidate, channel] = shift_center(state, targets[channel])
    return result


def evaluate(
    developed: np.ndarray,
    parameters: np.ndarray,
    solo: list[dict],
    contexts: tuple[tuple[str, int, int], ...],
    *,
    size: int,
    post_steps: int,
    competition: float,
    seed_base: int,
) -> list[dict]:
    count = len(parameters)
    results = [
        {
            "candidate": index,
            "both_solo_viable": solo[index]["both_viable"],
            "solo": solo[index],
            "contexts": [],
        }
        for index in range(count)
    ]
    centers = parameters[:, :, 0]
    widths = parameters[:, :, 1]
    for context in contexts:
        initial = encounter_states(developed, context, size=size, seed_base=seed_base)
        control = CoupledWorld(
            initial, centers, widths, CoupledConfig(size=size, competition=0)
        )
        coupled = CoupledWorld(
            initial, centers, widths, CoupledConfig(size=size, competition=competition)
        )
        control.step(post_steps)
        coupled.step(post_steps)
        control_final = control.numpy()
        coupled_final = coupled.numpy()
        initial_mass, _ = channel_metrics(initial)
        control_mass, control_occupied = channel_metrics(control_final)
        coupled_mass, coupled_occupied = channel_metrics(coupled_final)
        for candidate in range(count):
            channels = []
            for channel in range(2):
                control_ratio = float(
                    control_mass[candidate, channel]
                    / max(initial_mass[candidate, channel], 1e-12)
                )
                mass_ratio = float(coupled_mass[candidate, channel] / max(control_mass[candidate, channel], 1e-12))
                occupied_ratio = float(
                    coupled_occupied[candidate, channel]
                    / max(control_occupied[candidate, channel], 1e-12)
                )
                similarity = aligned_similarity(
                    coupled_final[candidate, channel], control_final[candidate, channel]
                )
                control_valid = bool(
                    control_mass[candidate, channel] >= 10
                    and 0.5 <= control_ratio <= 2.0
                    and control_occupied[candidate, channel] <= 0.15
                )
                persists = bool(
                    solo[candidate]["channels"][channel]["viable"]
                    and control_valid
                    and 0.5 <= mass_ratio <= 1.5
                    and 0.5 <= occupied_ratio <= 2.0
                    and coupled_occupied[candidate, channel] <= 0.15
                    and similarity >= 0.5
                )
                channels.append(
                    {
                        "persists": persists,
                        "control_valid": control_valid,
                        "mass_ratio": mass_ratio,
                        "occupied_ratio": occupied_ratio,
                        "control_similarity": similarity,
                    }
                )
            results[candidate]["contexts"].append(
                {
                    "geometry": context[0],
                    "orientation_degrees": context[1],
                    "noise_seed": context[2],
                    "coexists": all(channel["persists"] for channel in channels),
                    "channels": channels,
                }
            )
    for result in results:
        contexts_result = result["contexts"]
        result["coexistence_trials"] = sum(row["coexists"] for row in contexts_result)
        result["trials"] = len(contexts_result)
        result["coexistence_rate"] = result["coexistence_trials"] / len(contexts_result)
        result["median_minimum_similarity"] = float(
            np.median(
                [min(channel["control_similarity"] for channel in row["channels"]) for row in contexts_result]
            )
        )
        result["median_minimum_mass_ratio"] = float(
            np.median(
                [min(channel["mass_ratio"] for channel in row["channels"]) for row in contexts_result]
            )
        )
    return results


def parameter_distance(parameters: np.ndarray, parent: np.ndarray) -> np.ndarray:
    scales = np.asarray([[0.012, 0.004], [0.028, 0.012]], np.float32)
    return np.sqrt((((parameters - parent) / scales) ** 2).sum(axis=(1, 2)))


def compact(result: dict, parameters: np.ndarray, distance: float) -> dict:
    return {
        "candidate": result["candidate"],
        "parameters": {
            "O4i": {"growth_center": float(parameters[0, 0]), "growth_width": float(parameters[0, 1])},
            "P4cl": {"growth_center": float(parameters[1, 0]), "growth_width": float(parameters[1, 1])},
        },
        "parameter_distance": float(distance),
        "both_solo_viable": result["both_solo_viable"],
        "coexistence_trials": result["coexistence_trials"],
        "trials": result["trials"],
        "coexistence_rate": result["coexistence_rate"],
        "median_minimum_similarity": result["median_minimum_similarity"],
        "median_minimum_mass_ratio": result["median_minimum_mass_ratio"],
        "solo": result["solo"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--out", default="runs/coexistence-evolution.json")
    parser.add_argument("--web-summary", default="web/data/coexistence-evolution.json")
    parser.add_argument("--training-phase", type=int, default=240)
    parser.add_argument("--held-out-phase", type=int, default=300)
    parser.add_argument("--post-steps", type=int, default=180)
    args = parser.parse_args()

    specimens = tuple(load_specimen(path) for _, path in TARGETS)
    parameters = candidate_parameters(specimens, args.candidates, args.seed)
    parent_parameters = parameters[0].copy()
    developed = develop(specimens, parameters, args.training_phase, 128)
    solo = solo_metrics(developed)
    training = evaluate(
        developed,
        parameters,
        solo,
        TRAIN_CONTEXTS,
        size=128,
        post_steps=args.post_steps,
        competition=1.0,
        seed_base=202609040,
    )
    distances = parameter_distance(parameters, parent_parameters)
    eligible = [result for result in training if result["both_solo_viable"]]
    eligible.sort(
        key=lambda result: (
            result["coexistence_rate"],
            result["median_minimum_similarity"],
            result["median_minimum_mass_ratio"],
            -distances[result["candidate"]],
        ),
        reverse=True,
    )
    selected = eligible[0]
    selected_index = selected["candidate"]

    validation_parameters = np.stack([parent_parameters, parameters[selected_index]])
    held_developed = develop(specimens, validation_parameters, args.held_out_phase, 128)
    held_solo = solo_metrics(held_developed)
    held_out_contexts = tuple(HELD_OUT_CONTEXTS)
    held_out = evaluate(
        held_developed,
        validation_parameters,
        held_solo,
        held_out_contexts,
        size=128,
        post_steps=240,
        competition=1.0,
        seed_base=202609050,
    )
    baseline_validation, candidate_validation = held_out
    improvement = candidate_validation["coexistence_rate"] - baseline_validation["coexistence_rate"]
    promoted = bool(
        selected["coexistence_rate"] >= 0.70
        and candidate_validation["coexistence_rate"] >= 0.50
        and improvement >= 0.25
        and candidate_validation["both_solo_viable"]
    )

    ranked = [
        compact(result, parameters[result["candidate"]], distances[result["candidate"]])
        for result in eligible[:25]
    ]
    record = {
        "schema": "genesis.coexistence-evolution/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": "Can paired growth-rule mutation rescue robust coexistence in the reproducibly difficult O4i × P4cl encounter?",
        "random_seed": args.seed,
        "protocol": {
            "candidate_count": args.candidates,
            "target_pair": ["O4i", "P4cl"],
            "competition": 1.0,
            "training_phase": args.training_phase,
            "training_contexts": [
                {"geometry": row[0], "orientation_degrees": row[1], "noise_seed": row[2]}
                for row in TRAIN_CONTEXTS
            ],
            "held_out_phase": args.held_out_phase,
            "held_out_contexts": [
                {"geometry": row[0], "orientation_degrees": row[1], "noise_seed": row[2]}
                for row in held_out_contexts
            ],
            "selection": "coexistence, then morphological similarity, mass retention, and minimum parameter distance; solo viability required",
            "solo_identity_gate": "each organism retains 0.75–1.25x parental mass and occupied area with >=0.5 translation-aligned similarity",
            "promotion_gate": "training >=70%, held-out >=50%, >=25 percentage-point held-out improvement, both solo viable",
        },
        "summary": {
            "solo_viable_candidates": len(eligible),
            "training_baseline": compact(training[0], parameters[0], distances[0]),
            "selected_training": compact(selected, parameters[selected_index], distances[selected_index]),
            "held_out_baseline": compact(baseline_validation, validation_parameters[0], 0.0),
            "held_out_candidate": compact(candidate_validation, validation_parameters[1], distances[selected_index]),
            "held_out_improvement": improvement,
            "promoted": promoted,
        },
        "top_candidates": ranked,
        "selected_training_contexts": selected["contexts"],
        "held_out_results": held_out,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    web_record = {
        key: record[key]
        for key in ("schema", "created_at", "question", "protocol", "summary", "top_candidates")
    }
    web_destination = Path(args.web_summary)
    web_destination.parent.mkdir(parents=True, exist_ok=True)
    web_destination.write_text(json.dumps(web_record, indent=2) + "\n")
    print(json.dumps(record["summary"], indent=2))


if __name__ == "__main__":
    main()
