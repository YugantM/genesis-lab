"""Evolve pressure sensitivity, boundary response, and memory for a hard pair.

This run keeps both bodies and native Lenia growth rules fixed.  Only the way
each organism responds to another organism's local field may change.  Search
and held-out contexts are separated before candidate generation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from .core import GenesisConfig, GenesisWorld
from .coupled_ecology import aligned_similarity, channel_metrics, shift_center
from .genome import Specimen, load_specimen
from .interaction_ecology import InteractionConfig, ResponsiveCoupledWorld
from .multispecies_benchmark import centered_state


TARGETS = (
    ("O4i", Path("web/specimens/catalog/synorbium-ignis.json")),
    ("P4cl", Path("web/specimens/catalog/paraptera-cavus-labens.json")),
)
TRAIN_CONTEXTS = (
    ("contact", 0, 30),
    ("contact", 90, 31),
    ("overlap", 0, 30),
    ("overlap", 90, 31),
)
HELD_OUT_CONTEXTS = tuple(
    (geometry, orientation, noise_seed)
    for geometry in ("contact", "overlap")
    for orientation in (180, 270)
    for noise_seed in (40, 41)
)


def interaction_candidates(count: int, seed: int) -> np.ndarray:
    """Return (candidate, channel, [sensitivity, response, memory])."""
    rng = np.random.default_rng(seed)
    result = np.empty((count, 2, 3), np.float32)
    result[:, :, 0] = np.clip(rng.normal(1.0, 0.16, (count, 2)), 0.72, 1.28)
    result[:, :, 1] = rng.beta(2.0, 2.5, (count, 2)) * 0.65
    result[:, :, 2] = rng.beta(2.0, 1.8, (count, 2)) * 0.95
    result[0, :, 0] = 1.0
    result[0, :, 1] = 0.0
    result[0, :, 2] = 0.0
    return result


def develop(specimens: tuple[Specimen, Specimen], phase: int, size: int) -> np.ndarray:
    states = np.stack([centered_state(specimen, size) for specimen in specimens])
    world = GenesisWorld(GenesisConfig(size=size, batch=2, radius=13, dt=0.1))
    world.state = mx.array(states)
    world.set_growth_parameters(
        [specimen.parameters["growth_center"] for specimen in specimens],
        [specimen.parameters["growth_width"] for specimen in specimens],
    )
    world.step(phase)
    return world.numpy()


def encounter(
    developed: np.ndarray,
    context: tuple[str, int, int],
    count: int,
    *,
    size: int,
    seed_base: int,
) -> np.ndarray:
    geometry, orientation, noise_seed = context
    separation = 24 if geometry == "contact" else 12
    targets = (
        (size / 2, size / 2 - separation / 2),
        (size / 2, size / 2 + separation / 2),
    )
    rng = np.random.default_rng(seed_base + noise_seed)
    channels = []
    for channel in range(2):
        state = np.rot90(
            developed[channel], k=orientation // 90 + (2 if channel else 0)
        ).copy()
        active = state > 0
        state[active] = np.clip(
            state[active] + rng.normal(0, 0.0015, int(active.sum())), 0, 1
        )
        channels.append(shift_center(state, targets[channel]))
    # Every candidate receives the exact same perturbed encounter.
    return np.repeat(np.stack(channels)[None], count, axis=0)


def evaluate(
    developed: np.ndarray,
    interactions: np.ndarray,
    specimens: tuple[Specimen, Specimen],
    contexts: tuple[tuple[str, int, int], ...],
    *,
    size: int,
    post_steps: int,
    seed_base: int,
) -> list[dict]:
    count = len(interactions)
    centers = np.repeat(
        np.asarray(
            [[specimen.parameters["growth_center"] for specimen in specimens]],
            np.float32,
        ),
        count,
        axis=0,
    )
    widths = np.repeat(
        np.asarray(
            [[specimen.parameters["growth_width"] for specimen in specimens]],
            np.float32,
        ),
        count,
        axis=0,
    )
    results = [
        {"candidate": candidate, "contexts": []} for candidate in range(count)
    ]
    for context in contexts:
        initial = encounter(
            developed, context, count, size=size, seed_base=seed_base
        )
        control = ResponsiveCoupledWorld(
            initial,
            centers,
            widths,
            interactions[:, :, 0],
            interactions[:, :, 1],
            interactions[:, :, 2],
            InteractionConfig(size=size, competition=0.0),
        )
        coupled = ResponsiveCoupledWorld(
            initial,
            centers,
            widths,
            interactions[:, :, 0],
            interactions[:, :, 1],
            interactions[:, :, 2],
            InteractionConfig(size=size, competition=1.0),
        )
        control.step(post_steps)
        coupled.step(post_steps)
        initial_mass, _ = channel_metrics(initial)
        control_final = control.numpy()
        coupled_final = coupled.numpy()
        control_mass, control_area = channel_metrics(control_final)
        coupled_mass, coupled_area = channel_metrics(coupled_final)
        for candidate in range(count):
            channels = []
            for channel in range(2):
                reference_mass = max(control_mass[0, channel], 1e-12)
                reference_area = max(control_area[0, channel], 1e-12)
                identity_mass_ratio = float(
                    control_mass[candidate, channel] / reference_mass
                )
                identity_area_ratio = float(
                    control_area[candidate, channel] / reference_area
                )
                identity_similarity = aligned_similarity(
                    control_final[candidate, channel], control_final[0, channel]
                )
                own_mass = max(control_mass[candidate, channel], 1e-12)
                own_area = max(control_area[candidate, channel], 1e-12)
                mass_ratio = float(coupled_mass[candidate, channel] / own_mass)
                area_ratio = float(coupled_area[candidate, channel] / own_area)
                similarity = aligned_similarity(
                    coupled_final[candidate, channel],
                    control_final[candidate, channel],
                )
                control_valid = bool(
                    control_mass[candidate, channel] >= 10
                    and 0.5
                    <= control_mass[candidate, channel]
                    / max(initial_mass[candidate, channel], 1e-12)
                    <= 2.0
                    and control_area[candidate, channel] <= 0.15
                    and 0.75 <= identity_mass_ratio <= 1.25
                    and 0.75 <= identity_area_ratio <= 1.25
                    and identity_similarity >= 0.5
                )
                persists = bool(
                    control_valid
                    and 0.5 <= mass_ratio <= 1.5
                    and 0.5 <= area_ratio <= 2.0
                    and coupled_area[candidate, channel] <= 0.15
                    and similarity >= 0.5
                )
                channels.append(
                    {
                        "control_valid": control_valid,
                        "persists": persists,
                        "identity_mass_ratio": identity_mass_ratio,
                        "identity_area_ratio": identity_area_ratio,
                        "identity_similarity": identity_similarity,
                        "coupled_to_control_mass_ratio": mass_ratio,
                        "coupled_to_control_area_ratio": area_ratio,
                        "coupled_to_control_similarity": similarity,
                    }
                )
            results[candidate]["contexts"].append(
                {
                    "geometry": context[0],
                    "orientation_degrees": context[1],
                    "noise_seed": context[2],
                    "identity_safe": all(row["control_valid"] for row in channels),
                    "coexists": all(row["persists"] for row in channels),
                    "channels": channels,
                }
            )
    for result in results:
        rows = result["contexts"]
        result["identity_safe"] = all(row["identity_safe"] for row in rows)
        result["coexistence_trials"] = sum(row["coexists"] for row in rows)
        result["trials"] = len(rows)
        result["coexistence_rate"] = result["coexistence_trials"] / len(rows)
        result["median_minimum_similarity"] = float(
            np.median(
                [
                    min(
                        channel["coupled_to_control_similarity"]
                        for channel in row["channels"]
                    )
                    for row in rows
                ]
            )
        )
    return results


def distance(interactions: np.ndarray) -> np.ndarray:
    parent = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
    scale = np.asarray([[0.16, 0.25, 0.4], [0.16, 0.25, 0.4]], np.float32)
    return np.sqrt((((interactions - parent) / scale) ** 2).sum(axis=(1, 2)))


def compact(
    result: dict,
    traits: np.ndarray,
    trait_distance: float,
    *,
    candidate_label: int | None = None,
) -> dict:
    return {
        "candidate": result["candidate"] if candidate_label is None else candidate_label,
        "interaction_genes": {
            "O4i": {
                "pressure_sensitivity": float(traits[0, 0]),
                "boundary_response": float(traits[0, 1]),
                "signal_memory": float(traits[0, 2]),
            },
            "P4cl": {
                "pressure_sensitivity": float(traits[1, 0]),
                "boundary_response": float(traits[1, 1]),
                "signal_memory": float(traits[1, 2]),
            },
        },
        "trait_distance": float(trait_distance),
        "identity_safe": result["identity_safe"],
        "coexistence_trials": result["coexistence_trials"],
        "trials": result["trials"],
        "coexistence_rate": result["coexistence_rate"],
        "median_minimum_similarity": result["median_minimum_similarity"],
    }


def run(
    *,
    candidate_count: int = 1024,
    seed: int = 20260905,
    size: int = 128,
    training_phase: int = 240,
    held_out_phase: int = 300,
    post_steps: int = 240,
) -> dict:
    if candidate_count < 2:
        raise ValueError("candidate_count must include a parent and mutations")
    specimens = tuple(load_specimen(path) for _, path in TARGETS)
    traits = interaction_candidates(candidate_count, seed)
    developed = develop(specimens, training_phase, size)
    training = evaluate(
        developed,
        traits,
        specimens,
        TRAIN_CONTEXTS,
        size=size,
        post_steps=post_steps,
        seed_base=202609050,
    )
    distances = distance(traits)
    eligible = [row for row in training if row["identity_safe"]]
    eligible.sort(
        key=lambda row: (
            row["coexistence_rate"],
            row["median_minimum_similarity"],
            -distances[row["candidate"]],
        ),
        reverse=True,
    )
    selected = eligible[0]
    selected_index = selected["candidate"]
    held_traits = np.stack([traits[0], traits[selected_index]])
    held_developed = develop(specimens, held_out_phase, size)
    held = evaluate(
        held_developed,
        held_traits,
        specimens,
        HELD_OUT_CONTEXTS,
        size=size,
        post_steps=post_steps,
        seed_base=202609060,
    )
    baseline, candidate = held
    improvement = candidate["coexistence_rate"] - baseline["coexistence_rate"]
    promoted = bool(
        selected["coexistence_rate"] >= 0.75
        and candidate["coexistence_rate"] >= 0.5
        and improvement >= 0.25
        and candidate["identity_safe"]
    )
    return {
        "schema": "genesis.interaction-evolution/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": "Can evolvable pressure sensitivity, costly boundary response, and contact memory rescue robust coexistence without changing either body?",
        "random_seed": seed,
        "protocol": {
            "candidate_count": candidate_count,
            "target_pair": ["O4i", "P4cl"],
            "body_parameters": "fixed parental values",
            "mutable_traits": [
                "pressure_sensitivity",
                "boundary_response",
                "signal_memory",
            ],
            "trait_bounds": {
                "pressure_sensitivity": [0.72, 1.28],
                "boundary_response": [0.0, 0.65],
                "signal_memory": [0.0, 0.95],
            },
            "boundary_maintenance_cost": 0.035,
            "training_phase": training_phase,
            "training_contexts": [
                {"geometry": row[0], "orientation_degrees": row[1], "noise_seed": row[2]}
                for row in TRAIN_CONTEXTS
            ],
            "held_out_phase": held_out_phase,
            "held_out_contexts": [
                {"geometry": row[0], "orientation_degrees": row[1], "noise_seed": row[2]}
                for row in HELD_OUT_CONTEXTS
            ],
            "matched_noise": "all candidates receive identical perturbations within each context",
            "identity_gate": "every isolated context retains 0.75–1.25x parental mass and occupied area with >=0.5 aligned similarity",
            "promotion_gate": "training >=75%, held-out >=50%, held-out improvement >=25 points, identity safe",
        },
        "summary": {
            "identity_safe_candidates": len(eligible),
            "training_baseline": compact(training[0], traits[0], distances[0]),
            "selected_training": compact(
                selected, traits[selected_index], distances[selected_index]
            ),
            "held_out_baseline": compact(baseline, held_traits[0], 0.0),
            "held_out_candidate": compact(
                candidate,
                held_traits[1],
                distances[selected_index],
                candidate_label=selected_index,
            ),
            "held_out_improvement": improvement,
            "promoted": promoted,
        },
        "top_candidates": [
            compact(row, traits[row["candidate"]], distances[row["candidate"]])
            for row in eligible[:25]
        ],
        "selected_training_contexts": selected["contexts"],
        "held_out_results": held,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--out", default="runs/interaction-evolution.json")
    parser.add_argument("--web-summary", default="web/data/interaction-evolution.json")
    args = parser.parse_args()
    record = run(candidate_count=args.candidates, seed=args.seed)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    web_output = Path(args.web_summary)
    web_output.parent.mkdir(parents=True, exist_ok=True)
    web_output.write_text(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "schema",
                    "created_at",
                    "question",
                    "protocol",
                    "summary",
                    "top_candidates",
                )
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(record["summary"], indent=2))


if __name__ == "__main__":
    main()
