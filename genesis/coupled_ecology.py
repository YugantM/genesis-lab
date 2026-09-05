"""Reproducible pairwise ecology for real continuous Lenia organisms.

Each organism retains its native single-channel Lenia rule. Two channels share
space through a symmetric, local competition term: neighbourhood mass from the
other channel subtracts from intrinsic growth. This is a deliberately minimal
ecological bridge, not a claim that classic Lenia already defines ecology.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from .core import GenesisConfig, GenesisWorld, _kernel
from .genome import Specimen, load_specimen
from .metrics import aligned_similarity, center_of_mass, occupied_fraction
from .multispecies_benchmark import centered_state, load_manifest


@dataclass(frozen=True)
class CoupledConfig:
    size: int = 128
    dt: float = 0.1
    radius: float = 13.0
    competition: float = 0.5


class CoupledWorld:
    """A batch of two-channel Lenia worlds with symmetric local competition."""

    def __init__(
        self,
        states: np.ndarray,
        centers: np.ndarray,
        widths: np.ndarray,
        config: CoupledConfig,
    ) -> None:
        if states.ndim != 4 or states.shape[1] != 2:
            raise ValueError("states must have shape (batch, 2, size, size)")
        expected = states.shape[:2]
        if centers.shape != expected or widths.shape != expected:
            raise ValueError(f"growth parameters must have shape {expected}")
        if states.shape[-2:] != (config.size, config.size):
            raise ValueError("state size does not match configuration")
        if np.any(widths <= 0) or config.competition < 0:
            raise ValueError("widths must be positive and competition non-negative")
        self.config = config
        self.state = mx.array(states.astype(np.float32, copy=False))
        self.centers = mx.array(centers.astype(np.float32))[:, :, None, None]
        self.widths = mx.array(widths.astype(np.float32))[:, :, None, None]
        kernel_config = GenesisConfig(size=config.size, radius=config.radius)
        self.kernel_fft = mx.fft.fft2(_kernel(kernel_config))[None, None]
        mx.eval(self.state, self.centers, self.widths, self.kernel_fft)

    def step(self, count: int = 1) -> mx.array:
        for _ in range(count):
            neighbourhood = mx.fft.ifft2(
                mx.fft.fft2(self.state) * self.kernel_fft
            ).real
            intrinsic = 2.0 * mx.exp(
                -((neighbourhood - self.centers) ** 2) / (2.0 * self.widths**2)
            ) - 1.0
            other_neighbourhood = mx.sum(
                neighbourhood, axis=1, keepdims=True
            ) - neighbourhood
            growth = intrinsic - self.config.competition * other_neighbourhood
            self.state = mx.clip(
                self.state + self.config.dt * growth, 0.0, 1.0
            )
        mx.eval(self.state)
        return self.state

    def numpy(self) -> np.ndarray:
        return np.asarray(self.state)


def shift_center(state: np.ndarray, target_yx: tuple[float, float]) -> np.ndarray:
    """Toroidally translate a state so its centre of mass reaches a target."""
    centre = center_of_mass(state[None])[0]
    shift = np.rint(np.asarray(target_yx) - centre).astype(int)
    return np.roll(state, tuple(shift), axis=(0, 1))


def develop_specimens(
    catalogue: list[tuple[dict, Specimen]], size: int, phase: int
) -> list[tuple[dict, Specimen, np.ndarray]]:
    states = np.stack([centered_state(specimen, size) for _, specimen in catalogue])
    centers = np.asarray(
        [specimen.parameters["growth_center"] for _, specimen in catalogue],
        np.float32,
    )
    widths = np.asarray(
        [specimen.parameters["growth_width"] for _, specimen in catalogue],
        np.float32,
    )
    world = GenesisWorld(
        GenesisConfig(size=size, batch=len(catalogue), radius=13, dt=0.1)
    )
    world.state = mx.array(states)
    world.set_growth_parameters(centers, widths)
    world.step(phase)
    return [
        (entry, specimen, state.copy())
        for (entry, specimen), state in zip(catalogue, world.numpy(), strict=True)
    ]


def load_panel(manifest: Path, genesis_path: Path) -> list[tuple[dict, Specimen]]:
    genesis = load_specimen(genesis_path)
    genesis_entry = {
        "code": "GEN001",
        "name": genesis.name,
        "specimen_id": genesis.id,
        "path": str(genesis_path),
    }
    return [(genesis_entry, genesis), *load_manifest(manifest)]


def build_encounters(
    developed: list[tuple[dict, Specimen, np.ndarray]],
    size: int,
    *,
    orientations: tuple[int, ...] = (0, 90),
    noise_seeds: tuple[int, ...] = (0, 1),
    seed_base: int = 20260904,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    geometries = {
        "near": ((size / 2, size / 2 - 28), (size / 2, size / 2 + 28)),
        "contact": ((size / 2, size / 2 - 12), (size / 2, size / 2 + 12)),
        "overlap": ((size / 2, size / 2 - 6), (size / 2, size / 2 + 6)),
    }
    states: list[np.ndarray] = []
    centers: list[list[float]] = []
    widths: list[list[float]] = []
    metadata: list[dict] = []
    for (left_index, left), (right_index, right) in itertools.combinations(
        enumerate(developed), 2
    ):
        left_entry, left_specimen, left_state = left
        right_entry, right_specimen, right_state = right
        for geometry, (left_target, right_target) in geometries.items():
            for orientation in orientations:
                k = orientation // 90
                rotated_left = np.rot90(left_state, k=k).copy()
                rotated_right = np.rot90(right_state, k=k + 2).copy()
                for noise_seed in noise_seeds:
                    rng = np.random.default_rng(
                        seed_base
                        + left_index * 10000
                        + right_index * 1000
                        + orientation * 10
                        + noise_seed
                    )
                    channels = []
                    for source, target in (
                        (rotated_left, left_target),
                        (rotated_right, right_target),
                    ):
                        perturbed = source.copy()
                        active = perturbed > 0
                        perturbed[active] = np.clip(
                            perturbed[active]
                            + rng.normal(0, 0.001, int(active.sum())),
                            0,
                            1,
                        )
                        channels.append(shift_center(perturbed, target))
                    states.append(np.stack(channels))
                    centers.append(
                        [
                            float(left_specimen.parameters["growth_center"]),
                            float(right_specimen.parameters["growth_center"]),
                        ]
                    )
                    widths.append(
                        [
                            float(left_specimen.parameters["growth_width"]),
                            float(right_specimen.parameters["growth_width"]),
                        ]
                    )
                    metadata.append(
                        {
                            "left_code": left_entry["code"],
                            "left_name": left_entry["name"],
                            "right_code": right_entry["code"],
                            "right_name": right_entry["name"],
                            "geometry": geometry,
                            "orientation_degrees": orientation,
                            "noise_seed": noise_seed,
                        }
                    )
    return (
        np.stack(states).astype(np.float32),
        np.asarray(centers, np.float32),
        np.asarray(widths, np.float32),
        metadata,
    )


def channel_metrics(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = states.reshape(*states.shape[:2], -1)
    masses = flat.sum(axis=-1)
    occupied = (flat >= 0.1).mean(axis=-1)
    return masses, occupied


def run_experiment(
    *,
    manifest: Path,
    genesis_path: Path,
    size: int = 128,
    phase: int = 240,
    post_steps: int = 240,
    strengths: tuple[float, ...] = (0.25, 0.5, 1.0),
    orientations: tuple[int, ...] = (0, 90),
    noise_seeds: tuple[int, ...] = (0, 1),
    seed_base: int = 20260904,
) -> dict:
    panel = load_panel(manifest, genesis_path)
    developed = develop_specimens(panel, size, phase)
    initial, centers, widths, metadata = build_encounters(
        developed,
        size,
        orientations=orientations,
        noise_seeds=noise_seeds,
        seed_base=seed_base,
    )
    initial_mass, initial_occupied = channel_metrics(initial)

    control = CoupledWorld(
        initial,
        centers,
        widths,
        CoupledConfig(size=size, competition=0.0),
    )
    control.step(post_steps)
    control_final = control.numpy()
    control_mass, control_occupied = channel_metrics(control_final)

    rows: list[dict] = []
    for strength in strengths:
        coupled = CoupledWorld(
            initial,
            centers,
            widths,
            CoupledConfig(size=size, competition=strength),
        )
        coupled.step(post_steps)
        final = coupled.numpy()
        final_mass, final_occupied = channel_metrics(final)
        for index, meta in enumerate(metadata):
            persistence = []
            similarities = []
            channel_rows = []
            for channel, side in enumerate(("left", "right")):
                control_mass_ratio = float(
                    control_mass[index, channel] / max(initial_mass[index, channel], 1e-12)
                )
                mass_ratio = float(
                    final_mass[index, channel] / max(control_mass[index, channel], 1e-12)
                )
                occupied_ratio = float(
                    final_occupied[index, channel]
                    / max(control_occupied[index, channel], 1e-12)
                )
                control_valid = bool(
                    control_mass[index, channel] >= 10
                    and 0.5 <= control_mass_ratio <= 2.0
                    and control_occupied[index, channel] <= 0.15
                )
                similarity = aligned_similarity(
                    final[index, channel], control_final[index, channel]
                )
                persists = bool(
                    control_valid
                    and 0.5 <= mass_ratio <= 1.5
                    and 0.5 <= occupied_ratio <= 2.0
                    and final_occupied[index, channel] <= 0.15
                    and similarity >= 0.5
                )
                persistence.append(persists)
                similarities.append(similarity)
                channel_rows.append(
                    {
                        "side": side,
                        "code": meta[f"{side}_code"],
                        "control_valid": control_valid,
                        "initial_mass": float(initial_mass[index, channel]),
                        "control_final_mass": float(control_mass[index, channel]),
                        "coupled_final_mass": float(final_mass[index, channel]),
                        "coupled_to_control_mass_ratio": mass_ratio,
                        "coupled_to_control_occupied_ratio": occupied_ratio,
                        "aligned_control_similarity": similarity,
                        "persists": persists,
                    }
                )
            valid = all(item["control_valid"] for item in channel_rows)
            if not valid:
                outcome = "invalid-control"
            elif all(persistence):
                outcome = "coexistence"
            elif any(persistence):
                outcome = "dominance"
            else:
                outcome = "mutual-collapse"
            rows.append(
                {
                    **meta,
                    "competition": strength,
                    "outcome": outcome,
                    "minimum_similarity": float(min(similarities)),
                    "channels": channel_rows,
                }
            )

    grouped: dict[tuple[float, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["competition"], row["left_code"], row["right_code"])].append(row)
    pair_summary = []
    for key, group in sorted(grouped.items()):
        strength, left_code, right_code = key
        valid = [row for row in group if row["outcome"] != "invalid-control"]
        counts = {
            outcome: sum(row["outcome"] == outcome for row in valid)
            for outcome in ("coexistence", "dominance", "mutual-collapse")
        }
        pair_summary.append(
            {
                "competition": strength,
                "left_code": left_code,
                "right_code": right_code,
                "trials": len(group),
                "valid_trials": len(valid),
                **{f"{name.replace('-', '_')}_trials": count for name, count in counts.items()},
                "coexistence_rate": counts["coexistence"] / max(len(valid), 1),
                "median_minimum_similarity": float(
                    np.median([row["minimum_similarity"] for row in valid])
                )
                if valid
                else 0.0,
            }
        )

    strength_summary = []
    for strength in strengths:
        group = [
            row
            for row in rows
            if row["competition"] == strength and row["outcome"] != "invalid-control"
        ]
        strength_summary.append(
            {
                "competition": strength,
                "valid_trials": len(group),
                "coexistence_trials": sum(row["outcome"] == "coexistence" for row in group),
                "dominance_trials": sum(row["outcome"] == "dominance" for row in group),
                "mutual_collapse_trials": sum(row["outcome"] == "mutual-collapse" for row in group),
                "coexistence_rate": float(
                    np.mean([row["outcome"] == "coexistence" for row in group])
                )
                if group
                else 0.0,
            }
        )

    geometry_summary = []
    for strength in strengths:
        for geometry in ("near", "contact", "overlap"):
            group = [
                row
                for row in rows
                if row["competition"] == strength
                and row["geometry"] == geometry
                and row["outcome"] != "invalid-control"
            ]
            geometry_summary.append(
                {
                    "competition": strength,
                    "geometry": geometry,
                    "valid_trials": len(group),
                    "coexistence_trials": sum(
                        row["outcome"] == "coexistence" for row in group
                    ),
                    "coexistence_rate": float(
                        np.mean([row["outcome"] == "coexistence" for row in group])
                    )
                    if group
                    else 0.0,
                }
            )

    return {
        "schema": "genesis.coupled-ecology/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": "Can pairs of real continuous Lenia organisms retain both identities under symmetric shared-space pressure?",
        "model": {
            "intrinsic_growth": "native Gaussian Lenia growth per channel",
            "coupling": "growth_i = intrinsic_i - competition * kernel(other_channel)",
            "interpretation": "minimal symmetric space competition; no resource, predation, or reproduction mechanism",
        },
        "protocol": {
            "species_count": len(panel),
            "pair_count": len(panel) * (len(panel) - 1) // 2,
            "development_phase": phase,
            "post_encounter_steps": post_steps,
            "world_size": size,
            "geometries": ["near", "contact", "overlap"],
            "orientations_degrees": list(orientations),
            "noise_seeds": list(noise_seeds),
            "seed_base": seed_base,
            "initial_state_noise_sigma": 0.001,
            "competition_strengths": list(strengths),
            "trials_per_pair_strength": 3 * len(orientations) * len(noise_seeds),
            "outcome_rule": "coexistence requires both channels to retain 0.5–1.5x matched-control mass, 0.5–2.0x occupied area, and >=0.5 translation-aligned cosine similarity",
        },
        "summary": {
            "trial_count": len(rows),
            "invalid_control_trials": sum(row["outcome"] == "invalid-control" for row in rows),
            "strengths": strength_summary,
            "geometries": geometry_summary,
        },
        "pair_summary": pair_summary,
        "results": rows,
    }


def web_summary(record: dict) -> dict:
    return {
        key: record[key]
        for key in ("schema", "created_at", "question", "model", "protocol", "summary", "pair_summary")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="web/specimens/species-benchmark.json")
    parser.add_argument("--genesis", default="web/specimens/genesis-001.json")
    parser.add_argument("--out", default="runs/coupled-ecology.json")
    parser.add_argument("--web-summary", default="web/data/coupled-ecology.json")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--phase", type=int, default=240)
    parser.add_argument("--post-steps", type=int, default=240)
    parser.add_argument("--strengths", default="0.25,0.5,1.0")
    parser.add_argument("--orientations", default="0,90")
    parser.add_argument("--noise-seeds", default="0,1")
    parser.add_argument("--seed-base", type=int, default=20260904)
    args = parser.parse_args()
    strengths = tuple(float(value) for value in args.strengths.split(","))
    orientations = tuple(int(value) for value in args.orientations.split(","))
    noise_seeds = tuple(int(value) for value in args.noise_seeds.split(","))
    record = run_experiment(
        manifest=Path(args.manifest),
        genesis_path=Path(args.genesis),
        size=args.size,
        phase=args.phase,
        post_steps=args.post_steps,
        strengths=strengths,
        orientations=orientations,
        noise_seeds=noise_seeds,
        seed_base=args.seed_base,
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    web_destination = Path(args.web_summary)
    web_destination.parent.mkdir(parents=True, exist_ok=True)
    web_destination.write_text(json.dumps(web_summary(record), indent=2) + "\n")
    print(json.dumps(record["summary"], indent=2))


if __name__ == "__main__":
    main()
