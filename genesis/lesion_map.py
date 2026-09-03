"""Map the consequence of a calibrated lesion at every active Orbium cell."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from .core import GenesisWorld
from .damage_sweep import calibrate_radius, disk_mask
from .genome import load_specimen
from .metrics import center_of_mass, mass, toroidal_displacement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", default="web/specimens/orbium-unicaudatus.json")
    parser.add_argument("--out", default="runs/lesion-map.json")
    parser.add_argument("--csv", default="runs/lesion-map.csv")
    parser.add_argument("--phase", type=int, default=300)
    parser.add_argument("--target", type=float, default=0.05)
    parser.add_argument("--post-steps", type=int, default=300)
    parser.add_argument("--activity-threshold", type=float, default=0.1)
    args = parser.parse_args()

    specimen = load_specimen(args.specimen)
    template = GenesisWorld.from_specimen(specimen)
    template.step(args.phase - 20)
    prior_center = center_of_mass(template.numpy())[0]
    template.step(20)
    state = template.numpy()[0]
    centre = center_of_mass(state[None, :, :])[0]
    motion = toroidal_displacement(prior_center, centre, template.config.size)
    forward = motion / max(float(np.linalg.norm(motion)), 1e-12)
    lateral_axis = np.array([-forward[1], forward[0]])

    candidate_cells = np.argwhere(state >= args.activity_threshold)
    injured_states, conditions = [], []
    for cell in candidate_cells:
        radius, actual = calibrate_radius(state, cell.astype(float), args.target)
        injured = state.copy()
        injured[disk_mask(template.config.size, cell, radius)] = 0.0
        injured_states.append(injured)
        relative = toroidal_displacement(centre, cell, template.config.size)
        conditions.append(
            {
                "cell_y": int(cell[0]),
                "cell_x": int(cell[1]),
                "relative_forward": float(np.dot(relative, forward)),
                "relative_lateral": float(np.dot(relative, lateral_axis)),
                "local_activity": float(state[tuple(cell)]),
                "radius": radius,
                "actual_removed_fraction": actual,
            }
        )

    config = type(template.config)(**{**template.config.__dict__, "batch": len(injured_states)})
    worlds = GenesisWorld(config)
    worlds.state = mx.array(np.stack(injured_states))
    mx.eval(worlds.state)
    extinction = np.full(len(conditions), -1, np.int32)
    for step in range(1, args.post_steps + 1):
        worlds.step()
        if step % 10 == 0:
            measured = mass(worlds.numpy())
            newly_extinct = (extinction < 0) & (measured < 1.0)
            extinction[newly_extinct] = step

    final_mass = mass(worlds.numpy())
    pre_mass = float(state.sum())
    rows = []
    for index, condition in enumerate(conditions):
        value = float(final_mass[index])
        rows.append(
            {
                **condition,
                "final_mass": value,
                "final_mass_ratio": value / pre_mass,
                "survived": value >= 1.0,
                "extinction_step": None if extinction[index] < 0 else int(extinction[index]),
            }
        )

    record = {
        "schema": "genesis.lesion-map/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specimen_id": specimen.id,
        "protocol": {
            "phase": args.phase,
            "target_removed_fraction": args.target,
            "post_steps": args.post_steps,
            "activity_threshold": args.activity_threshold,
            "candidate_count": len(rows),
        },
        "frame": {
            "center_yx": centre.tolist(),
            "forward_yx": forward.tolist(),
        },
        "summary": {
            "survivors": sum(row["survived"] for row in rows),
            "survival_rate": float(np.mean([row["survived"] for row in rows])),
            "mean_calibration_error": float(np.mean([abs(row["actual_removed_fraction"] - args.target) for row in rows])),
        },
        "results": rows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    csv_path = Path(args.csv)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Mapped {len(rows)} active-cell lesions")
    print(f"Survival: {record['summary']['survivors']}/{len(rows)} ({record['summary']['survival_rate']:.1%})")
    print(f"Mean calibration error: {record['summary']['mean_calibration_error']:.3%}")


if __name__ == "__main__":
    main()
