"""Export verified local evidence summaries for the static observatory."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read(name):
    return json.loads(Path(name).read_text())


def write(name, record):
    destination=Path(name)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")


def main():
    validation=read("runs/genesis-magnitude-validation.json")
    for aggregate in validation["aggregate"]:
        rows=[row for row in validation["results"] if row["genotype"]==aggregate["genotype"]
              and row["target_removed_fraction"]==aggregate["target_removed_fraction"]]
        seeds=validation["protocol"]["held_out_initial_condition_seeds"]
        rates=[np.mean([row["functionally_recovered"] for row in rows if row["initial_condition_seed"]==seed]) for seed in seeds]
        if not np.isclose(np.mean(rates),aggregate["mean_seed_recovery_rate"]):
            raise ValueError("held-out aggregate does not reconcile to independent seed rates")
        if len(rows)!=len(seeds)*len(validation["protocol"]["held_out_phases"]):
            raise ValueError("held-out trial matrix incomplete")
    # Raw trials included so browser claims remain independently checkable.
    write("web/data/genesis-magnitude-validation.json",validation)
    replicated=read("runs/replicated-damage-sweep.json")
    write("web/data/replicated-damage-sweep.json",replicated)
    record={"schema":"genesis.research-audit/v1","created_at":datetime.now(timezone.utc).isoformat(),
        "metrics":"survival, functional recovery, strict mass return, translation-aligned shape and sustained recovery times separated",
        "replication":"20 independent noisy initial conditions; one phase per seed; damage and location repeated within seed",
        "magnitude_candidate_decision":validation["acceptance"],
        "charter_comparison_status":"coverage pending",
        "evidence":["genesis-magnitude-validation.json","replicated-damage-sweep.json","multispecies-transfer.json"]}
    coverage_path=Path("runs/charter-comparison/coverage.json")
    if coverage_path.exists():
        coverage=read(coverage_path)
        write("web/data/charter-coverage.json",coverage)
        write("web/data/charter-protocol.json",read("runs/charter-comparison/protocol.json"))
        record["coverage_gate"]=coverage
        record["charter_comparison_status"]="coverage passed; comparison pending" if coverage["gate_passed"] else "coverage failed; comparison locked"
        record["evidence"].append("charter-coverage.json")
        record["evidence"].append("charter-protocol.json")
    comparison_path=Path("runs/charter-comparison/comparison.json")
    if comparison_path.exists():
        comparison=read(comparison_path)
        write("web/data/charter-comparison.json",comparison)
        record["charter_comparison_status"]="comparison complete; ablation pending"
        record["evidence"].append("charter-comparison.json")
        if Path("web/data/charter-analysis.json").exists():
            record["evidence"].extend(["charter-analysis.json","charter-trials.json"])
    write("web/data/research-audit.json",record)
    print(record["charter_comparison_status"])


if __name__=="__main__":
    main()
