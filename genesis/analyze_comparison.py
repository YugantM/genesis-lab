"""Reconcile the completed charter trial matrix and summarize paired seed effects."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .selection_comparison import METHODS, HELDOUT, confidence_interval, save


def median_event(rows, field):
    values=[row[field] for row in rows if row.get(field) is not None]
    return {"events":len(values),"censored_or_invalid":len(rows)-len(values),
            "median_steps_among_events":float(np.median(values)) if values else None}


def main():
    root=Path("runs/charter-comparison")
    complete=json.loads((root/"comparison.json").read_text())
    protocol=json.loads((root/"protocol.json").read_text())["protocol"]
    seeds=protocol["search_seeds"]
    summaries=[]
    contrasts=[]
    seed_differences={"performance":[],"direct-regeneration":[]}
    public_trials=[]
    selected_genomes=[]
    common_valid_sensitivity=[]
    for species in protocol["species"]:
        record=json.loads((root/f"comparison-{species}.json").read_text())
        if record["protocol_sha256"]!=complete["protocol_sha256"]:
            raise ValueError("mixed protocols in completed comparison")
        if not record["eligible"]:
            summaries.append({"species":species,"excluded":True})
            continue
        rows=record["heldout"]
        keys={(row["seed"],row["method"],row["intervention"]) for row in rows}
        expected={(seed,method,condition) for seed in seeds for method in METHODS for condition in HELDOUT}
        if len(rows)!=len(keys) or keys!=expected:
            raise ValueError(f"incomplete or duplicate trial matrix for {species}")
        public_trials.extend({"species":species,**{key:value for key,value in row.items() if key!="trajectory"}} for row in rows)
        selected_genomes.extend({"species":species,"seed":row["seed"],"method":row["method"],"champion":row["champion"]} for row in record["searches"])
        seed_rates={}
        valid_seeds={}
        for method in METHODS:
            subset=[row for row in rows if row["method"]==method]
            rates=np.array([np.mean([row["functionally_recovered"] for row in subset if row["seed"]==seed]) for seed in seeds])
            seed_rates[method]=rates
            valid_seeds[method]={seed for seed in seeds if all(r["control_valid"] for r in subset if r["seed"]==seed)}
            valid_rows=[row for row in subset if row["control_valid"]]
            summaries.append({"species":species,"method":method,"independent_search_seeds":len(seeds),
                "repeated_trials":len(subset),"functional_recovery_rate":float(np.mean(rates)),
                "survival_rate":float(np.mean([row["survived"] for row in subset])),
                "control_validity":float(np.mean([row["control_valid"] for row in subset])),
                "recovery_among_valid_controls":float(np.mean([r["functionally_recovered"] for r in valid_rows])) if valid_rows else None,
                "seed_rates":rates.tolist(),"ci95":confidence_interval(rates),
                "functional_recovery_time":median_event(subset,"functional_recovery_time_steps"),
                "mass_recovery_time":median_event(subset,"mass_recovery_time_steps"),
                "per_intervention":[{"intervention":name,
                    "functional_recovery_rate":float(np.mean([r["functionally_recovered"] for r in subset if r["intervention"]==name])),
                    "survival_rate":float(np.mean([r["survived"] for r in subset if r["intervention"]==name])),
                    "functional_recovery_time":median_event([r for r in subset if r["intervention"]==name],"functional_recovery_time_steps")}
                    for name in HELDOUT]})
        for comparator in ("performance","direct-regeneration"):
            delta=seed_rates["map-elites"]-seed_rates[comparator]
            seed_differences[comparator].append(delta)
            contrasts.append({"species":species,"contrast":f"map-elites minus {comparator}",
                              "mean_difference":float(np.mean(delta)),"ci95":confidence_interval(delta)})
        common=sorted(set.intersection(*(valid_seeds[method] for method in METHODS)))
        indices=[seeds.index(seed) for seed in common]
        common_valid_sensitivity.append({"species":species,"common_valid_search_seeds":common,"n":len(common),
            "method_rates":{method:float(np.mean(seed_rates[method][indices])) if indices else None for method in METHODS},
            "status":"post-selection sensitivity only; excluding unstable selections changes the population"})
    aggregate=[]
    for comparator,by_species in seed_differences.items():
        if by_species:
            delta=np.mean(by_species,axis=0)
            aggregate.append({"contrast":f"map-elites minus {comparator}","mean_difference":float(np.mean(delta)),
                              "ci95":confidence_interval(delta),"eligible_species":len(by_species)})
    overall=[]
    for method in METHODS:
        group=[row for row in summaries if row.get("method")==method]
        overall.append({"method":method,"mean_species_success_rate":float(np.mean([r["functional_recovery_rate"] for r in group])),
            "mean_species_control_validity":float(np.mean([r["control_validity"] for r in group])),
            "mean_species_survival":float(np.mean([r["survival_rate"] for r in group]))})
    result={"schema":"genesis.charter-analysis/v1","protocol_sha256":complete["protocol_sha256"],
        "scope":"64 evaluations per method per seed, inherited anatomy and two growth parameters within each species",
        "independent_unit":"search seed within species; interventions repeated within seed",
        "uncertainty":"descriptive95% paired seed-bootstrap intervals; species contrasts not multiplicity-adjusted; not a broad superiority claim",
        "survival_definition":"mass>=1 at final observation; no viable selected specimen is scored as failure",
        "primary_estimand":"selection-pipeline success yield: valid undamaged control AND functional recovery; invalid controls count as failed selection, not demonstrated injury susceptibility",
        "time_definition":"first3-sample sustained return; medians among events with censored counts; later relapse possible",
        "summary":summaries,"overall":overall,"species_contrasts":contrasts,"aggregate_contrasts":aggregate,
        "common_valid_sensitivity":common_valid_sensitivity}
    save(root/"analysis.json",result)
    save(Path("web/data/charter-analysis.json"),result)
    save(Path("web/data/charter-trials.json"),{"protocol_sha256":complete["protocol_sha256"],
                                             "selected_genomes":selected_genomes,"trials":public_trials})
    for row in aggregate:
        print(f"{row['contrast']}: {row['mean_difference']:+.3%};95%CI {row['ci95']}")


if __name__=="__main__":
    main()
