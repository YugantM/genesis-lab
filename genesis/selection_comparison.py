"""Gated, within-species comparison of selection methods on Apple Silicon.

Coverage runs before any held-out comparison. Both stages write their full
protocol before evaluation and retain per-species, per-seed results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .core import GenesisWorld
from .metrics import (
    aligned_similarity, center_of_mass, functional_recovery, functional_score, functional_recovery_time,
    mass, occupied_fraction, recovery_time, toroidal_displacement,
)
from .multispecies_benchmark import load_manifest
from .robust_search import calibrated_damage, make_world

METHODS = ("performance", "direct-regeneration", "map-elites")
TRAIN_DAMAGE = (0.04, 0.06, 0.08)
HELDOUT = ("central-5", "central-7", "central-10", "central-12", "repeated-7", "fragmentation", "communication-noise", "kernel-radius", "diffusion", "moving-resource")


def save(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def confidence_interval(values, seed=84321) -> list[float]:
    """Seed-level bootstrap; never resample the within-seed injury trials."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    estimates = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def cell_for(mass_ratio: float, motion_ratio: float) -> tuple[int, int]:
    # Fixed before running: bounded 8x8 cells, overflow clipped and reported.
    return (int(np.clip((mass_ratio - 0.5) / 1.5 * 8, 0, 7)),
            int(np.clip(motion_ratio / 2 * 8, 0, 7)))


def update_archive(archive: dict, row: dict) -> None:
    if not row["valid"]:
        return
    cell = tuple(row["cell"])
    if cell not in archive or row["performance"] > archive[cell]["performance"]:
        archive[cell] = row


def perturbed_state(template, seed: int, sigma: float):
    state = template.numpy()[0].copy()
    active = state > 0
    rng = np.random.default_rng(seed)
    state[active] = np.clip(state[active] + rng.normal(0, sigma, int(active.sum())), 0, 1)
    return state


def evolve_measure(template, states, parameters, steps=300):
    world = make_world(template, states, parameters[:, 0], parameters[:, 1])
    world.step(steps - 50)
    earlier = center_of_mass(world.numpy())
    world.step(50)
    final = world.numpy().copy()
    motion = np.linalg.norm(toroidal_displacement(earlier, center_of_mass(final), template.config.size), axis=1)
    return final, mass(final), occupied_fraction(final), motion


def evaluate(template, states, parameters, parent_mass, parent_motion, parent_valid, *, damage=False):
    pre, pre_mass, pre_area, pre_motion = evolve_measure(template, states, parameters)
    final, final_mass, final_area, final_motion = evolve_measure(template, pre, parameters)
    retention = final_mass / np.maximum(pre_mass, 1e-12)
    relative_mass = final_mass / np.maximum(parent_mass, 1e-12)
    relative_motion = final_motion / np.maximum(parent_motion, 1.0)
    valid = (parent_valid & (pre_mass >= 10) & (final_mass >= 10) & (pre_area <= .15) & (final_area <= .15)
             & (retention >= .75) & (retention <= 1.25)
             & (relative_mass >= .5) & (relative_mass <= 2.0))
    performance = np.where(valid, np.minimum(retention, 1 / np.maximum(retention, 1e-12))
                           * (.5 + .5 * np.minimum(relative_motion, 2) / 2), 0)
    recovery_scores = np.zeros(len(states))
    if damage:
        injured = []
        for state in pre:
            for magnitude in TRAIN_DAMAGE:
                injured.append(calibrated_damage(state, center_of_mass(state[None])[0], magnitude)[0])
        injured_final, injured_mass, injured_area, injured_motion = evolve_measure(
            template, np.stack(injured), np.repeat(parameters, len(TRAIN_DAMAGE), axis=0))
        for i in range(len(states)):
            scores = []
            for offset in range(len(TRAIN_DAMAGE)):
                j = i * len(TRAIN_DAMAGE) + offset
                motion_ratio = injured_motion[j] / max(final_motion[i], 1e-12) if final_motion[i] >= 1 else 1.0
                scores.append(functional_score(float(injured_mass[j] / max(final_mass[i], 1e-12)),
                    float(motion_ratio), bool(valid[i] and injured_area[j] <= .15)))
            recovery_scores[i] = np.mean(scores)
    return [{
        "parameters": parameters[i].tolist(), "valid": bool(valid[i]),
        "performance": float(performance[i]), "training_recovery_score": float(recovery_scores[i]),
        "cell": list(cell_for(float(relative_mass[i]), float(relative_motion[i]))),
        "parent_relative_mass": float(relative_mass[i]), "parent_relative_motion": float(relative_motion[i]),
        "descriptor_clipped": bool(relative_mass[i] < .5 or relative_mass[i] >= 2 or relative_motion[i] >= 2),
    } for i in range(len(states))]


def search_species(specimen, species_index, methods, protocol):
    seeds = protocol["search_seeds"]
    population = protocol["population"]
    budget = protocol["evaluations_per_method_per_seed"]
    template = GenesisWorld.from_specimen(specimen)
    parent = np.array([specimen.parameters["growth_center"], specimen.parameters["growth_width"]])
    starts = np.stack([perturbed_state(template, 510000 + species_index*10000 + seed, .012) for seed in seeds])
    parent_params = np.repeat(parent[None], len(seeds), axis=0)
    parent_pre, parent_pre_mass, parent_pre_area, _ = evolve_measure(template, starts, parent_params)
    _, parent_mass, parent_area, parent_motion = evolve_measure(template, parent_pre, parent_params)
    parental_retention = parent_mass / np.maximum(parent_pre_mass, 1e-12)
    parent_valid = ((parent_pre_mass >=10) & (parent_mass >=10) & (parent_pre_area<=.15) & (parent_area<=.15)
                    & (parental_retention>=.75) & (parental_retention<=1.25))
    results = []
    for method in methods:
        # Same initial population and RNG stream per seed; divergence follows selection.
        rngs = [np.random.default_rng(610000 + species_index*10000 + seed) for seed in seeds]
        archives = [{} for _ in seeds]
        history = [[] for _ in seeds]
        before = time.monotonic()
        for generation in range(budget // population):
            params = []
            for i, rng in enumerate(rngs):
                if generation == 0 or method == "random":
                    candidates = parent * np.column_stack((rng.uniform(.8, 1.2, population), rng.uniform(.5, 1.5, population)))
                    if generation == 0:
                        candidates[0] = parent
                else:
                    if method == "map-elites":
                        pool = list(archives[i].values())
                    else:
                        key = "performance" if method == "performance" else "training_recovery_score"
                        pool = sorted(history[i], key=lambda row: row[key], reverse=True)[:4]
                    if not pool:
                        pool = [{"parameters": parent.tolist()}]
                    chosen = np.array([pool[int(rng.integers(len(pool)))]["parameters"] for _ in range(population)])
                    candidates = chosen * np.exp(rng.normal(0, [.025, .10], (population, 2)))
                    candidates = np.clip(candidates, parent * [.8, .5], parent * [1.2, 1.5])
                params.extend(candidates)
            measured = evaluate(template, np.repeat(starts, population, axis=0), np.asarray(params, np.float32),
                np.repeat(parent_mass, population), np.repeat(parent_motion, population), np.repeat(parent_valid,population), damage=method == "direct-regeneration")
            for i in range(len(seeds)):
                for offset, row in enumerate(measured[i*population:(i+1)*population]):
                    row["evaluation"] = generation * population + offset
                    history[i].append(row)
                    update_archive(archives[i], row)
        elapsed = time.monotonic() - before
        for i, seed in enumerate(seeds):
            if method == "map-elites":
                # Uniform cell choice is fixed before any held-out observations.
                pool = sorted(archives[i].values(), key=lambda row: row["cell"])
                pick = np.random.default_rng(710000 + species_index*10000 + seed)
                champion = pool[int(pick.integers(len(pool)))] if pool else None
            else:
                key = "training_recovery_score" if method == "direct-regeneration" else "performance"
                viable = [row for row in history[i] if row["valid"]]
                champion = max(viable, key=lambda row: row[key]) if viable else None
            results.append({"seed": seed, "method": method, "coverage": len(archives[i]),
                "parent_control_valid":bool(parent_valid[i]),
                "evaluations": len(history[i]), "viable_candidates": sum(row["valid"] for row in history[i]),
                "champion": champion, "archive": list(archives[i].values()), "evaluated_candidates": history[i],
                "method_wall_seconds_all_seeds": elapsed,
                "simulated_world_steps_per_seed": budget * (1500 if method == "direct-regeneration" else 600)})
        print(f"{specimen.name}: {method} mean coverage {np.mean([len(a) for a in archives]):.2f}, {elapsed:.1f}s", flush=True)
    return results


def intervention(world, name, elapsed, rng):
    """Persistent interventions apply to the injured trial only."""
    import mlx.core as mx
    if name == "communication-noise":
        neighborhood = mx.fft.ifft2(mx.fft.fft2(world.state)*world._kernel_fft).real
        sensed = np.asarray(neighborhood).copy()
        active = sensed > 1e-6
        sensed[active] += rng.normal(0,.015,int(active.sum()))
        growth=2*mx.exp(-((mx.array(sensed)-world._growth_center)**2)/(2*world._growth_width**2))-1
        world.state=mx.clip(world.state+world.config.dt*growth,0,1)
        mx.eval(world.state)
        return
    world.step()
    if name == "diffusion":
        state = world.numpy()
        neighbors = sum(np.roll(state, shift, axis) for shift, axis in ((1,-1),(-1,-1),(1,-2),(-1,-2))) / 4
        world.state = mx.array(.9 * state + .1 * neighbors)
    elif name == "moving-resource":
        # An imposed moving growth modifier, not a resource-aware organism model.
        state = world.numpy()
        axis = np.arange(state.shape[-1])
        modifier = .002 * np.cos(2*np.pi*(axis - elapsed*.5)/state.shape[-1])
        world.state = mx.array(np.clip(state + modifier[None, None, :] * (state > 0), 0, 1))


def heldout_species(specimen, species_index, searches, protocol):
    import mlx.core as mx
    from .core import _kernel
    template = GenesisWorld.from_specimen(specimen)
    rows = []
    for result in searches:
        if result["method"] not in METHODS:
            continue
        champion = result["champion"]
        if champion is None:
            for name in HELDOUT:
                rows.append({"seed": result["seed"], "method": result["method"], "intervention": name,
                             "control_valid": False, "functionally_recovered": False, "survived": False,
                             "no_viable_selection": True, "mass_recovery_time_steps": None,"functional_recovery_time_steps":None})
            continue
        seed = result["seed"]
        phase = (220, 360, 500, 640)[seed % 4]
        start = perturbed_state(template, 910000 + species_index*10000 + seed, .015)
        parameters = np.array([champion["parameters"]], np.float32)
        pre, pre_mass, _, pre_motion = evolve_measure(template, start[None], parameters, phase)
        ctrl_params = np.repeat(parameters, len(HELDOUT), axis=0)
        control = make_world(template, np.repeat(pre, len(HELDOUT), axis=0), ctrl_params[:,0], ctrl_params[:,1])
        damaged = []
        actuals = []
        centre = center_of_mass(pre)[0]
        for name in HELDOUT:
            state = pre[0].copy()
            removed = 0.
            if name.startswith("central-") or name.startswith("repeated-"):
                state, _, removed = calibrated_damage(state, centre, int(name.split("-")[1])/100)
            elif name == "fragmentation":
                yy, xx = np.indices(state.shape)
                deltas = toroidal_displacement(centre, np.stack((yy,xx), axis=-1), state.shape[0])
                state[np.abs(deltas[...,1]) <= 1.5] = 0
                removed = float(1 - state.sum()/max(pre_mass[0], 1e-12))
            damaged.append(state)
            actuals.append(removed)
        worlds = [make_world(template, state[None], parameters[:,0], parameters[:,1]) for state in damaged]
        radius_index = HELDOUT.index("kernel-radius")
        config = type(template.config)(**{**template.config.__dict__, "radius":template.config.radius*1.1})
        worlds[radius_index]._kernel_fft = mx.fft.fft2(_kernel(config))
        samples = [[] for _ in HELDOUT]
        positions = {}
        control_positions = {}
        noise_rngs = [np.random.default_rng(1010000 + species_index*10000 + seed*20+i) for i in range(len(HELDOUT))]
        for step in range(301):
            if step == 150:
                i = HELDOUT.index("repeated-7")
                state = worlds[i].numpy()[0]
                wounded, _, _ = calibrated_damage(state, center_of_mass(state[None])[0], .07)
                worlds[i].state = mx.array(wounded[None])
            if step % 10 == 0:
                controls = control.numpy()
                states = np.stack([world.numpy()[0] for world in worlds])
                positions[step] = center_of_mass(states)
                control_positions[step] = center_of_mass(controls)
                masses, cmasses = mass(states), mass(controls)
                areas, careas = occupied_fraction(states), occupied_fraction(controls)
                for i in range(len(HELDOUT)):
                    speed = cspeed = None
                    if step >= 50:
                        speed = float(np.linalg.norm(toroidal_displacement(positions[step-50][i], positions[step][i],128)))
                        cspeed = float(np.linalg.norm(toroidal_displacement(control_positions[step-50][i],control_positions[step][i],128)))
                    samples[i].append({"step":step, "mass":float(masses[i]), "control_mass":float(cmasses[i]),
                        "occupied":float(areas[i]),"control_occupied":float(careas[i]),"motion":speed,"control_motion":cspeed})
            if step < 300:
                control.step()
                for i, world in enumerate(worlds):
                    intervention(world, HELDOUT[i], step+1, noise_rngs[i])
        controls = control.numpy()
        for i, name in enumerate(HELDOUT):
            final = samples[i][-1]
            valid = bool(pre_mass[0]>=10 and final["control_mass"]>=10 and .75<=final["control_mass"]/max(pre_mass[0],1e-12)<=1.25 and final["control_occupied"]<=.15)
            mass_ratio = final["mass"]/max(final["control_mass"],1e-12)
            motion_ratio = final["motion"]/max(final["control_motion"],1e-12) if final["control_motion"]>=1 else 1.
            times = [sample for sample in samples[i] if sample["step"] >= (150 if name=="repeated-7" else 0)]
            recovered = functional_recovery(control_valid=valid,mass_ratio=mass_ratio,motion_ratio=motion_ratio,occupied=final["occupied"],maximum_occupied=.15)
            rolling=[s for s in times if s["step"] >= (200 if name=="repeated-7" else 50)]
            function_time=functional_recovery_time(
                [s["step"]-(150 if name=="repeated-7" else 0) for s in rolling],
                [s["mass"]/max(s["control_mass"],1e-12) for s in rolling],
                [s["motion"]/max(s["control_motion"],1e-12) if s["control_motion"]>=1 else 1. for s in rolling],
                [s["occupied"] for s in rolling],
                [valid and s["control_mass"]>=10 and s["control_occupied"]<=.15 for s in rolling],maximum_occupied=.15)
            rows.append({"seed":seed,"method":result["method"],"intervention":name,"phase":phase,
                "control_valid":valid,"functionally_recovered":recovered,"survived":final["mass"]>=1,
                "actual_removed_fraction":actuals[i],"mass_ratio_to_control":mass_ratio,
                "motion_ratio_to_control":motion_ratio,"motion_ratio_to_pre_injury":float(final["motion"]/max(pre_motion[0],1e-12)),
                "aligned_similarity_to_control":aligned_similarity(worlds[i].numpy()[0],controls[i]),
                "functional_recovery_time_steps":function_time if valid else None,
                "mass_recovery_time_steps":recovery_time([s["step"]-(150 if name=="repeated-7" else 0) for s in times],
                    [s["mass"] for s in times],[s["control_mass"] for s in times]) if valid else None,
                "trajectory":samples[i]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("coverage", "comparison"), default="coverage")
    parser.add_argument("--manifest", default="web/specimens/species-benchmark.json")
    parser.add_argument("--out-dir", default="runs/charter-comparison")
    parser.add_argument("--evaluations", type=int, default=64)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    if args.seeds < 20 or args.population < 1 or args.evaluations < args.population or args.evaluations % args.population:
        parser.error("requires >=20 seeds and an evaluation budget divisible by population")
    catalogue = load_manifest(Path(args.manifest))
    out = Path(args.out_dir)
    protocol = {"schema":"genesis.charter-comparison-protocol/v1", "search_seeds":list(range(args.seeds)),
        "source_sha256":{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ("selection_comparison.py","core.py","metrics.py","robust_search.py")},
        "evaluations_per_method_per_seed":args.evaluations,"population":args.population,
        "species":[entry["code"] for entry,_ in catalogue],
        "specimen_sha256":{entry["code"]:hashlib.sha256((Path(args.manifest).parent/entry["path"]).read_bytes()).hexdigest() for entry,_ in catalogue},
        "train_damage":list(TRAIN_DAMAGE),"heldout_interventions":list(HELDOUT),
        "independent_unit":"search seed within species; ten heldout interventions are repeated measurements",
        "initial_condition_noise_sigma":{"train":.012,"heldout":.015},
        "search_space":"native per-species growth centre x[0.8,1.2], growth width x[0.5,1.5]; fixed inherited anatomy",
        "descriptors":"fixed8x8 parent-relative final mass[0.5,2] and50-step displacement[0,2]; clipped overflow",
        "map_quality":"undamaged performance; uniform archive-cell reproduction and uniform final archive-cell choice",
        "performance_quality":"valid bounded persistence x(0.5+0.5*capped parent-relative motion/2)",
        "direct_quality":"mean mass closeness x retained motion across4/6/8%; no shape reward",
        "budget":"equal candidate evaluations; coverage also equal simulated world steps; direct selection has explicitly reported extra injury compute",
        "coverage_gate":"mean species-stratified paired seed coverage difference lower95% bootstrap bound>0",
        "gate_scope":"coverage gate precedes heldout comparison; small fixed budget, no scientific novelty claim",
        "heldout_scope":"fresh initial-state seeds and specified interventions; some injury magnitudes have historical exploratory exposure",
        "heldout_definition":"300 steps; repeated7% at0 and150; fragmentation3-cell central vertical gap; sensed-neighborhood noise sigma.015 eachstep; radius+10%; diffusion10%4-neighbor mixing eachstep; moving imposed growth field amplitude.002 cosine speed.5cells/step (no explicit resource uptake)",
        "species_eligibility":"at least90% valid parental controls; excluded species retained in diagnostics",
        "analysis":"primary functional recovery and survival averaged within seed, species reported separately; anatomy secondary; no tuning after results"}
    protocol_path = out/"protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())["protocol"] != protocol:
        raise RuntimeError("existing locked protocol differs; use a distinct output directory")
    if not protocol_path.exists():
        save(protocol_path,{"locked_at":datetime.now(timezone.utc).isoformat(),"sha256":digest(protocol),"protocol":protocol})
    if args.stage == "comparison":
        gate = json.loads((out/"coverage.json").read_text())
        if gate["protocol_sha256"] != digest(protocol) or not gate["gate_passed"]:
            raise RuntimeError("coverage gate has not passed for this protocol; heldout comparison remains locked")
    species_results = []
    for species_index,(entry,specimen) in enumerate(catalogue):
        path = out/f"{args.stage}-{entry['code']}.json"
        if path.exists():
            cached=json.loads(path.read_text())
            if cached["protocol_sha256"]!=digest(protocol):
                raise RuntimeError("cached species record does not match protocol")
            species_results.append(cached)
            continue
        methods=("map-elites","random") if args.stage=="coverage" else METHODS
        searches=search_species(specimen,species_index,methods,protocol)
        result={"species":entry["code"],"protocol_sha256":digest(protocol),"searches":searches}
        if args.stage=="comparison":
            result["parent_control_validity"]=float(np.mean([r["parent_control_valid"] for r in searches]))
            result["eligible"]=result["parent_control_validity"]>=.9
            result["heldout"]=heldout_species(specimen,species_index,searches,protocol) if result["eligible"] else []
        save(path,result)
        species_results.append(result)
    if args.stage=="coverage":
        per_species=[]
        for record in species_results:
            lookup={(row["seed"],row["method"]):row["coverage"] for row in record["searches"]}
            delta=[lookup[(seed,"map-elites")]-lookup[(seed,"random")] for seed in protocol["search_seeds"]]
            parent_rate=float(np.mean([r["parent_control_valid"] for r in record["searches"] if r["method"]=="random"]))
            per_species.append({"species":record["species"],"eligible":parent_rate>=.9,"parent_control_validity":parent_rate,"seed_differences":delta,"mean_difference":float(np.mean(delta)),"ci95":confidence_interval(delta)})
        # Same seed index across species is resampled as a block, conservatively.
        eligible=[row for row in per_species if row["eligible"]]
        differences=np.mean([row["seed_differences"] for row in eligible],axis=0) if eligible else np.zeros(args.seeds)
        interval=confidence_interval(differences)
        result={"protocol_sha256":digest(protocol),"gate_passed":bool(eligible) and interval[0]>0,
            "mean_coverage_difference":float(np.mean(differences)),"ci95":interval,"per_species":per_species}
    else:
        summaries=[]
        for record in species_results:
            if not record["eligible"]:
                summaries.append({"species":record["species"],"excluded":True,"parent_control_validity":record["parent_control_validity"]})
                continue
            for method in METHODS:
                rows=[r for r in record["heldout"] if r["method"]==method]
                seed_rates=[float(np.mean([r["functionally_recovered"] for r in rows if r["seed"]==seed])) for seed in protocol["search_seeds"]]
                summaries.append({"species":record["species"],"method":method,"mean_seed_recovery":float(np.mean(seed_rates)),
                    "ci95":confidence_interval(seed_rates),"seed_rates":seed_rates,
                    "control_validity":float(np.mean([r["control_valid"] for r in rows])),
                    "per_intervention":[{"intervention":name,"recovery_rate":float(np.mean([r["functionally_recovered"] for r in rows if r["intervention"]==name]))} for name in HELDOUT]})
        result={"protocol_sha256":digest(protocol),"independent_search_seeds_per_method_per_species":args.seeds,"summary":summaries}
    save(out/f"{args.stage}.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
