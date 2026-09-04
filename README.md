# Genesis Lab

An open laboratory for discovering robust, self-organising systems on Apple
Silicon.

**Live observatory:** <https://yugantm.github.io/genesis-lab/>

Genesis begins with a continuous cellular universe inspired by Lenia. The
research core runs batched simulations in MLX; the Genesis Zoo is a
dependency-free WebGL observatory where nine validated species live side by
side and can be inspected or injured without leaving the page.

## First run

```bash
cd genesis-lab
python3 -m genesis.benchmark
python3 -m http.server 8000 -d web
```

Then open <http://localhost:8000>. The first instrument is a shared ecosystem:
nine founding lineages seek renewable energy, compete for space, reproduce,
inherit mutated traits, and occasionally split into new lineages. Select a
lifeform to inspect it, create a nutrient bloom, or apply a climate shock.

The isolation laboratory underneath preserves the controlled experiment.
Select any habitat to inspect its native growth rule, apply an exact 5% central
injury to one creature or the full population, pause time, and restore every
genome to its recorded initial state.

The common-world ecosystem is an exploratory organism-level model, not an
extension of the validated Lenia evidence. That boundary is shown in the UI:
claims from the ecology layer must be reproduced in the MLX laboratory before
they enter the research ledger.

The browser renderer advances the nine WebGL habitats incrementally and
suspends both simulations while their canvases are off-screen. This keeps the
observatory responsive without changing the recorded MLX experiments.

Run the calibrated Specimen Zero experiment with an undamaged control:

```bash
python3 -m genesis.specimen_zero
```

Run the automated mass-calibrated damage sweep:

```bash
python3 -m genesis.damage_sweep
```

Replicate the sweep across organism phases and rotations:

```bash
python3 -m genesis.replicated_sweep
```

Map anatomy, search robust parameter mutants, select across multiple conditions,
and run the locked held-out validation:

```bash
python3 -m genesis.lesion_map
python3 -m genesis.robust_search
python3 -m genesis.select_candidate
python3 -m genesis.validate_mutant
```

Import the pinned eight-species panel and run the paired cross-species transfer
benchmark:

```bash
python3 -m genesis.import_species
python3 -m genesis.multispecies_benchmark
```

Run the real two-channel ecology experiment and its separately held-out
validation:

```bash
python3 -m genesis.coupled_ecology
python3 -m genesis.validate_ecology
```

Run the first paired co-evolution search against the hardest reproducible pair:

```bash
python3 -m genesis.evolve_coexistence --candidates 2048
```

The benchmark removes exactly 5% of organism mass at core, leading, and
trailing anatomical anchors. Every injured trial is compared with an undamaged
control sharing species, parameters, phase, rotation, and noise seed.

## Research premise

The first falsifiable question is:

> Does selection for behavioural diversity produce systems that recover from
> unseen damage better than systems selected directly for regeneration?

The project will compare performance selection, explicit regeneration
selection, and MAP-Elites under identical held-out damage protocols.

## Layout

- `genesis/core.py` — batched MLX continuous cellular automaton
- `genesis/metrics.py` — deterministic behavioural measurements
- `genesis/benchmark.py` — hardware sanity check and throughput benchmark
- `genesis/multispecies_benchmark.py` — normalized cross-species injury test
- `genesis/coupled_ecology.py` — real two-channel Lenia competition experiment
- `genesis/validate_ecology.py` — held-out ecological candidate validation
- `genesis/evolve_coexistence.py` — paired growth-rule search with promotion gate
- `tests/` — invariants for the scientific core
- `web/` — nine-habitat WebGL artificial-life observatory
- `docs/research-charter.md` — hypothesis, controls, and research gates

## Verify

```bash
python3 -m unittest discover -s tests -v
```

Genesis is an experiment platform, not evidence by itself. Every future gallery
specimen must retain its genome, seed, simulator version, and evaluation record.
