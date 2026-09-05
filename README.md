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
python3 -m genesis.dashboard_server
```

Then open <http://127.0.0.1:8765>. The local server batches all nine Lenia
worlds through MLX on the Apple GPU and streams a single rendered atlas to the
browser. Static hosting still uses the automatic WebGL fallback.

The observatory leads with the rejected eight-species transfer hypothesis and
the limits of the current damage search. Evidence records precede the live view.
The optional shared ecosystem is collapsed and idle until opened.

The isolation laboratory demonstrates the controlled cellular dynamics.
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

Replicate the sweep across 20 independent noisy initial conditions, with balanced
injury phases from 180 to 500 steps. Rotations are not counted as replicates:

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

The search trains on 4/6/8% damage and validates on 5/7/10/12%, using separate
initial-state seeds and phase ranges. Functional recovery measures matched-control
mass and motion; translation-aligned anatomy is secondary. Strict mass, shape,
and functional recovery times are reported separately. Some validation magnitudes
were explored historically, so this is not a wholly unseen injury family.

The corrected candidate 806 recovered at 5% and 7%, collapsed in all 10% trials,
and recovered in 1 of 32 repeated contexts at 12%. Its 50.8% average failed the
65% promotion gate. These 32 contexts contain eight independent initial-state
seeds with four repeated phases, not 32 independent replicates.

Run the charter coverage gate, then the three-method comparison (the latter
refuses to run without a passing matching coverage record):

```bash
python3 -m genesis.selection_comparison --stage coverage
python3 -m genesis.selection_comparison --stage comparison
python3 -m genesis.analyze_comparison
python3 -m genesis.export_research
```

The locked [comparison protocol](docs/selection-comparison-protocol.md) uses
20 independent search seeds and 64 candidate evaluations per method within each
of eight species. Per-species checkpoint files support resuming interrupted runs.
The [metric audit](docs/recovery-metric-audit.md) defines the outcomes and limits.

The first full comparison is complete: mean success was 32.4% for performance,
40.6% for MAP-Elites, and 48.4% for direct regeneration. Success includes valid
undamaged controls; these rates combine selection stability and injury recovery.
See the [full research update](docs/research-update-2026-09-05.md) for species
breakdowns, control failures, uncertainty, and the unequal training-compute caveat.

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

Keep both bodies fixed and evolve pressure sensitivity, costly boundary
response, and memory of recent contact:

```bash
python3 -m genesis.evolve_interactions --candidates 1024
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
- `genesis/interaction_ecology.py` — evolvable inter-species boundary dynamics
- `genesis/evolve_interactions.py` — held-out interaction-genome search
- `genesis/dashboard_server.py` — local MLX simulation and streamed PNG renderer
- `tests/` — invariants for the scientific core
- `web/` — nine-habitat WebGL artificial-life observatory
- `docs/research-charter.md` — hypothesis, controls, and research gates

## Verify

```bash
python3 -m unittest discover -s tests -v
```

Genesis is an experiment platform, not evidence by itself. Every future gallery
specimen must retain its genome, seed, simulator version, and evaluation record.
