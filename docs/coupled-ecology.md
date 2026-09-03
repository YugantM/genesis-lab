# Coupled Ecology Experiment 003

## Decision

**Share with caveats.** The experiment establishes a reproducible minimal
interaction model for real continuous Lenia organisms. It does not establish a
general law of ecology, and the apparent ecological advantage of Genesis 001
did not pass held-out validation.

## Question

Can two independently viable continuous organisms preserve both forms when a
symmetric local competition term makes them share space?

## Model

Each organism occupies its own continuous channel and retains its native Lenia
kernel and Gaussian growth parameters. The coupled update is:

`growth_i = native_growth_i - competition × kernel(other_channel)`

The interaction is symmetric and contains no handcrafted winner, resource,
predation, reproduction, or species-specific coefficient. It should be read as
a minimal space-pressure experiment—not as a biological ecology model.

## Discovery protocol

- As of: 2026-09-03 UTC
- Population: Genesis 001 plus eight pinned compatible catalogue species
- Pairings: all 36 unordered pairs
- Encounter geometry: near, contact, and overlapping starts
- Orientations: 0° and 90°
- Noise seeds: 0 and 1, sigma 0.001 on active cells
- Competition strengths: 0.25, 0.5, and 1.0
- Trials: 1,296, with 432 at each strength
- Controls: the exact two initial channels evolved independently at zero coupling
- Coexistence: both channels retain 0.5–1.5× control mass, 0.5–2.0× control
  occupied area, no more than 15% world occupancy, and at least 0.5 maximum
  translation-aligned cosine similarity to their matched controls

All 1,296 matched controls were valid.

## Discovery results

| Competition | Coexistence | Dominance | Mutual collapse |
| --- | ---: | ---: | ---: |
| 0.25 | 369 / 432 (85.4%) | 49 | 14 |
| 0.50 | 275 / 432 (63.7%) | 87 | 70 |
| 1.00 | 194 / 432 (44.9%) | 111 | 127 |

At competition 1.0, coexistence was 84.7% for near starts, 41.7% at
contact, and 8.3% for overlapping starts. The distance gradient is therefore
much larger than any apparent single-lineage advantage.

## Candidate and held-out validation

During discovery, Genesis 001 appeared to coexist more often than its Orbium
parent against identical third-species opponents. The promotion gate was fixed
before validation: at least +5 percentage points, at least twice as many
Genesis-only as parent-only paired wins, and two-sided exact paired p ≤ 0.05.

The held-out run changed the developmental phase from 240 to 300, orientations
from 0°/90° to 180°/270°, noise seeds from 0/1 to 2/3, and the random seed base.
It produced another 1,296 trials and 252 direct matched Genesis-versus-parent
contexts.

- Genesis 001 coexistence: 226 / 252 (89.7%)
- Orbium parent coexistence: 221 / 252 (87.7%)
- Difference: +2.0 percentage points
- Discordant outcomes: Genesis-only 5, parent-only 0
- Two-sided exact paired p: 0.0625
- Promotion decision: **failed**

The candidate remains an observation. The evidence does not support claiming
that Genesis 001 has a general ecological-persistence advantage.

## Reproducibility

- Discovery implementation: `genesis/coupled_ecology.py`
- Validation implementation: `genesis/validate_ecology.py`
- Discovery record: `runs/coupled-ecology.json`
- Validation record: `runs/coupled-ecology-validation.json`
- Integrity checks: `tests/test_ecology_records.py`

Run both records from the repository root:

```bash
python3 -m genesis.coupled_ecology
python3 -m genesis.validate_ecology
python3 -m unittest discover -s tests -v
```

## Required caveats

- Trials share organisms, parameters, and deterministic structures, so the
  1,296 rows are not 1,296 fully independent biological observations.
- Competition is an explicit experimental term chosen by this project.
- Only nine related classic single-kernel Lenia forms were studied.
- Pairwise persistence does not demonstrate reproduction, heredity, food webs,
  or open-ended evolution.
