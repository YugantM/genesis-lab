# Interaction Evolution Run 002

## Decision

**No promotion.** An evolvable boundary response improved the difficult pair in
some contexts, but the training-selected candidate did not generalize strongly
enough to qualify as a robust social trait.

## Question

Can Synorbium ignis (O4i) and Paraptera cavus labens (P4cl) improve coexistence
by changing only how they respond to one another, while their bodies and native
Lenia growth rules remain fixed?

## Interaction model

Each organism receives three bounded genes:

- pressure sensitivity: 0.72–1.28
- boundary response: 0–0.65
- signal memory: 0–0.95

The other organism's local convolution field acts as a signal. A stateful trace
retains recent signal, so protection can persist briefly after contact.
Boundary response attenuates incoming pressure in proportion to this trace and
pays a maintenance cost of 0.035 against the organism's own active mass, with
an additional memory multiplier. Protection is therefore not free.

## Locked protocol

- 1,024 deterministic candidates; seed 20260905
- four training encounters at phase 240
- eight held-out encounters at phase 300
- contact and overlap geometry only
- held-out orientations, noise seeds, phase, and seed base
- identical perturbation within a context for every candidate
- isolated identity gate: 75–125% parental mass and area, with at least 0.5
  aligned similarity
- promotion: at least 3/4 training coexistence, 4/8 held-out coexistence,
  +2/8 improvement over the held-out parent, and preserved identity

## Result

- identity-safe: 1,024 / 1,024
- parental training baseline: 0 / 4
- candidate 717 training: 2 / 4
- parental held-out baseline: 1 / 8
- candidate 717 held-out: 3 / 8
- held-out improvement: +2 / 8, or 25 percentage points
- held-out contact: 0 / 4 baseline → 2 / 4 candidate
- held-out overlap: 1 / 4 baseline → 1 / 4 candidate
- promotion: failed

Candidate 717 evolved the following interaction genes:

| Organism | Pressure sensitivity | Boundary response | Signal memory |
|---|---:|---:|---:|
| O4i | 0.8897 | 0.1210 | 0.6142 |
| P4cl | 1.0226 | 0.3563 | 0.3914 |

The candidate selected on training data was evaluated once on held-out data;
held-out outcomes were not used to reselect a different genome.

## Interpretation

Local sensing and memory changed outcomes without changing anatomy, confirming
that the interaction layer is operational. The improvement was specific:
memory rescued two contact cases and no additional overlap cases. Protection
appears useful while boundaries remain legible, but not after bodies deeply
interpenetrate. The next model should add active avoidance or a multi-species
community context while retaining the same matched-noise and identity controls.

## Reproduce

```bash
python3 -m genesis.evolve_interactions --candidates 1024
python3 -m unittest discover -s tests -v
```
