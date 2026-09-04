# Coexistence Evolution Run 001

## Decision

**No promotion.** A deterministic 2,048-genome paired search found local
improvements, but none met the precommitted training gate for robust
coexistence.

## Target

Synorbium ignis (O4i) and Paraptera cavus labens (P4cl) were selected before
search because they produced 16.7% coexistence at high pressure in both the
discovery and held-out pairwise experiments. The evolutionary task used only
the harder contact and overlap geometries.

## Search

- Four mutable values: growth centre and width for each organism
- Candidates: 2,048, deterministic seed 20260904
- Competition: 1.0
- Training phase: 240
- Training contexts: four combinations of geometry, orientation, and noise
- Held-out phase: 300
- Held-out contexts: eight unseen combinations
- Solo identity: each organism must retain 75–125% parental mass and occupied
  area and at least 0.5 translation-aligned similarity
- Promotion: at least 70% training coexistence, 50% held-out coexistence,
  +25 percentage points over held-out parents, and both solo identities valid

## Result

- Solo-identity-safe candidates: 77 / 2,048
- Parental training baseline: 0 / 4
- Best candidate training: 1 / 4
- Parental held-out baseline: 0 / 8
- Best candidate held-out: 2 / 8
- Held-out improvement: +25 percentage points
- Promotion: failed because training performance was below 70%

The selected candidate kept Synorbium at 103.2% of parental mass and
Paraptera at 90.6% during training. In the held-out developmental phase those
ratios were 103.0% and 89.5%, respectively. Its partial rescue therefore did
not depend on shrinking either organism toward the identity boundary.

## Important failed attempt

An earlier permissive gate allowed 50–200% parental mass. The search exploited
that objective by shrinking Synorbium to 52% of its parent, producing an
apparently stronger coexistence result. That candidate was rejected and the
solo-identity interval was tightened to 75–125% before the final run. This is
recorded as objective gaming, not a discovery.

## Interpretation

Four growth parameters can rescue some encounters, but this neighborhood did
not contain a robust co-adaptation under the tested conditions. The next search
should add interaction sensitivity or spatial signalling rather than simply
spending more compute on the same four-dimensional family.

## Reproduce

```bash
python3 -m genesis.evolve_coexistence --candidates 2048
python3 -m unittest discover -s tests -v
```
