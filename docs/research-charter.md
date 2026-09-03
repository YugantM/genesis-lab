# Research charter 001 — Diversity and unseen damage

## Question

Does behavioural diversity produce self-organising systems that recover from
unseen damage better than systems selected directly for regeneration?

## Compared conditions

1. Selection for survival and locomotion only.
2. Selection for recovery from a fixed training damage distribution.
3. MAP-Elites selection for behavioural and morphological diversity, without a
   direct recovery reward.

Each condition receives the same evaluation budget and is repeated across at
least 20 independent random seeds.

## Held-out interventions

- Internal circular excision
- Repeated injury
- Communication noise
- Changed kernel radius and diffusion
- Fragmentation into disconnected halves
- Moving resource field

## Primary outcomes

- Functional recovery relative to pre-injury performance
- Survival probability
- Recovery time
- Performance retained after repeated injury

Shape similarity is secondary: an organism may recover its function without
recovering its original anatomy.

## Research gates

1. Reproduce a known stable continuous-CA organism.
2. Validate that metrics separate persistence, explosion, drift, and death.
3. Demonstrate MAP-Elites coverage above random search at equal compute.
4. Pre-register held-out damage before the main comparison.
5. Explain robust examples with intervention and ablation, not visual judgment.

## Multi-species gate

Before comparing selection methods, run the same paired injury protocol across
at least eight stable classic-Lenia parental species. A parameter mechanism is
called transferable only when it improves recovery by at least 10 percentage
points in at least half of eligible species and makes at most one species less
viable without injury. Species with fewer than 90% valid parental controls are
excluded rather than counted as failures.

The first transfer benchmark rejected the universal-transfer hypothesis:
Genesis 001's relative growth-parameter shift benefited Orbium and Circium,
was neutral in one species, and made five transfers nonviable. The main comparison must
therefore evolve candidates within each species.
