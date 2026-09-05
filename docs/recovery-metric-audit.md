# Recovery metric audit — 5 September 2026

The recovery pipeline now distinguishes retained matter, retained motion, and
retained anatomy. A surviving state alone is insufficient evidence of recovery.
The metric contract is tested against persistence, translation, death, and diffuse
explosion, with matched-control failure cases included.

## Outcome definitions

| Outcome | Definition | Interpretation |
| --- | --- | --- |
| Survival | Final mass at least 1 | Matter persists; no claim about function |
| Functional recovery | Valid matched control; mass ratio 0.75–1.25; motion ratio at least 0.25; occupied fraction at most 0.03 | The current operational functional endpoint |
| Functional recovery time | First of three consecutive eligible samples satisfying the functional endpoint | Motion uses a completed 50-step post-injury window |
| Strict mass recovery time | First of three samples within 2% of matched-control mass | Separate, stricter mass endpoint |
| Shape similarity | Maximum cosine similarity over all toroidal translations | Position-independent anatomical comparison; secondary outcome |
| Shape recovery time | First of three valid-control samples with similarity at least 0.9 | Anatomical retention or restoration, independent of functional recovery |

The thresholds are operational research choices, not established biological
definitions. A shape time of zero means the wounded state was already inside the
shape tolerance band. A functional time of 50 means the criterion holds at the
first eligible motion window; it does not locate an exact biological event at
step 50. Sampling is normally every 10 steps and recovery must persist for three
samples. Later relapse remains possible and final recovery is reported separately.

Times are null when no qualifying sustained interval occurs during follow-up or
the matched control is invalid. Control validity is recorded so these cases can
be distinguished. Reporting medians only among recovered trials introduces
survivor selection; recovery fractions and the censored count must accompany
such medians.

## Correctness checks

`tests/test_recovery_metrics.py` verifies FFT alignment against exhaustive
translation on a rectangular world, translation invariance, empty-state handling,
sustained bands, moving matched controls, invalid samples, and finite strictly
increasing time axes. Joint extinction and infinite control values cannot count
as mass recovery. The functional endpoint rejects stationary persistence,
death, mass explosion, diffuse occupancy, and invalid controls. The functional
time uses the same endpoint.

The specimen and sweep pipelines now use pre-injury and post-injury control
viability, completed rolling motion windows, and separately recorded functional,
mass, and shape outcomes. The final state is always sampled even when the run
length is not divisible by the sampling interval. A dead specimen has no reported
total displacement; the toroidal centroid of an empty world is not locomotion.

Verification run: 25 core and recovery-metric tests passed. Additional manual
runs at settle/pre=303, post=57, sample interval=13 confirmed terminal snapshots
at the actual final step.

## Canonical rerun

At the default 300-step pre-injury phase and 300-step follow-up, the matched
Orbium control remained viable. The approximately 25% centered injury in
`runs/specimen-zero.json` caused extinction; functional, mass, and shape recovery
times are all null.

In the exact-magnitude sweep, only trailing injuries removing 5% or 10% of mass
survived and met the functional endpoint. Their strict mass recovery times were
20 and 30 steps; their shape times were zero because similarity never initially
left the 0.9 band. Both met the functional criterion by the first eligible
50-step window, retaining approximately 99.8% of control motion at the final
measurement. Center and leading injuries at 5% failed, as did all locations at
15%, 20%, and 25%. These are condition-specific observations of one initial
state, not 15 independent replicates.

## Inference limits for the subsequent search

Shape must remain secondary under charter 001; multiplying similarity into the
primary search objective would change that charter question. Invalid controls
and diffuse injured states should also be excluded from the objective, not only
from the binary endpoint.

Independent noisy initial-condition seeds are the replication unit. Phases and
damage magnitudes within a seed are repeated measurements. Report seed and
condition breakdowns; do not use their product as an independent sample size.
Independent starting draws may converge toward the same attractor during
settling, so seed independence does not establish broad morphological diversity.

The 5%, 7%, 10%, and 12% damage levels are held out from the current 4%, 6%, and
8% selection objective. Some magnitudes, including 5% and 10%, were examined in
earlier project experiments. They should not be described as wholly unseen
historical interventions. This audit validates the operational metric gate; it
does not establish the diversity hypothesis or replace the coverage gate and
pre-registered comparison.
