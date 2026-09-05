# Research update — 5 September 2026

The six-point review exposed a real mismatch between the existing survival
measurements and the recovery claims. The repaired pipeline now reports matter,
motion and anatomy separately, with matched controls and sustained recovery times.

## Completed corrections

- The shared metric module now contains toroidal translation alignment. Injury
  experiments use it directly, and ecology imports the same implementation.
- Recovery times distinguish strict mass return (within 2% of control), functional
  return (mass within 25%, at least 25% of control motion, bounded occupancy), and
  anatomical return (aligned similarity at least 0.9). Three consecutive samples
  are required. Dead or invalid controls cannot count as recovery.
- The search objective averages functional recovery across 4%, 6% and 8% exact
  mass removal. Shape is secondary under the charter and is not a search reward.
- Selection uses four independent noisy initial states, measured at three phases;
  validation uses eight different noisy starts and four different phases. Neither
  rotation nor a repeated phase is counted as an independent replicate.
- The canonical replicated sweep now uses 20 distinct initial states, one balanced
  phase per seed over 180–500 steps, and noise standard deviation 0.012. Magnitude
  and wound location are repeated conditions within each seed.
- The website leads with the eight-species transfer rejection, identifies legacy
  dependent observations, and places the live-view disclaimer above each canvas.
  The common world initializes only when opened and stops rendering when closed.
- Genesis 001 is credited to Genesis Lab; its parental author remains credited.

## Recovery results

The canonical approximately 25% central injury caused extinction. Its mass,
functional and shape recovery times are all null. In the single-state exact
sweep, only trailing 5% and 10% wounds recovered. Their strict mass return times
were 20 and 30 steps. Both met the functional criterion by the first complete
50-step motion window. Their shape time of zero means the injured states never
initially left the shape tolerance band; it does not mean instantaneous healing.

The independent replication confirmed 20/20 recoveries for trailing 5% and 10%
wounds, found 3/20 recoveries for leading 5% wounds, and found none for central
wounds or any location at 15% or above. All 20 matched controls remained viable.
These are 300 repeated-condition trials from 20 independent starting draws.

The corrected magnitude search evaluated 1,024 parameter candidates. Ten passed
all three training magnitudes in the first screen. Candidate 806 was selected
after the independent training contexts and recovered in all 36 repeated training
trials. Its subsequent validation was:

| Exact central mass removal | Functional recoveries | Independent initial states |
| --- | ---: | ---: |
| 5% | 32/32 phase contexts | 8 |
| 7% | 32/32 phase contexts | 8 |
| 10% | 0/32 phase contexts | 8 |
| 12% | 1/32 phase contexts | 8 |

The 50.8% mean failed the 65% gate. All controls were valid. This is consistent
with moving a local collapse boundary, while broad damage robustness remains
unsupported. It does not by itself establish a bistability mechanism. Candidate
806 is distinct from the earlier Genesis 001, and has not been promoted.

## Charter gates

Gate 2 is supported by adversarial metric tests that distinguish stationary
persistence, translation, diffuse explosion and death, along with invalid-control
and recovery-time tests. Shape remains secondary to function.

Gate 3 was run before the main comparison at equal simulation compute: MAP-Elites
and random search each received 64 candidate evaluations per seed, with 20 seeds
within each of the eight species. All parental controls were valid. The mean
paired coverage difference was +0.78125 occupied bins, with a 95% seed-bootstrap
interval of [0.575, 0.9875], passing the prespecified aggregate gate.

The result is species-dependent. Orbium favored random search by 0.7 bins;
Helicium cavus pedes favored MAP-Elites by 2.7 bins. The archive and random control
both count every occupied behavior bin they encounter. The advantage is not
created by discarding random-search candidates.

The three-condition, 20-seed comparison was unlocked after this gate. Its full
protocol, intervention definitions, source hashes, seed schedule, and specimen
hashes are saved before held-out evaluation. Per-species records are written under
`runs/charter-comparison/`. The protocol is described in
[selection-comparison-protocol.md](selection-comparison-protocol.md).

## Completed three-condition comparison

All eight species completed 20 independent search seeds under each of the three
methods, totaling 480 selected specimens and 4,800 matched intervention trials.
Each search used 64 candidate evaluations. Results below are pipeline success:
the selected specimen must have a valid undamaged control and retain function
after intervention. Invalid controls count as failed selections; they are not
evidence that injury itself caused the failure.

| Method | Mean species success | Undamaged controls valid | Final survival |
| --- | ---: | ---: | ---: |
| Performance | 32.4% | 80.6% | 62.8% |
| MAP-Elites | 40.6% | 91.9% | 66.6% |
| Direct regeneration | 48.4% | 94.4% | 71.0% |

MAP-Elites minus performance was +8.25 percentage points, with a descriptive
95% paired seed-bootstrap interval of [+3.94, +12.63]. MAP-Elites minus direct
regeneration was −7.81 points, interval [−11.88, −3.50]. These intervals describe
this fixed species panel and protocol; they are not broad generalization bounds.

| Species | Performance | MAP-Elites | Direct regeneration |
| --- | ---: | ---: | ---: |
| Orbium | 28.0% | 31.5% | 40.0% |
| Gyrorbium | 18.5% | 28.0% | 34.0% |
| Synorbium | 20.0% | 31.0% | 41.5% |
| Scutium | 26.0% | 33.5% | 49.0% |
| Paraptera | 38.5% | 51.5% | 60.5% |
| Helicium solidus | 39.0% | 45.0% | 61.5% |
| Helicium cavus pedes | 6.5% | 24.5% | 23.5% |
| Circium | 82.5% | 80.0% | 77.5% |

Direct regeneration leads in six species; MAP-Elites leads narrowly in one and
performance leads in one. The registered hypothesis that diversity selection
beats direct regeneration is not supported by this pilot. MAP-Elites did improve
pipeline success over performance-only selection in seven species, but increased
coverage alone does not establish the mechanism behind those differences.

The primary outcome mixes generalization of undamaged stability with recovery.
A separate post-selection sensitivity retains only seed indices with valid
controls under all three methods. The same ordering holds in the six species
where direct selection leads. Those subsets contain 4–20 seeds depending on
species and change the population; they do not replace the registered result.
Control failures are especially substantial for performance-selected Helicium
cavus pedes (only 25% valid), and must remain visible.

The candidate budgets are equal as specified by the charter, but direct
regeneration's training reward costs 2.5 times as many simulated world steps.
This experiment cannot establish superiority at equal total compute. The
MAP-Elites specimen is sampled uniformly from the archive, while the other
methods return their objective's best viable candidate. This explicit selection
rule also limits interpretation.

All raw records are in `runs/charter-comparison/comparison-*.json`. Reconciled
seed-level estimates, control-validity sensitivity, event/censoring counts, and
per-intervention breakdowns are in `runs/charter-comparison/analysis.json`.
The static site exposes the compact analysis and all trial outcomes with selected
genomes. Reproduce the final analysis with `python3 -m genesis.analyze_comparison`.

The next scientific step is a separately registered, equal-compute replication
with an archive-selection ablation and stronger training checks on undamaged
stability. Gate 5 remains open; no new ecology module was added.

## Verification

All 63 automated tests pass, including reconstruction of the 4,800 trial endpoints
and equal-budget coverage counts from raw records. Browser checks passed on desktop
and at a 390-pixel mobile width, with no page-wide horizontal overflow or JavaScript
errors. All six displayed evidence links returned HTTP 200. Selecting a species
while paused preserved every habitat's mass. The local Apple M4 Pro renderer was
observed at 23.8 frames/second and 71.5 simulation steps/second with a 30 fps target.
These changes are local; the public GitHub Pages site has not been republished.

## Limits

Independent noise draws can settle toward the same attractor; independent seeds
do not guarantee independent morphology. Some validation damage magnitudes were
examined historically, so only exclusion from the current training objective and
fresh initial-state draws are claimed. The comparison is restricted to inherited
anatomy and two native growth parameters, at a small fixed evaluation budget.
Any broader finding requires replication and explanatory ablations under gate 5.
