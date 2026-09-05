# Charter comparison protocol — 5 September 2026

This is a bounded first comparison in the native two-parameter growth space of
each of the eight catalogue species. It does not test unrestricted evolution or
establish a new artificial-life mechanism.

Run `python3 -m genesis.selection_comparison --stage coverage` first. The default
budget is64 candidate evaluations for each method, species, and independent
search seed, using20 seeds and generations of16 candidates. Before simulation,
the runner saves the exact protocol, source hashes, specimen hashes and random
seeds. Interrupted runs resume only from matching per-species records.

Coverage compares MAP-Elites with random sampling under the same parameter
bounds, initial population, noisy starting states, evaluation count and simulated
world-step budget. The archive has fixed8×8 bins for mass and50-step displacement
relative to the native parent. Viable candidates replace a cell's elite only when
undamaged performance improves. Later MAP-Elites populations mutate uniformly
sampled occupied cells; random search continues sampling the original domain.
Coverage counts all occupied cells encountered by either method, not the number
retained in an arbitrarily small random subset.

Gate3 passes only if the lower95% bootstrap bound of the mean paired coverage
difference is above zero. Each bootstrap draw resamples search seeds, carrying
all species together. Species with fewer than90% valid parental controls are
excluded from the gate, and all exclusions and species-specific outcomes remain
visible. No descriptor bounds or search budget are tuned after seeing the result.

Only a passing matching coverage record unlocks
`python3 -m genesis.selection_comparison --stage comparison`. The three methods
then receive the same64 candidate evaluations across20 independent search seeds
within each species: undamaged performance selection, direct regeneration
selection over4/6/8% central wounds, and MAP-Elites. Performance and direct search
mutate their best four candidates. MAP-Elites samples uniformly from its archive.
The final MAP-Elites specimen is a uniformly selected archive cell; the others
use their own objective's best viable specimen. Failed searches are retained as
failures rather than silently excluded.

Candidate evaluations are equal across the three methods; direct regeneration
uses extra simulated steps for its injury reward, which are reported explicitly.
The stronger equal-compute requirement applies to the coverage gate and is met
there. Shape similarity is a secondary outcome and never selects a candidate.

The held-out matrix uses fresh noisy initial-state draws, balanced phases220,
360,500,640, and ten conditions per selected specimen:5/7/10/12% central wounds,
two7% wounds150 steps apart, a central fragmentation gap, noisy neighborhood
sensing,10% larger interaction radius, explicit diffusion, and an imposed moving
growth field. The growth field is a controlled perturbation; the model does not
contain resource uptake. Some wound magnitudes have historical exploratory
exposure; only their exclusion from the current training objective and the fresh
initial-state seeds are claimed.

Follow-up lasts300 steps. Every trial retains its matched native control.
Primary outputs are functional recovery, survival, mass and functional recovery
times, and repeated-injury performance. Mass within25% of control plus at least
25% of its motion constitutes functional recovery; controls moving less than one
cell per50 steps have no locomotion requirement. Occupancy must remain at most
15% for this multi-species protocol. Strict mass return uses a2% tolerance;
shape remains separately measured. These species-normalized thresholds differ
from the narrower Orbium-only validity bounds; rates should not be pooled
between protocols.

The independent unit is the search seed within species. Interventions are
repeated measurements, reported individually and averaged within each seed.
The experimental matrix answers the operational selection question at this
limited budget; a broad or mechanistic conclusion still requires follow-up
interventions and ablations under charter gate5.
