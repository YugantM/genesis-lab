import unittest
from collections import Counter

import numpy as np

from genesis.replicated_sweep import (
    NOISE_SIGMA,
    PHASES,
    condition_summary,
    initial_condition_plan,
    perturbed_initial_state,
)


class ReplicatedSweepTests(unittest.TestCase):
    def test_independent_seed_is_the_replicate_axis(self):
        plan = initial_condition_plan()
        self.assertEqual(len({item["initial_condition_seed"] for item in plan}), 20)
        self.assertEqual(Counter(item["phase"] for item in plan), dict.fromkeys(PHASES, 4))
        self.assertGreaterEqual(max(PHASES) - min(PHASES), 320)
        self.assertTrue(all("orientation" not in item for item in plan))

    def test_perturbations_are_independent_reproducible_and_local(self):
        initial = np.zeros((64, 64), np.float32)
        initial[8:56, 8:56] = 0.5
        states = [perturbed_initial_state(initial, item["initial_condition_seed"]) for item in initial_condition_plan()]
        self.assertEqual(len({state.tobytes() for state in states}), 20)
        np.testing.assert_array_equal(states[0], perturbed_initial_state(initial, initial_condition_plan()[0]["initial_condition_seed"]))
        self.assertTrue(all(np.all(state[initial == 0] == 0) for state in states))
        observed_sigma = float(np.std((states[0] - initial)[initial > 0]))
        self.assertAlmostEqual(observed_sigma, NOISE_SIGMA, delta=0.001)
        self.assertGreater(NOISE_SIGMA, 0.01)

    def test_condition_summary_rejects_pooled_repeated_seed(self):
        row = {
            "initial_condition_seed": 1,
            "functionally_recovered": True,
            "survived": True,
            "control_valid": True,
            "mass_recovery_time_steps": 20,
            "functional_recovery_time_steps": 50,
            "extinction_step": None,
            "actual_removed_fraction": 0.05,
        }
        with self.assertRaises(ValueError):
            condition_summary([row, row])
        second = {**row, "initial_condition_seed": 2, "functionally_recovered": False,
                  "mass_recovery_time_steps": None, "functional_recovery_time_steps": None}
        summary = condition_summary([row, second])
        self.assertEqual(summary["independent_initial_conditions"], 2)
        self.assertEqual(summary["functional_recovery_rate"], 0.5)
        self.assertEqual(summary["mass_recovery_events"], 1)
        self.assertEqual(summary["median_mass_recovery_time_steps_among_events"], 20)


if __name__ == "__main__":
    unittest.main()
