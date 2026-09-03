import json
import unittest
from pathlib import Path

from genesis.validate_ecology import paired_advantage


DISCOVERY = Path("runs/coupled-ecology.json")
VALIDATION = Path("runs/coupled-ecology-validation.json")


class EcologyRecordTests(unittest.TestCase):
    def test_discovery_record_is_complete_and_unique(self):
        record = json.loads(DISCOVERY.read_text())
        rows = record["results"]
        self.assertEqual(record["summary"]["trial_count"], 1296)
        self.assertEqual(len(rows), 1296)
        keys = {
            (
                row["left_code"], row["right_code"], row["competition"],
                row["geometry"], row["orientation_degrees"], row["noise_seed"],
            )
            for row in rows
        }
        self.assertEqual(len(keys), len(rows))
        self.assertEqual(record["summary"]["invalid_control_trials"], 0)

    def test_strength_outcomes_reconcile_to_valid_trials(self):
        record = json.loads(DISCOVERY.read_text())
        for row in record["summary"]["strengths"]:
            total = (
                row["coexistence_trials"]
                + row["dominance_trials"]
                + row["mutual_collapse_trials"]
            )
            self.assertEqual(total, row["valid_trials"])
            self.assertAlmostEqual(
                row["coexistence_rate"],
                row["coexistence_trials"] / row["valid_trials"],
            )

    def test_validation_comparison_recomputes_from_raw_trials(self):
        record = json.loads(VALIDATION.read_text())
        recomputed = paired_advantage(record["experiment"])
        self.assertEqual(recomputed, record["comparison"])
        self.assertEqual(recomputed["matched_contexts"], 252)

    def test_validation_conditions_are_held_out(self):
        discovery = json.loads(DISCOVERY.read_text())["protocol"]
        validation = json.loads(VALIDATION.read_text())
        held_out = validation["separation_from_discovery"]
        self.assertTrue(set(discovery["orientations_degrees"]).isdisjoint(held_out["orientations_degrees"]))
        self.assertTrue(set(discovery["noise_seeds"]).isdisjoint(held_out["noise_seeds"]))
        self.assertNotEqual(discovery["development_phase"], held_out["development_phase"])


if __name__ == "__main__":
    unittest.main()
