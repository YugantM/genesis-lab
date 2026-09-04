import json
import unittest
from pathlib import Path


RECORD = Path("runs/interaction-evolution.json")


class InteractionEvolutionRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(RECORD.read_text())

    def test_protocol_separates_training_and_held_out_conditions(self):
        protocol = self.record["protocol"]
        self.assertEqual(protocol["candidate_count"], 1024)
        self.assertEqual(protocol["body_parameters"], "fixed parental values")
        training = {
            (row["orientation_degrees"], row["noise_seed"])
            for row in protocol["training_contexts"]
        }
        held = {
            (row["orientation_degrees"], row["noise_seed"])
            for row in protocol["held_out_contexts"]
        }
        self.assertTrue(training.isdisjoint(held))

    def test_promotion_decision_recomputes(self):
        summary = self.record["summary"]
        training = summary["selected_training"]["coexistence_rate"]
        held = summary["held_out_candidate"]["coexistence_rate"]
        improvement = held - summary["held_out_baseline"]["coexistence_rate"]
        recomputed = (
            training >= 0.75
            and held >= 0.5
            and improvement >= 0.25
            and summary["held_out_candidate"]["identity_safe"]
        )
        self.assertAlmostEqual(improvement, summary["held_out_improvement"])
        self.assertEqual(recomputed, summary["promoted"])
        self.assertFalse(recomputed)

    def test_selected_candidate_was_not_reselected_on_held_out_data(self):
        summary = self.record["summary"]
        self.assertEqual(summary["selected_training"]["candidate"], 717)
        self.assertEqual(summary["held_out_candidate"]["candidate"], 717)
        self.assertEqual(summary["held_out_baseline"]["coexistence_trials"], 1)
        self.assertEqual(summary["held_out_candidate"]["coexistence_trials"], 3)

    def test_held_out_identity_metrics_respect_gate(self):
        candidate = self.record["held_out_results"][1]
        for context in candidate["contexts"]:
            self.assertTrue(context["identity_safe"])
            for channel in context["channels"]:
                self.assertGreaterEqual(channel["identity_mass_ratio"], 0.75)
                self.assertLessEqual(channel["identity_mass_ratio"], 1.25)
                self.assertGreaterEqual(channel["identity_area_ratio"], 0.75)
                self.assertLessEqual(channel["identity_area_ratio"], 1.25)
                self.assertGreaterEqual(channel["identity_similarity"], 0.5)


if __name__ == "__main__":
    unittest.main()
