import json
import unittest
from pathlib import Path


RECORD = Path("runs/coexistence-evolution.json")


class EvolutionRecordTests(unittest.TestCase):
    def test_search_record_matches_locked_protocol(self):
        record = json.loads(RECORD.read_text())
        self.assertEqual(record["protocol"]["candidate_count"], 2048)
        self.assertEqual(record["protocol"]["target_pair"], ["O4i", "P4cl"])
        self.assertEqual(record["protocol"]["competition"], 1.0)
        self.assertEqual(len(record["protocol"]["training_contexts"]), 4)
        self.assertEqual(len(record["protocol"]["held_out_contexts"]), 8)

    def test_selected_candidate_respects_solo_identity_gate(self):
        record = json.loads(RECORD.read_text())
        selected = record["summary"]["selected_training"]
        self.assertTrue(selected["both_solo_viable"])
        for channel in selected["solo"]["channels"]:
            self.assertGreaterEqual(channel["mass_ratio_to_parent"], 0.75)
            self.assertLessEqual(channel["mass_ratio_to_parent"], 1.25)
            self.assertGreaterEqual(channel["occupied_ratio_to_parent"], 0.75)
            self.assertLessEqual(channel["occupied_ratio_to_parent"], 1.25)
            self.assertGreaterEqual(channel["parent_similarity"], 0.5)

    def test_promotion_decision_recomputes(self):
        summary = json.loads(RECORD.read_text())["summary"]
        training = summary["selected_training"]
        candidate = summary["held_out_candidate"]
        baseline = summary["held_out_baseline"]
        improvement = candidate["coexistence_rate"] - baseline["coexistence_rate"]
        promoted = (
            training["coexistence_rate"] >= 0.70
            and candidate["coexistence_rate"] >= 0.50
            and improvement >= 0.25
            and candidate["both_solo_viable"]
        )
        self.assertAlmostEqual(improvement, summary["held_out_improvement"])
        self.assertEqual(promoted, summary["promoted"])
        self.assertFalse(promoted)


if __name__ == "__main__":
    unittest.main()
