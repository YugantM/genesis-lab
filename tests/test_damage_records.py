"""Verify that saved research claims reconcile with their trial records."""
import json
import unittest
from pathlib import Path

import numpy as np

from genesis.metrics import functional_recovery


ROOT = Path(__file__).resolve().parents[1]


class DamageRecordTests(unittest.TestCase):
    def test_validation_has_unique_matched_contexts_and_seed_level_rates(self):
        record=json.loads((ROOT/"runs/genesis-magnitude-validation.json").read_text())
        rows=record["results"]
        keys=[(r["genotype"],r["initial_condition_seed"],r["phase"],r["target_removed_fraction"]) for r in rows]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(len(keys),256)
        for aggregate in record["aggregate"]:
            subset=[r for r in rows if r["genotype"]==aggregate["genotype"] and r["target_removed_fraction"]==aggregate["target_removed_fraction"]]
            rates=[np.mean([r["functionally_recovered"] for r in subset if r["initial_condition_seed"]==seed])
                   for seed in record["protocol"]["held_out_initial_condition_seeds"]]
            self.assertAlmostEqual(aggregate["mean_seed_recovery_rate"],float(np.mean(rates)))
            self.assertEqual(len(subset),32)
        acceptance=record["acceptance"]
        self.assertEqual(acceptance["accepted"],acceptance["candidate_mean_seed_recovery_rate"]>=.65
                         and acceptance["difference"]>=.25 and acceptance["minimum_control_validity"]>=.95)

    def test_replicated_wounds_are_exact_and_outcomes_follow_declared_endpoint(self):
        record=json.loads((ROOT/"runs/replicated-damage-sweep.json").read_text())
        rows=record["results"]
        self.assertEqual(len({r["initial_condition_seed"] for r in rows}),20)
        self.assertEqual(len(rows),300)
        self.assertEqual(len({(r["initial_condition_seed"],r["target_removed_fraction"],r["location"]) for r in rows}),300)
        for row in rows:
            self.assertAlmostEqual(row["target_removed_fraction"],row["actual_removed_fraction"],places=5)
            self.assertEqual(row["functionally_recovered"],functional_recovery(
                control_valid=row["control_valid"],mass_ratio=row["mass_ratio_to_control"],
                motion_ratio=row["motion_ratio_to_control"],occupied=row["final_occupied_fraction"]))
            if row["functional_recovery_time_steps"] is not None:
                self.assertGreaterEqual(row["functional_recovery_time_steps"],50)

    def test_coverage_gate_uses_equal_budgets_and_actual_occupied_cells(self):
        directory=ROOT/"runs/charter-comparison"
        if not (directory/"coverage.json").exists():
            self.skipTest("coverage record has not been generated")
        gate=json.loads((directory/"coverage.json").read_text())
        protocol=json.loads((directory/"protocol.json").read_text())["protocol"]
        differences=[]
        for species in protocol["species"]:
            record=json.loads((directory/f"coverage-{species}.json").read_text())
            rows=record["searches"]
            self.assertEqual(len(rows),40)
            lookup={(r["seed"],r["method"]):r for r in rows}
            self.assertEqual(len(lookup),40)
            for seed in protocol["search_seeds"]:
                first,second=lookup[(seed,"map-elites")],lookup[(seed,"random")]
                self.assertEqual(first["simulated_world_steps_per_seed"],second["simulated_world_steps_per_seed"])
                for row in (first,second):
                    candidates=row["evaluated_candidates"]
                    self.assertEqual(len(candidates),protocol["evaluations_per_method_per_seed"])
                    cells={tuple(r["cell"]) for r in candidates if r["valid"]}
                    self.assertEqual(row["coverage"],len(cells))
                differences.append(first["coverage"]-second["coverage"])
        self.assertAlmostEqual(float(np.mean(differences)),gate["mean_coverage_difference"])
        self.assertEqual(gate["gate_passed"],gate["ci95"][0]>0)

    def test_completed_comparison_reconciles_all_4800_trial_endpoints(self):
        directory=ROOT/"runs/charter-comparison"
        if not (directory/"comparison.json").exists():
            self.skipTest("comparison has not completed")
        protocol=json.loads((directory/"protocol.json").read_text())["protocol"]
        total=0
        for species in protocol["species"]:
            record=json.loads((directory/f"comparison-{species}.json").read_text())
            rows=record["heldout"]
            self.assertEqual(len(rows),600)
            self.assertEqual(len({(r["seed"],r["method"],r["intervention"]) for r in rows}),600)
            for row in rows:
                if row.get("no_viable_selection"):
                    self.assertFalse(row["functionally_recovered"])
                    continue
                final=row["trajectory"][-1]
                self.assertEqual(final["step"],300)
                self.assertEqual(row["functionally_recovered"],functional_recovery(
                    control_valid=row["control_valid"],mass_ratio=row["mass_ratio_to_control"],
                    motion_ratio=row["motion_ratio_to_control"],occupied=final["occupied"],maximum_occupied=.15))
                self.assertEqual(row["survived"],final["mass"]>=1)
                if not row["control_valid"]:
                    self.assertIsNone(row["mass_recovery_time_steps"])
                    self.assertIsNone(row["functional_recovery_time_steps"])
            total+=len(rows)
        self.assertEqual(total,4800)
