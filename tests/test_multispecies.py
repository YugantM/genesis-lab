import json
import unittest
from pathlib import Path

import numpy as np

from genesis.genome import load_specimen
from genesis.multispecies_benchmark import (
    GENESIS_001,
    lesion_anchor,
    load_manifest,
    transferred_parameters,
)


MANIFEST = "web/specimens/species-benchmark.json"


class MultiSpeciesTests(unittest.TestCase):
    def test_manifest_has_eight_compatible_species(self):
        catalogue = load_manifest(Path(MANIFEST))
        self.assertEqual(len(catalogue), 8)
        for _, specimen in catalogue:
            self.assertEqual(specimen.simulator, "lenia/classic-v1")
            self.assertEqual(specimen.parameters["radius"], 13)
            self.assertEqual(specimen.parameters["time_resolution"], 10)

    def test_orbium_transfer_reproduces_genesis_parameters(self):
        specimen = load_specimen("web/specimens/orbium-unicaudatus.json")
        mu, sigma = transferred_parameters(specimen)
        self.assertAlmostEqual(mu, GENESIS_001[0])
        self.assertAlmostEqual(sigma, GENESIS_001[1])

    def test_anatomical_anchors_land_on_opposite_active_extremes(self):
        state = np.zeros((16, 16), np.float32)
        state[8, 5:12] = 1
        centre = np.array([8.0, 8.0])
        motion = np.array([0.0, 2.0])
        leading = lesion_anchor(state, centre, motion, "leading")
        trailing = lesion_anchor(state, centre, motion, "trailing")
        self.assertGreater(leading[1], centre[1])
        self.assertLess(trailing[1], centre[1])
        self.assertGreaterEqual(state[tuple(leading.astype(int))], 0.1)
        self.assertGreaterEqual(state[tuple(trailing.astype(int))], 0.1)

    def test_saved_benchmark_has_no_invalid_parent_species(self):
        record = json.loads(Path("runs/multispecies-transfer.json").read_text())
        self.assertEqual(record["summary"]["trial_count"], 1728)
        self.assertEqual(record["summary"]["excluded_parent_unstable_species"], 0)
        self.assertFalse(record["summary"]["transferable"])


if __name__ == "__main__":
    unittest.main()
