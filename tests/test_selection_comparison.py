import unittest
import numpy as np

from genesis.selection_comparison import cell_for, confidence_interval, update_archive, intervention
from genesis.core import GenesisConfig, GenesisWorld


class SelectionComparisonTests(unittest.TestCase):
    def test_archive_requires_viability_and_keeps_best_per_cell(self):
        archive = {}
        update_archive(archive, {"valid": False, "cell": [1, 2], "performance": 10})
        self.assertEqual(archive, {})
        first = {"valid": True, "cell": [1, 2], "performance": .8, "training_recovery_score": 0}
        update_archive(archive, first)
        update_archive(archive, {**first, "performance": .7, "training_recovery_score": 1})
        self.assertIs(archive[(1, 2)], first)
        better = {**first, "performance": .9}
        update_archive(archive, better)
        self.assertIs(archive[(1, 2)], better)

    def test_archive_coverage_counts_distinct_cells_not_evaluations(self):
        archive = {}
        for i in range(100):
            update_archive(archive, {"valid":True,"cell":[i % 2, 3],"performance":i / 100})
        self.assertEqual(len(archive), 2)

    def test_descriptor_boundaries_are_fixed(self):
        self.assertEqual(cell_for(.5, 0), (0,0))
        self.assertEqual(cell_for(1.25, 1), (4,4))
        self.assertEqual(cell_for(2, 2), (7,7))
        self.assertEqual(cell_for(9, 9), (7,7))

    def test_bootstrap_preserves_paired_seed_differences(self):
        self.assertEqual(confidence_interval(np.ones(20)), [1,1])
        self.assertEqual(confidence_interval(np.zeros(20)), [0,0])
        self.assertEqual(confidence_interval(np.arange(20)), confidence_interval(np.arange(20)))

    def test_diffusion_conserves_mass_of_the_immediate_poststep_state(self):
        control=GenesisWorld(GenesisConfig(size=32,batch=1,seed=51))
        injured=GenesisWorld(GenesisConfig(size=32,batch=1,seed=51))
        control.step()
        intervention(injured,"diffusion",1,np.random.default_rng(9))
        self.assertAlmostEqual(float(control.numpy().sum()),float(injured.numpy().sum()),places=4)
        self.assertFalse(np.array_equal(control.numpy(),injured.numpy()))

    def test_sensing_noise_is_reproducible_and_does_not_mutate_control(self):
        config=GenesisConfig(size=32,batch=1,seed=51)
        control=GenesisWorld(config)
        first,second=GenesisWorld(config),GenesisWorld(config)
        control.step()
        before=control.numpy().copy()
        intervention(first,"communication-noise",1,np.random.default_rng(9))
        intervention(second,"communication-noise",1,np.random.default_rng(9))
        np.testing.assert_array_equal(first.numpy(),second.numpy())
        np.testing.assert_array_equal(control.numpy(),before)
        self.assertFalse(np.array_equal(first.numpy(),control.numpy()))
