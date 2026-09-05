import unittest

import mlx.core as mx
import numpy as np

from genesis.core import GenesisConfig, GenesisWorld
from genesis.genome import decode_lenia_rle, load_specimen
from genesis.metrics import (
    aligned_similarity,
    center_of_mass,
    mass,
    occupied_fraction,
    recovery_time,
    survival,
    threshold_recovery_time,
)
from genesis.robust_search import calibrated_damage

SPECIMEN = "web/specimens/orbium-unicaudatus.json"
MUTANT = "web/specimens/genesis-001.json"


class GenesisCoreTests(unittest.TestCase):
    def test_state_is_bounded_and_reproducible(self):
        config = GenesisConfig(size=32, batch=2, radius=6, seed=7)
        a, b = GenesisWorld(config), GenesisWorld(config)
        np.testing.assert_array_equal(a.step(3), b.step(3))
        self.assertGreaterEqual(a.numpy().min(), 0.0)
        self.assertLessEqual(a.numpy().max(), 1.0)

    def test_damage_removes_mass(self):
        world = GenesisWorld(GenesisConfig(size=32, batch=2, seed=2))
        before = mass(world.numpy())
        world.damage(0.5)
        self.assertTrue(np.all(mass(world.numpy()) < before))

    def test_metrics_return_one_value_per_world(self):
        state = np.zeros((3, 8, 8), np.float32)
        state[:, 3, 3] = 1
        self.assertEqual(mass(state).shape, (3,))
        self.assertEqual(occupied_fraction(state).shape, (3,))
        self.assertTrue(survival(state).all())

    def test_orbium_is_canonical_and_persists(self):
        specimen = load_specimen(SPECIMEN)
        self.assertEqual(specimen.cells.shape, (20, 20))
        self.assertAlmostEqual(float(specimen.cells.sum()), 76.86275, places=4)
        world = GenesisWorld.from_specimen(specimen, size=64)
        world.step(300)
        final_mass = float(mass(world.numpy())[0])
        self.assertGreater(final_mass, 65.0)
        self.assertLess(final_mass, 80.0)

    def test_toroidal_center_handles_boundary(self):
        state = np.zeros((1, 16, 16), np.float32)
        state[0, 8, [0, 15]] = 1
        y, x = center_of_mass(state)[0]
        self.assertAlmostEqual(y, 8.0, places=5)
        self.assertTrue(x < 1 or x > 15)

    def test_rle_repeat_syntax(self):
        np.testing.assert_array_equal(
            decode_lenia_rle("2A2.$2B!"),
            np.array([[1, 1, 0, 0], [2, 2, 0, 0]], np.float32) / 255,
        )

    def test_per_world_growth_parameters_diverge(self):
        specimen = load_specimen(SPECIMEN)
        world = GenesisWorld.from_specimen(specimen, size=64, batch=2)
        world.set_growth_parameters([0.15, 0.18], [0.015, 0.015])
        world.step(20)
        self.assertFalse(np.array_equal(world.numpy()[0], world.numpy()[1]))

    def test_growth_parameter_shape_is_checked(self):
        world = GenesisWorld(GenesisConfig(size=32, batch=2))
        with self.assertRaises(ValueError):
            world.set_growth_parameters([0.15], [0.015])

    def test_genesis_001_survives_calibrated_center_damage(self):
        specimen = load_specimen(MUTANT)
        world = GenesisWorld.from_specimen(specimen)
        world.step(300)
        state = world.numpy()[0]
        centre = center_of_mass(state[None, :, :])[0]
        damaged, _, actual = calibrated_damage(state, centre, 0.05)
        self.assertAlmostEqual(actual, 0.05, places=5)
        world.state = mx.array(damaged[None, :, :])
        world.step(300)
        self.assertGreater(float(mass(world.numpy())[0]), 40.0)

    def test_calibrated_damage_is_exact_for_fractional_centres(self):
        state = np.ones((32, 32), np.float32)
        _, _, actual = calibrated_damage(state, np.array([15.1234, 16.9876]), 0.05)
        self.assertAlmostEqual(actual, 0.05, places=6)

    def test_aligned_similarity_ignores_toroidal_translation(self):
        state = np.zeros((16, 16), np.float32)
        state[3:6, 5:9] = np.arange(12, dtype=np.float32).reshape(3, 4)
        shifted = np.roll(state, (7, -4), axis=(0, 1))
        self.assertAlmostEqual(aligned_similarity(state, shifted), 1.0, places=6)

    def test_recovery_time_requires_sustained_control_band(self):
        steps = [0, 10, 20, 30, 40, 50]
        observed = [80, 99, 80, 99, 100, 101]
        controls = [100] * len(steps)
        self.assertEqual(
            recovery_time(
                steps,
                observed,
                controls,
                relative_tolerance=0.02,
                consecutive_samples=3,
            ),
            30,
        )
        self.assertEqual(
            threshold_recovery_time(
                steps, [0.5, 0.91, 0.8, 0.91, 0.92, 0.93],
                threshold=0.9, consecutive_samples=3,
            ),
            30,
        )


if __name__ == "__main__":
    unittest.main()
