import unittest

import numpy as np

from genesis.coupled_ecology import CoupledConfig, CoupledWorld, aligned_similarity
from genesis.validate_ecology import exact_sign_test


class CoupledEcologyTests(unittest.TestCase):
    def test_parameter_shapes_are_checked(self):
        states = np.zeros((2, 2, 32, 32), np.float32)
        with self.assertRaises(ValueError):
            CoupledWorld(states, np.ones((2, 1)), np.ones((2, 2)), CoupledConfig(size=32))

    def test_zero_coupling_keeps_identical_channels_identical(self):
        rng = np.random.default_rng(4)
        state = rng.random((1, 1, 32, 32), dtype=np.float32)
        states = np.repeat(state, 2, axis=1)
        parameters = np.full((1, 2), 0.15, np.float32)
        widths = np.full((1, 2), 0.025, np.float32)
        world = CoupledWorld(
            states, parameters, widths, CoupledConfig(size=32, competition=0)
        )
        final = world.step(3)
        self.assertTrue(np.allclose(final[:, 0], final[:, 1]))

    def test_competition_never_increases_total_mass(self):
        rng = np.random.default_rng(7)
        states = rng.random((1, 2, 32, 32), dtype=np.float32) * 0.3
        centers = np.full((1, 2), 0.15, np.float32)
        widths = np.full((1, 2), 0.025, np.float32)
        control = CoupledWorld(states, centers, widths, CoupledConfig(size=32, competition=0))
        coupled = CoupledWorld(states, centers, widths, CoupledConfig(size=32, competition=1))
        self.assertLessEqual(float(coupled.step().sum()), float(control.step().sum()) + 1e-5)

    def test_similarity_is_translation_invariant(self):
        state = np.zeros((32, 32), np.float32)
        state[11:16, 7:14] = np.arange(35, dtype=np.float32).reshape(5, 7) / 35
        shifted = np.roll(state, (8, -9), axis=(0, 1))
        self.assertGreater(aligned_similarity(state, shifted), 0.999)

    def test_exact_sign_test_is_symmetric(self):
        self.assertAlmostEqual(exact_sign_test(8, 2), exact_sign_test(2, 8))
        self.assertEqual(exact_sign_test(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
