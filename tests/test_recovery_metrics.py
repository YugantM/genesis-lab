"""Adversarial contracts for the matched-control recovery measurements."""

import unittest

import numpy as np

from genesis.metrics import (
    aligned_similarity,
    center_of_mass,
    functional_recovery,
    functional_recovery_time,
    functional_score,
    mass,
    occupied_fraction,
    recovery_time,
    survival,
    threshold_recovery_time,
    toroidal_displacement,
)


class RecoveryMetricTests(unittest.TestCase):
    def test_fft_alignment_matches_exhaustive_translations(self):
        rng = np.random.default_rng(531)
        left = rng.random((5, 7))
        right = rng.random((5, 7))
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        exhaustive = max(
            float(np.sum(left * np.roll(right, (dy, dx), axis=(0, 1))))
            / denominator
            for dy in range(left.shape[0])
            for dx in range(left.shape[1])
        )
        self.assertAlmostEqual(aligned_similarity(left, right), exhaustive, places=12)

    def test_empty_world_is_not_shape_recovery(self):
        empty = np.zeros((8, 8))
        body = empty.copy()
        body[3:5, 3:5] = 1
        self.assertEqual(aligned_similarity(empty, empty), 0)
        self.assertEqual(aligned_similarity(empty, body), 0)

    def test_joint_extinction_is_not_mass_recovery(self):
        self.assertIsNone(recovery_time([0, 10, 20], [0, 0, 0], [0, 0, 0]))

    def test_invalid_samples_interrupt_the_sustained_band(self):
        steps = [0, 10, 20, 30, 40, 50, 60]
        observed = [99, 100, np.nan, 101, 100, 99, 100]
        self.assertEqual(recovery_time(steps, observed, [100] * 7), 30)

    def test_infinite_control_is_not_mass_recovery(self):
        self.assertIsNone(
            recovery_time([0, 10, 20], [1, 1, 1], [np.inf, np.inf, np.inf])
        )

    def test_band_tracks_the_matched_control_not_initial_mass(self):
        self.assertEqual(
            recovery_time([0, 10, 20, 30], [80, 105, 110, 115], [100, 105, 110, 115]),
            10,
        )

    def test_insufficient_followup_is_censored(self):
        self.assertIsNone(recovery_time([0, 10], [100, 100], [100, 100]))
        self.assertIsNone(
            threshold_recovery_time([0, 10], [1, 1], threshold=0.9)
        )

    def test_time_axis_must_be_finite_and_strictly_increasing(self):
        for steps in ([10, 0, 20], [0, 0, 20], [0, np.nan, 20]):
            with self.subTest(steps=steps):
                with self.assertRaises(ValueError):
                    recovery_time(steps, [100] * 3, [100] * 3)
                with self.assertRaises(ValueError):
                    threshold_recovery_time(steps, [1] * 3, threshold=0.9)

    def test_threshold_recovery_ignores_nonfinite_measurements(self):
        self.assertIsNone(
            threshold_recovery_time([0, 10, 20], [np.inf] * 3, threshold=0.9)
        )

    def test_functional_endpoint_separates_persistence_and_recovery(self):
        base = dict(control_valid=True, mass_ratio=1.0, motion_ratio=1.0, occupied=0.01)
        self.assertTrue(functional_recovery(**base))
        for changed in (
            {"control_valid": False},
            {"motion_ratio": 0.0},
            {"mass_ratio": 0.0},
            {"mass_ratio": 1.5},
            {"occupied": 0.5},
            {"mass_ratio": np.nan},
            {"motion_ratio": np.inf},
        ):
            with self.subTest(changed=changed):
                self.assertFalse(functional_recovery(**{**base, **changed}))

    def test_functional_score_penalizes_both_growth_and_loss(self):
        self.assertEqual(functional_score(1.0, 1.0, True), 1.0)
        self.assertEqual(functional_score(1.0, 2.0, True), 1.0)
        self.assertEqual(functional_score(0.5, 1.0, True), 0.5)
        self.assertEqual(functional_score(2.0, 1.0, True), 0.5)
        self.assertEqual(functional_score(1.0, 0.0, True), 0.0)
        self.assertEqual(functional_score(1.0, 1.0, False), 0.0)
        self.assertEqual(functional_score(np.nan, 1.0, True), 0.0)

    def test_functional_time_uses_the_same_sustained_endpoint(self):
        self.assertEqual(
            functional_recovery_time(
                [50, 60, 70, 80, 90, 100],
                [1.0] * 6,
                [1.0, 0.0, 0.9, 1.0, 1.0, 1.0],
                [0.01] * 6,
                [True] * 6,
            ),
            70,
        )
        self.assertIsNone(
            functional_recovery_time(
                [50, 60, 70], [1.0] * 3, [1.0] * 3, [0.01] * 3, [False] * 3
            )
        )

    def test_mass_shape_motion_and_occupancy_separate_failure_modes(self):
        # Movement is deliberately measured separately from shape. A shifted
        # body has identical anatomy; matching anatomy alone does not establish
        # resumed locomotion after injury.
        body = np.zeros((32, 32))
        body[14:17, 13:18] = 1
        translated = np.roll(body, (0, 4), axis=(0, 1))
        death = np.zeros_like(body)
        explosion = np.ones_like(body)
        states = np.stack([body, translated, death, explosion])
        np.testing.assert_array_equal(survival(states), [True, True, False, True])
        np.testing.assert_allclose(mass(states), [15, 15, 0, 1024])
        self.assertAlmostEqual(aligned_similarity(body, translated), 1)
        self.assertLess(aligned_similarity(body, explosion), 0.2)
        self.assertLess(float(occupied_fraction(body)), 0.03)
        self.assertEqual(float(occupied_fraction(explosion)), 1)
        centers = center_of_mass(states[:2])
        np.testing.assert_allclose(toroidal_displacement(centers[0], centers[1], 32), [0, 4])
        np.testing.assert_array_equal(toroidal_displacement(centers[0], centers[0], 32), [0, 0])


if __name__ == "__main__":
    unittest.main()
