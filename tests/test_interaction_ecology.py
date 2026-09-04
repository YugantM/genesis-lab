import unittest

import numpy as np

from genesis.interaction_ecology import InteractionConfig, ResponsiveCoupledWorld


class ResponsiveCoupledWorldTests(unittest.TestCase):
    def fixtures(self):
        states = np.zeros((2, 2, 32, 32), np.float32)
        states[:, :, 13:19, 13:19] = 0.5
        values = np.full((2, 2), 0.15, np.float32)
        widths = np.full((2, 2), 0.025, np.float32)
        sensitivities = np.ones((2, 2), np.float32)
        responses = np.zeros((2, 2), np.float32)
        memory = np.zeros((2, 2), np.float32)
        return states, values, widths, sensitivities, responses, memory

    def test_zero_response_matches_identical_candidates(self):
        states, centers, widths, sensitivities, responses, memory = self.fixtures()
        world = ResponsiveCoupledWorld(
            states, centers, widths, sensitivities, responses, memory,
            InteractionConfig(size=32, radius=6, competition=0.5),
        )
        world.step(3)
        np.testing.assert_array_equal(world.numpy()[0], world.numpy()[1])

    def test_boundary_response_changes_contact_dynamics(self):
        states, centers, widths, sensitivities, responses, memory = self.fixtures()
        responses[1] = 0.65
        memory[1] = 0.8
        world = ResponsiveCoupledWorld(
            states, centers, widths, sensitivities, responses, memory,
            InteractionConfig(size=32, radius=6, competition=1.0),
        )
        world.step(5)
        self.assertFalse(np.array_equal(world.numpy()[0], world.numpy()[1]))

    def test_parameter_shapes_are_checked(self):
        states, centers, widths, sensitivities, responses, memory = self.fixtures()
        with self.assertRaises(ValueError):
            ResponsiveCoupledWorld(
                states, centers, widths, sensitivities[:, :1], responses, memory,
                InteractionConfig(size=32),
            )


if __name__ == "__main__":
    unittest.main()
