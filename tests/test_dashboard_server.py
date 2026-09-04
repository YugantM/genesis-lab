import struct
import unittest

import numpy as np

from genesis.dashboard_server import encode_png, render_atlas


class DashboardRenderingTests(unittest.TestCase):
    def test_png_encoder_writes_expected_dimensions(self):
        image = np.zeros((7, 11, 3), np.uint8)
        encoded = encode_png(image)
        self.assertEqual(encoded[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", encoded[16:24])
        self.assertEqual((width, height), (11, 7))

    def test_atlas_has_nine_tiles_and_visible_body(self):
        states = np.zeros((9, 128, 128), np.float32)
        states[:, 60:68, 60:68] = 0.8
        image = render_atlas(states)
        self.assertEqual(image.shape, (384, 384, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(int(image.max()), 180)


if __name__ == "__main__":
    unittest.main()
