import unittest

import numpy as np

from region_guided_v8.native_inference import (
    HEIGHT,
    WIDTH,
    NativeRegionGuidedV8,
)


class NativeInferenceInputTest(unittest.TestCase):
    def setUp(self):
        self.fixed = np.full((HEIGHT, WIDTH), 100, dtype=np.uint8)
        self.moving = np.full((HEIGHT, WIDTH), 200, dtype=np.uint8)
        self.fixed_masks = np.zeros((5, HEIGHT, WIDTH), dtype=np.uint8)
        self.moving_masks = np.zeros((5, HEIGHT, WIDTH), dtype=np.uint8)

        self.fixed_masks[0, 10:20, 30:40] = 1
        self.fixed_masks[1, 12:16, 32:36] = 1
        self.moving_masks[0, 50:60, 70:80] = 1
        self.moving_masks[2, 52:56, 72:76] = 1

    def test_build_input_shape_channels_and_masked_frames(self):
        x = NativeRegionGuidedV8.build_input(
            self.fixed, self.moving, self.fixed_masks, self.moving_masks
        )

        self.assertEqual(x.shape, (1, 480, 640, 12))
        self.assertEqual(x.dtype, np.float32)

        # Frames are scaled to [0, 1] and masked by each outer mask.
        self.assertAlmostEqual(float(x[0, 15, 35, 0]), 100 / 255)
        self.assertEqual(float(x[0, 0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(x[0, 55, 75, 1]), 200 / 255)
        self.assertEqual(float(x[0, 0, 0, 1]), 0.0)

        np.testing.assert_array_equal(x[0, :, :, 2:7], self.fixed_masks.transpose(1, 2, 0))
        np.testing.assert_array_equal(x[0, :, :, 7:12], self.moving_masks.transpose(1, 2, 0))

    def test_rejects_wrong_frame_shape(self):
        with self.assertRaises(ValueError):
            NativeRegionGuidedV8.build_input(
                self.fixed[:100], self.moving, self.fixed_masks, self.moving_masks
            )

    def test_rejects_wrong_mask_shape(self):
        with self.assertRaises(ValueError):
            NativeRegionGuidedV8.build_input(
                self.fixed, self.moving, self.fixed_masks[:4], self.moving_masks
            )

    def test_prediction_returns_native_moving_to_fixed_values_without_negation(self):
        expected = np.arange(10, dtype=np.float32).reshape(5, 2)

        class FakeModel:
            def __call__(self, x, training=False):
                self.seen_shape = x.shape
                return expected.reshape(1, 10)

        adapter = NativeRegionGuidedV8.__new__(NativeRegionGuidedV8)
        adapter.model = FakeModel()

        result = adapter.predict_moving_to_fixed(
            self.fixed, self.moving, self.fixed_masks, self.moving_masks
        )

        self.assertEqual(adapter.model.seen_shape, (1, 480, 640, 12))
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
