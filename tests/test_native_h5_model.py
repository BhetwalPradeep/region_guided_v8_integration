import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from region_guided_v8.native_inference import NativeRegionGuidedV8
from region_guided_v8.preprocessing import AdapterPreprocessing


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "model/V8-RG-deployment-matched-640x480-20261026"


class NativeH5ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = NativeRegionGuidedV8(ARTIFACT / "model.weights.h5")
        cls.saved = tf.saved_model.load(str(ARTIFACT / "saved_model"))

    def test_h5_loader_matches_complete_export_and_legacy_input_builder(self):
        rng = np.random.default_rng(42)
        reference = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        current = np.roll(reference, 40, axis=1)
        masks = np.zeros((5, 480, 640), np.uint8)
        masks[:, 80:400, 120:480] = 1
        moving_masks = np.roll(masks, 40, axis=2)
        x = NativeRegionGuidedV8.build_input(reference, current, masks, moving_masks)
        unmasked = x.copy()
        unmasked[0, ..., 0] = reference.astype(np.float32) / 255
        unmasked[0, ..., 1] = current.astype(np.float32) / 255
        expected = self.saved.signatures["serving_default"](
            input=tf.constant(unmasked)
        )["corrections"].numpy()
        actual = self.model.model(x, training=False).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-5)
        np.testing.assert_allclose(
            self.model.predict_moving_to_fixed(reference, current, masks, moving_masks),
            expected.reshape(5, 2), atol=1e-4, rtol=1e-5,
        )
        self.assertEqual(self.model.model.input_shape, (None, 480, 640, 12))
        self.assertEqual(self.model.model.output_shape, (None, 10))

    def test_normalization_once_and_masks_unchanged(self):
        x = np.zeros((1, 8, 8, 12), np.float32)
        x[0, ..., 0] = np.arange(64).reshape(8, 8) / 255
        x[0, ..., 1] = x[0, ..., 0]
        x[:, 2:6, 2:6, 2:] = 1
        actual = AdapterPreprocessing()(x).numpy()
        pixels = x[0, 2:6, 2:6, 0]
        expected = (pixels - pixels.mean()) / pixels.std()
        np.testing.assert_allclose(actual[0, 2:6, 2:6, 0], expected, atol=1e-6)
        np.testing.assert_array_equal(actual[..., 2:], x[..., 2:])
        np.testing.assert_array_equal(actual[:, :2, :, :2], 0)

    def test_empty_constant_masks_are_finite(self):
        x = np.zeros((1, 8, 8, 12), np.float32)
        x[..., :2] = 0.5
        x[..., 7] = 1
        actual = AdapterPreprocessing()(x).numpy()
        np.testing.assert_array_equal(actual[..., :2], 0)
        self.assertTrue(np.isfinite(actual).all())

    def test_invalid_input_is_rejected(self):
        for channel, value in ((0, 255), (2, 0.5), (0, np.nan)):
            with self.subTest(channel=channel, value=value):
                x = np.zeros((1, 8, 8, 12), np.float32)
                x[..., channel] = value
                with self.assertRaises(tf.errors.InvalidArgumentError):
                    AdapterPreprocessing()(x)

    def test_unknown_preprocessing_is_rejected_before_loading_weights(self):
        config = json.loads((ARTIFACT / "config.json").read_text())
        config["internal_preprocessing"] = "unknown"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Unsupported internal_preprocessing"):
                NativeRegionGuidedV8(ARTIFACT / "model.weights.h5", path)

    def test_legacy_checkpoint_still_has_unwrapped_architecture(self):
        legacy = NativeRegionGuidedV8(
            ROOT / "model/V8-RG-native-640x480-20261016/model.weights.h5"
        )
        self.assertFalse(any(isinstance(layer, AdapterPreprocessing) for layer in legacy.model.layers))
        self.assertTrue(any(isinstance(layer, AdapterPreprocessing) for layer in self.model.model.layers))


if __name__ == "__main__":
    unittest.main()
