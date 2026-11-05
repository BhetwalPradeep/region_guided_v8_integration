"""Inference regression tests; requires this bundle's actual checkpoint."""
import os
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf

from loader import Predictor
from tiered_flow.geometry import translate


class CompiledPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        explicit = os.environ.get("TIERED_TEST_CHECKPOINT")
        checkpoint = Path(explicit) if explicit else Path(__file__).with_name("model.weights.h5")
        if not explicit and (not checkpoint.exists() or checkpoint.stat().st_size < 1024):
            raise unittest.SkipTest("Fetch model.weights.h5 with git lfs pull first")
        cls.predictor = Predictor(checkpoint=checkpoint)
        image = tf.random.stateless_uniform([1, 480, 640, 1], [41, 52])
        yy, xx = tf.meshgrid(tf.range(480), tf.range(640), indexing="ij")
        outer = tf.cast((yy > 60) & (yy < 420) & (xx > 80) & (xx < 560), tf.float32)
        masks = tf.repeat(outer[None, ..., None], 5, axis=-1)
        cls.packed = tf.concat([
            image, translate(image, [[20., -10.]]),
            masks, translate(masks, [[20., -10.]], "nearest"),
        ], axis=-1)

    def assert_outputs_close(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            a, b = expected[key].numpy(), actual[key].numpy()
            if a.dtype.kind in "biu":
                np.testing.assert_array_equal(a, b, err_msg=key)
            else:
                np.testing.assert_allclose(a, b, atol=1e-3, rtol=1e-4, err_msg=key)

    def test_eager_compiled_outputs(self):
        for translation in ((20., -10.), (-3.5, 2.25)):
            packed = self.packed
            if translation != (20., -10.):
                packed = tf.concat([
                    packed[..., :1], translate(packed[..., :1], [translation]),
                    packed[..., 2:7], translate(packed[..., 2:7], [translation], "nearest"),
                ], axis=-1)
            with self.subTest(translation=translation):
                self.assert_outputs_close(
                    self.predictor._predict_impl(packed),
                    self.predictor.predict(packed))
        self.assertEqual(self.predictor.predict(self.packed)["flow"].shape, (1, 480, 640, 2))

    def test_outer_only_and_invalid_outer(self):
        packed = self.packed.numpy()
        packed[..., 3:7] = 0
        packed[..., 8:12] = 0
        result = self.predictor.predict(packed)
        np.testing.assert_array_equal(result["valid"], [[True, False, False, False, False]])
        np.testing.assert_array_equal(result["shifts"].numpy()[:, 1:], 0)
        shifts = self.predictor.predict_moving_to_fixed(packed)
        np.testing.assert_allclose(shifts, tf.reshape(result["shifts"], [1, 10]))
        packed[..., 7] = 0
        self.assertFalse(bool(self.predictor.predict(packed)["global_valid"][0]))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.predictor.predict_moving_to_fixed(packed)

    def test_validation_and_graph_reuse(self):
        with self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
            self.predictor.predict(tf.zeros([1, 480, 640, 11]))
        invalid = self.packed.numpy()
        invalid[0, 0, 0, 0] = np.nan
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.predictor.predict(invalid)
        self.predictor.predict(self.packed)
        count = self.predictor._compiled_predict.experimental_get_tracing_count()
        self.predictor.predict(self.packed.numpy())
        self.assertEqual(self.predictor._compiled_predict.experimental_get_tracing_count(), count)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
