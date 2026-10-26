"""Model-side preprocessing for deployment-matched native checkpoints."""
import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="native_shift")
class AdapterPreprocessing(keras.layers.Layer):
    """Normalize already unit-range frames inside the supplied outer masks."""

    def call(self, inputs):
        frames = inputs[..., :2]
        masks = inputs[..., 2:]
        checks = [
            tf.debugging.assert_all_finite(inputs, "Adapter input must be finite"),
            tf.debugging.assert_greater_equal(frames, 0.0, message="Grayscale must be in [0,1]"),
            tf.debugging.assert_less_equal(frames, 1.0, message="Grayscale is already divided by 255"),
            tf.debugging.assert_greater_equal(masks, 0.0, message="Masks must be binary 0/1"),
            tf.debugging.assert_less_equal(masks, 1.0, message="Masks must be binary 0/1"),
            tf.debugging.assert_equal(masks, tf.round(masks), message="Masks must be binary 0/1"),
        ]
        with tf.control_dependencies(checks):
            frames = tf.identity(frames)
            outer = tf.stack([masks[..., 0], masks[..., 5]], axis=-1)
            count = tf.reduce_sum(outer, axis=[1, 2], keepdims=True)
            denominator = tf.maximum(count, 1.0)
            mean = tf.reduce_sum(frames * outer, axis=[1, 2], keepdims=True) / denominator
            variance = tf.reduce_sum(
                tf.square(frames - mean) * outer, axis=[1, 2], keepdims=True,
            ) / denominator
            sigma = tf.sqrt(variance)
            sigma = tf.where(sigma < 1e-6, tf.ones_like(sigma), sigma)
            return tf.concat([(frames - mean) / sigma * outer, masks], axis=-1)


def wrap_adapter_input(core):
    if core.input_shape[1:] != (480, 640, 12) or core.output_shape[1:] != (10,):
        raise ValueError(f"Unexpected native core shapes: {core.input_shape}, {core.output_shape}")
    inputs = keras.Input(shape=(480, 640, 12), dtype="float32", name="input")
    prepared = AdapterPreprocessing(name="adapter_preprocessing")(inputs)
    return keras.Model(inputs, core(prepared), name="native_adapter_compatible")
