"""Inference building blocks copied from the trained architecture."""
import tensorflow as tf
from tensorflow import keras

from .sampling import pixel_grid, sample_at


def block(channels, name):
    return keras.Sequential([
        keras.layers.Conv2D(channels, 3, padding="same"),
        keras.layers.LayerNormalization(axis=-1),
        keras.layers.Activation("swish"),
        keras.layers.Conv2D(channels, 3, padding="same"),
        keras.layers.LayerNormalization(axis=-1),
        keras.layers.Activation("swish"),
    ], name=name)


class ConvGRU(keras.layers.Layer):
    def __init__(self, channels=32, **kwargs):
        super().__init__(**kwargs)
        self.gates = keras.layers.Conv2D(2 * channels, 3, padding="same", activation="sigmoid")
        self.candidate = keras.layers.Conv2D(channels, 3, padding="same", activation="tanh")

    def call(self, hidden, context):
        reset, update = tf.split(self.gates(tf.concat([hidden, context], -1)), 2, -1)
        candidate = self.candidate(tf.concat([reset * hidden, context], -1))
        return (1. - update) * hidden + update * candidate


def local_correlation(fixed, moving, native_flow, native_size, radius=2):
    shape = tf.shape(moving)
    scale = tf.cast(tf.stack([shape[2], shape[1]]), tf.float32)
    scale /= tf.cast(tf.stack([native_size[1], native_size[0]]), tf.float32)
    offsets = pixel_grid(2 * radius + 1, 2 * radius + 1) - float(radius)
    coordinates = pixel_grid(shape[1], shape[2])[None, ..., None, :]
    coordinates = coordinates + native_flow[..., None, :] * scale
    coordinates = coordinates + tf.reshape(offsets, [1, 1, 1, -1, 2])
    sampled = sample_at(tf.math.l2_normalize(fixed, -1), coordinates)
    return tf.reduce_sum(sampled * tf.math.l2_normalize(moving, -1)[..., None, :], -1)
