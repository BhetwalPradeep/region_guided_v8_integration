"""Standalone runtime for the experimental coarse-to-fine matching model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from region_guided_v8.preprocessing import AdapterPreprocessing


def _correlate(a, b, shape):
    a = tf.transpose(a, [0, 3, 1, 2])
    b = tf.transpose(b, [0, 3, 1, 2])
    fa = tf.signal.rfft2d(a, shape)
    fb = tf.signal.rfft2d(b, shape)
    return tf.signal.irfft2d(tf.reduce_sum(fa * tf.math.conj(fb), axis=1), shape)


def correlation_scores(fixed, moving, fixed_mask, moving_mask, radius):
    height, width = fixed.shape[1:3]
    shape = [2 * height, 2 * width]
    numerator = _correlate(fixed * fixed_mask, moving * moving_mask, shape)
    fixed_energy = _correlate(
        tf.reduce_sum(tf.square(fixed), -1, keepdims=True) * fixed_mask,
        moving_mask, shape,
    )
    moving_energy = _correlate(
        fixed_mask, tf.reduce_sum(tf.square(moving), -1, keepdims=True) * moving_mask, shape,
    )
    overlap = tf.maximum(_correlate(fixed_mask, moving_mask, shape), 0.)
    minimum = tf.minimum(tf.reduce_sum(fixed_mask, [1, 2, 3]),
                         tf.reduce_sum(moving_mask, [1, 2, 3]))
    valid = ((overlap >= tf.maximum(.25 * minimum[:, None, None], 4.))
             & (fixed_energy > 1e-6) & (moving_energy > 1e-6))
    scores = numerator / tf.sqrt(tf.maximum(fixed_energy * moving_energy, 1e-12))
    scores = tf.where(valid, tf.clip_by_value(scores, -1., 1.), -1e4)
    ys = tf.math.floormod(tf.range(-radius, radius + 1), shape[0])
    xs = tf.math.floormod(tf.range(-radius, radius + 1), shape[1])
    return tf.gather(tf.gather(scores, ys, axis=1), xs, axis=2)


def peak_shift(scores, stride, center=None, residual_radius=None):
    radius = (scores.shape[1] - 1) // 2
    yy, xx = tf.meshgrid(tf.range(-radius, radius + 1, dtype=tf.float32),
                         tf.range(-radius, radius + 1, dtype=tf.float32), indexing="ij")
    coords = tf.stack([xx, yy], -1) * float(stride)
    if center is not None:
        allowed = tf.reduce_all(
            tf.abs(coords[None] - tf.stop_gradient(center)[:, None, None]) <= residual_radius,
            axis=-1,
        )
        scores = tf.where(allowed, scores, -1e4)
    flat = tf.reshape(scores, [tf.shape(scores)[0], -1])
    indices = tf.argmax(flat, -1, output_type=tf.int32)
    size = 2 * radius + 1
    iy, ix = indices // size, indices % size

    def gather(y, x):
        return tf.gather_nd(scores, tf.stack([tf.range(tf.shape(scores)[0]), y, x], -1))

    top = gather(iy, ix)

    def offset(a, b, interior):
        denominator = a - 2 * top + b
        valid = interior & (denominator < -1e-6) & (a > -2.) & (b > -2.)
        safe = tf.where(valid, denominator, -1.)
        return tf.where(valid, tf.clip_by_value(.5 * (a - b) / safe, -.5, .5), 0.)

    dx = offset(gather(iy, tf.maximum(ix - 1, 0)),
                gather(iy, tf.minimum(ix + 1, size - 1)), (ix > 0) & (ix < size - 1))
    dy = offset(gather(tf.maximum(iy - 1, 0), ix),
                gather(tf.minimum(iy + 1, size - 1), ix), (iy > 0) & (iy < size - 1))
    hard = tf.stack([tf.cast(ix - radius, tf.float32) + dx,
                     tf.cast(iy - radius, tf.float32) + dy], -1) * float(stride)
    probabilities = tf.nn.softmax(flat * 12., axis=-1)
    soft = tf.matmul(probabilities, tf.reshape(coords, [-1, 2]))
    valid = tf.reduce_max(flat, -1) > -2.
    result = hard + soft - tf.stop_gradient(soft)
    return tf.where(valid[:, None], result, tf.zeros_like(result))


@keras.utils.register_keras_serializable(package="coarse_matching")
class ExplicitMatching(keras.layers.Layer):
    def call(self, inputs):
        fine_f, fine_m, coarse_f, coarse_m, fixed_masks, moving_masks = inputs
        fine_masks_f = tf.image.resize(fixed_masks, fine_f.shape[1:3], method="area")
        fine_masks_m = tf.image.resize(moving_masks, fine_m.shape[1:3], method="area")
        coarse_masks_f = tf.image.resize(fixed_masks, coarse_f.shape[1:3], method="area")
        coarse_masks_m = tf.image.resize(moving_masks, coarse_m.shape[1:3], method="area")
        shifts = []
        for region in range(5):
            coarse = peak_shift(correlation_scores(
                coarse_f, coarse_m, coarse_masks_f[..., region:region + 1],
                coarse_masks_m[..., region:region + 1], 20,
            ), 8)
            fine = peak_shift(correlation_scores(
                fine_f, fine_m, fine_masks_f[..., region:region + 1],
                fine_masks_m[..., region:region + 1], 40,
            ), 4, coarse, 16.)
            shift = fine + coarse - tf.stop_gradient(coarse)
            valid = ((tf.reduce_sum(fixed_masks[..., region], [1, 2]) >= 64)
                     & (tf.reduce_sum(moving_masks[..., region], [1, 2]) >= 64))
            shifts.append(tf.where(valid[:, None], shift, tf.zeros_like(shift)))
        return tf.concat(shifts, -1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 10


def build_model():
    image = keras.Input((480, 640, 1))
    x = keras.layers.Conv2D(8, 5, strides=2, padding="same", activation="tanh",
                            use_bias=False)(image)
    fine = keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="tanh",
                               use_bias=False)(x)
    coarse = keras.layers.Conv2D(12, 3, strides=2, padding="same", activation="tanh",
                                 use_bias=False)(fine)
    shared = keras.Model(image, [fine, coarse], name="shared_feature_pyramid")
    inputs = keras.Input((480, 640, 12), name="adapter_input")
    prepared = AdapterPreprocessing(name="adapter_preprocessing")(inputs)
    ff, cf = shared(prepared[..., 0:1])
    fm, cm = shared(prepared[..., 1:2])
    outputs = ExplicitMatching(name="explicit_matching")(
        [ff, fm, cf, cm, prepared[..., 2:7], prepared[..., 7:12]],
    )
    return keras.Model(inputs, outputs, name="coarse_to_fine_native_matching")


def load_export(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    required = {
        "model": "coarse_to_fine_explicit_v1",
        "input_shape": [None, 480, 640, 12],
        "output_shape": [None, 10],
        "target_direction": "moving_to_fixed",
        "output_units": "native_pixels",
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected export {key}: {config.get(key)!r}")
    model = build_model()
    model.load_weights(str(directory / "model.weights.h5"))
    return model


class CoarseToFineMatching:
    """Public native-pixel inference API; this is not the V8 adapter class."""

    def __init__(self, checkpoint, config=None):
        self.checkpoint = Path(checkpoint)
        directory = Path(config).parent if config else self.checkpoint.parent
        if self.checkpoint.name != "model.weights.h5":
            raise ValueError("Expected the matching bundle's model.weights.h5")
        self.model = load_export(directory)

    @staticmethod
    def build_input(fixed_gray_u8, moving_gray_u8, fixed_masks, moving_masks):
        frames = (fixed_gray_u8, moving_gray_u8)
        if any(np.asarray(frame).dtype != np.uint8
               or np.asarray(frame).shape != (480, 640) for frame in frames):
            raise ValueError("Frames must be 640x480 uint8 grayscale")
        masks = [np.asarray(value) for value in (fixed_masks, moving_masks)]
        if any(mask.shape != (5, 480, 640) for mask in masks):
            raise ValueError("Masks must have shape [5,480,640]")
        masks = [(mask > 0).astype(np.float32) for mask in masks]
        x = np.empty((1, 480, 640, 12), dtype=np.float32)
        x[0, ..., :2] = np.stack(
            [frames[0].astype(np.float32) / 255., frames[1].astype(np.float32) / 255.],
            axis=-1,
        )
        x[0, ..., 2:7] = masks[0].transpose(1, 2, 0)
        x[0, ..., 7:12] = masks[1].transpose(1, 2, 0)
        x[0, ..., 0] *= masks[0][0]
        x[0, ..., 1] *= masks[1][0]
        return x

    def predict_moving_to_fixed(self, fixed_gray_u8, moving_gray_u8,
                                fixed_masks, moving_masks):
        inputs = self.build_input(fixed_gray_u8, moving_gray_u8, fixed_masks, moving_masks)
        return self.model(inputs, training=False).numpy()[0].reshape(5, 2)
