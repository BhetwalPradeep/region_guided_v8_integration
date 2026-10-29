"""TensorFlow-native exact component-wise medians and rigid-mask overlap metrics."""
import tensorflow as tf
from tensorflow import keras

from . import REGIONS
from .geometry import translate, validate_masks


def masked_component_median(flow, active, minimum_pixels=100):
    """Sort foreground values and gather the middle pair (exact odd/even median).

    tf.experimental.numpy.median is not present in the supported TensorFlow build.
    Sorting with +inf padding is graph-safe and excludes background zero values.
    """
    count = tf.reduce_sum(tf.cast(active, tf.int32), [1, 2])
    values = tf.reshape(tf.where(active[..., None], flow, tf.constant(float("inf"), tf.float32)),
                        [tf.shape(flow)[0], -1, 2])
    values = tf.sort(values, axis=1)
    batch = tf.range(tf.shape(flow)[0])
    lower = tf.stack([batch, tf.maximum((count - 1) // 2, 0)], -1)
    upper = tf.stack([batch, tf.maximum(count // 2, 0)], -1)
    median = .5 * (tf.gather_nd(values, lower) + tf.gather_nd(values, upper))
    valid = count >= minimum_pixels
    return tf.where(valid[:, None], median, 0.), valid, count


def overlap_scores(fixed, moving):
    """Binary or soft overlap; empty/empty has no tracking evidence and scores zero."""
    intersection = tf.reduce_sum(fixed * moving, [1, 2])
    area = tf.reduce_sum(fixed + moving, [1, 2])
    dice = tf.math.divide_no_nan(2. * intersection, area)
    iou = tf.math.divide_no_nan(intersection, area - intersection)
    return dice, iou


@keras.utils.register_keras_serializable(package="tiered_flow")
class RegionalMedianAggregator(keras.layers.Layer):
    def __init__(self, minimum_pixels=100, **kwargs):
        super().__init__(**kwargs)
        if minimum_pixels < 1:
            raise ValueError("minimum_pixels must be positive")
        self.minimum_pixels = int(minimum_pixels)

    def call(self, inputs):
        flow, fixed_masks, moving_masks = inputs
        flow = tf.cast(flow, tf.float32)
        fixed_masks, moving_masks = validate_masks(fixed_masks), validate_masks(moving_masks)
        tf.debugging.assert_all_finite(flow, "Cannot aggregate nonfinite flow")
        tf.debugging.assert_equal(tf.shape(flow)[-1], 2)
        tf.debugging.assert_equal(tf.shape(flow)[:3], tf.shape(moving_masks)[:3])
        tf.debugging.assert_equal(tf.shape(fixed_masks), tf.shape(moving_masks))
        shifts, validity, counts, after = [], [], [], []
        fixed_count = tf.reduce_sum(fixed_masks, [1, 2])
        for region in range(len(REGIONS)):
            moving = moving_masks[..., region:region + 1]
            shift, valid, count = masked_component_median(
                flow, moving[..., 0] > .5, self.minimum_pixels)
            valid &= fixed_count[:, region] >= self.minimum_pixels
            shift = tf.where(valid[:, None], shift, 0.)
            # shift is a forward moving->fixed translation. Our pull warp samples
            # input(p - shift), so applying this shift (not -shift) aligns the mask.
            warped = translate(moving, shift)
            shifts.append(shift)
            validity.append(valid)
            counts.append(count)
            after.append(warped)
        valid = tf.stack(validity, -1)
        before_dice, before_iou = overlap_scores(fixed_masks, moving_masks)
        after_dice, after_iou = overlap_scores(fixed_masks, tf.concat(after, -1))
        shifts = tf.stack(shifts, axis=1)
        return dict(
            shifts=shifts, valid=valid, moving_pixel_count=tf.stack(counts, -1),
            global_couch_shift=shifts[:, 0], global_valid=valid[:, 0],
            dice_before=before_dice, iou_before=before_iou,
            dice_after=tf.where(valid, after_dice, 0.),
            iou_after=tf.where(valid, after_iou, 0.))

    def get_config(self):
        return dict(super().get_config(), minimum_pixels=self.minimum_pixels)
