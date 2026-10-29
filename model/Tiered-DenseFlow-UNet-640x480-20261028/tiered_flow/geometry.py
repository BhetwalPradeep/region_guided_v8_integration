"""All public flows use (x,y) pixels, with fixed(p + flow(p)) matching moving(p)."""
import tensorflow as tf

from .sampling import pixel_grid, sample_at


def dense_image_warp(image, backward_flow_xy, interpolation="bilinear"):
    """Pull warp: output(p) = image(p - backward_flow_xy(p)), zero outside.

    Unlike TensorFlow Addons' discontinued warp, channels are (x,y), not (y,x).
    This function is not installed into the tf.image namespace.
    """
    image = tf.cast(image, tf.float32)
    flow = tf.cast(backward_flow_xy, tf.float32)
    tf.debugging.assert_rank(image, 4)
    tf.debugging.assert_equal(tf.shape(image)[:3], tf.shape(flow)[:3])
    tf.debugging.assert_equal(tf.shape(flow)[-1], 2)
    tf.debugging.assert_all_finite(flow, "Warp flow must be finite")
    coordinates = pixel_grid(tf.shape(image)[1], tf.shape(image)[2])[None] - flow
    if interpolation == "nearest":
        coordinates = tf.floor(coordinates + .5)
    elif interpolation != "bilinear":
        raise ValueError(f"Unknown interpolation: {interpolation}")
    return sample_at(image, coordinates)


def translate(image, translation_xy, interpolation="bilinear"):
    """Apply a spatially constant forward translation to a batched image."""
    shape = tf.shape(image)
    translation_xy = tf.ensure_shape(tf.cast(translation_xy, tf.float32), [None, 2])
    flow = tf.broadcast_to(translation_xy[:, None, None], tf.concat([shape[:3], [2]], 0))
    return dense_image_warp(image, flow, interpolation)


def pull_fixed(fixed, native_flow, native_size):
    """Pull fixed features onto the moving grid, keeping stored flow in native pixels."""
    shape = tf.shape(native_flow)
    scale = tf.cast(tf.stack([shape[2], shape[1]]), tf.float32)
    scale /= tf.cast([native_size[1], native_size[0]], tf.float32)
    return sample_at(fixed, pixel_grid(shape[1], shape[2])[None] + native_flow * scale)


def validate_masks(masks):
    masks = tf.cast(masks, tf.float32)
    tf.debugging.assert_rank(masks, 4)
    tf.debugging.assert_equal(tf.shape(masks)[-1], 5)
    tf.debugging.assert_all_finite(masks, "Nonfinite segmentation mask")
    tf.debugging.assert_greater_equal(masks, 0.)
    tf.debugging.assert_less_equal(masks, 1.)
    tf.debugging.assert_equal(masks, tf.round(masks), message="Expected binary segmentation masks")
    return masks
