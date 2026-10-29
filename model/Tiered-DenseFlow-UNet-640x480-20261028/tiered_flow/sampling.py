"""Moving-grid flow: fixed-coordinate = moving-coordinate + flow, in (x,y) order."""
import tensorflow as tf


def pixel_grid(height, width):
    yy, xx = tf.meshgrid(tf.range(height), tf.range(width), indexing="ij")
    return tf.cast(tf.stack([xx, yy], -1), tf.float32)


def sample_at(image, coordinates):
    """Differentiable bilinear pull sampler, zero outside the image (no wrap)."""
    image = tf.convert_to_tensor(image, tf.float32)
    coordinates = tf.convert_to_tensor(coordinates, tf.float32)
    shape = tf.shape(image)
    x, y = coordinates[..., 0], coordinates[..., 1]
    x0, y0 = tf.floor(x), tf.floor(y)
    batch_shape = tf.concat([[shape[0]], tf.ones([tf.rank(x) - 1], tf.int32)], 0)
    batch = tf.broadcast_to(tf.reshape(tf.range(shape[0]), batch_shape), tf.shape(x))

    def gather(xx, yy):
        valid = (xx >= 0) & (xx < tf.cast(shape[2], tf.float32))
        valid &= (yy >= 0) & (yy < tf.cast(shape[1], tf.float32))
        xi = tf.cast(tf.clip_by_value(xx, 0., tf.cast(shape[2] - 1, tf.float32)), tf.int32)
        yi = tf.cast(tf.clip_by_value(yy, 0., tf.cast(shape[1] - 1, tf.float32)), tf.int32)
        return tf.gather_nd(image, tf.stack([batch, yi, xi], -1)) * tf.cast(valid[..., None], tf.float32)

    wx, wy = x - x0, y - y0
    return (
        gather(x0, y0) * ((1. - wx) * (1. - wy))[..., None]
        + gather(x0 + 1., y0) * (wx * (1. - wy))[..., None]
        + gather(x0, y0 + 1.) * ((1. - wx) * wy)[..., None]
        + gather(x0 + 1., y0 + 1.) * (wx * wy)[..., None]
    )


def pull_fixed(fixed, native_flow, native_size):
    """Sample fixed on moving grid; native_flow always remains in native pixels."""
    shape = tf.shape(native_flow)
    scale = tf.cast(tf.stack([shape[2], shape[1]]), tf.float32)
    scale /= tf.cast(tf.stack([native_size[1], native_size[0]]), tf.float32)
    coordinates = pixel_grid(shape[1], shape[2])[None] + native_flow * scale
    return sample_at(fixed, coordinates)


def robust_regions(flow, fixed_masks, moving_masks, minimum_pixels=16):
    """Three Huber-IRLS steps; absent regions have zero placeholders plus validity."""
    tf.debugging.assert_all_finite(flow, "Nonfinite predicted flow")
    count = tf.reduce_sum(moving_masks, [1, 2])
    valid = (count >= minimum_pixels) & (tf.reduce_sum(fixed_masks, [1, 2]) >= minimum_pixels)
    weight = moving_masks[..., None]
    values = flow[..., None, :]
    estimate = tf.math.divide_no_nan(
        tf.reduce_sum(values * weight, [1, 2]), count[..., None])
    for _ in range(3):
        distance = tf.norm(values - estimate[:, None, None], axis=-1, keepdims=True)
        robust = tf.stop_gradient(tf.minimum(1., 5. / tf.maximum(distance, 1e-6)))
        weights = weight * robust
        estimate = tf.math.divide_no_nan(
            tf.reduce_sum(values * weights, [1, 2]), tf.reduce_sum(weights, [1, 2]))
    estimate = tf.where(valid[..., None], estimate, 0.)
    return tf.reshape(estimate, [-1, 10]), valid
