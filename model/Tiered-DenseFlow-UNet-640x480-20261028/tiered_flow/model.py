"""Memory-safe global -> local recurrent -> native refinement U-Net."""
import tensorflow as tf
from tensorflow import keras

from .layers import block, ConvGRU, local_correlation
from .geometry import pixel_grid, pull_fixed, validate_masks


@keras.utils.register_keras_serializable(package="tiered_flow")
class GlobalCorrelation(keras.layers.Layer):
    def __init__(self, temperature=20., **kwargs):
        super().__init__(**kwargs)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def call(self, inputs):
        fixed, moving, fixed_outer = inputs
        shape = tf.shape(moving)
        tf.debugging.assert_less_equal(shape[1] * shape[2], 1200,
                                        message="All-pairs matching is restricted to <=30x40")
        f = tf.reshape(tf.math.l2_normalize(tf.cast(fixed, tf.float32), -1),
                       [shape[0], -1, shape[3]])
        m = tf.reshape(tf.math.l2_normalize(tf.cast(moving, tf.float32), -1),
                       [shape[0], -1, shape[3]])
        scores = tf.linalg.matmul(m, f, transpose_b=True) * self.temperature
        allowed = tf.reshape(fixed_outer > 0., [shape[0], 1, -1])
        tf.debugging.assert_equal(tf.reduce_any(allowed, -1), True,
                                  message="Cannot match an empty fixed outer mask")
        probabilities = tf.nn.softmax(tf.where(allowed, scores, -1e4), axis=-1)
        grid = pixel_grid(shape[1], shape[2])
        matched = tf.linalg.matmul(probabilities, tf.reshape(grid, [-1, 2]))
        flow = tf.reshape(matched, [shape[0], shape[1], shape[2], 2]) - grid[None]
        volume = tf.reshape(probabilities, [shape[0], shape[1], shape[2], -1])
        return flow, volume

    def get_config(self):
        return dict(super().get_config(), temperature=self.temperature)


@keras.utils.register_keras_serializable(package="tiered_flow")
class TieredFlowUNet(keras.Model):
    """Public call returns native dense (Vx,Vy); call(..., return_sequence=True) trains."""
    def __init__(self, input_size=(480, 640), recurrent_steps=6, **kwargs):
        super().__init__(**kwargs)
        if len(input_size) != 2 or any(s < 32 or s % 16 for s in input_size):
            raise ValueError("Input dimensions must be multiples of 16 and >=32")
        if input_size[0] * input_size[1] > 480 * 640:
            raise ValueError("Native input exceeds the verified 480x640 global-cost budget")
        if recurrent_steps != 6:
            raise ValueError("This tiered experiment requires exactly six recurrent steps")
        self.input_size = tuple(input_size)
        self.recurrent_steps = recurrent_steps
        self.encoder = [block(c, f"encoder_{i}") for i, c in enumerate((12, 16, 24, 32, 48))]
        self.pool = keras.layers.AveragePooling2D(2)
        self.global_match = GlobalCorrelation(name="global_correlation_1_16", dtype="float32")
        self.decoder = [block(c, f"decoder_skip_{i}") for i, c in enumerate((12, 16, 24, 32, 48))]
        self.residual = [keras.layers.Conv2D(
            2, 3, padding="same", activation=None, dtype="float32",
            kernel_initializer="zeros", name=f"linear_flow_residual_{i}") for i in range(5)]
        self.context = block(32, "quarter_resolution_context")
        self.gru = ConvGRU(32, name="shared_convgru")
        self.gru_delta = keras.layers.Conv2D(
            2, 3, padding="same", activation=None, dtype="float32",
            kernel_initializer="zeros", name="linear_recurrent_residual")

    def encode(self, image):
        pyramid = []
        for i, layer in enumerate(self.encoder):
            image = layer(image)
            pyramid.append(image)
            if i < 4:
                image = self.pool(image)
        return pyramid

    @staticmethod
    def standardize(frame, outer):
        count = tf.reduce_sum(outer, [1, 2], keepdims=True)
        mean = tf.math.divide_no_nan(tf.reduce_sum(frame * outer, [1, 2], keepdims=True), count)
        variance = tf.math.divide_no_nan(
            tf.reduce_sum(tf.square(frame - mean) * outer, [1, 2], keepdims=True), count)
        return (frame - mean) * tf.math.rsqrt(tf.maximum(variance, 1e-6)) * outer

    def call(self, inputs, training=False, return_sequence=False):
        fixed = tf.ensure_shape(tf.cast(inputs["fixed"], tf.float32), [None, *self.input_size, 1])
        moving = tf.ensure_shape(tf.cast(inputs["moving"], tf.float32), [None, *self.input_size, 1])
        fixed_masks = tf.ensure_shape(validate_masks(inputs["fixed_masks"]),
                                     [None, *self.input_size, 5])
        moving_masks = tf.ensure_shape(validate_masks(inputs["moving_masks"]),
                                      [None, *self.input_size, 5])
        tf.debugging.assert_equal(tf.shape(fixed), tf.shape(moving))
        tf.debugging.assert_equal(tf.shape(fixed_masks), tf.shape(moving_masks))
        tf.debugging.assert_equal(tf.shape(fixed)[0], tf.shape(fixed_masks)[0])
        for frame in (fixed, moving):
            tf.debugging.assert_all_finite(frame, "Nonfinite grayscale input")
            tf.debugging.assert_greater_equal(frame, 0.)
            tf.debugging.assert_less_equal(frame, 1., message="Divide grayscale by 255 exactly once")
        tf.debugging.assert_greater_equal(tf.reduce_sum(fixed_masks[..., 0], [1, 2]), 100.,
                                          message="Fixed outer mask must contain >=100 pixels")
        fixed = self.encode(tf.concat([self.standardize(fixed, fixed_masks[..., :1]), fixed_masks], -1))
        moving = self.encode(tf.concat([self.standardize(moving, moving_masks[..., :1]), moving_masks], -1))
        size = tf.shape(fixed[-1])[1:3]
        outer = tf.image.resize(fixed_masks[..., :1], size, method="area")
        flow, _ = self.global_match([fixed[-1], moving[-1], outer])
        flow *= tf.cast([self.input_size[1], self.input_size[0]], tf.float32) / tf.cast(
            tf.stack([size[1], size[0]]), tf.float32)
        sequence = []
        decoded = None
        for level in range(4, -1, -1):
            size = tf.shape(moving[level])[1:3]
            flow = tf.image.resize(flow, size, method="bilinear")
            pieces = [moving[level], pull_fixed(fixed[level], flow, self.input_size),
                      flow / float(2**level)]
            if decoded is not None:
                pieces.append(tf.image.resize(decoded, size))
            if level in (2, 3):
                pieces.append(local_correlation(fixed[level], moving[level], flow, self.input_size))
            decoded = self.decoder[level](tf.concat(pieces, -1), training=training)
            flow = flow + self.residual[level](decoded) * float(2**level)
            if level != 2:
                sequence.append(flow)
            else:
                hidden = self.context(decoded, training=training)
                for _ in range(self.recurrent_steps):
                    cost = local_correlation(fixed[level], moving[level], flow, self.input_size)
                    hidden = self.gru(hidden, tf.concat([decoded, cost, flow / 4.], -1))
                    flow = flow + self.gru_delta(hidden) * 4.
                    sequence.append(flow)
                decoded = tf.concat([decoded, hidden], -1)
        tf.debugging.assert_all_finite(flow, "Nonfinite predicted dense flow")
        return tuple(sequence) if return_sequence else flow

    def get_config(self):
        return dict(super().get_config(), input_size=self.input_size,
                    recurrent_steps=self.recurrent_steps)

    def get_build_config(self):
        return {"input_size": self.input_size}

    def build_from_config(self, config):
        if tuple(config["input_size"]) != self.input_size:
            raise ValueError("Saved build dimensions disagree with model configuration")
        frame = tf.zeros([1, *self.input_size, 1])
        masks = tf.ones([1, *self.input_size, 5])
        self(dict(fixed=frame, moving=frame, fixed_masks=masks, moving_masks=masks))
