"""Standalone experimental loader. No V8 loader or training repository required."""
import hashlib
import json
from pathlib import Path

import tensorflow as tf

from tiered_flow.model import TieredFlowUNet
from tiered_flow.aggregation import RegionalMedianAggregator


class Predictor:
    def __init__(self, directory=None):
        directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
        config = json.loads((directory / "config.json").read_text())
        expected = dict(
            architecture="tiered_correspondence_flow_unet",
            regions=["outer", "body", "arms", "face", "hair"],
            input_shape=[None, 480, 640, 12], direction="moving_to_fixed",
            units="native_pixels", flow_grid="moving", output_scale=1)
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"Unexpected {key}: {config.get(key)!r}")
        checkpoint = directory / "model.keras"
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != config["checkpoint_sha256"]:
            raise ValueError("Checkpoint SHA256 mismatch")
        self.model = tf.keras.models.load_model(checkpoint, compile=False)
        if not isinstance(self.model, TieredFlowUNet) or self.model.recurrent_steps != 6:
            raise ValueError("Unexpected checkpoint architecture")
        self.aggregator = RegionalMedianAggregator(minimum_pixels=100)

    def predict(self, packed):
        """Packed fixed/moving gray then five fixed and five moving binary masks."""
        packed = tf.ensure_shape(tf.cast(packed, tf.float32), [None, 480, 640, 12])
        inputs = dict(fixed=packed[..., :1], moving=packed[..., 1:2],
                      fixed_masks=packed[..., 2:7], moving_masks=packed[..., 7:12])
        flow = self.model(inputs, training=False)
        result = self.aggregator([flow, inputs["fixed_masks"], inputs["moving_masks"]])
        return dict(flow=flow, **result)

    def predict_moving_to_fixed(self, packed):
        """Return [B,10] native corrections; missing non-outer regions are placeholders."""
        result = self.predict(packed)
        tf.debugging.assert_equal(result["global_valid"], True,
                                  message="Outer region is not trackable; use predict() validity")
        return tf.reshape(result["shifts"], [-1, 10])
