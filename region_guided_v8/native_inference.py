from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

HEIGHT, WIDTH = 480, 640
MASK_TYPES = ("outer_masks", "body", "arms", "face", "hair")


class NativeRegionGuidedV8:
    """Offline native-resolution inference; output vectors are image pixels."""

    def __init__(self, checkpoint: str | Path, config: str | Path | None = None):
        self.checkpoint = Path(checkpoint)
        config_path = Path(config) if config else self.checkpoint.with_name("config.json")
        self.config: dict[str, Any] = json.loads(config_path.read_text())

        if (self.config.get("input_height"), self.config.get("input_width")) != (HEIGHT, WIDTH):
            raise ValueError("Expected a 640x480 native checkpoint")
        if self.config.get("target_direction") != "moving_to_fixed":
            raise ValueError("Checkpoint must declare target_direction=moving_to_fixed")
        if self.config.get("normalize") != "none":
            raise ValueError("This inference path matches normalize=none only")
        if tuple(self.config.get("mask_types", ())) != MASK_TYPES:
            raise ValueError("Checkpoint mask order does not match this adapter")

        # Import lazily so input preparation and unit tests do not require TensorFlow.
        from .train_consecutive import build_region_guided_attention

        self.model = build_region_guided_attention(
            input_shape=(HEIGHT, WIDTH, 12), n_masks=5, norm=self.config.get("norm", "layer")
        )
        self.model.load_weights(str(self.checkpoint))

    @staticmethod
    def build_input(
        fixed_gray_u8: np.ndarray,
        moving_gray_u8: np.ndarray,
        fixed_masks: np.ndarray,
        moving_masks: np.ndarray,
    ) -> np.ndarray:
        """Build the training-format tensor; frames must be grayscale uint8."""
        fixed = np.asarray(fixed_gray_u8)
        moving = np.asarray(moving_gray_u8)
        if fixed.dtype != np.uint8 or moving.dtype != np.uint8:
            raise ValueError("Frames must be uint8 grayscale arrays")
        if fixed.shape != (HEIGHT, WIDTH) or moving.shape != (HEIGHT, WIDTH):
            raise ValueError(f"Frames must have shape ({HEIGHT}, {WIDTH})")

        fm = (np.asarray(fixed_masks) > 0).astype(np.float32)
        mm = (np.asarray(moving_masks) > 0).astype(np.float32)
        expected = (5, HEIGHT, WIDTH)
        if fm.shape != expected or mm.shape != expected:
            raise ValueError(f"Mask arrays must have shape {expected}")

        # Training used Sapiens outer masks and normalize=none.
        f = (fixed.astype(np.float32) / 255.0) * fm[0]
        m = (moving.astype(np.float32) / 255.0) * mm[0]

        x = np.empty((1, HEIGHT, WIDTH, 12), dtype=np.float32)
        x[0, :, :, 0:2] = np.stack((f, m), axis=-1)
        x[0, :, :, 2:7] = np.transpose(fm, (1, 2, 0))
        x[0, :, :, 7:12] = np.transpose(mm, (1, 2, 0))
        return x

    def predict_moving_to_fixed(
        self,
        fixed_gray_u8: np.ndarray,
        moving_gray_u8: np.ndarray,
        fixed_masks: np.ndarray,
        moving_masks: np.ndarray,
    ) -> np.ndarray:
        """Return five (dx, dy) corrections in 640x480 image pixels."""
        x = self.build_input(fixed_gray_u8, moving_gray_u8, fixed_masks, moving_masks)
        raw = self.model(x, training=False)
        if hasattr(raw, "numpy"):
            raw = raw.numpy()
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != (1, 10) or not np.isfinite(raw).all():
            raise ValueError(f"Unexpected model output: shape={raw.shape}")
        # Training negated targets for moving_to_fixed; do not negate again.
        return raw[0].reshape(5, 2)
