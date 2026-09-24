from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from train_consecutive import build_region_guided_attention
from warp_fn import build_dvf, warp_with_dvf

H = W = 256
N_MASKS = 5
MASK_TYPES = ["outer_masks", "body", "arms", "face", "hair"]


class RegionGuidedV8:
    """Locked region-guided V8 inference wrapper.

    The network predicts fixed-to-moving displacement vectors. Clinical
    registration applies the inverse vectors to the moving image.
    """

    def __init__(self, checkpoint: str | Path, config: str | Path | None = None):
        self.checkpoint = Path(checkpoint)
        self.config_path = Path(config) if config else self.checkpoint.with_name("config.json")
        self.config: dict[str, Any] = json.loads(self.config_path.read_text()) if self.config_path.exists() else {}
        self.model = build_region_guided_attention(input_shape=(H, W, 12), n_masks=N_MASKS, norm="layer")
        self.model.load_weights(str(self.checkpoint))

    @staticmethod
    def mask_and_normalize(frame_u8: np.ndarray, yolo_mask: np.ndarray, sapiens_outer: np.ndarray) -> np.ndarray:
        frame = frame_u8.astype(np.float32) / 255.0
        mask = yolo_mask.astype(np.float32)
        if mask.sum() < 100:
            mask = sapiens_outer.astype(np.float32)
        frame = frame * mask
        if mask.sum() > 0:
            pixels = frame[mask > 0]
            frame = (frame - pixels.mean()) / max(float(pixels.std()), 1e-6)
            frame = frame * mask
        return frame.astype(np.float32)

    @staticmethod
    def build_input(
        fixed_frame_u8: np.ndarray,
        moving_frame_u8: np.ndarray,
        fixed_masks: np.ndarray,
        moving_masks: np.ndarray,
        fixed_yolo_mask: np.ndarray | None = None,
        moving_yolo_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        fixed_masks = np.asarray(fixed_masks)
        moving_masks = np.asarray(moving_masks)
        if fixed_masks.shape != (N_MASKS, H, W) or moving_masks.shape != (N_MASKS, H, W):
            raise ValueError("fixed_masks and moving_masks must have shape (5, 256, 256)")
        if fixed_yolo_mask is None:
            fixed_yolo_mask = fixed_masks[0]
        if moving_yolo_mask is None:
            moving_yolo_mask = moving_masks[0]
        x = np.empty((1, H, W, 12), np.float32)
        x[0, :, :, 0] = RegionGuidedV8.mask_and_normalize(fixed_frame_u8, fixed_yolo_mask, fixed_masks[0])
        x[0, :, :, 1] = RegionGuidedV8.mask_and_normalize(moving_frame_u8, moving_yolo_mask, moving_masks[0])
        x[0, :, :, 2:7] = np.transpose(fixed_masks.astype(np.float32), (1, 2, 0))
        x[0, :, :, 7:12] = np.transpose(moving_masks.astype(np.float32), (1, 2, 0))
        return x

    def predict_fixed_to_moving(
        self,
        fixed_frame_u8: np.ndarray,
        moving_frame_u8: np.ndarray,
        fixed_masks: np.ndarray,
        moving_masks: np.ndarray,
        fixed_yolo_mask: np.ndarray | None = None,
        moving_yolo_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        x = self.build_input(fixed_frame_u8, moving_frame_u8, fixed_masks, moving_masks, fixed_yolo_mask, moving_yolo_mask)
        return self.model(x, training=False).numpy()[0].reshape(N_MASKS, 2).astype(np.float32)

    @staticmethod
    def inverse_displacements(predicted_dxdy: np.ndarray) -> np.ndarray:
        return -np.asarray(predicted_dxdy, np.float32).reshape(N_MASKS, 2)

    def register_moving_to_fixed(
        self,
        moving_image: np.ndarray,
        moving_masks: np.ndarray,
        predicted_fixed_to_moving: np.ndarray,
        sigma: float = 8.0,
        nearest: bool = False,
    ) -> np.ndarray:
        moving_to_fixed = self.inverse_displacements(predicted_fixed_to_moving)
        dvf = build_dvf(moving_to_fixed, moving_masks, sigma=sigma)
        return warp_with_dvf(moving_image, dvf, nearest=nearest)

    def predict_and_register(
        self,
        fixed_frame_u8: np.ndarray,
        moving_frame_u8: np.ndarray,
        fixed_masks: np.ndarray,
        moving_masks: np.ndarray,
        fixed_yolo_mask: np.ndarray | None = None,
        moving_yolo_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        fixed_to_moving = self.predict_fixed_to_moving(
            fixed_frame_u8,
            moving_frame_u8,
            fixed_masks,
            moving_masks,
            fixed_yolo_mask,
            moving_yolo_mask,
        )
        registered = self.register_moving_to_fixed(moving_frame_u8, moving_masks, fixed_to_moving)
        return {
            "fixed_to_moving_dxdy": fixed_to_moving,
            "moving_to_fixed_dxdy": self.inverse_displacements(fixed_to_moving),
            "registered_moving": registered,
        }
