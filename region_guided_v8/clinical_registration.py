"""Clinical moving-to-fixed registration helpers for region-guided V8."""
from __future__ import annotations

import numpy as np

from warp_fn import build_dvf, warp_translate, warp_with_dvf


def inverse_region_displacements(predicted_dxdy: np.ndarray) -> np.ndarray:
    """Convert fixed-to-moving model vectors to moving-to-fixed vectors."""
    vectors = np.asarray(predicted_dxdy, dtype=np.float32).reshape(-1, 2)
    return -vectors


def register_moving_image_to_fixed(
    moving_image: np.ndarray,
    moving_masks: np.ndarray,
    predicted_dxdy: np.ndarray,
    sigma: float = 8.0,
    nearest: bool = False,
) -> np.ndarray:
    """Warp a later moving image back to the fixed image coordinate system.

    The model predicts fixed-to-moving vectors. This function applies their
    inverse to the moving image using a mask-guided dense displacement field.
    """
    moving_vectors = inverse_region_displacements(predicted_dxdy)
    dvf = build_dvf(moving_vectors, moving_masks, sigma=sigma)
    return warp_with_dvf(moving_image, dvf, nearest=nearest)


def register_moving_mask_to_fixed(
    moving_mask: np.ndarray,
    predicted_dxdy: np.ndarray,
    region_index: int,
) -> np.ndarray:
    """Warp one moving region mask back using its inverse predicted vector."""
    vectors = inverse_region_displacements(predicted_dxdy)
    dx, dy = vectors[region_index]
    return warp_translate(moving_mask, float(dx), float(dy), nearest=True)
