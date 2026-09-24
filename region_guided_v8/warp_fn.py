#!/usr/bin/env python3
"""
Warp functions: single-silhouette translation, and the 5-mask displacement field.

Follows the two hand-written specs:

  1masks:  img = concat[img1, img2];  dx, dy = model(img)
           warped = warp_fn(img, dx, dy)                      <- one global shift

  5masks:  convert dx_list, dy_list to 2d_dvf(dx_list, dy_list, mov_mask)
           warped = warp_fn(img, dvf)                         <- one dense field

The required invariant, stated by the user:

    if every entry of dx_list / dy_list is the same pair (dx, dy), the 5-mask
    result must be equivalent to the single whole-body silhouette result.

`build_dvf` therefore blends the per-part vectors as a *partition of unity*: at
every pixel the output is a weighted average of part vectors whose weights sum
to one. A weighted average of identical vectors is that vector, at every pixel
and for any masks whatsoever -- so the invariant holds exactly, not
approximately, and `test_warp_fn.py` asserts it bit-for-bit.

Pixels are blended rather than hard-assigned because a hard per-part assignment
tears the image at part boundaries: neighbouring pixels would move by different
amounts with no transition, which shows up as seams in the warped result.
"""
from __future__ import annotations

import cv2
import numpy as np

# Index convention matches MASK_TYPES in train_consecutive.py:
#   0 outer_masks, 1 body, 2 arms, 3 face, 4 hair
OUTER_INDEX = 0
PART_INDICES = (1, 2, 3, 4)

DEFAULT_SIGMA = 8.0  # px; softness of the part boundaries in the blend


def build_dvf(
    dxdy: np.ndarray,
    masks: np.ndarray,
    sigma: float = DEFAULT_SIGMA,
    outer_index: int = OUTER_INDEX,
    part_indices: tuple[int, ...] = PART_INDICES,
) -> np.ndarray:
    """Compose per-part shifts into a dense displacement field.

    Args:
        dxdy:  [M, 2] float, the (dx, dy) predicted for each mask.
        masks: [M, H, W], the moving masks (`mov_mask`) at time t. Non-zero =
               that part occupies the pixel. Blank masks simply contribute no
               weight, so an invalid part drops out on its own.
        sigma: Gaussian smoothing of the part memberships, in pixels.

    Returns:
        [H, W, 2] float32 displacement field, dvf[..., 0] = dx, [..., 1] = dy.

    The outer silhouette acts as the fallback: wherever no part claims a pixel
    (background, or a body region with no part label) the field falls back to
    the outer vector, so the field is defined everywhere.
    """
    dxdy = np.asarray(dxdy, np.float32).reshape(-1, 2)
    masks = np.asarray(masks)
    _, h, w = masks.shape

    d_outer = dxdy[outer_index].reshape(1, 1, 2)

    # Accumulate each part's offset *relative to the outer vector*. This is
    # algebraically identical to blending the vectors themselves --
    #   d_o + a*sum(w_k (d_k - d_o))/W  ==  a*sum(w_k d_k)/W + (1-a)*d_o
    # -- but when every d_k equals d_outer the numerator is exactly zero, so the
    # invariant survives float32 rounding bit-for-bit instead of to ~1e-5 px.
    num = np.zeros((h, w, 2), np.float32)   # sum_k w_k * (d_k - d_outer)
    wsum = np.zeros((h, w), np.float32)     # sum_k w_k

    for k in part_indices:
        m = masks[k].astype(np.float32)
        if m.max() <= 0:
            continue
        if sigma > 0:
            # BORDER_REPLICATE, not the default reflect-with-zero-fill: a
            # constant field must stay constant after smoothing.
            m = cv2.GaussianBlur(m, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        wsum += m
        num += m[..., None] * (dxdy[k] - dxdy[outer_index])

    # alpha = how strongly the parts claim this pixel, capped at 1.
    alpha = np.minimum(wsum, 1.0)[..., None]
    correction = num / np.maximum(wsum, 1e-6)[..., None]

    return (d_outer + alpha * correction).astype(np.float32)


def constant_dvf(dx: float, dy: float, shape: tuple[int, int]) -> np.ndarray:
    """The 1-mask case: a single global shift expressed as a field."""
    h, w = shape
    dvf = np.empty((h, w, 2), np.float32)
    dvf[..., 0] = dx
    dvf[..., 1] = dy
    return dvf


def warp_with_dvf(img: np.ndarray, dvf: np.ndarray, nearest: bool = False) -> np.ndarray:
    """Backward-warp `img` by `dvf`, i.e. out(x, y) = img(x - dx, y - dy).

    Same convention as cv2.warpAffine with M = [[1, 0, dx], [0, 1, dy]], so the
    constant-field case is identical to a plain translation.
    """
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xs - dvf[..., 0]
    map_y = ys - dvf[..., 1]
    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def warp_translate(img: np.ndarray, dx: float, dy: float, nearest: bool = False) -> np.ndarray:
    """The 1-mask warp_fn: shift the whole image/silhouette by one vector."""
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img, m, (w, h),
        flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def warp_fn(img, dx_list, dy_list=None, masks=None, sigma: float = DEFAULT_SIGMA,
            nearest: bool = False) -> np.ndarray:
    """Unified entry point matching the notes.

    warp_fn(img, dx, dy)                      -> single global shift (1 mask)
    warp_fn(img, dx_list, dy_list, mov_mask)  -> composed field     (5 masks)
    """
    if masks is None:
        return warp_translate(img, float(dx_list), float(dy_list), nearest=nearest)
    dxdy = np.stack([np.asarray(dx_list, np.float32), np.asarray(dy_list, np.float32)], axis=-1)
    return warp_with_dvf(img, build_dvf(dxdy, masks, sigma=sigma), nearest=nearest)


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    """Dice between two binary masks (1.0 when both are empty)."""
    a_bin = (a >= 0.5)
    b_bin = (b >= 0.5)
    total = int(a_bin.sum()) + int(b_bin.sum())
    if total == 0:
        return 1.0
    return 2.0 * int(np.logical_and(a_bin, b_bin).sum()) / total
