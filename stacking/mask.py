from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class AlignMaskResult:
    """Aligned frames along with a per-frame validity mask."""

    images: List[np.ndarray]
    masks: List[np.ndarray]


def masks_from_shifts(
    image_shape: Tuple[int, ...],
    shifts: List[np.ndarray],
) -> List[np.ndarray]:
    """Create boolean validity masks for shifted images.

    We assume shifts are (dy, dx) applied to the moving image to align it to the reference.
    Pixels that originate from outside the original image after shifting are invalid.

    Output mask shape is (H, W), True means valid.
    """

    h, w = image_shape[:2]
    masks: List[np.ndarray] = []

    for sh in shifts:
        dy, dx = float(sh[0]), float(sh[1])

        # After shifting by (dy, dx), the valid region is the intersection of the original
        # image bounds with the shifted bounds. For integer shifts it's exact; for subpixel
        # shifts we conservatively ceil.
        y_pad = int(np.ceil(abs(dy)))
        x_pad = int(np.ceil(abs(dx)))

        m = np.zeros((h, w), dtype=bool)
        m[
            y_pad : h - y_pad if (h - y_pad) > y_pad else y_pad,
            x_pad : w - x_pad if (w - x_pad) > x_pad else x_pad,
        ] = True
        masks.append(m)

    return masks


def masked_average(images: List[np.ndarray], masks: List[np.ndarray]) -> np.ndarray:
    """Compute per-pixel average ignoring invalid pixels.

    Supports grayscale (H,W) and RGB (H,W,C).
    """

    if len(images) == 0:
        raise RuntimeError("No images")

    if len(images) != len(masks):
        raise ValueError("images and masks must have the same length")

    img0 = images[0]
    h, w = img0.shape[:2]

    acc = np.zeros_like(img0, dtype=np.float32)
    denom = np.zeros((h, w), dtype=np.float32)

    for img, m in zip(images, masks):
        img_f = img.astype(np.float32, copy=False)
        if img_f.ndim == 2:
            acc += np.where(m, img_f, 0.0)
        else:
            acc += np.where(m[..., None], img_f, 0.0)
        denom += m.astype(np.float32)

    # Avoid division by zero
    if img0.ndim == 2:
        out = np.where(denom > 0, acc / (denom + 1e-12), 0.0)
    else:
        out = np.where(denom[..., None] > 0, acc / (denom[..., None] + 1e-12), 0.0)

    return out.astype(np.float32)
