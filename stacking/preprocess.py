import numpy as np


def apply_calibration(
    image: np.ndarray, bias=None, dark=None, flat=None, bad_pixel_mask=None
) -> np.ndarray:
    """Apply basic calibration frames and optional bad-pixel mask.

    Args:
        image: Input frame.
        bias/dark/flat: Optional calibration frames.
        bad_pixel_mask: Optional mask marking bad pixels.
            Supported formats:
            - boolean mask (True = bad)
            - uint8/uint16 mask where non-zero = bad

    Returns:
        Calibrated float32 image.
    """
    img = image.astype(np.float32)

    if bias is not None:
        img -= bias.astype(np.float32)
    if dark is not None:
        img -= dark.astype(np.float32)
    if flat is not None:
        f = flat.astype(np.float32)
        f[f == 0] = 1.0
        img /= f

    if bad_pixel_mask is not None:
        m = np.asarray(bad_pixel_mask)
        bad = m.astype(bool)
        # Replace bad pixels with local median approximation (fast + robust).
        # We do this per-channel if needed.
        if img.ndim == 2:
            med = np.median(img[~bad]) if np.any(~bad) else 0.0
            img[bad] = med
        else:
            for c in range(img.shape[-1]):
                plane = img[..., c]
                med = np.median(plane[~bad]) if np.any(~bad) else 0.0
                plane[bad] = med
                img[..., c] = plane

    return img
