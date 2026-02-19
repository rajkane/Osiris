from typing import Optional

import numpy as np


def apply_calibration(
    image: np.ndarray,
    bias: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply simple calibration: subtract bias and dark, divide by flat.

    Lightweight and intended for tests/initial pipeline.
    """
    img = image.astype(np.float64)
    if bias is not None:
        img = img - bias.astype(np.float64)
    if dark is not None:
        img = img - dark.astype(np.float64)
    if flat is not None:
        # prevent division by zero
        flat_f = flat.astype(np.float64)
        flat_f[flat_f == 0] = 1.0
        img = img / flat_f
    return img
