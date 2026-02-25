import numpy as np
from typing import Optional

def apply_calibration(image: np.ndarray, bias=None, dark=None, flat=None) -> np.ndarray:
    img = image.astype(np.float32)
    if bias is not None: img -= bias.astype(np.float32)
    if dark is not None: img -= dark.astype(np.float32)
    if flat is not None:
        f = flat.astype(np.float32)
        f[f == 0] = 1.0
        img /= f
    return img