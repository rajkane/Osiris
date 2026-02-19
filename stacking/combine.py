from typing import List

import numpy as np


def stack_images(images: List[np.ndarray], method: str = "average") -> np.ndarray:
    """Combine a list of images into a single stacked image.

    Supported methods: 'average', 'median'. The function promotes to float64 for
    the computation and returns a float64 array (caller can cast if needed).
    """
    if not images:
        raise ValueError("No images to stack")

    arr = np.stack([img.astype(np.float64) for img in images], axis=0)

    if method == "average":
        result = np.mean(arr, axis=0)
    elif method == "median":
        result = np.median(arr, axis=0)
    else:
        raise ValueError(f"Unknown stacking method: {method}")

    return result
