from typing import List, Protocol

import numpy as np


class CombineStrategy(Protocol):
    def combine(self, images: List[np.ndarray]) -> np.ndarray:
        ...


class AverageStrategy:
    def combine(self, images: List[np.ndarray]) -> np.ndarray:
        arr = np.stack([img.astype(np.float64) for img in images], axis=0)
        return np.mean(arr, axis=0)


class MedianStrategy:
    def combine(self, images: List[np.ndarray]) -> np.ndarray:
        arr = np.stack([img.astype(np.float64) for img in images], axis=0)
        return np.median(arr, axis=0)


def get_combine_strategy(method: str) -> CombineStrategy:
    method = method.lower()
    if method == "average":
        return AverageStrategy()
    if method == "median":
        return MedianStrategy()
    raise ValueError(f"Unknown stacking method: {method}")


def stack_images(images: List[np.ndarray], method: str = "average") -> np.ndarray:
    """Combine a list of images into a single stacked image using a strategy.

    Supported methods: 'average', 'median'.
    """
    if not images:
        raise ValueError("No images to stack")

    strategy = get_combine_strategy(method)
    return strategy.combine(images)
