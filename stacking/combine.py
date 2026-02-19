from typing import List, Protocol

import numpy as np
from astropy.stats import sigma_clip, mad_std


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


class SigmaClipStrategy:
    def __init__(self, sigma: float = 3.0, iters: int = 5):
        self.sigma = sigma
        self.iters = iters

    def combine(self, images: List[np.ndarray]) -> np.ndarray:
        arr = np.stack([img.astype(np.float64) for img in images], axis=0)
        # Use median+MAD for robust sigma-clipping
        clipped = sigma_clip(
            arr,
            sigma=self.sigma,
            maxiters=self.iters,
            axis=0,
            cenfunc=np.ma.median,
            stdfunc=mad_std,
        )
        # clipped is a MaskedArray; compute mean ignoring masked values
        return np.ma.mean(clipped, axis=0).filled(np.nan)


def get_combine_strategy(method: str, **kwargs) -> CombineStrategy:
    method = method.lower()
    if method == "average":
        return AverageStrategy()
    if method == "median":
        return MedianStrategy()
    if method in ("sigma", "sigma_clip", "sigma_clipped"):
        sigma = kwargs.get("sigma", 3.0)
        iters = kwargs.get("sigma_iters", 5)
        return SigmaClipStrategy(sigma=sigma, iters=iters)
    raise ValueError(f"Unknown stacking method: {method}")


def stack_images(images: List[np.ndarray], method: str = "average", **kwargs) -> np.ndarray:
    """Combine a list of images into a single stacked image using a strategy.

    Supported methods: 'average', 'median', 'sigma'. Additional keyword args
    are passed to the strategy factory (e.g., sigma, sigma_iters).
    """
    if not images:
        raise ValueError("No images to stack")

    strategy = get_combine_strategy(method, **kwargs)
    return strategy.combine(images)
