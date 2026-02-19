from typing import Iterable, List, Protocol

import numpy as np
from astropy.stats import mad_std, sigma_clip


class CombineStrategy(Protocol):
    def combine(self, images: List[np.ndarray], **kwargs) -> np.ndarray:
        ...


class AverageStrategy:
    def combine(self, images: List[np.ndarray]) -> np.ndarray:
        arr = np.stack([img.astype(np.float64) for img in images], axis=0)
        return np.mean(arr, axis=0)

    def combine_iter(self, images: Iterable[np.ndarray]) -> np.ndarray:
        """Streaming average: compute running mean without stacking all frames."""
        count = 0
        mean = None
        for img in images:
            a = img.astype(np.float64)
            if mean is None:
                mean = np.array(a, dtype=np.float64)
                count = 1
            else:
                count += 1
                # incremental mean: new_mean = old_mean + (a - old_mean)/count
                mean += (a - mean) / count
        if mean is None:
            raise ValueError("No images to stack")
        return mean


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
        chunk_size = kwargs.get("chunk_size", None)
        if chunk_size:
            return ChunkedSigmaClipStrategy(sigma=sigma, iters=iters, chunk_size=chunk_size)
        else:
            return SigmaClipStrategy(sigma=sigma, iters=iters)
    raise ValueError(f"Unknown stacking method: {method}")


def stack_images_stream(image_iter: Iterable[np.ndarray], method: str = "average", **kwargs) -> np.ndarray:
    """Combine images provided as iterator/stream. Only 'average' supported streaming for now."""
    method = method.lower()
    if method == "average":
        strat = AverageStrategy()
        return strat.combine_iter(image_iter)
    else:
        # fallback: collect into list and call stack_images
        imgs = list(image_iter)
        return stack_images(imgs, method=method, **kwargs)


def stack_images(images: List[np.ndarray], method: str = "average", stream: bool = False, **kwargs) -> np.ndarray:
    """Combine a list of images into a single stacked image using a strategy.

    Supported methods: 'average', 'median', 'sigma'. Additional keyword args
    are passed to the strategy factory (e.g., sigma, sigma_iters). If stream=True,
    images is expected to be an iterable and streaming combine will be used when
    available.
    """
    if stream:
        # images is an iterable/generator
        return stack_images_stream(images, method=method, **kwargs)

    if not images:
        raise ValueError("No images to stack")

    strategy = get_combine_strategy(method, **kwargs)
    return strategy.combine(images)


def _chunked_sigma_clip(
    images: List[np.ndarray], sigma: float, iters: int, chunk_size: int = 100
) -> np.ndarray:
    """Perform sigma-clipping in chunks to reduce memory usage.

    Args:
        images: List of input images.
        sigma: Sigma value for clipping.
        iters: Number of iterations for clipping.
        chunk_size: Number of images to process in each chunk.

    Returns:
        The sigma-clipped image.
    """
    # Initialize with zeros; will be updated in-place
    shape = images[0].shape
    mean_image = np.zeros(shape, dtype=np.float64)
    count = 0

    for i in range(0, len(images), chunk_size):
        chunk = images[i : i + chunk_size]
        arr = np.stack([img.astype(np.float64) for img in chunk], axis=0)

        # Use median+MAD for robust sigma-clipping
        clipped = sigma_clip(
            arr,
            sigma=sigma,
            maxiters=iters,
            axis=0,
            cenfunc=np.ma.median,
            stdfunc=mad_std,
        )

        # Update mean_image with the mean of the clipped chunk
        mean_image += np.ma.mean(clipped, axis=0).filled(0)
        count += 1

    # Finalize the mean image
    mean_image /= count

    return mean_image


class ChunkedSigmaClipStrategy:
    def __init__(self, sigma: float = 3.0, iters: int = 5, chunk_size: int = 100):
        self.sigma = sigma
        self.iters = iters
        self.chunk_size = chunk_size

    def combine(self, images: List[np.ndarray]) -> np.ndarray:
        return _chunked_sigma_clip(images, self.sigma, self.iters, self.chunk_size)
