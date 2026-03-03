import numpy as np
from astropy.stats import mad_std, sigma_clip


def photometric_normalization(images, reference=None):
    """Simple per-frame median normalization.

    The user-requested behavior is intentionally simple:
        img = img / median(img)

    Notes:
    - Works for grayscale (H,W) and RGB (H,W,C).
    - For RGB we normalize each channel independently (median per channel).
    - NaNs are ignored (useful for aligned frames with NaN borders).
    - `reference` is accepted for API stability but not used.
    """

    eps = 1e-12
    normed = []

    for img in images:
        img_f = img.astype(np.float32, copy=False)

        if img_f.ndim == 2:
            med = float(np.nanmedian(img_f))
            scale = 1.0 / (med + eps)
            out = img_f * scale
        else:
            # Per-channel medians: shape (C,)
            med = np.nanmedian(img_f.reshape(-1, img_f.shape[-1]), axis=0).astype(
                np.float32
            )
            scale = 1.0 / (med + eps)
            out = img_f * scale.reshape((1,) * (img_f.ndim - 1) + (img_f.shape[-1],))

        normed.append(out.astype(np.float32, copy=False))

    return normed


def weighted_sigma_clip_stack(images, sigma=3.0, iters=5, show_progress: bool = False):
    """Sigma-clip stack with robust per-frame weights.

    Args:
        images: list of ndarray (H,W) or (H,W,C)
        sigma: sigma threshold
        iters: max iterations
        show_progress: show a small tqdm progress bar (phase-based)
    """

    if len(images) == 0:
        raise RuntimeError("No images found to stack")

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    def _step(desc: str, it):
        if show_progress and tqdm is not None:
            return tqdm(it, desc=desc)
        return it

    stack = np.stack(images, axis=0).astype(np.float32)

    clipped = sigma_clip(
        stack,
        sigma=sigma,
        maxiters=iters,
        axis=0,
        cenfunc=np.median,
        stdfunc=mad_std,
    )

    # Convert masked array to float array with NaNs for clipped elements
    clipped_filled = np.ma.filled(clipped, np.nan).astype(np.float32)

    # Robust weight per frame (noise estimate)
    weights = []
    for img in _step("Sigma clip: weights", images):
        # Ignore NaNs (e.g., NaN borders after alignment)
        noise = mad_std(img, ignore_nan=True)
        weights.append(1.0 / (float(noise) + 1e-6))

    weights = np.array(weights, dtype=np.float32)
    weights_sum = float(weights.sum())
    if weights_sum == 0:
        weights = np.ones_like(weights) / float(len(weights))
    else:
        weights /= weights_sum

    # Weighted nan-aware aggregation per-pixel: ignore clipped frames per-pixel
    mask = ~np.isnan(clipped_filled)

    # Broadcast weights to match stack shape: (N, 1, 1[, 1])
    # NOTE: stack.ndim includes the leading N dimension.
    weights_reshaped = weights.reshape((len(weights),) + (1,) * (stack.ndim - 1))

    weighted_vals = np.where(mask, clipped_filled * weights_reshaped, 0.0)
    numerator = np.sum(weighted_vals, axis=0)

    denom = np.sum(np.where(mask, weights_reshaped, 0.0), axis=0)

    # Avoid division by zero
    result = np.where(denom == 0, np.nan, numerator / (denom + 1e-12))

    return result.astype(np.float32)


def _chunk_iter(seq, chunk_size: int):
    for i in range(0, len(seq), chunk_size):
        yield seq[i : i + chunk_size]


def low_memory_sigma_stack(
    images, sigma=3.0, iters=5, chunk_size: int = 10, show_progress: bool = False
):
    """Low-memory approximation for sigma clipping.

    We avoid building a full (N,H,W[,C]) stack. Instead:
    1) run sigma clip per chunk -> partial result
    2) aggregate partial results weighting by chunk size

    This is not identical to global sigma clipping across all frames, but it is
    robust and dramatically reduces peak memory.
    """

    if len(images) == 0:
        raise RuntimeError("No images found to stack")

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    if len(images) <= chunk_size:
        return weighted_sigma_clip_stack(
            images, sigma=sigma, iters=iters, show_progress=show_progress
        )

    chunks = list(_chunk_iter(images, chunk_size))
    iterator = (
        tqdm(chunks, desc="Sigma clip (chunked)")
        if show_progress and tqdm is not None
        else chunks
    )

    partials = []
    weights = []
    for ch in iterator:
        partials.append(
            weighted_sigma_clip_stack(ch, sigma=sigma, iters=iters, show_progress=False)
        )
        weights.append(len(ch))

    weights = np.asarray(weights, dtype=np.float32)
    weights /= float(weights.sum())

    acc = np.zeros_like(partials[0], dtype=np.float32)
    for w, p in zip(weights, partials):
        acc += p.astype(np.float32, copy=False) * float(w)
    return acc.astype(np.float32)


def stack_images(
    images,
    method="sigma",
    sigma=3.0,
    sigma_iters=5,
    chunk_size=None,
    stream=False,
    use_memmap=False,
    show_progress: bool = False,
    **kwargs,
):
    """Stack images using sigma clipping (the only supported method).

    Notes:
    - `method` is kept for backward compatibility but only "sigma" is supported.
    - If `chunk_size` is provided (or `stream` / `use_memmap`), a chunked low-memory
      approximation is used.
    """

    if method not in (None, "sigma"):
        raise ValueError("Only method='sigma' is supported")

    if chunk_size is None:
        chunk_size = kwargs.get("chunk_size")

    images_n = photometric_normalization(images)

    # Low-memory preference
    if chunk_size is not None or stream or use_memmap:
        effective_chunk = int(chunk_size) if chunk_size is not None else 10
        return low_memory_sigma_stack(
            images_n,
            sigma=sigma,
            iters=sigma_iters,
            chunk_size=effective_chunk,
            show_progress=show_progress,
        )

    return weighted_sigma_clip_stack(
        images_n, sigma=sigma, iters=sigma_iters, show_progress=show_progress
    )


# Strategy classes expected by tests: provide lightweight wrappers around
# the functional implementations.
class SigmaClipStrategy:
    def __init__(self, sigma=3.0, iters=5):
        self.sigma = sigma
        self.iters = iters

    def combine(self, images):
        images_n = photometric_normalization(images)
        return weighted_sigma_clip_stack(
            images_n, sigma=self.sigma, iters=self.iters, show_progress=False
        )


class ChunkedSigmaClipStrategy:
    def __init__(self, sigma=3.0, iters=5, chunk_size=10):
        self.sigma = sigma
        self.iters = iters
        self.chunk_size = chunk_size

    def combine(self, images):
        images_n = photometric_normalization(images)
        return low_memory_sigma_stack(
            images_n,
            sigma=self.sigma,
            iters=self.iters,
            chunk_size=self.chunk_size,
            show_progress=False,
        )
