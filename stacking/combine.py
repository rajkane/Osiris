import numpy as np
import tempfile
import os
from astropy.stats import mad_std, sigma_clip


class ChunkedSigmaClipStrategy:
    def __init__(self, sigma=3.0, iters=5, chunk_size=3):
        self.sigma = sigma
        self.iters = iters
        self.chunk_size = chunk_size

    def combine(self, images):
        if not images: return None
        shape = images[0].shape
        result = np.zeros(shape, dtype=np.float32)
        weights = np.zeros(shape, dtype=np.float32)

        # Process in chunks to prevent "Killed" (OOM) errors
        for i in range(0, len(images), self.chunk_size):
            chunk = images[i: i + self.chunk_size]
            # Use masked array to ignore NaNs from alignment
            arr = np.ma.masked_invalid(np.stack(chunk, axis=0))

            clipped = sigma_clip(arr, sigma=self.sigma, maxiters=self.iters, axis=0,
                                 cenfunc=np.ma.median, stdfunc=mad_std)

            result += clipped.filled(0).sum(axis=0)
            weights += (~clipped.mask).sum(axis=0)
            del arr, clipped

        weights[weights == 0] = 1
        return (result / weights).astype(np.float32)


def stack_images(images, method="average", use_memmap=False, **kwargs):
    if not images: return None

    # Memory Mapping Logic to handle large datasets on low RAM
    if use_memmap:
        fd, temp_path = tempfile.mkstemp(suffix='.npy')
        os.close(fd)
        shape = (len(images), *images[0].shape)
        mmap_arr = np.memmap(temp_path, dtype='float32', mode='w+', shape=shape)

        for i, img in enumerate(images):
            mmap_arr[i] = img.astype(np.float32)
        mmap_arr.flush()

        if method == "median":
            res = np.nanmedian(mmap_arr, axis=0)
        else:
            res = np.nanmean(mmap_arr, axis=0)

        del mmap_arr
        if os.path.exists(temp_path): os.remove(temp_path)
        return res.astype(np.float32)

    # Fallback to Sigma Clipping with Chunking
    if method == "sigma":
        strategy = ChunkedSigmaClipStrategy(
            sigma=kwargs.get("sigma", 3.0),
            iters=kwargs.get("sigma_iters", 5),
            chunk_size=kwargs.get("chunk_size", 3)
        )
        return strategy.combine(images)

    # Standard NumPy stack (only if enough RAM)
    arr = np.stack(images, axis=0)
    if method == "median":
        return np.nanmedian(arr, axis=0).astype(np.float32)
    return np.nanmean(arr, axis=0).astype(np.float32)