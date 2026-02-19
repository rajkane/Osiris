import os
from typing import List, Iterable

import imageio.v3 as iio
import numpy as np

try:
    from astropy.io import fits
except Exception:
    fits = None


class FileLoader:
    """Loads images from a directory into numpy arrays."""

    @staticmethod
    def load_images_from_dir(
        directory: str, extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".fits", ".fit")
    ) -> List[np.ndarray]:
        images = []
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(extensions):
                path = os.path.join(directory, fname)
                if (path.lower().endswith('.fits') or path.lower().endswith('.fit')) and fits is not None:
                    with fits.open(path) as hdul:
                        img = hdul[0].data
                else:
                    img = iio.imread(path)
                images.append(img)
        return images

    @staticmethod
    def iter_images_from_dir(
        directory: str, extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".fits", ".fit")
    ) -> Iterable[np.ndarray]:
        """Yield images from a directory one by one (streaming-friendly)."""
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(extensions):
                path = os.path.join(directory, fname)
                if (path.lower().endswith('.fits') or path.lower().endswith('.fit')) and fits is not None:
                    with fits.open(path) as hdul:
                        img = hdul[0].data
                else:
                    img = iio.imread(path)
                yield img
