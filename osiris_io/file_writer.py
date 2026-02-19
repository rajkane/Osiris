import imageio.v3 as iio
import numpy as np

try:
    from astropy.io import fits
except Exception:
    fits = None


class FileWriter:
    """Saves numpy array images to disk."""

    @staticmethod
    def save_image(path: str, image: np.ndarray):
        if (path.lower().endswith('.fits') or path.lower().endswith('.fit')) and fits is not None:
            hdu = fits.PrimaryHDU(image)
            hdu.writeto(path, overwrite=True)
        else:
            iio.imwrite(path, image)
