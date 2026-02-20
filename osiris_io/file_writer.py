import imageio.v3 as iio
import numpy as np

try:
    from astropy.io import fits
except Exception:
    fits = None


class FileWriter:
    """Saves numpy array images to disk.

    If a FITS header is provided, it will be attached to the primary HDU
    when writing FITS output.
    """

    @staticmethod
    def save_image(path: str, image: "np.ndarray", header=None):
        if (
            path.lower().endswith('.fits')
            or path.lower().endswith('.fit')
        ) and fits is not None:
            # If a header is given, attach it to the Primary HDU
            if header is not None:
                hdu = fits.PrimaryHDU(image, header=header)
            else:
                hdu = fits.PrimaryHDU(image)
            hdu.writeto(path, overwrite=True)
        else:
            iio.imwrite(path, image)
