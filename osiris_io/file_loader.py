import os
from typing import Iterable, List

import imageio.v3 as iio

try:
    from astropy.io import fits
except Exception:
    fits = None


EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".fits", ".fit")


class FileLoader:
    """Loads images from a directory into numpy arrays.

    Several helper methods support optional returning of FITS headers. By
    default functions return only the image arrays to preserve backward
    compatibility.
    """

    @staticmethod
    def load_images_from_dir(
        directory: str,
        extensions=EXTENSIONS,
        return_headers: bool = False,
    ) -> List:
        """Load all images in `directory`.

        If `return_headers` is True, returns a list of (image, header) tuples
        where header is an astropy header for FITS files or None for other
        formats. Otherwise returns a list of numpy arrays.
        """
        images = []
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(extensions):
                path = os.path.join(directory, fname)
                if (
                    path.lower().endswith(".fits")
                    or path.lower().endswith(".fit")
                ) and fits is not None:
                    with fits.open(path) as hdul:
                        img = hdul[0].data
                        hdr = hdul[0].header
                else:
                    img = iio.imread(path)
                    hdr = None
                images.append((img, hdr) if return_headers else img)
        return images

    @staticmethod
    def iter_images_from_dir(
        directory: str,
        extensions=EXTENSIONS,
        return_headers: bool = False,
    ) -> Iterable:
        """Yield images (or (image, header)) from a directory one by one.

        Useful for streaming processing. `return_headers` same semantics as
        in `load_images_from_dir`.
        """
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(extensions):
                path = os.path.join(directory, fname)
                if (
                    path.lower().endswith(".fits")
                    or path.lower().endswith(".fit")
                ) and fits is not None:
                    with fits.open(path) as hdul:
                        img = hdul[0].data
                        hdr = hdul[0].header
                else:
                    img = iio.imread(path)
                    hdr = None
                yield (img, hdr) if return_headers else img

    @staticmethod
    def load_image(path: str, return_header: bool = False):
        """Load a single image from a path.

        If `return_header` is True and the file is FITS, returns (image, header),
        otherwise returns image array.
        """
        p = path
        if (
            p.lower().endswith(".fits")
            or p.lower().endswith(".fit")
        ) and fits is not None:
            with fits.open(p) as hdul:
                img = hdul[0].data
                hdr = hdul[0].header
            return (img, hdr) if return_header else img
        img = iio.imread(p)
        return (img, None) if return_header else img
