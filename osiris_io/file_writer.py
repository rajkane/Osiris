import imageio.v3 as iio
import numpy as np


class FileWriter:
    """Saves numpy array images to disk."""

    @staticmethod
    def save_image(path: str, image: np.ndarray):
        iio.imwrite(path, image)
