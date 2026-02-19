import os
import numpy as np
import imageio.v3 as iio
from typing import List


class FileLoader:
    """Loads images from a directory into numpy arrays."""

    @staticmethod
    def load_images_from_dir(
        directory: str, extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff")
    ) -> List[np.ndarray]:
        images = []
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(extensions):
                path = os.path.join(directory, fname)
                img = iio.imread(path)
                images.append(img)
        return images
