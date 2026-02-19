import pytest
import numpy as np
import os
import imageio.v3 as iio
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from osiris_io.file_loader import FileLoader


@pytest.fixture
def tmp_image_dir(tmp_path):
    # Create a temporary directory with a few images
    img = np.zeros((10, 10), dtype=np.uint8)
    for i in range(3):
        path = tmp_path / f"test_{i}.png"
        iio.imwrite(str(path), img)
    return tmp_path


def test_load_images_from_dir(tmp_image_dir):
    images = FileLoader.load_images_from_dir(str(tmp_image_dir))
    assert len(images) == 3
    for img in images:
        assert isinstance(img, np.ndarray)
        assert img.shape == (10, 10)
