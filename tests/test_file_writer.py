import os
import sys

import imageio.v3 as iio
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from osiris_io.file_writer import FileWriter


@pytest.fixture
def tmp_image(tmp_path):
    img = np.ones((5, 5), dtype=np.uint8) * 127
    path = tmp_path / "out.png"
    return img, str(path)


def test_save_image(tmp_image):
    img, path = tmp_image
    FileWriter.save_image(path, img)
    loaded = iio.imread(path)
    assert np.array_equal(img, loaded)
