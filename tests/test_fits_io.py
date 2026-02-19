import os
import numpy as np

try:
    from astropy.io import fits
except Exception:
    fits = None

from osiris_io.file_writer import FileWriter
from osiris_io.file_loader import FileLoader


def test_fits_roundtrip(tmp_path):
    if fits is None:
        # astropy not installed in environment; skip
        return

    img = np.arange(16, dtype=np.float32).reshape((4, 4))
    path = tmp_path / "test.fits"
    FileWriter.save_image(str(path), img)
    imgs = FileLoader.load_images_from_dir(str(tmp_path))
    assert len(imgs) == 1
    loaded = imgs[0]
    assert loaded.shape == img.shape
    assert np.allclose(loaded, img)

