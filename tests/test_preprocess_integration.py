import os
import numpy as np
import imageio.v3 as iio
from cli import run_pipeline


def test_preprocess_applied(tmp_path):
    # create input images with known offset
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for i in range(3):
        img = np.ones((6, 6), dtype=np.uint8) * 10
        path = input_dir / f"img_{i}.png"
        iio.imwrite(str(path), img)

    # create bias (value 1) and dark (value 2) and flat (all ones)
    bias = np.ones((6, 6), dtype=np.uint8) * 1
    dark = np.ones((6, 6), dtype=np.uint8) * 2
    flat = np.ones((6, 6), dtype=np.uint8) * 1

    bias_path = tmp_path / "bias.png"
    dark_path = tmp_path / "dark.png"
    flat_path = tmp_path / "flat.png"
    iio.imwrite(str(bias_path), bias)
    iio.imwrite(str(dark_path), dark)
    iio.imwrite(str(flat_path), flat)

    output_path = str(tmp_path / "out_default.png")

    res = run_pipeline(
        str(input_dir),
        output_path,
        method="average",
        align=False,
        verbose=False,
        bias=str(bias_path),
        dark=str(dark_path),
        flat=str(flat_path),
    )

    assert os.path.exists(res)
    out = iio.imread(res)
    # each input pixel: 10 - bias(1) - dark(2) = 7; average stays 7
    assert out.mean() == 7

