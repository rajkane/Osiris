import os

import imageio.v3 as iio
import numpy as np

from cli import run_pipeline


def test_run_pipeline_end_to_end(tmp_path):
    # prepare temporary input directory with synthetic images
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for i in range(3):
        img = np.ones((10, 10), dtype=np.uint8) * (i * 10)
        path = input_dir / f"img_{i}.png"
        iio.imwrite(str(path), img)

    output_path = str(tmp_path / "out_default.png")

    res = run_pipeline(
        str(input_dir),
        output_path,
        method="average",
        align=False,
        verbose=False,
    )
    assert os.path.exists(res)
    out = iio.imread(res)
    assert out.shape == (10, 10)
    # average of 0,10,20 -> 10
    assert out.mean() == 10
