import os

import imageio.v3 as iio
import numpy as np

from cli import run_pipeline


def test_pipeline_with_sigma_and_feature(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for i in range(4):
        img = np.ones((20, 20), dtype=np.uint8) * (i * 5)
        path = input_dir / f"img_{i}.png"
        iio.imwrite(str(path), img)

    output_path = str(tmp_path / "out_default.png")

    res = run_pipeline(
        str(input_dir),
        output_path,
        method="sigma",
        align=True,
        verbose=False,
        progress=False,
        sigma=2.5,
        sigma_iters=3,
        align_method="feature",
        align_kp=200,
    )

    assert os.path.exists(res)
    out = iio.imread(res)
    assert out.shape == (20, 20)
