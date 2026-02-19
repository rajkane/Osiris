import os
import numpy as np
import imageio.v3 as iio
from cli import run_pipeline


def test_streaming_average_pipeline(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for i in range(5):
        img = np.ones((8, 8), dtype=np.uint8) * (i * 10)
        path = input_dir / f"img_{i}.png"
        iio.imwrite(str(path), img)

    output_path = str(tmp_path / "out_stream.png")

    res = run_pipeline(
        str(input_dir),
        output_path,
        method="average",
        align=False,
        verbose=False,
        progress=False,
        stream=True,
    )

    assert os.path.exists(res)
    out = iio.imread(res)
    assert out.shape == (8, 8)
    assert out.mean() == 20

