import os
import tempfile

import imageio.v3 as iio
import numpy as np

from cli import run_pipeline

d = tempfile.mkdtemp()
indir = os.path.join(d, "in")
os.makedirs(indir, exist_ok=True)
for i in range(3):
    img = np.ones((10, 10), dtype=np.uint8) * (i * 10)
    path = os.path.join(indir, f"img_{i}.png")
    iio.imwrite(path, img)
output = os.path.join(d, "out.png")
print("input dir", indir)
res = run_pipeline(
    indir,
    output,
    align=True,
    verbose=True,
    progress=True,
    sigma=3.0,
    sigma_iters=5,
    chunk_size=10,
)
print("returned path", res)
out = iio.imread(res)
print("out dtype", out.dtype, "shape", out.shape, "mean", out.mean())
print("sample pixels:", out[0, 0], out[0, 1], out[-1, -1])
print("done")
