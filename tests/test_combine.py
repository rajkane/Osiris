import numpy as np

from stacking.combine import stack_images


def test_stack_sigma_clips_outlier():
    base = np.ones((4, 4), dtype=np.float32) * 10
    imgs = [base.copy() for _ in range(5)]
    imgs[2] = base * 1000  # outlier

    res = stack_images(imgs, method="sigma", sigma=3.0, sigma_iters=2)
    assert res.shape == (4, 4)
    # After per-frame median normalization, the base frames become ~1 everywhere.
    assert np.allclose(res, 1.0, atol=1e-4) or np.allclose(
        np.nanmean(res), 1.0, atol=1e-4
    )
