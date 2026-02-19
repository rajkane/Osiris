import numpy as np
from stacking.combine import SigmaClipStrategy


def test_sigma_clip_combines():
    # create stack with outlier in middle frame
    base = np.ones((4, 4), dtype=np.float64) * 10
    imgs = [base.copy() for _ in range(5)]
    imgs[2] = base * 1000  # outlier

    strat = SigmaClipStrategy(sigma=3.0, iters=2)
    res = strat.combine(imgs)
    # the mean ignoring outlier should be close to 10
    assert np.allclose(res, 10, atol=1e-6) or np.allclose(np.nanmean(res), 10, atol=1e-6)

