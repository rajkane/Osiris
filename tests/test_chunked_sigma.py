import numpy as np

from stacking.combine import ChunkedSigmaClipStrategy


def test_chunked_sigma_simple():
    base = np.ones((6, 6), dtype=np.float64) * 10
    imgs = [base.copy() for _ in range(10)]
    # insert outliers in various chunks
    imgs[2] = base * 1000
    imgs[7] = base * 500

    strat = ChunkedSigmaClipStrategy(sigma=3.0, iters=2, chunk_size=3)
    res = strat.combine(imgs)
    # result should be close to 10 (ignoring outliers)
    ok1 = np.allclose(res, 10, atol=1e-6)
    ok2 = np.allclose(np.nanmean(res), 10, atol=1e-6)
    assert ok1 or ok2
