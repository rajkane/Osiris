import numpy as np

from stacking.align import FeatureMatchAlignStrategy


def create_simple_star(shape=(50, 50), pos=(25, 25)):
    img = np.zeros(shape)
    img[pos] = 1.0
    return img


def test_feature_match_align_fallback():
    # Create two simple images; feature matching likely fails, should not raise
    imgs = [create_simple_star(), create_simple_star(pos=(27, 23))]
    strat = FeatureMatchAlignStrategy()
    aligned = strat.align(imgs)
    assert len(aligned) == 2
    # ensure arrays returned
    assert all(isinstance(a, np.ndarray) for a in aligned)
