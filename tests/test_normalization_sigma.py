import numpy as np

from stacking.combine import photometric_normalization, stack_images


def test_photometric_normalization_divides_by_median_grayscale():
    img = np.arange(1, 17, dtype=np.float32).reshape(4, 4)  # median is 8.5
    (normed,) = photometric_normalization([img])
    assert np.isclose(np.median(normed), 1.0, atol=1e-6)


def test_stack_sigma_is_exposure_invariant_after_normalization():
    # Two identical scenes but different exposure scales should stack to ~1 after normalization.
    base = np.ones((8, 8), dtype=np.float32) * 10
    imgs = [base.copy(), base.copy() * 2.0, base.copy() * 0.5]

    res = stack_images(imgs, method="sigma", sigma=3.0, sigma_iters=2)
    assert np.allclose(res, 1.0, atol=1e-4) or np.allclose(
        np.nanmean(res), 1.0, atol=1e-4
    )


def test_photometric_normalization_rgb_per_channel():
    # Each channel has a different median; each should be normalized independently to median ~1.
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[..., 0] = 10
    img[..., 1] = 20
    img[..., 2] = 40

    (normed,) = photometric_normalization([img])
    med = np.median(normed.reshape(-1, 3), axis=0)
    assert np.allclose(med, [1.0, 1.0, 1.0], atol=1e-6)
