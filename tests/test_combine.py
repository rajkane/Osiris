import numpy as np

from stacking.combine import stack_images


def test_stack_average():
    imgs = [np.ones((4, 4)) * i for i in range(1, 4)]
    res = stack_images(imgs, method="average")
    assert res.shape == (4, 4)
    assert np.allclose(res, np.mean(np.stack(imgs, axis=0), axis=0))


def test_stack_median():
    imgs = [np.zeros((3, 3)), np.ones((3, 3)) * 10, np.ones((3, 3)) * 20]
    res = stack_images(imgs, method="median")
    assert np.allclose(res, np.median(np.stack(imgs, axis=0), axis=0))
