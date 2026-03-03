import numpy as np

from stacking.align import align_images


def create_simple_star(shape=(50, 50), pos=(25, 25)):
    img = np.zeros(shape)
    img[pos] = 1.0
    return img


def test_astroalign_align_images_no_raise():
    # Simple synthetic images; alignment may be identity fallback depending on astroalign availability.
    imgs = [create_simple_star(), create_simple_star(pos=(27, 23))]
    aligned = align_images(imgs, show_progress=False)

    assert len(aligned) == 2
    assert all(isinstance(a, np.ndarray) for a in aligned)
    assert aligned[0].shape == imgs[0].shape
    assert aligned[1].shape == imgs[1].shape
