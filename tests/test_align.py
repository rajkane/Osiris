import numpy as np

from stacking.align import align_images


def create_shifted_image(shape=(20, 20), shift=(2, -3)):
    img = np.zeros(shape)
    # place a single bright pixel
    cx, cy = shape[0] // 2, shape[1] // 2
    img[cx, cy] = 1.0
    from scipy.ndimage import shift as ndi_shift

    return ndi_shift(img, shift=shift, order=1)


def test_align_images_center():
    ref = np.zeros((20, 20))
    ref[10, 10] = 1.0
    shifted = create_shifted_image()
    imgs = [ref, shifted]
    aligned = align_images(imgs, reference_index=0)
    # after alignment the maximum pixel should be near center
    maxpos = np.unravel_index(np.argmax(aligned[1]), aligned[1].shape)
    assert abs(maxpos[0] - 10) <= 1
    assert abs(maxpos[1] - 10) <= 1
