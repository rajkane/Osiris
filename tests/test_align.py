import numpy as np

from stacking.align import align_images


def create_shifted_image(shape=(20, 20), shift=(2, -3)):
    img = np.zeros(shape)
    # place a single bright pixel
    cx, cy = shape[0] // 2, shape[1] // 2
    img[cx, cy] = 1.0
    from scipy.ndimage import shift as ndi_shift

    return ndi_shift(img, shift=shift, order=1)


def test_align_images_runs_and_preserves_shape():
    ref = np.zeros((20, 20))
    ref[10, 10] = 1.0
    shifted = create_shifted_image()
    imgs = [ref, shifted]

    aligned = align_images(imgs, reference_index=0)

    assert len(aligned) == 2
    assert aligned[0].shape == (20, 20)
    assert aligned[1].shape == (20, 20)


def test_align_images_accepts_legacy_kwargs():
    # Older CLI/tests passed crop/border args; astroalign-only implementation should ignore them.
    ref = np.zeros((20, 20))
    ref[10, 10] = 1.0
    shifted = create_shifted_image(shift=(3, -4))

    aligned = align_images(
        [ref, shifted],
        reference_index=0,
        crop=True,
        border_mode="constant",
        border_cval=0.0,
    )

    assert len(aligned) == 2
    assert aligned[0].shape == (20, 20)
    assert aligned[1].shape == (20, 20)
