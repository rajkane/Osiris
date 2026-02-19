from typing import List

import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation


def align_images(
    images: List[np.ndarray],
    reference_index: int = 0,
) -> List[np.ndarray]:
    """Align a list of images to the reference image using phase
    cross-correlation.
    """
    if not images:
        raise ValueError("No images to align")

    ref = images[reference_index].astype(float)
    aligned = []

    for img in images:
        img_f = img.astype(float)
        # compute shift that maps img -> ref
        shift, error, diffphase = phase_cross_correlation(
            ref, img_f, upsample_factor=10
        )
        # apply negative shift to img to align to ref
        corrected = ndi_shift(img_f, shift=shift, order=1, mode="constant", cval=0.0)
        aligned.append(corrected)

    return aligned
