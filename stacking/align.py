from typing import List, Protocol, Optional

import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation
from tqdm import tqdm


class AlignStrategy(Protocol):
    def align(
        self,
        images: List[np.ndarray],
        reference_index: int = 0,
        show_progress: bool = False,
    ) -> List[np.ndarray]:
        ...


class PhaseCorrelationAlignStrategy:
    """Align using phase cross-correlation."""

    def align(
        self,
        images: List[np.ndarray],
        reference_index: int = 0,
        show_progress: bool = False,
    ) -> List[np.ndarray]:
        if not images:
            raise ValueError("No images to align")

        ref = images[reference_index].astype(float)
        aligned: List[np.ndarray] = []

        iterator = enumerate(images)
        if show_progress:
            iterator = tqdm(enumerate(images), total=len(images), desc="Aligning")

        for idx, img in iterator:
            img_f = img.astype(float)
            # compute shift that maps img -> ref
            shift, error, diffphase = phase_cross_correlation(
                ref, img_f, upsample_factor=10
            )
            # apply negative shift to img to align to ref
            corrected = ndi_shift(
                img_f, shift=shift, order=1, mode="constant", cval=0.0
            )
            aligned.append(corrected)

        return aligned


def get_align_strategy(method: Optional[str] = None) -> AlignStrategy:
    # For now we only have phase correlation, keep factory for extensibility
    return PhaseCorrelationAlignStrategy()


def align_images(
    images: List[np.ndarray],
    reference_index: int = 0,
    show_progress: bool = False,
    strategy: Optional[AlignStrategy] = None,
) -> List[np.ndarray]:
    """Align a list of images to the reference image using the chosen strategy.

    Default strategy is phase cross-correlation. Optionally show progress
    with tqdm.
    """
    if strategy is None:
        strategy = get_align_strategy()
    return (
        strategy.align(
            images,
            reference_index=reference_index,
            show_progress=show_progress,
        )
    )
