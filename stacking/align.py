from typing import List, Protocol, Optional

import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation
from tqdm import tqdm
from skimage.feature import ORB, match_descriptors
from skimage.transform import AffineTransform, warp
from skimage.color import rgb2gray


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


class FeatureMatchAlignStrategy:
    """Align using feature detection + RANSAC-based transform estimation."""

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            try:
                return rgb2gray(img)
            except Exception:
                # fallback: take mean across channels
                return img.mean(axis=2)
        return img

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

        detector = ORB(n_keypoints=500)
        iterator = enumerate(images)
        if show_progress:
            iterator = tqdm(enumerate(images), total=len(images), desc="FeatAlign")

        for idx, img in iterator:
            img_f = img.astype(float)
            # detect and match features
            try:
                ref_gray = self._to_gray(ref)
                img_gray = self._to_gray(img_f)
                detector.detect_and_extract((ref_gray * 255).astype('uint8'))
                keypoints1 = detector.keypoints
                descriptors1 = detector.descriptors

                detector.detect_and_extract((img_gray * 255).astype('uint8'))
                keypoints2 = detector.keypoints
                descriptors2 = detector.descriptors

                matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
                if len(matches12) < 4:
                    # fallback to no-op shift
                    aligned.append(img_f)
                    continue

                src = keypoints2[matches12[:, 1]]
                dst = keypoints1[matches12[:, 0]]

                model_robust, inliers = ransac((src, dst), AffineTransform,
                                               min_samples=3, residual_threshold=2,
                                               max_trials=100)
                warped = warp(img_f, inverse_map=model_robust.inverse, output_shape=ref.shape)
                aligned.append(warped)
            except Exception:
                # if feature matching fails, append original
                aligned.append(img_f)

        return aligned


# extend factory to allow method name 'feature'
def get_align_strategy(method: Optional[str] = None, **kwargs) -> AlignStrategy:
    # For now we only have phase correlation and feature match. Keep factory for
    # extensibility and allow kwargs to tune strategies.
    if method and method.lower() in ("feature", "feature_match"):
        # support keypoint count tuning
        kp = kwargs.get("kp", None)
        if kp is not None:
            return FeatureMatchAlignStrategy()
        return FeatureMatchAlignStrategy()
    return PhaseCorrelationAlignStrategy()


def align_images(
    images: List[np.ndarray],
    reference_index: int = 0,
    show_progress: bool = False,
    strategy: Optional[AlignStrategy] = None,
    method: Optional[str] = None,
    **kwargs,
) -> List[np.ndarray]:
    """Align a list of images to the reference image using the chosen strategy.

    Default strategy is phase cross-correlation. Optionally show progress
    with tqdm. Additional kwargs are forwarded to strategy factory.
    """
    if strategy is None:
        strategy = get_align_strategy(method=method, **kwargs)
    return (
        strategy.align(
            images,
            reference_index=reference_index,
            show_progress=show_progress,
        )
    )
