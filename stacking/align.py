import numpy as np
import gc
from scipy.ndimage import shift as ndi_shift
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, warp
from tqdm import tqdm

class PhaseCorrelationAlignStrategy:
    def align(self, images, reference_index=0, show_progress=False):
        ref = images[reference_index].astype(np.float32)
        aligned = []
        iterator = tqdm(images, desc="Aligning") if show_progress else images
        for img in iterator:
            img_f = img.astype(np.float32)
            shift, _, _ = phase_cross_correlation(ref, img_f, upsample_factor=10)
            corrected = ndi_shift(img_f, shift=shift, order=1, mode="constant", cval=np.nan)
            aligned.append(corrected)
            gc.collect()
        return aligned

class FeatureMatchAlignStrategy:
    def __init__(self, n_keypoints=5000):
        self.n_keypoints = n_keypoints

    def align(self, images, reference_index=0, show_progress=False):
        ref = images[reference_index].astype(np.float32)
        ref_gray = rgb2gray(ref) if ref.ndim == 3 else ref
        aligned = []
        detector = ORB(n_keypoints=self.n_keypoints)
        iterator = tqdm(images, desc="Aligning") if show_progress else images
        for img in iterator:
            img_f = img.astype(np.float32)
            try:
                img_gray = rgb2gray(img_f) if img_f.ndim == 3 else img_f
                detector.detect_and_extract((ref_gray * 255).astype("uint8"))
                kp1, des1 = detector.keypoints, detector.descriptors
                detector.detect_and_extract((img_gray * 255).astype("uint8"))
                kp2, des2 = detector.keypoints, detector.descriptors
                matches = match_descriptors(des1, des2, cross_check=True)
                src, dst = kp2[matches[:, 1]], kp1[matches[:, 0]]
                model, _ = ransac((src, dst), SimilarityTransform, min_samples=2, residual_threshold=2, max_trials=500)
                warped = warp(img_f, inverse_map=model.inverse, output_shape=ref.shape, cval=np.nan)
                aligned.append(warped.astype(np.float32))
            except:
                aligned.append(img_f)
            gc.collect()
        return aligned

def align_images(images, method=None, **kwargs):
    strategy = FeatureMatchAlignStrategy(n_keypoints=kwargs.get("kp", 5000)) if method == "feature" else PhaseCorrelationAlignStrategy()
    aligned = strategy.align(images, show_progress=kwargs.get("show_progress", False))
    # replace NaN with zeros to avoid downstream argmax/nan issues
    aligned = [np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0) for a in aligned]
    return aligned
