import numpy as np
from tqdm import tqdm


def _to_gray_float32(img: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale float32 for registration.

    Keeps behavior consistent for both grayscale and RGB inputs.
    """
    img_f = img.astype(np.float32, copy=False)
    if img_f.ndim == 2:
        return img_f
    if img_f.ndim == 3:
        rgb = img_f[:, :, :3]
        return (
            0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        ).astype(np.float32)
    # Fallback: flatten extra dims cautiously
    return np.mean(img_f, axis=-1).astype(np.float32)


def _apply_nan_border(aligned_img: np.ndarray) -> np.ndarray:
    """Convert constant-fill borders (typically 0) to NaN.

    This is a pragmatic way to avoid stacking artifacts at the edges after
    geometric alignment. We only touch pixels that are exactly 0.0 in all
    channels, which is consistent with constant-fill warps.

    Note: This assumes images are non-negative (typical for astro frames).
    """

    out = aligned_img.astype(np.float32, copy=False)

    if out.ndim == 2:
        border = out == 0.0
        if np.any(border):
            out = out.copy()
            out[border] = np.nan
        return out

    if out.ndim == 3:
        border = np.all(out[:, :, :3] == 0.0, axis=-1)
        if np.any(border):
            out = out.copy()
            out[border, :] = np.nan
        return out

    return out


class AstroalignAlignStrategy:
    """Align using the `astroalign` library.

    Notes:
    - We estimate the transform on a grayscale representation for robustness.
    - Then we apply the transform to the original image (including RGB).
    - If astroalign isn't available or matching fails, we fall back to identity
      alignment (no movement) instead of using another method.
    """

    def align(self, images, reference_index=0, show_progress=False):
        try:
            import astroalign as aa
        except Exception:
            # astroalign not available; keep frames unchanged.
            return [img.astype(np.float32, copy=False) for img in images]

        if not images:
            return []

        ref = images[reference_index]
        ref_gray = _to_gray_float32(ref)

        aligned = []
        iterator = tqdm(images, desc="Aligning") if show_progress else images

        for img in iterator:
            try:
                img_gray = _to_gray_float32(img)

                # astroalign.find_transform returns (transform, (src, dst))
                transf, _ = aa.find_transform(img_gray, ref_gray)

                # Apply transform to full image. Keep output shape consistent with reference.
                aligned_img, _ = aa.apply_transform(transf, img, ref)

                # Mark outside-of-frame / fill regions as NaN so stacking can ignore them.
                aligned_img = _apply_nan_border(aligned_img)

                aligned.append(aligned_img.astype(np.float32, copy=False))
            except Exception:
                # If alignment fails for a frame, keep it unchanged.
                aligned.append(img.astype(np.float32, copy=False))

        return aligned


def align_images(
    images,
    reference_index=0,
    show_progress=False,
    **_kwargs,
):
    """Align a list of images using astroalign.

    Args:
        images: List of numpy arrays (H,W) or (H,W,C)
        reference_index: Index of reference frame
        show_progress: Whether to show tqdm progress bar

    Returns:
        List of aligned images, same shapes as inputs.
    """
    strategy = AstroalignAlignStrategy()
    return strategy.align(
        images, reference_index=reference_index, show_progress=show_progress
    )
