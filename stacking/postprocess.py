import numpy as np


def normalize_image(image: np.ndarray, out_dtype=np.uint8) -> np.ndarray:
    """Normalize image to full dynamic range of out_dtype (default uint8).

    Returns an array with dtype = out_dtype.
    """
    img = image.astype(np.float64)
    mn, mx = np.nanmin(img), np.nanmax(img)
    if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img)

    if out_dtype == np.uint8:
        img = (img * 255.0).round().astype(np.uint8)
    else:
        return img.astype(out_dtype)

    return img
