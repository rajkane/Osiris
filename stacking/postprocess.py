import numpy as np


def normalize_image(image, out_dtype=np.uint8):
    img = image.astype(np.float32, copy=True)

    # Background neutralization
    bg = np.percentile(img, 2)
    img -= bg

    img = np.clip(img, 0, None)

    # White point
    white = np.percentile(img, 99.8)
    if white > 0:
        img /= white

    img = np.clip(img, 0, 1)

    # Soft gamma (astro friendly)
    img = np.power(img, 1 / 2.4)

    if out_dtype == np.uint8:
        return (img * 255.0).astype(np.uint8)

    return img
