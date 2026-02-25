import numpy as np


def normalize_image(image, out_dtype=np.uint8):
    """
    Per-channel background neutralization and Gamma stretch.
    """
    img = image.astype(np.float32, copy=True)

    if img.ndim == 3:
        for c in range(3):
            channel = img[:, :, c]
            valid_mask = ~np.isnan(channel)
            if np.any(valid_mask):
                # Subtract background (removes green tint / light pollution)
                bg = np.nanpercentile(channel[valid_mask], 1)
                img[:, :, c] -= bg

                # Normalize channel to its own 99.9th percentile
                white_point = np.nanpercentile(img[:, :, c][valid_mask], 99.9)
                if white_point > 0:
                    img[:, :, c] /= white_point

    # Fill NaNs and clip
    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, 0, 1)

    # Gamma stretch to reveal faint nebula details
    img = np.power(img, 1 / 2.2)

    if out_dtype == np.uint8:
        return (img * 255).astype(np.uint8)
    return img