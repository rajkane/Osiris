import argparse
import os
from typing import Optional


def run_pipeline(
    input_dir: str,
    output_path: str,
    method: str = "average",
    align: bool = False,
    verbose: bool = False,
    mem_limit_mb: Optional[float] = None,
    logger=None,
    **kwargs,
):
    """Run the full stacking pipeline: load -> preprocess -> align -> combine
    -> postprocess -> save.

    Comments in English inside code as requested.
    """
    # Local imports to avoid import-time side-effects and keep top-level light
    from osiris_io.file_loader import FileLoader
    from osiris_io.file_writer import FileWriter
    from stacking import (
        align_images,
        normalize_image,
        stack_images,
    )
    from utils import ErrorManager, LogManager

    if logger is None:
        logger = LogManager.get_logger()

    try:
        if verbose:
            logger.info(f"Loading images from: {input_dir}")
        images = FileLoader.load_images_from_dir(input_dir)

        if not images:
            raise ValueError("No images found in input directory")

        # optional alignment
        if align:
            if verbose:
                logger.info("Aligning images...")
            align_method = kwargs.get("align_method", None)
            align_kp = kwargs.get("align_kp", None)
            images = align_images(
                images,
                show_progress=kwargs.get("progress", False) or verbose,
                method=align_method,
                kp=align_kp,
            )

        # combine / stack
        if verbose:
            logger.info(f"Stacking images using method: {method}")
        stacked = stack_images(images, method=method, **kwargs)

        # postprocess (prepare float image)
        if verbose:
            logger.info("Postprocessing result image...")
        # We don't assign the result to an unused variable
        _ = normalize_image(stacked, out_dtype=None)

        # ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # save (cast to uint8 for file storage)
        # Prefer preserving absolute intensity when possible (no global
        # normalization
        import numpy as np

        min_val = np.nanmin(stacked)
        max_val = np.nanmax(stacked)

        if np.isfinite(min_val) and np.isfinite(max_val) and max_val - min_val > 0:
            # If values already in 0-255 range, just clip and cast
            if 0 <= min_val and max_val <= 255:
                final = np.clip(np.rint(stacked), 0, 255).astype(np.uint8)
            else:
                # otherwise normalize to 0..255
                scaled = (stacked - min_val) / (max_val - min_val)
                final = np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)
        else:
            # degenerate case (constant image). Cast constant value to uint8
            val = 0 if not np.isfinite(min_val) else float(min_val)
            v = int(round(max(0, min(255, val))))
            final = np.full_like(stacked, fill_value=v, dtype=np.uint8)

        FileWriter.save_image(output_path, final)

        if verbose:
            logger.info(f"Saved stacked image to: {output_path}")

        return output_path

    except Exception as e:
        ErrorManager().handle_error(e, logger=logger)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="CLI application for stacking astrophotography images."
    )

    parser.add_argument(
        "--input", "-i", required=True, help="Path to input image directory"
    )

    parser.add_argument(
        "--output", "-o", required=True, help="Path to output stacked image"
    )

    parser.add_argument(
        "--method",
        "-m",
        choices=["average", "median"],
        default="average",
        help="Stacking method to use (default: average or median)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Sigma value for sigma-clipping combine strategy",
    )
    parser.add_argument(
        "--sigma-iters",
        type=int,
        default=5,
        help="Max iterations for sigma-clipping combine strategy",
    )

    parser.add_argument(
        "--align",
        "-a",
        action="store_true",
        default=False,
        help="Align images before stacking (default: False)",
    )
    parser.add_argument(
        "--align-method",
        type=str,
        default=None,
        help="Alignment strategy to use, e.g. 'feature' or 'phase'",
    )
    parser.add_argument(
        "--align-kp",
        type=int,
        default=500,
        help="Number of keypoints for feature-based alignment (ORB)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose output (default: False)",
    )

    parser.add_argument(
        "--mem-limit",
        type=float,
        default=None,
        help="Memory limit in MB (optional safeguard)",
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars during long operations",
    )

    args = parser.parse_args()

    # Run the pipeline with parsed arguments
    run_pipeline(
        input_dir=args.input,
        output_path=args.output,
        method=args.method,
        align=args.align,
        verbose=args.verbose,
        mem_limit_mb=args.mem_limit,
        logger=None,
        # forward strategy kwargs
        **{
            "sigma": args.sigma,
            "sigma_iters": args.sigma_iters,
            "align_method": args.align_method,
            "align_kp": args.align_kp,
        },
    )


if __name__ == "__main__":
    main()
