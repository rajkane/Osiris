import argparse
import os
import gc
import numpy as np
from typing import Optional


def run_pipeline(input_dir, output_path, method="average", align=False, verbose=False, **kwargs):
    from osiris_io.file_loader import FileLoader
    from osiris_io.file_writer import FileWriter
    from stacking import align_images, normalize_image, stack_images
    from utils import LogManager

    logger = LogManager.get_logger()

    # 1. Load
    images = FileLoader.load_images_from_dir(input_dir)

    # 2. Align (Uses NaN for borders)
    if align:
        if verbose: logger.info("Aligning frames...")
        aligned = align_images(images, method=kwargs.get("align_method"), show_progress=True)
        images = aligned  # Overwrite to free old references
        gc.collect()

    # 3. Stack (Memory Safe)
    if verbose: logger.info(f"Stacking images (Method: {method})...")
    stacked = stack_images(images, method=method, **kwargs)

    # Clear RAM
    del images
    gc.collect()

    # 4. Postprocess (Fixed assignment!)
    if verbose: logger.info("Performing Color Neutralization and Stretch...")
    final_image = normalize_image(stacked, out_dtype=np.uint8)

    # 5. Save
    FileWriter.save_image(output_path, final_image)
    if verbose: logger.info(f"Successfully saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Osiris Stacker CLI")
    parser.add_argument("--input", "-i")
    parser.add_argument("--output", "-o")
    parser.add_argument("--method", "-m")
    parser.add_argument("--profile", "-p")
    parser.add_argument("--align", "-a", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    parser.add_argument("--use-memmap", action="store_true")

    args = parser.parse_args()

    # Profile integration
    import tomllib
    profiles = {}
    if os.path.exists("osiris.toml"):
        with open("osiris.toml", "rb") as f:
            profiles = tomllib.load(f)

    profile_data = profiles.get(args.profile, {}) if args.profile else {}

    # Merge CLI and Profile (CLI has priority)
    input_path = args.input or profile_data.get("input")
    output_path = args.output or profile_data.get("output")

    if not input_path or not output_path:
        parser.error("Input and Output paths required via CLI or Profile.")

    # Prepare safe arguments
    run_params = {
        "method": args.method or profile_data.get("method", "average"),
        "align": args.align or profile_data.get("align", False),
        "use_memmap": args.use_memmap or profile_data.get("use_memmap", False),
        "chunk_size": profile_data.get("chunk_size", 3),
        "sigma": profile_data.get("sigma", 3.0),
        "align_method": profile_data.get("align_method", "phase")
    }

    run_pipeline(
        input_dir=input_path,
        output_path=output_path,
        verbose=args.verbose,
        **run_params
    )


if __name__ == "__main__":
    main()