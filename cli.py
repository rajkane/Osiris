import argparse
import os

import numpy as np


def _load_profiles(path: str = "osiris.toml") -> dict:
    import tomllib

    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _coalesce(*values):
    for v in values:
        if v is not None:
            return v
    return None


def run_pipeline(
    input_dir,
    output_path,
    method="sigma",
    align=False,
    verbose=False,
    **kwargs,
):
    from osiris_io.file_loader import FileLoader
    from osiris_io.file_writer import FileWriter
    from stacking import align_images, normalize_image, stack_images
    from stacking.preprocess import apply_calibration
    from utils import LogManager

    logger = LogManager.get_logger()

    if not input_dir or not output_path:
        raise ValueError("input_dir and output_path must be provided")

    use_memmap = bool(kwargs.get("use_memmap", False))
    show_progress = bool(kwargs.get("progress", False))

    # calibration frames
    bias = None
    dark = None
    flat = None
    bad_pixel_mask = None
    bias_path = kwargs.get("bias")
    dark_path = kwargs.get("dark")
    flat_path = kwargs.get("flat")
    mask_path = kwargs.get("mask")

    if bias_path:
        bias = FileLoader.load_image(bias_path, use_memmap=use_memmap)
    if dark_path:
        dark = FileLoader.load_image(dark_path, use_memmap=use_memmap)
    if flat_path:
        flat = FileLoader.load_image(flat_path, use_memmap=use_memmap)
    if mask_path:
        bad_pixel_mask = FileLoader.load_image(mask_path, use_memmap=use_memmap)

    # Optional: preserve FITS header if output is FITS.
    header = None

    if verbose:
        logger.info(f"Loading images from: {input_dir}")

    images = FileLoader.load_images_from_dir(input_dir, use_memmap=use_memmap)
    if len(images) == 0:
        raise RuntimeError("No images found in input directory")

    # apply calibration if provided
    if (
        bias is not None
        or dark is not None
        or flat is not None
        or bad_pixel_mask is not None
    ):
        images = [
            apply_calibration(
                img, bias=bias, dark=dark, flat=flat, bad_pixel_mask=bad_pixel_mask
            )
            for img in images
        ]

    # 2) Align
    if align:
        if verbose:
            logger.info("Zarovnávam snímky...")
        images = align_images(images, show_progress=show_progress)

    # 3) Stack (sigma clipping)
    if verbose:
        logger.info("Spájam snímky (metóda: sigma)...")

    stacked = stack_images(
        images,
        method="sigma",
        sigma=kwargs.get("sigma", 3.0),
        sigma_iters=kwargs.get("sigma_iters", 5),
        chunk_size=kwargs.get("chunk_size"),
        stream=False,
        use_memmap=use_memmap,
        show_progress=show_progress,
    )

    # 4) Output formatting
    normalize = bool(kwargs.get("normalize", False))
    out_dtype = kwargs.get("out_dtype")  # "uint8" | "float32" | None

    to_save = stacked

    if normalize:
        # normalize_image returns uint8 by default
        to_save = normalize_image(to_save, out_dtype=np.uint8)
    else:
        # Keep float output when saving FITS unless user explicitly requests uint8.
        is_fits = str(output_path).lower().endswith((".fits", ".fit"))
        if out_dtype == "uint8" or (out_dtype is None and not is_fits):
            to_save = np.clip(to_save, 0, 255).astype(np.uint8)
        else:
            # Default for FITS/scientific output
            to_save = to_save.astype(np.float32, copy=False)

    FileWriter.save_image(output_path, to_save, header=header)
    if verbose:
        logger.info(f"Hotovo! Obrázok uložený: {output_path}")

    return output_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="osiris")

    # Profiles
    p.add_argument("--profile", "-p", help="Name of profile in osiris.toml")
    p.add_argument(
        "--list-profiles", action="store_true", help="List available profiles and exit"
    )

    # Core IO
    p.add_argument("-i", "--input", dest="input_dir", help="Input frames directory")
    p.add_argument("-o", "--output", dest="output_path", help="Output image path")

    # Sigma-clipping options
    p.add_argument("--sigma", type=float, help="Sigma threshold for sigma clipping")
    p.add_argument("--sigma-iters", type=int, help="Iterations for sigma clipping")
    p.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk size for chunked sigma clipping (low memory)",
    )

    # Pipeline switches
    p.add_argument("--align", action="store_true", help="Enable alignment (astroalign)")
    p.add_argument("--no-align", action="store_true", help="Disable alignment")

    p.add_argument(
        "--use-memmap",
        action="store_true",
        help="Use FITS memmap loading when possible",
    )

    # Show progress by default; can be disabled explicitly
    p.add_argument(
        "--progress",
        dest="progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars (default: on)",
    )

    # Calibration
    p.add_argument("--bias", help="Bias frame path")
    p.add_argument("--dark", help="Dark frame path")
    p.add_argument("--flat", help="Flat frame path")
    p.add_argument(
        "--mask", help="Bad-pixel mask path (non-zero/True pixels will be corrected)"
    )

    # Output
    p.add_argument(
        "--normalize", action="store_true", help="Normalize for display output (uint8)"
    )
    p.add_argument(
        "--out-dtype", choices=["uint8", "float32"], help="Force output dtype"
    )

    # Logging
    p.add_argument(
        "--verbose",
        dest="verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verbose logging (default: on)",
    )

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    profiles = _load_profiles("osiris.toml")

    if args.list_profiles:
        for name in sorted(profiles.keys()):
            print(name)
        return

    profile_data = profiles.get(args.profile, {}) if args.profile else {}
    if args.profile and args.profile not in profiles:
        raise ValueError(f"Profile '{args.profile}' not found in osiris.toml")

    # Resolve values with precedence: CLI > profile > defaults
    input_dir = _coalesce(args.input_dir, profile_data.get("input"))
    output_path = _coalesce(args.output_path, profile_data.get("output"))

    # align: explicit CLI overrides profile
    if args.no_align:
        align = False
    elif args.align:
        align = True
    else:
        align = bool(profile_data.get("align", False))

    run_pipeline(
        input_dir=input_dir,
        output_path=output_path,
        method="sigma",
        align=align,
        verbose=bool(args.verbose) if args.verbose is not None else bool(profile_data.get("verbose", True)),
        progress=bool(args.progress) if args.progress is not None else bool(profile_data.get("progress", True)),
        sigma=_coalesce(args.sigma, profile_data.get("sigma"), 3.0),
        sigma_iters=_coalesce(args.sigma_iters, profile_data.get("sigma_iters"), 5),
        chunk_size=_coalesce(args.chunk_size, profile_data.get("chunk_size")),
        use_memmap=bool(args.use_memmap) or bool(profile_data.get("use_memmap", False)),
        bias=_coalesce(args.bias, profile_data.get("bias")),
        dark=_coalesce(args.dark, profile_data.get("dark")),
        flat=_coalesce(args.flat, profile_data.get("flat")),
        mask=_coalesce(args.mask, profile_data.get("mask")),
        normalize=bool(args.normalize) or bool(profile_data.get("normalize", False)),
        out_dtype=_coalesce(args.out_dtype, profile_data.get("out_dtype")),
    )


if __name__ == "__main__":
    main()
