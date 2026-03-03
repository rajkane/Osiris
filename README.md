# Osiris — Astronomical Image Stacking CLI

Osiris is a small, test-driven CLI tool for stacking astrophotography frames.

Current pipeline (simplified):
- optional calibration (bias/dark/flat + bad-pixel mask)
- optional alignment
- per-frame median normalization (`img = img / median(img)`)
- sigma-clipping stack (optionally chunked to reduce peak RAM)

Contents
- `osiris_io` — image IO helpers (PNG/TIFF and optional FITS via astropy)
- `stacking` — stacking (sigma) and alignment
- `utils` — LogManager, MemoryManager, ErrorManager
- `tests` — pytest test suite
- `cli.py` — CLI entrypoint wrapper (calls `run_pipeline`)

Quick start (development)

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Basic usage (stack PNG/JPEG/TIFF in a folder):

```bash
python main.py -i /path/to/frames -o out.png
```

3. Align frames and show progress:

```bash
python main.py -i /path/to/frames -o out.png --align --progress
```

4. Use chunked sigma-clipping (reduces peak RAM):

```bash
python main.py -i /path/to/frames -o out.png --chunk-size 50
```

Notes and gotchas
- Calibration: use `--bias`, `--dark`, and `--flat` to supply calibration frames (single image file each). Calibration is applied before alignment and stacking.

```bash
python main.py -i /path/to/frames -o out.fits --bias /path/to/bias.fits --dark /path/to/dark.fits --flat /path/to/flat.fits --align
```

- FITS: If `astropy` is installed, Osiris will read and write FITS files.
- Memory: Use `--chunk-size` (and optionally `--use-memmap` for FITS) to reduce peak RAM usage.

Preprocess details
------------------

The `preprocess` step implements a simple calibration function `apply_calibration(image, bias, dark, flat)` that performs the following in order:

- `image = image.astype(float)`
- if `bias` provided: `image -= bias`
- if `dark` provided: `image -= dark`
- if `flat` provided: `image /= flat`  (zeros in flat are replaced with 1.0 to avoid division by zero)

Practical CLI examples
----------------------

1) Typical calibrated sigma-clipping stack (recommended):

```bash
python main.py -i /data/frames -o result.fits --bias calib/bias.fits \
  --dark calib/dark.fits --flat calib/flat.fits --align \
  --sigma 3.0 --sigma-iters 5 --chunk-size 100 --verbose
```

2) Quick run without alignment (fastest, less accurate if frames are shifted):

```bash
python main.py -i /data/frames -o result.png --no-align --chunk-size 25
```

Profiles & shortcuts (quick commands)
-------------------------------------

If you run the same groups of options repeatedly, use profiles stored in `./osiris.toml`.

List available profiles:

```bash
python main.py --list-profiles
```

Run a profile by name:

```bash
python main.py --profile deep_sky
```

Override a single option on top of a profile:

```bash
python main.py --profile deep_sky --chunk-size 10
```

Precedence (how values are chosen)
- CLI flags (highest priority)
- Profile values in TOML (`./osiris.toml`)
- Built-in defaults (lowest priority)

Options reference
-----------------

| Flag | Purpose | Notes |
|---|---|---|
| `--sigma` | Sigma threshold | Default comes from profile or 3.0 |
| `--sigma-iters` | Sigma clipping iterations | More iters = slower, sometimes more robust |
| `--chunk-size N` | Chunked sigma (low memory) | Tradeoff RAM vs precision |
| `--align` / `--no-align` | Enable/disable alignment | Alignment is recommended for deep-sky stacks |
| `--use-memmap` | FITS memmap loading when possible | Mostly helps with FITS inputs |
| `--progress` | Show tqdm progress bars | Useful for alignment |
| `--bias/--dark/--flat` | Calibration frames | Applied before alignment |
| `--mask` | Bad-pixel mask | Non-zero/True pixels will be corrected |

Development & Tests

- Run tests: `pytest -q`
- Lint: `ruff check . --select E,F,W`

License

See `LICENSE` in the repository root.

