# Osiris — Astronomical Image Stacking CLI

Osiris is a small, test-driven CLI tool for stacking astrophotography frames. It supports several stacking strategies (average, median, sigma-clip), optional alignment (phase-correlation or feature-based), streaming processing for low-memory runs, FITS input/output, and basic memory safeguards.

This repository contains a minimal, well-tested implementation intended as a starting point for a production-ready tool.

Contents
- `osiris_io` — image IO helpers (PNG/TIFF and optional FITS via astropy)
- `stacking` — stacking strategies and alignment implementations
- `utils` — LogManager, MemoryManager, ErrorManager
- `tests` — pytest test suite
- `cli.py` — command-line entrypoint wrapper (calls `run_pipeline`)

Quick start (development)

1. Create a virtual environment and install dependencies (development):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Basic usage (stack PNG/JPEG/TIFF in a folder):

```bash
python main.py -i /path/to/frames -o out_default.png --method average
```

3. Align frames using feature-based alignment and verbose logging:

```bash
python main.py -i /path/to/frames -o out_default.png --align --align-method feature --verbose
```

4. Use sigma-clipping with chunked processing (reduces peak RAM):

```bash
python main.py -i /path/to/frames -o out_default.png --method sigma --sigma 3.0 --sigma-iters 5 --chunk-size 50
```

5. Streaming average (low-memory):

```bash
python main.py -i /path/to/frames -o out_default.png --method average --stream
```

Notes and gotchas
- Preprocess / Calibration: use `--bias`, `--dark`, and `--flat` to supply calibration frames (single image file each). The pipeline will apply calibration before alignment and stacking. Example:

```bash
python main.py -i /path/to/frames -o out_default.fits --bias /path/to/bias.fits --dark /path/to/dark.fits --flat /path/to/flat.fits --method average --align
```

- FITS: If `astropy` is installed, Osiris will read and write FITS files. The loader can optionally return FITS headers and the writer will preserve a provided primary header when writing FITS output. When saving to FITS, Osiris will capture a header from the first input frame (if available) and attach it to the output primary HDU.
- Streaming: Streaming mode is only supported for streaming‑friendly methods (currently `average`). Operations requiring global access (e.g., true median or sigma-clip) are not supported in one-pass streaming; use `--chunk-size` to enable chunked sigma-clip as a tradeoff between memory and accuracy.
- Alignment: Feature‑based alignment uses ORB (scikit-image) + RANSAC. If feature alignment fails for frames, the pipeline falls back to leaving those frames unmodified. Use `--verbose` to see diagnostic messages.
- Memory: Use `--use-memmap` to instruct the pipeline to use on-disk numpy memmap for intermediate stacking to reduce peak RAM usage. This can be slower due to disk IO but avoids OOM on large stacks.

Preprocess details
------------------

The `preprocess` step implements a simple calibration function `apply_calibration(image, bias, dark, flat)` that performs the following in order:

- `image = image.astype(float)`
- if `bias` provided: `image -= bias`
- if `dark` provided: `image -= dark`
- if `flat` provided: `image /= flat`  (zeros in flat are replaced with 1.0 to avoid division by zero)

This step runs automatically if any of `--bias`, `--dark`, or `--flat` are supplied. In streaming mode the calibration frames are preloaded and applied per-frame as they are yielded from disk.

Practical CLI examples
----------------------

1) Typical calibrated sigma‑clip stack (recommended):

```bash
python main.py -i /data/frames -o result.fits --bias calib/bias.fits \
  --dark calib/dark.fits --flat calib/flat.fits --align --align-method phase \
  --method sigma --sigma 3.0 --sigma-iters 5 --chunk-size 100 --verbose
```

2) Low memory average with streaming:

```bash
python main.py -i /data/frames -o result.png --method average --stream --use-memmap
```

3) Quick median stack without calibration (only if frames are pre-calibrated):

```bash
python main.py -i /precalibrated -o median.png --method median
```

Profiles & shortcuts (quick commands)
-------------------------------------

If you run the same groups of options repeatedly, use profiles stored in a TOML file so you only type a short command.

Where to put profiles
- Per-user config: `~/.config/osiris/config.toml` (preferred for a user on your machine)
- Project-local: `./osiris.toml` (keeps config with the project and works for CI or other users)
- Examples shipped with this repo: `examples/config.toml` or the root example `osiris.toml`.

List available profiles

```bash
python main.py --list-profiles
```

Run a profile by name

```bash
python main.py --profile deep_sky
```

Override a single option on top of a profile

```bash
python main.py --profile deep_sky --method average
```

Precedence (how values are chosen)
- CLI flags (highest priority)
- Environment variables: `OSIRIS_<UPPER_NAME>` (for example `OSIRIS_BIAS`)
- Profile values in TOML (`~/.config/osiris/config.toml` or `./osiris.toml`)
- Built-in defaults (lowest priority)

Example profile (the project contains `osiris.toml` and `examples/config.toml`):

```toml
[deep_sky]
input = "./data/frames/deep_frames"
output = "./data/out_deep/deep_stack.jpg"
method = "sigma"
sigma = 3.5
sigma_iters = 5
chunk_size = 200
align = true
align_method = "feature"
use_memmap = true
```

Quick aliases and scripts

- Shell alias (bash/zsh):

```bash
# put into ~/.bashrc or ~/.zshrc
alias osiris-default='python /path/to/main.py --profile default'
```

- Small wrapper script (`osiris-run`) you can add to your PATH:

```bash
#!/usr/bin/env bash
python /path/to/main.py --profile "$1"
```

Environment variables

- You can export environment variables as shortcuts. Example:

```bash
export OSIRIS_BIAS=/home/user/calib/bias.fits
export OSIRIS_DARK=/home/user/calib/dark.fits
python main.py --profile quick
```

Notes and tips
- Keep calibration file paths (bias/dark/flat) absolute or relative to where you run the command.
- For reproducibility, commit a project-local `osiris.toml` in your project repository (others can use it directly).
- Use `--profile` + occasional CLI overrides for the fastest, repeatable workflows.

Options reference
-----------------

| Flag | Purpose | Notes |
|---|---|---|
| --bias PATH | Bias frame to subtract | Single image file (FITS/PNG). Applied to all frames before align. |
| --dark PATH | Dark frame to subtract | Should match exposure characteristics when possible. |
| --flat PATH | Flat frame to divide | Prevent zero-values in flat; tool replaces zeros with 1.0. |
| --stream | Stream images (low memory) | Only supported for average (one‑pass) or when using chunked approaches. |
| --chunk-size N | Process N frames per chunk for sigma | Tradeoff memory vs. clipping accuracy. |

Development & Tests

- Run unit and integration tests: `pytest -q`
- Lint: `ruff check . --select E,F,W`

Contributing

Contributions are welcome. Suggested next steps if you want to extend the project:
- Preserve FITS headers/WCS across pipeline steps (already implemented when writing primary header from first input).  
- Add more robust streaming sigma-clip algorithms (2-pass or merged histograms)
- Add CI with coverage and black formatting

License

See `LICENSE` in the repository root.

