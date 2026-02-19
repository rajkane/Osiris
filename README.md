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
python main.py -i /path/to/frames -o out.png --method average
```

3. Align frames using feature-based alignment and verbose logging:

```bash
python main.py -i /path/to/frames -o out.png --align --align-method feature --verbose
```

4. Use sigma-clipping with chunked processing (reduces peak RAM):

```bash
python main.py -i /path/to/frames -o out.png --method sigma --sigma 3.0 --sigma-iters 5 --chunk-size 50
```

5. Streaming average (low-memory):

```bash
python main.py -i /path/to/frames -o out.png --method average --stream
```

Notes and gotchas
- FITS: If `astropy` is installed, Osiris will read and write FITS files. The loader currently reads the primary HDU data and writer preserves the image data; headers are available internally but must be handled if you need WCS preservation.
- Streaming: Streaming mode is only supported for streaming‑friendly methods (currently `average`). Operations requiring global access (e.g., true median or sigma-clip) are not supported in one-pass streaming; use `--chunk-size` to enable chunked sigma-clip as a tradeoff between memory and accuracy.
- Alignment: Feature‑based alignment uses ORB (scikit-image) + RANSAC. If feature alignment fails for frames, the pipeline falls back to leaving those frames unmodified. Use `--verbose` to see diagnostic messages.
- Memory: Use `--use-memmap` to instruct the pipeline to use on-disk numpy memmap for intermediate stacking to reduce peak RAM usage. This can be slower due to disk IO but avoids OOM on large stacks.

Development & Tests

- Run tests: `pytest -q`
- Lint: `ruff check . --select E,F,W`

Contributing

Contributions are welcome. Suggested next steps if you want to extend the project:
- Preserve FITS headers/WCS across pipeline steps
- Add more robust streaming sigma-clip algorithms (2-pass or merged histograms)
- Add CI with coverage and black formatting

License

See `LICENSE` in the repository root.

