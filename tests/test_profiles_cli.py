import imageio.v3 as iio
import numpy as np


def test_profile_deep_sky_defaults_to_sigma(monkeypatch, tmp_path):
    """The deep_sky profile in osiris.toml does not set `method`, but it sets
    sigma parameters. The CLI should treat this as sigma stacking.

    The test creates its own osiris.toml in an isolated CWD to avoid depending
    on the repository file.
    """
    import cli

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for i in range(3):
        img = np.ones((10, 10), dtype=np.uint8) * (i * 10)
        iio.imwrite(str(input_dir / f"img_{i}.png"), img)

    out_path = tmp_path / "out.png"

    # Create isolated profile config
    (tmp_path / "osiris.toml").write_text(
        """
[deep_sky]
# method intentionally missing
sigma = 2.5
sigma_iters = 3
align = false
use_memmap = true
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    cli.main(
        [
            "--profile",
            "deep_sky",
            "-i",
            str(input_dir),
            "-o",
            str(out_path),
            "--no-align",
        ]
    )

    assert out_path.exists()
