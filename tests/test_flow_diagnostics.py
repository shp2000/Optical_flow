import subprocess
import sys

import numpy as np
from PIL import Image

from optical_flow.flow_diagnostics import diagnose_flow, flow_to_rgb


def test_warp_reconstructs_translation_and_checks_round_trip():
    reference = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
    following = np.roll(reference, shift=(-1, 2), axis=(0, 1))
    forward = np.broadcast_to([2.0, -1.0], (*reference.shape[:2], 2))
    backward = -forward
    backward[1, 5] += (2, 0)
    result = diagnose_flow(reference, following, forward, backward, fb_threshold=1)

    expected_in_bounds = np.zeros(reference.shape[:2], dtype=bool)
    expected_in_bounds[1:, :-2] = True
    expected_consistent = expected_in_bounds.copy()
    expected_consistent[2, 3] = False
    np.testing.assert_array_equal(result.in_bounds, expected_in_bounds)
    np.testing.assert_array_equal(result.consistent, expected_consistent)
    np.testing.assert_array_equal(
        result.warped_next[result.in_bounds], reference[result.in_bounds]
    )
    assert (result.warp_residual[result.in_bounds] == 0).all()
    assert np.isnan(result.warp_residual[~result.in_bounds]).all()
    assert (result.unwarped_residual[result.in_bounds] > 0).all()


def test_hsv_flow_distinguishes_direction_and_magnitude():
    flow = np.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]]])
    expected = [[0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 255, 255]]
    np.testing.assert_allclose(flow_to_rgb(flow, max_motion=2)[0], expected, atol=1)


def test_diagnostics_cli_writes_panels_and_animated_reconstruction(tmp_path):
    image = np.random.default_rng(301).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    paths = [tmp_path / f"frame_{index}.png" for index in range(3)]
    for index, path in enumerate(paths):
        Image.fromarray(np.roll(image, index, axis=1)).save(path)
    output = tmp_path / "diagnostics"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "optical_flow.diagnostics_cli",
            "--output-dir",
            str(output),
            *(str(path) for path in paths),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    for index in (1, 2):
        pair = output / f"pair_{index:03d}"
        with Image.open(pair / "overview.png") as overview:
            assert overview.width > image.shape[1]
        for name in ("flow", "warped_next", "residual", "confidence", "consistency"):
            with Image.open(pair / f"{name}.png") as preview:
                assert preview.size == (image.shape[1], image.shape[0])
    with Image.open(output / "reconstruction.gif") as animation:
        assert animation.is_animated
        assert animation.n_frames == 4
