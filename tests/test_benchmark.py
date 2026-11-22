import csv
import json
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from optical_flow.benchmark import flow_errors
from optical_flow.benchmark_data import list_pairs, read_flo


def _write_flo(path: Path, flow: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = flow.shape[:2]
    path.write_bytes(
        b"PIEH" + struct.pack("<ii", width, height) + flow.astype("<f4").tobytes()
    )


def test_flo_preserves_components_and_rejects_truncated_payload(tmp_path):
    flow = np.arange(12).reshape(2, 3, 2)
    path = tmp_path / "flow.flo"
    _write_flo(path, flow)
    np.testing.assert_array_equal(read_flo(path), flow)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="payload size"):
        read_flo(path)


def test_flow_errors_ignore_invalid_truth_and_use_strict_bad_pixel_threshold():
    truth = np.array([[[0, 0], [0, 0], [0, 0], [0, 1], [1e10, 0], [0, np.nan], [2, 1]]])
    prediction = np.array(
        [[[0, 0], [3, 0], [0, 4], [1, 0], [np.nan, 0], [0, np.nan], [np.nan, 0]]]
    )
    invalid = np.array([[0, 0, 0, 0, 0, 0, 255]])
    result = flow_errors(prediction, truth, bad_threshold=3, invalid=invalid).averages()
    assert result == pytest.approx(
        {
            "valid_pixels": 4,
            "mean_epe_px": (7 + np.sqrt(2)) / 4,
            "mean_angular_error_deg": (60 + np.degrees(np.arctan([3, 4])).sum()) / 4,
            "bad_pixel_pct": 25,
        }
    )
    with pytest.raises(ValueError, match="Non-finite prediction"):
        flow_errors(prediction, truth, bad_threshold=3)


def test_sintel_pairs_use_selected_pass_next_frame_and_invalid_mask(tmp_path):
    training = tmp_path / "training"
    flow = training / "flow" / "alley_1" / "frame_0009.flo"
    _write_flo(flow, np.zeros((2, 3, 2)))
    [pair] = list_pairs(tmp_path, "sintel", "final")
    assert pair.name == "alley_1/frame_0009"
    assert pair.first == training / "final" / "alley_1" / "frame_0009.png"
    assert pair.second == training / "final" / "alley_1" / "frame_0010.png"
    assert pair.invalid == training / "invalid" / "alley_1" / "frame_0009.png"


def test_middlebury_cli_writes_pixel_weighted_accuracy_and_performance_reports(
    tmp_path,
):
    image = np.zeros((8, 10), dtype=np.uint8)
    encoded, buffer = cv2.imencode(".png", image)
    assert encoded
    truth = np.zeros((*image.shape, 2))
    for scene in ("scene_A", "scene_B"):
        frames = tmp_path / "other-data" / scene
        frames.mkdir(parents=True)
        for index in (10, 11):
            (frames / f"frame{index}.png").write_bytes(buffer.tobytes())
        if scene == "scene_B":
            truth[...] = 1e10
            truth[0, 0] = (3, 4)
        _write_flo(tmp_path / "other-gt-flow" / scene / "flow10.flo", truth)

    output = tmp_path / "reports"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "optical_flow.benchmark",
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--repeats",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    metadata = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert metadata["pairs"] == 2
    methods = [summary["method"] for summary in metadata["methods"]]
    assert methods == ["lk-original", "lk-pyramid", "farneback"]
    for summary in metadata["methods"]:
        assert summary["dense"] is True
        assert summary["valid_pixels"] == image.size + 1
        assert summary["mean_epe_px"] == pytest.approx(5 / summary["valid_pixels"])
        assert summary["bad_pixel_pct"] == pytest.approx(100 / summary["valid_pixels"])
        assert summary["peak_rss_mib"] > 0
        assert summary["fps"] == pytest.approx(2 / summary["runtime_seconds"])
    with (output / "per_pair.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["method"], row["pair"]) for row in rows] == [
        (method, scene) for method in methods for scene in ("scene_A", "scene_B")
    ]
    assert [float(row["mean_epe_px"]) for row in rows] == [0, 5] * len(methods)
    assert (
        (output / "accuracy_vs_speed.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    )
    table = (output / "comparison.md").read_text(encoding="utf-8")
    assert "| Method | EPE ↓ (px) | Runtime ↓ (s) | FPS ↑ | Dense |" in table
    assert "| Original Lucas–Kanade |" in table
    assert "| Optimized pyramidal Lucas–Kanade |" in table
    assert "| OpenCV Farnebäck |" in table
    assert table.count("| Yes |") == len(methods)
