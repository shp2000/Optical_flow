import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from optical_flow import video_flow
from optical_flow.improved_cli import main


def _moving_frames() -> list[np.ndarray]:
    frame = np.random.default_rng(224).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return [np.roll(frame, 3 * index, axis=1) for index in range(4)]


def _read_video(path: Path) -> tuple[float, list[np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, image = capture.read()
            if not ok:
                return fps, frames
            frames.append(image)
    finally:
        capture.release()


def test_improved_main_exports_resized_video_with_flow_and_source_playback_rate(
    tmp_path,
):
    source = tmp_path / "input.mp4"
    output = tmp_path / "nested" / "flow.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12.5, (96, 64)
    )
    try:
        assert writer.isOpened()
        for frame in _moving_frames():
            writer.write(frame)
    finally:
        writer.release()

    entry_point = Path(__file__).resolve().parents[1] / "improved_main.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(entry_point),
            "--video",
            str(source),
            "--output-video",
            str(output),
            "--device",
            "cpu",
            "--width",
            "63",
            "--max-frames",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Average processing FPS" in completed.stdout
    fps, frames = _read_video(output)
    assert fps == pytest.approx(12.5)
    assert len(frames) == 3
    assert all(frame.shape == (74, 126, 3) for frame in frames)
    first_flow = frames[0][32:, 63:]
    next_flow = frames[1][32:, 63:]
    assert next_flow.mean() > first_flow.mean() + 5


def test_webcam_stream_releases_capture_and_falls_back_after_cuda_failure(
    tmp_path, monkeypatch, capsys
):
    class Camera:
        def __init__(self):
            self.frames = iter(_moving_frames())
            self.reads = 0
            self.released = False

        def isOpened(self):
            return True

        def get(self, _property):
            return 0.0

        def read(self):
            self.reads += 1
            return True, next(self.frames)

        def release(self):
            self.released = True

    camera = Camera()
    open_capture = cv2.VideoCapture

    def capture(source):
        if isinstance(source, int):
            assert source == 2
            return camera
        return open_capture(source)

    def failing_cuda(_first, _second):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(cv2, "VideoCapture", capture)
    monkeypatch.setattr(video_flow, "_cuda_estimator", lambda: failing_cuda)
    output = tmp_path / "camera.mp4"
    main(
        [
            "--webcam",
            "2",
            "--no-show",
            "--height",
            "41",
            "--max-frames",
            "2",
            "--output-video",
            str(output),
        ]
    )

    assert camera.reads == 2
    assert camera.released
    assert "Switching to CPU" in capsys.readouterr().err
    fps, frames = _read_video(output)
    assert fps == pytest.approx(30)
    assert len(frames) == 2
    assert frames[0].shape == (74, 124, 3)
