"""Stream dense optical flow from video files or cameras into an MP4."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Callable
from contextlib import suppress
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np

from .benchmark_raft import RAFT_SETTINGS, build_raft_estimator
from .flow_diagnostics import flow_to_rgb
from .improved_compute_flow import flow_lk_pyramid

FlowEstimator = Callable[[np.ndarray, np.ndarray], np.ndarray]
WINDOW_TITLE = "Optical flow video"
HEADER_HEIGHT = 32
CPU_BACKEND = "CPU LK"


def add_video_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="process a video file")
    source.add_argument(
        "--webcam",
        type=int,
        nargs="?",
        const=0,
        help="record a webcam (default index: 0)",
    )
    video = parser.add_argument_group("video options")
    video.add_argument("--output-video", type=Path, help="destination MP4")
    video.add_argument("--width", type=int, help="processing width in pixels")
    video.add_argument("--height", type=int, help="processing height in pixels")
    video.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    video.add_argument(
        "--preview",
        action="store_true",
        help="show the processing window for a video file",
    )
    video.add_argument(
        "--max-frames", type=int, help="stop after exporting this many frames"
    )
    video.add_argument(
        "--output-fps", type=float, help="override the MP4 playback frame rate"
    )
    video.add_argument(
        "--flow-max",
        type=float,
        default=10,
        help="motion in pixels at full HSV brightness",
    )


def _estimate_cpu(first: np.ndarray, second: np.ndarray, settings: dict) -> np.ndarray:
    flow, _ = flow_lk_pyramid(
        cv2.cvtColor(first, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(second, cv2.COLOR_BGR2GRAY),
        **settings,
    )
    return flow


def _cuda_estimator() -> FlowEstimator | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    raft = build_raft_estimator("raft-small", **RAFT_SETTINGS, device="cuda")

    def estimate(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return raft(
            cv2.cvtColor(first, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(second, cv2.COLOR_BGR2RGB),
        )

    return estimate


def _select_estimator(requested: str, cpu: FlowEstimator) -> tuple[str, FlowEstimator]:
    if requested == "cpu":
        return CPU_BACKEND, cpu
    try:
        cuda = _cuda_estimator()
    except (OSError, RuntimeError) as error:
        if requested == "cuda":
            raise
        print(f"CUDA initialization failed. Switching to CPU. {error}", file=sys.stderr)
    else:
        if cuda is not None:
            return "CUDA RAFT Small", cuda
        if requested == "cuda":
            raise RuntimeError(
                "CUDA is unavailable. Install a GPU-enabled PyTorch and Torchvision build."
            )
    return CPU_BACKEND, cpu


def _resize_target(
    frame: np.ndarray, width: int | None, height: int | None
) -> tuple[int, int]:
    source_height, source_width = frame.shape[:2]
    if width is None and height is None:
        return source_width, source_height
    if width is None:
        width = max(1, round(source_width * height / source_height))
    if height is None:
        height = max(1, round(source_height * width / source_width))
    return width, height


def _resize(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    if size == (width, height):
        return frame
    interpolation = (
        cv2.INTER_AREA if size[0] <= width and size[1] <= height else cv2.INTER_LINEAR
    )
    return cv2.resize(frame, size, interpolation=interpolation)


def _compose(
    frame: np.ndarray, flow_view: np.ndarray, backend: str, fps: float | None
) -> np.ndarray:
    height, width = frame.shape[:2]
    # MP4 encoders may truncate odd dimensions. Pad the canvas, leaving the image grid intact.
    canvas = np.zeros(
        (height + HEADER_HEIGHT + height % 2, 2 * width, 3), dtype=np.uint8
    )
    canvas[:HEADER_HEIGHT] = (45, 34, 25)
    canvas[HEADER_HEIGHT : HEADER_HEIGHT + height, :width] = frame
    canvas[HEADER_HEIGHT : HEADER_HEIGHT + height, width:] = flow_view

    label = (
        f"Input | {fps:.1f} processing FPS"
        if fps is not None
        else "Input | First frame"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    for start, text in ((0, label), (width, f"HSV flow | {backend}")):
        margin = min(8, width // 8)
        text_width, _ = cv2.getTextSize(text, font, 1.0, 1)[0]
        scale = min(0.56, (width - 2 * margin) / text_width)
        cv2.putText(
            canvas,
            text,
            (start + margin, 22),
            font,
            scale,
            (244, 244, 244),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _preview_should_stop(frame: np.ndarray) -> bool:
    cv2.imshow(WINDOW_TITLE, frame)
    if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
        return True
    # Some HighGUI backends do not expose window visibility.
    with suppress(cv2.error):
        return cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
    return False


def _write_frames(
    capture: cv2.VideoCapture,
    writer: cv2.VideoWriter,
    first: np.ndarray,
    size: tuple[int, int],
    args: argparse.Namespace,
    preview: bool,
) -> tuple[int, float | None]:
    settings = {
        "levels": args.levels,
        "iterations": args.iterations,
        "window_size": args.window_size,
    }
    cpu = partial(_estimate_cpu, settings=settings)
    backend, estimate = _select_estimator(args.device, cpu)
    print(f"Processing video with {backend}.", flush=True)

    previous = _resize(first, size)
    count = 0
    processing_seconds = 0.0
    fps = None
    try:
        while True:
            if count == 0:
                frame = previous
                flow_view = np.zeros_like(frame)
            else:
                started = perf_counter()
                ok, following = capture.read()
                if not ok:
                    break
                frame = _resize(following, size)
                try:
                    flow = estimate(previous, frame)
                except RuntimeError as error:
                    if args.device != "auto" or estimate is cpu:
                        raise
                    backend, estimate = CPU_BACKEND, cpu
                    print(
                        f"CUDA processing failed. Switching to CPU. {error}",
                        file=sys.stderr,
                    )
                    flow = estimate(previous, frame)
                flow_view = cv2.cvtColor(
                    flow_to_rgb(flow, args.flow_max), cv2.COLOR_RGB2BGR
                )
                processing_seconds += perf_counter() - started
                fps = count / processing_seconds

            canvas = _compose(frame, flow_view, backend, fps)
            writer.write(canvas)
            count += 1
            previous = frame
            if preview and _preview_should_stop(canvas):
                break
            if count == args.max_frames:
                break

    except KeyboardInterrupt:
        if count == 0:
            raise
        print("Recording stopped. Saving the processed frames.", flush=True)
    return count, fps


def run_video(args: argparse.Namespace) -> None:
    for option, value in (
        ("--width", args.width),
        ("--height", args.height),
        ("--max-frames", args.max_frames),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{option} must be positive")
    for option, value in (
        ("--output-fps", args.output_fps),
        ("--flow-max", args.flow_max),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{option} must be finite and positive")
    if (
        args.levels < 1
        or args.iterations < 1
        or args.window_size < 3
        or args.window_size % 2 == 0
    ):
        raise ValueError(
            "Levels and iterations must be positive, and window size must be odd and at least 3"
        )
    if args.webcam is not None and args.webcam < 0:
        raise ValueError("Webcam index must be non-negative")

    source = args.video if args.video is not None else args.webcam
    output = args.output_video or args.output_dir / "flow.mp4"
    if output.suffix.lower() != ".mp4":
        raise ValueError("The output video must use the .mp4 extension")
    if isinstance(source, Path):
        if not source.is_file():
            raise FileNotFoundError(f"Input video does not exist: {source}")
        if source.resolve() == output.resolve():
            raise ValueError("The input and output video paths must differ")
        source = str(source)
    preview = not args.no_show and (args.preview or args.webcam is not None)
    if (
        preview
        and sys.platform.startswith("linux")
        and not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
    ):
        raise RuntimeError(
            "Video preview requires a graphical session. Use --no-show for headless export."
        )

    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open video source {source}")
        ok, first = capture.read()
        if not ok:
            raise OSError(f"No decodable frames from video source {source}")
        width, height = _resize_target(first, args.width, args.height)
        playback_fps = args.output_fps or capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(playback_fps) or playback_fps <= 0:
            playback_fps = 30.0

        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".flow-", dir=output.parent) as temporary:
            staged = Path(temporary) / "video.mp4"
            writer = cv2.VideoWriter(
                str(staged),
                cv2.VideoWriter_fourcc(*"mp4v"),
                playback_fps,
                (2 * width, height + HEADER_HEIGHT + height % 2),
            )
            try:
                if not writer.isOpened():
                    raise OSError(f"Could not create an MP4 for {output}")
                count, fps = _write_frames(
                    capture, writer, first, (width, height), args, preview
                )
            finally:
                writer.release()

            exported = cv2.VideoCapture(str(staged))
            try:
                ok, _ = exported.read()
            finally:
                exported.release()
            if not ok:
                raise OSError(f"The encoded MP4 could not be read for {output}")
            staged.replace(output)
    finally:
        capture.release()
        if preview:
            with suppress(cv2.error):
                cv2.destroyWindow(WINDOW_TITLE)

    print(f"Saved {count} frames at {playback_fps:g} playback FPS to {output}.")
    if fps is not None:
        print(f"Average processing FPS {fps:.2f}")
