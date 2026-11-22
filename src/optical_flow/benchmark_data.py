"""Training pairs and ground-truth flow for the quantitative benchmark."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowPair:
    name: str
    first: Path
    second: Path
    ground_truth: Path
    invalid: Path | None = None


def list_pairs(root: Path, dataset: str, sintel_pass: str = "clean") -> list[FlowPair]:
    """Find pairs from the extracted training archives, in benchmark order."""
    pairs = []
    if dataset == "middlebury":
        for flow in sorted((root / "other-gt-flow").glob("*/flow10.flo")):
            scene = flow.parent.name
            frames = root / "other-data" / scene
            pairs.append(
                FlowPair(scene, frames / "frame10.png", frames / "frame11.png", flow)
            )
        if not pairs:
            raise ValueError(f"No Middlebury flow files in {root / 'other-gt-flow'}")
    elif dataset == "sintel":
        training = root / "training" if (root / "training").is_dir() else root
        for flow in sorted((training / "flow").glob("*/frame_*.flo")):
            match = re.fullmatch(r"frame_(\d{4})", flow.stem)
            if match is None:
                raise ValueError(f"Unexpected Sintel flow filename: {flow}")
            scene = flow.parent.name
            frames = training / sintel_pass / scene
            next_frame = f"frame_{int(match[1]) + 1:04d}.png"
            first_name = flow.with_suffix(".png").name
            pairs.append(
                FlowPair(
                    f"{scene}/{flow.stem}",
                    frames / first_name,
                    frames / next_frame,
                    flow,
                    training / "invalid" / scene / first_name,
                )
            )
        if not pairs:
            raise ValueError(f"No Sintel flow files in {training / 'flow'}")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return pairs


def read_flo(path: Path) -> np.ndarray:
    """Read a little-endian .flo file into an H × W × 2 array of (u, v)."""
    content = path.read_bytes()
    if len(content) < 12 or content[:4] != b"PIEH":
        raise ValueError(f"Invalid .flo header: {path}")
    width, height = struct.unpack_from("<ii", content, 4)
    if width <= 0 or height <= 0 or len(content) != 12 + width * height * 8:
        raise ValueError(f"Invalid .flo dimensions or payload size: {path}")
    return np.frombuffer(content, dtype="<f4", offset=12).reshape(height, width, 2)


def read_grayscale(path: Path) -> np.ndarray:
    """Decode a frame or mask, including paths with non-ASCII characters."""
    return _read_image(path, cv2.IMREAD_GRAYSCALE)


def read_rgb(path: Path) -> np.ndarray:
    """Decode a frame in RGB channel order."""
    return cv2.cvtColor(_read_image(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _read_image(path: Path, mode: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, mode) if data.size else None
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image
