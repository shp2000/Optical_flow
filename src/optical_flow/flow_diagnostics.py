"""Compute dense flow diagnostics on the reference image grid."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowDiagnostics:
    warped_next: np.ndarray
    unwarped_residual: np.ndarray
    warp_residual: np.ndarray
    in_bounds: np.ndarray
    consistent: np.ndarray


def diagnose_flow(
    reference: np.ndarray,
    next_frame: np.ndarray,
    forward: np.ndarray,
    backward: np.ndarray,
    fb_threshold: float = 1.0,
) -> FlowDiagnostics:
    """Reconstruct the reference and check the flow round trip in pixels."""
    if (
        reference.ndim != 3
        or reference.shape[2] != 3
        or next_frame.shape != reference.shape
    ):
        raise ValueError("Input frames must be equally sized RGB images")
    height, width = reference.shape[:2]
    if (
        min(height, width) < 1
        or forward.shape != (height, width, 2)
        or backward.shape != forward.shape
    ):
        raise ValueError("Both flow fields must match the input frame dimensions")
    if not np.isfinite(fb_threshold) or fb_threshold < 0:
        raise ValueError(
            "The forward–backward threshold must be finite and non-negative"
        )

    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    x += forward[..., 0].astype(np.float32)
    y += forward[..., 1].astype(np.float32)
    in_bounds = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x[~in_bounds] = y[~in_bounds] = -1

    reference = reference.astype(np.float32)
    next_frame = next_frame.astype(np.float32)
    warped = cv2.remap(next_frame, x, y, cv2.INTER_LINEAR)
    reverse = cv2.remap(backward.astype(np.float32), x, y, cv2.INTER_LINEAR)
    round_trip = forward + reverse
    consistent = in_bounds & (
        np.hypot(round_trip[..., 0], round_trip[..., 1]) <= fb_threshold
    )

    unwarped_residual = np.abs(reference - next_frame).mean(axis=-1)
    warp_residual = np.abs(reference - warped).mean(axis=-1)
    unwarped_residual[~in_bounds] = warp_residual[~in_bounds] = np.nan
    return FlowDiagnostics(
        warped, unwarped_residual, warp_residual, in_bounds, consistent
    )


def flow_to_rgb(flow: np.ndarray, max_motion: float) -> np.ndarray:
    """Encode direction as HSV hue and motion magnitude as brightness."""
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("Expected an H × W × 2 flow field")
    if not np.isfinite(max_motion) or max_motion <= 0:
        raise ValueError("The flow brightness limit must be finite and positive")

    u, v = np.moveaxis(
        np.where(np.isfinite(flow).all(axis=-1, keepdims=True), flow, 0), -1, 0
    )
    hue = np.degrees(np.arctan2(v, u)) % 360
    value = np.clip(np.hypot(u, v) / max_motion, 0, 1)
    hsv = np.stack((hue, np.ones_like(hue), value), axis=-1).astype(np.float32)
    return np.rint(255 * cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)).astype(np.uint8)
