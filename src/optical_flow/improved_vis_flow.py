"""Efficient, subsampled optical-flow visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_flow_subsampled(
    image: np.ndarray,
    flow: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 10,
    step: int = 8,
    vector_scale: float = 10.0,
) -> None:
    """Plot reliable vectors on a regular grid without per-pixel Python loops."""
    if step < 1:
        raise ValueError("step must be positive")
    if flow.shape[:2] != image.shape or confidence.shape != image.shape:
        raise ValueError("image, flow, and confidence dimensions must agree")

    rows = np.arange(0, image.shape[0], step)
    columns = np.arange(0, image.shape[1], step)
    x, y = np.meshgrid(columns, rows)
    sampled_flow = flow[np.ix_(rows, columns)]
    sampled_confidence = confidence[np.ix_(rows, columns)]
    keep = sampled_confidence > threshold

    plt.imshow(image, cmap="gray")
    plt.quiver(
        x[keep],
        y[keep],
        sampled_flow[..., 0][keep] * vector_scale,
        sampled_flow[..., 1][keep] * vector_scale,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="red",
        width=0.002,
    )


def plot_sparse_flow(
    image: np.ndarray,
    points: np.ndarray,
    vectors: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 10,
    vector_scale: float = 10.0,
) -> None:
    """Plot flow that was computed only at the supplied grid points."""
    if points.shape != vectors.shape or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points and vectors must have shape (n, 2)")
    if confidence.shape != (points.shape[0],):
        raise ValueError("confidence must have shape (n,)")

    keep = confidence > threshold
    plt.imshow(image, cmap="gray")
    plt.quiver(
        points[keep, 0],
        points[keep, 1],
        vectors[keep, 0] * vector_scale,
        vectors[keep, 1] * vector_scale,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="red",
        width=0.002,
    )
