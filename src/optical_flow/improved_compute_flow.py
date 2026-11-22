"""Vectorized and coarse-to-fine Lucas–Kanade optical flow."""

from __future__ import annotations

import cv2
import numpy as np


def flow_lk_vectorized(
    Ix: np.ndarray,
    Iy: np.ndarray,
    It: np.ndarray,
    window_size: int = 5,
    min_eigenvalue: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve every local 2x2 Lucas–Kanade system in one vectorized pass.

    Box filters compute the six windowed sums required by the normal equations.
    The returned confidence is the smaller eigenvalue of the local structure
    tensor, a conventional measure of whether motion is constrained in both
    image directions.
    """
    if Ix.shape != Iy.shape or Ix.shape != It.shape or Ix.ndim != 2:
        raise ValueError("Ix, Iy, and It must be equally sized 2-D arrays")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer of at least 3")

    def window_sum(values: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(
            values,
            ddepth=-1,
            ksize=(window_size, window_size),
            normalize=False,
            borderType=cv2.BORDER_REFLECT101,
        )

    Ix = np.asarray(Ix, dtype=np.float64)
    Iy = np.asarray(Iy, dtype=np.float64)
    It = np.asarray(It, dtype=np.float64)

    sxx = window_sum(Ix * Ix)
    syy = window_sum(Iy * Iy)
    sxy = window_sum(Ix * Iy)
    sxt = window_sum(Ix * It)
    syt = window_sum(Iy * It)

    trace = sxx + syy
    discriminant = np.sqrt(np.maximum((sxx - syy) ** 2 + 4.0 * sxy**2, 0.0))
    confidence = 0.5 * (trace - discriminant)
    determinant = sxx * syy - sxy**2

    scale = np.maximum(trace**2, 1.0)
    valid = (confidence > min_eigenvalue) & (np.abs(determinant) > 1e-12 * scale)

    u = np.zeros_like(Ix)
    v = np.zeros_like(Iy)
    u[valid] = (sxy[valid] * syt[valid] - syy[valid] * sxt[valid]) / determinant[valid]
    v[valid] = (sxy[valid] * sxt[valid] - sxx[valid] * syt[valid]) / determinant[valid]

    flow = np.stack((u, v), axis=-1)
    confidence[~valid] = 0.0
    return flow, confidence


def _build_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [np.asarray(image, dtype=np.float64)]
    for _ in range(1, levels):
        height, width = pyramid[-1].shape
        if min(height, width) < 16:
            break
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def flow_lk_pyramid(
    first: np.ndarray,
    second: np.ndarray,
    levels: int = 4,
    iterations: int = 3,
    window_size: int = 7,
    min_eigenvalue: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate dense flow with iterative warping from coarse to fine.

    ``flow[y, x]`` maps a pixel in ``first`` to its position in ``second``.
    Large displacement is recovered at coarse resolution and refined after the
    flow is upsampled at every finer pyramid level.
    """
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("first and second must be equally sized grayscale images")
    if levels < 1 or iterations < 1:
        raise ValueError("levels and iterations must be positive")

    first_pyramid = _build_pyramid(first, levels)
    second_pyramid = _build_pyramid(second, len(first_pyramid))
    flow: np.ndarray | None = None
    confidence: np.ndarray | None = None

    for level in range(len(first_pyramid) - 1, -1, -1):
        reference = first_pyramid[level]
        target = second_pyramid[level]
        height, width = reference.shape

        if flow is None:
            flow = np.zeros((height, width, 2), dtype=np.float64)
        else:
            old_height, old_width = flow.shape[:2]
            flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= width / old_width
            flow[..., 1] *= height / old_height

        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )

        for _ in range(iterations):
            warped = cv2.remap(
                target,
                (grid_x + flow[..., 0]).astype(np.float32),
                (grid_y + flow[..., 1]).astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            average = 0.5 * (reference + warped)
            Ix = cv2.Sobel(average, cv2.CV_64F, 1, 0, ksize=3, scale=0.125)
            Iy = cv2.Sobel(average, cv2.CV_64F, 0, 1, ksize=3, scale=0.125)
            It = warped - reference
            increment, confidence = flow_lk_vectorized(
                Ix,
                Iy,
                It,
                window_size=window_size,
                min_eigenvalue=min_eigenvalue,
            )
            flow += increment

    assert flow is not None and confidence is not None
    return flow, confidence


def flow_lk_sparse_pyramid(
    first: np.ndarray,
    second: np.ndarray,
    step: int = 8,
    levels: int = 4,
    window_size: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pyramidal Lucas–Kanade flow only at a regular point grid.

    This path avoids allocating and solving a dense flow field when the caller
    only needs a visualization. OpenCV tracks the selected points through its
    image pyramid; the returned confidence comes from a box-filtered structure
    tensor sampled at those same points.
    """
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("first and second must be equally sized grayscale images")
    if step < 1 or levels < 1:
        raise ValueError("step and levels must be positive")

    height, width = first.shape
    margin = step // 2
    rows = np.arange(margin, height, step, dtype=np.float32)
    columns = np.arange(margin, width, step, dtype=np.float32)
    x, y = np.meshgrid(columns, rows)
    points = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        np.clip(first, 0, 255).astype(np.uint8),
        np.clip(second, 0, 255).astype(np.uint8),
        points.reshape(-1, 1, 2),
        None,
        winSize=(window_size, window_size),
        maxLevel=levels - 1,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    next_points = next_points.reshape(-1, 2)
    status = status.ravel().astype(bool)

    Ix = cv2.Sobel(first, cv2.CV_64F, 1, 0, ksize=3, scale=0.125)
    Iy = cv2.Sobel(first, cv2.CV_64F, 0, 1, ksize=3, scale=0.125)
    _, dense_confidence = flow_lk_vectorized(
        Ix,
        Iy,
        np.zeros_like(first),
        window_size=window_size,
    )
    sample_rows = points[:, 1].astype(int)
    sample_columns = points[:, 0].astype(int)
    confidence = dense_confidence[sample_rows, sample_columns]

    valid = status & np.isfinite(next_points).all(axis=1)
    return points[valid], (next_points - points)[valid], confidence[valid]
