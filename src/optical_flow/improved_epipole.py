"""Quality-guided, vectorized RANSAC epipole estimation."""

from __future__ import annotations

import numpy as np


def epipole_guided(
    u: np.ndarray,
    v: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    num_iterations: int = 1000,
    max_points: int = 20_000,
    inlier_threshold: float = 1.5,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a finite epipole from a high-quality subset of flow vectors.

    The best-confidence candidates are retained before RANSAC. Line equations
    are normalized, so ``inlier_threshold`` is an approximate pixel distance
    rather than an unscaled algebraic error.
    """
    if u.shape != v.shape or u.shape != confidence.shape or u.ndim != 2:
        raise ValueError("u, v, and confidence must be equally sized 2-D arrays")
    if max_points < 2:
        raise ValueError("max_points must be at least 2")

    height, width = u.shape
    flat_u = np.asarray(u, dtype=np.float64).ravel()
    flat_v = np.asarray(v, dtype=np.float64).ravel()
    quality = np.asarray(confidence, dtype=np.float64).ravel()
    magnitude = np.hypot(flat_u, flat_v)

    eligible = np.flatnonzero(
        np.isfinite(flat_u)
        & np.isfinite(flat_v)
        & np.isfinite(quality)
        & (quality > threshold)
        & (magnitude > 1e-8)
    )
    if eligible.size < 2:
        raise ValueError("fewer than two flow vectors pass the quality threshold")

    if eligible.size > max_points:
        top = np.argpartition(quality[eligible], -max_points)[-max_points:]
        eligible = eligible[top]

    rows, columns = np.divmod(eligible, width)
    x = columns.astype(np.float64) - (width - 1) / 2.0
    y = rows.astype(np.float64) - (height - 1) / 2.0

    # x_homogeneous cross flow_direction gives the flow line.
    lines = np.column_stack(
        (-flat_v[eligible], flat_u[eligible], x * flat_v[eligible] - y * flat_u[eligible])
    )
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)

    rng = np.random.default_rng(random_seed)
    best_inliers = np.empty(0, dtype=np.int64)
    best_error = np.inf

    for _ in range(num_iterations):
        sample = rng.choice(lines.shape[0], size=2, replace=False)
        candidate = np.cross(lines[sample[0]], lines[sample[1]])
        if abs(candidate[2]) < 1e-12:
            continue
        candidate /= candidate[2]

        distances = np.abs(lines @ candidate)
        inliers = np.flatnonzero(distances < inlier_threshold)
        error = float(distances[inliers].mean()) if inliers.size else np.inf
        if inliers.size > best_inliers.size or (
            inliers.size == best_inliers.size and error < best_error
        ):
            best_inliers = inliers
            best_error = error

    if best_inliers.size < 2:
        raise RuntimeError("RANSAC could not find a valid epipole")

    _, _, vt = np.linalg.svd(lines[best_inliers], full_matrices=False)
    best_epipole = vt[-1]
    if abs(best_epipole[2]) < 1e-12:
        raise RuntimeError("the fitted epipole is at infinity")
    best_epipole /= best_epipole[2]
    return best_epipole, eligible[best_inliers]
