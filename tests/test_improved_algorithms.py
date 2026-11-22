import cv2
import numpy as np

from optical_flow.improved_compute_flow import flow_lk_pyramid, flow_lk_sparse_pyramid
from optical_flow.improved_epipole import epipole_guided


def test_pyramid_flow_recovers_translation():
    rng = np.random.default_rng(4)
    first = cv2.GaussianBlur(rng.normal(127, 35, (96, 96)), (0, 0), 1.2)
    transform = np.float32([[1, 0, 4], [0, 1, -3]])
    second = cv2.warpAffine(
        first,
        transform,
        (96, 96),
        borderMode=cv2.BORDER_REFLECT101,
    )

    flow, confidence = flow_lk_pyramid(
        first,
        second,
        levels=4,
        iterations=4,
        window_size=9,
    )
    interior = np.s_[16:-16, 16:-16]
    reliable = confidence[interior] > 10
    recovered = np.median(flow[interior][reliable], axis=0)

    np.testing.assert_allclose(recovered, [4, -3], atol=0.25)


def test_sparse_pyramid_flow_only_tracks_grid_points():
    rng = np.random.default_rng(8)
    first = cv2.GaussianBlur(rng.normal(127, 35, (96, 96)), (0, 0), 1.2)
    transform = np.float32([[1, 0, 4], [0, 1, -3]])
    second = cv2.warpAffine(
        first,
        transform,
        (96, 96),
        borderMode=cv2.BORDER_REFLECT101,
    )

    points, vectors, confidence = flow_lk_sparse_pyramid(
        first,
        second,
        step=12,
        levels=4,
        window_size=15,
    )

    assert 0 < points.shape[0] <= 64
    assert vectors.shape == points.shape
    assert confidence.shape == (points.shape[0],)
    np.testing.assert_allclose(np.median(vectors, axis=0), [4, -3], atol=0.25)


def test_quality_guided_ransac_recovers_epipole():
    height, width = 80, 100
    rows, columns = np.indices((height, width))
    x = columns - (width - 1) / 2.0
    y = rows - (height - 1) / 2.0
    expected = np.array([20.0, -10.0, 1.0])

    u = 0.03 * (x - expected[0])
    v = 0.03 * (y - expected[1])
    confidence = np.full((height, width), 100.0)

    estimated, inliers = epipole_guided(
        u,
        v,
        confidence,
        threshold=10,
        num_iterations=50,
        max_points=1000,
        inlier_threshold=0.1,
    )

    np.testing.assert_allclose(estimated, expected, atol=1e-8)
    assert inliers.size == 1000
