import numpy as np


def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """Solve one Lucas–Kanade least-squares system over a clipped image patch."""
    height, width = Ix.shape
    radius = size // 2
    patch = np.s_[
        max(0, y - radius) : min(height, y + radius + 1),
        max(0, x - radius) : min(width, x + radius + 1),
    ]
    gradients = np.column_stack((Ix[patch].ravel(), Iy[patch].ravel())).astype(
        np.float64, copy=False
    )
    flow, _, _, singular_values = np.linalg.lstsq(
        gradients, -It[patch].ravel(), rcond=None
    )
    return list(flow), singular_values.min()


def flow_lk(Ix, Iy, It, size=5):
    """Run the original independent Lucas–Kanade solve at every pixel."""
    height, width = Ix.shape
    image_flow = np.zeros((height, width, 2))
    confidence = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            flow, conf = flow_lk_patch(Ix, Iy, It, x, y, size=size)
            image_flow[y, x] = flow
            confidence[y, x] = conf
    return image_flow, confidence
