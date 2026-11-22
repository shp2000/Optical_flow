"""Torchvision RAFT adapter with optional dependencies loaded on demand."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

RAFT_METHODS = ("raft-small", "raft-large")
RAFT_SETTINGS = {"weights_name": "C_T_V2", "num_flow_updates": 12, "cpu_threads": 1}
RAFT_INSTALL = 'Install RAFT support with python -m pip install -e ".[raft]".'


def build_raft_estimator(
    method: str,
    weights_name: str,
    num_flow_updates: int,
    cpu_threads: int,
    *,
    device: str = "cpu",
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    try:
        import torch
        from torch.nn.functional import pad
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            f"{RAFT_INSTALL} Could not import PyTorch or Torchvision: {error}"
        ) from error

    if device == "cpu":
        torch.set_num_threads(cpu_threads)
    if method == "raft-small":
        weights = Raft_Small_Weights[weights_name]
        model = raft_small(weights=weights, progress=False)
    else:
        weights = Raft_Large_Weights[weights_name]
        model = raft_large(weights=weights, progress=False)
    model = model.to(device).eval()
    preprocess = weights.transforms()
    min_height, min_width = weights.meta["min_size"]

    @torch.inference_mode()
    def estimate(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        first_tensor = torch.from_numpy(first).permute(2, 0, 1).unsqueeze(0).to(device)
        second_tensor = (
            torch.from_numpy(second).permute(2, 0, 1).unsqueeze(0).to(device)
        )
        first_tensor, second_tensor = preprocess(first_tensor, second_tensor)
        height, width = first.shape[:2]
        padded_height = max(min_height, (height + 7) // 8 * 8)
        padded_width = max(min_width, (width + 7) // 8 * 8)
        padding = (0, padded_width - width, 0, padded_height - height)
        # Padding keeps the original pixel scale. Crop back to the ground-truth grid.
        flow = model(
            pad(first_tensor, padding, mode="replicate"),
            pad(second_tensor, padding, mode="replicate"),
            num_flow_updates=num_flow_updates,
        )[-1]
        return flow[0, :, :height, :width].permute(1, 2, 0).contiguous().cpu().numpy()

    return estimate
