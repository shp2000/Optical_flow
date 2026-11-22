"""Compare dense optical-flow methods on labeled training data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
from dataclasses import dataclass
from functools import partial
from importlib.util import find_spec
from multiprocessing.connection import Connection
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.ndimage import convolve1d

from .benchmark_data import FlowPair, list_pairs, read_flo, read_grayscale, read_rgb
from .benchmark_raft import (
    RAFT_INSTALL,
    RAFT_METHODS,
    RAFT_SETTINGS,
    build_raft_estimator,
)
from .compute_flow import flow_lk
from .compute_grad import KERNEL_G, compute_Ix, compute_Iy
from .improved_compute_flow import flow_lk_pyramid

DEFAULT_METHODS = ("lk-original", "lk-pyramid", "farneback")
METHOD_LABELS = {
    "lk-original": "Original Lucas–Kanade",
    "lk-pyramid": "Optimized pyramidal Lucas–Kanade",
    "farneback": "OpenCV Farnebäck",
    "lk-single": "Vectorized single-scale Lucas–Kanade",
    "raft-small": "RAFT Small",
    "raft-large": "RAFT Large",
}
FARNEBACK = {
    "pyr_scale": 0.5,
    "levels": 4,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}
MIB = 1024**2


@dataclass(frozen=True)
class FlowErrors:
    valid_pixels: int
    endpoint_sum: float
    angular_sum: float
    bad_pixels: int

    def averages(self) -> dict:
        pixels = self.valid_pixels
        return {
            "valid_pixels": pixels,
            "mean_epe_px": self.endpoint_sum / pixels if pixels else None,
            "mean_angular_error_deg": self.angular_sum / pixels if pixels else None,
            "bad_pixel_pct": 100 * self.bad_pixels / pixels if pixels else None,
        }


def flow_errors(
    predicted: np.ndarray,
    truth: np.ndarray,
    bad_threshold: float,
    invalid: np.ndarray | None = None,
) -> FlowErrors:
    """Accumulate errors over valid ground-truth pixels for weighted aggregation."""
    if truth.ndim != 3 or truth.shape[2] != 2 or predicted.shape != truth.shape:
        raise ValueError(
            f"Expected matching H × W × 2 flow fields, got {predicted.shape} and {truth.shape}"
        )
    valid = np.isfinite(truth).all(axis=-1) & (np.abs(truth) <= 1e9).all(axis=-1)
    if invalid is not None:
        if invalid.shape != truth.shape[:2]:
            raise ValueError(
                f"Invalid mask shape {invalid.shape} for flow shape {truth.shape}"
            )
        valid &= invalid == 0

    pred_u, pred_v = predicted[valid].astype(np.float64, copy=False).T
    true_u, true_v = truth[valid].astype(np.float64, copy=False).T
    if not (np.isfinite(pred_u).all() and np.isfinite(pred_v).all()):
        raise ValueError("Non-finite prediction on valid ground truth")

    endpoint = np.hypot(pred_u - true_u, pred_v - true_v)
    # atan2 of the cross-product norm and dot product gives the angle between
    # (pred_u, pred_v, 1) and (true_u, true_v, 1), including zero motion.
    cross_z = pred_u * true_v - pred_v * true_u
    dot = 1 + pred_u * true_u + pred_v * true_v
    angular = np.degrees(np.arctan2(np.hypot(endpoint, cross_z), dot))
    return FlowErrors(
        endpoint.size,
        float(endpoint.sum()),
        float(angular.sum()),
        int(np.count_nonzero(endpoint > bad_threshold)),
    )


def _peak_rss_bytes() -> int:
    if sys.platform != "win32":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024

    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    current_process = ctypes.WinDLL("kernel32").GetCurrentProcess
    current_process.restype = wintypes.HANDLE
    memory_info = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    memory_info.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not memory_info(current_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize


def _estimate(
    method: str, first: np.ndarray, second: np.ndarray, settings: dict
) -> np.ndarray:
    if method == "farneback":
        return cv2.calcOpticalFlowFarneback(first, second, None, **settings)
    if method == "lk-original":
        first = first.astype(np.float64)
        # Labeled pairs provide two frames. Use the original spatial kernels at
        # their midpoint and a spatially smoothed one-frame temporal difference.
        midpoint = (0.5 * (first + second))[..., None]
        ix = compute_Ix(midpoint)[..., 0]
        iy = compute_Iy(midpoint)[..., 0]
        it = convolve1d(second - first, KERNEL_G, axis=0)
        it = convolve1d(it, KERNEL_G, axis=1)
        flow, _ = flow_lk(ix, iy, it, **settings)
    else:
        flow, _ = flow_lk_pyramid(first, second, **settings)
    return flow


def _estimate_pairs(
    connection: Connection,
    method: str,
    settings: dict,
    repeats: int,
) -> None:
    # The caller requests each pair after scoring the previous one. The worker's
    # peak memory excludes scoring, and scoring does not compete for CPU time.
    with connection:
        cv2.setNumThreads(1)
        uses_raft = method in RAFT_METHODS
        read_frame = read_rgb if uses_raft else read_grayscale
        needs_warmup = True
        while (pair := connection.recv()) is not None:
            try:
                first = read_frame(pair.first)
                second = read_frame(pair.second)
                if first.shape != second.shape:
                    raise ValueError(
                        f"Frame dimensions differ: {first.shape} and {second.shape}"
                    )
                if needs_warmup:
                    estimate = (
                        build_raft_estimator(method, **settings)
                        if uses_raft
                        else partial(_estimate, method, settings=settings)
                    )
                    estimate(first, second)
                    needs_warmup = False
                started = perf_counter()
                for _ in range(repeats - 1):
                    estimate(first, second)
                flow = estimate(first, second)
                seconds = (perf_counter() - started) / repeats
                peak_mib = _peak_rss_bytes() / MIB
            except (OSError, RuntimeError, ValueError, cv2.error) as error:
                connection.send((str(error), None))
                return
            connection.send((None, (seconds, peak_mib, flow.shape, flow.dtype.str)))
            connection.send_bytes(flow)
            del flow, first, second


def _benchmark_method(
    pairs: list[FlowPair],
    method: str,
    settings: dict,
    repeats: int,
    bad_threshold: float,
) -> tuple[dict, list[dict]]:
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    errors = []
    rows = []
    with parent, child:
        worker = context.Process(
            target=_estimate_pairs, args=(child, method, settings, repeats)
        )
        worker.start()
        child.close()
        try:
            for pair in pairs:
                try:
                    parent.send(pair)
                    message, measurement = parent.recv()
                    if message is not None:
                        raise RuntimeError(f"{method}, {pair.name}: {message}")
                    seconds, peak_mib, shape, dtype = measurement
                    predicted = np.frombuffer(parent.recv_bytes(), dtype=dtype).reshape(
                        shape
                    )
                    truth = read_flo(pair.ground_truth)
                    invalid = read_grayscale(pair.invalid) if pair.invalid else None
                    result = flow_errors(predicted, truth, bad_threshold, invalid)
                except EOFError as error:
                    raise RuntimeError(
                        f"{method}, {pair.name}: estimator process ended unexpectedly"
                    ) from error
                except (OSError, ValueError, cv2.error) as error:
                    raise ValueError(f"{method}, {pair.name}: {error}") from error
                errors.append(result)
                rows.append(
                    {
                        "method": method,
                        "pair": pair.name,
                        **result.averages(),
                        "runtime_seconds": seconds,
                        "fps": 1 / seconds,
                    }
                )
            parent.send(None)
            worker.join()
            if worker.exitcode != 0:
                raise RuntimeError(
                    f"{method}: estimator process exited with code {worker.exitcode}"
                )
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join()
            worker.close()

    total = FlowErrors(
        sum(result.valid_pixels for result in errors),
        sum(result.endpoint_sum for result in errors),
        sum(result.angular_sum for result in errors),
        sum(result.bad_pixels for result in errors),
    )
    runtime = sum(row["runtime_seconds"] for row in rows)
    return {
        "method": method,
        "dense": True,
        "parameters": settings,
        **total.averages(),
        "runtime_seconds": runtime,
        "fps": len(rows) / runtime,
        "peak_rss_mib": peak_mib,
    }, rows


def _write_comparison(path: Path, title: str, methods: list[dict]) -> None:
    lines = [
        f"# {title}",
        "",
        "| Method | EPE ↓ (px) | Runtime ↓ (s) | FPS ↑ | Dense |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for result in methods:
        epe = result["mean_epe_px"]
        error = f"{epe:.3f}" if epe is not None else "n/a"
        dense = "Yes" if result["dense"] else "No"
        lines.append(
            f"| {METHOD_LABELS[result['method']]} | {error} | "
            f"{result['runtime_seconds']:.3f} | {result['fps']:.3f} | {dense} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_chart(
    path: Path, title: str, methods: list[dict], bad_threshold: float
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4), layout="constrained")
    FigureCanvasAgg(figure)
    figure.suptitle(title)
    axes = figure.subplots(1, 3, sharex=True)
    metrics = (
        ("mean_epe_px", "Mean endpoint error (px)"),
        ("mean_angular_error_deg", "Mean angular error (°)"),
        ("bad_pixel_pct", f"Pixels with EPE > {bad_threshold:g} px (%)"),
    )
    for axis, (metric, label) in zip(axes, metrics):
        for result in methods:
            if result[metric] is not None:
                axis.scatter(
                    result["fps"], result[metric], s=55, label=result["method"]
                )
        axis.set(xlabel="Frames per second", ylabel=label)
        axis.grid(alpha=0.25)
    axes[0].set_xlim(left=0)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels)
    figure.savefig(path, dpi=150)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark dense optical flow on labeled training pairs."
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="root of the extracted dataset"
    )
    parser.add_argument(
        "--dataset", choices=("middlebury", "sintel"), default="middlebury"
    )
    parser.add_argument("--sintel-pass", choices=("clean", "final"), default="clean")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_LABELS),
        default=DEFAULT_METHODS,
        help="estimators to compare (default: lk-original lk-pyramid farneback)",
    )
    parser.add_argument(
        "--bad-threshold",
        type=float,
        default=3.0,
        help="bad-pixel EPE threshold in pixels",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="timed estimates per pair"
    )
    parser.add_argument(
        "--limit", type=int, help="benchmark only the first N labeled pairs"
    )
    parser.add_argument(
        "--levels", type=int, default=4, help="pyramid levels for lk-pyramid"
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="refinements per level for lk-pyramid"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="odd vectorized Lucas–Kanade window size",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.bad_threshold) or args.bad_threshold < 0:
        parser.error("--bad-threshold must be finite and non-negative")
    if min(args.repeats, args.levels, args.iterations) < 1 or (
        args.limit is not None and args.limit < 1
    ):
        parser.error("--repeats, --levels, --iterations, and --limit must be positive")
    if args.window_size < 3 or args.window_size % 2 == 0:
        parser.error("--window-size must be an odd integer of at least 3")
    if any(method in RAFT_METHODS for method in args.methods) and any(
        find_spec(dependency) is None for dependency in ("torch", "torchvision")
    ):
        parser.error(RAFT_INSTALL)

    settings = {
        "lk-original": {"size": 5},
        "lk-single": {"levels": 1, "iterations": 1, "window_size": args.window_size},
        "lk-pyramid": {
            "levels": args.levels,
            "iterations": args.iterations,
            "window_size": args.window_size,
        },
        "farneback": FARNEBACK,
        **dict.fromkeys(RAFT_METHODS, RAFT_SETTINGS),
    }
    try:
        pairs = list_pairs(args.data_dir, args.dataset, args.sintel_pass)[: args.limit]
        summary = []
        per_pair = []
        for method in dict.fromkeys(args.methods):
            print(f"Benchmarking {method} on {len(pairs)} pairs", flush=True)
            result, rows = _benchmark_method(
                pairs, method, settings[method], args.repeats, args.bad_threshold
            )
            summary.append(result)
            per_pair.extend(rows)

        output = args.output_dir
        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "dataset": args.dataset,
            "sintel_pass": args.sintel_pass if args.dataset == "sintel" else None,
            "pairs": len(pairs),
            "bad_threshold_px": args.bad_threshold,
            "timed_repeats": args.repeats,
            "device": "cpu",
            "opencv_threads": 1,
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "methods": summary,
        }
        (output / "summary.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        with (output / "per_pair.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_pair[0]))
            writer.writeheader()
            writer.writerows(per_pair)
        title = (
            "Middlebury training"
            if args.dataset == "middlebury"
            else f"MPI Sintel training ({args.sintel_pass})"
        )
        title = f"{title} · {len(pairs)} {'pair' if len(pairs) == 1 else 'pairs'}"
        _write_comparison(output / "comparison.md", title, summary)
        _write_chart(
            output / "accuracy_vs_speed.png",
            title,
            summary,
            args.bad_threshold,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")

    print(
        f"\n{'Method':<13} {'EPE (px)':>10} {'Angle (deg)':>12} {'Bad (%)':>10} {'Time (s)':>10} {'FPS':>9} {'Peak (MiB)':>11}"
    )
    for result in summary:
        values = [
            result[key]
            for key in (
                "mean_epe_px",
                "mean_angular_error_deg",
                "bad_pixel_pct",
                "runtime_seconds",
                "fps",
                "peak_rss_mib",
            )
        ]
        formatted = [f"{value:.3f}" if value is not None else "n/a" for value in values]
        epe, angle, bad, seconds, fps, memory = formatted
        print(
            f"{result['method']:<13} {epe:>10} {angle:>12} {bad:>10} {seconds:>10} {fps:>9} {memory:>11}"
        )
    print(f"\nReports saved to {output}")


if __name__ == "__main__":
    main()
