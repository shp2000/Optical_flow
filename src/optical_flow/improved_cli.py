"""Command-line interface for the optimized optical-flow pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .depth import depth
from .improved_compute_flow import flow_lk_pyramid, flow_lk_sparse_pyramid
from .improved_epipole import epipole_guided
from .improved_vis_flow import plot_flow_subsampled, plot_sparse_flow
from .planar_flow import compute_planar_params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run vectorized, pyramidal Lucas-Kanade optical flow."
    )
    parser.add_argument("--all", action="store_true", help="generate every output")
    parser.add_argument("--epipole", action="store_true")
    parser.add_argument("--plot-flow", action="store_true")
    parser.add_argument("--planar-flow", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--threshmin", type=float, default=10.0)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--plot-step", type=int, default=8)
    parser.add_argument("--ransac-iterations", type=int, default=1000)
    parser.add_argument("--max-ransac-points", type=int, default=20_000)
    parser.add_argument("--data-dir", type=Path, default=Path("data/frames"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/improved")
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def _load_frames(data_dir: Path) -> list[np.ndarray]:
    paths = [data_dir / f"insight{i}.png" for i in range(20, 27)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input frames: " + ", ".join(missing))
    return [
        cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float64)
        for path in paths
    ]


def _finish_figure(path: Path, no_show: bool) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if not no_show:
        plt.show()
    plt.close()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.all:
        args.plot_flow = args.epipole = args.depth = args.planar_flow = True
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    frames = _load_frames(args.data_dir)
    center_index = 3
    center = frames[center_index]
    following = frames[center_index + 1]

    flow_started = perf_counter()
    needs_dense_flow = args.epipole or args.depth or args.planar_flow
    if args.plot_flow and not needs_dense_flow:
        points, vectors, sparse_confidence = flow_lk_sparse_pyramid(
            center,
            following,
            step=args.plot_step,
            levels=args.levels,
            window_size=args.window_size,
        )
        flow_seconds = perf_counter() - flow_started
        plt.figure()
        plot_sparse_flow(
            center,
            points,
            vectors,
            sparse_confidence,
            threshold=args.threshmin,
        )
        _finish_figure(args.output_dir / "flow.png", args.no_show)
        total_seconds = perf_counter() - started
        timing_report = (
            f"flow_mode=sparse_grid\n"
            f"flow_points={points.shape[0]}\n"
            f"flow_seconds={flow_seconds:.6f}\n"
            f"total_seconds={total_seconds:.6f}\n"
            f"pyramid_levels={args.levels}\n"
            f"window_size={args.window_size}\n"
        )
        (args.output_dir / "timings.txt").write_text(timing_report, encoding="utf-8")
        print(timing_report, end="")
        return

    flow, confidence = flow_lk_pyramid(
        center,
        following,
        levels=args.levels,
        iterations=args.iterations,
        window_size=args.window_size,
    )
    flow_seconds = perf_counter() - flow_started

    calibration = np.array(
        [[1118.0, 0.0, 357.0], [0.0, 1121.0, 268.0], [0.0, 0.0, 1.0]]
    )

    if args.plot_flow:
        plt.figure()
        plot_flow_subsampled(
            center,
            flow,
            confidence,
            threshold=args.threshmin,
            step=args.plot_step,
        )
        _finish_figure(args.output_dir / "flow.png", args.no_show)

    estimated_epipole = None
    inliers = None
    if args.epipole or args.depth:
        estimated_epipole, inliers = epipole_guided(
            flow[..., 0],
            flow[..., 1],
            confidence,
            threshold=args.threshmin,
            num_iterations=args.ransac_iterations,
            max_points=args.max_ransac_points,
        )

        plt.figure()
        plot_flow_subsampled(
            center,
            flow,
            confidence,
            threshold=args.threshmin,
            step=args.plot_step,
        )
        rows, columns = np.divmod(inliers, center.shape[1])
        plt.scatter(columns, rows, c="blue", s=0.2, label="RANSAC inliers")
        plt.scatter(
            estimated_epipole[0] + (center.shape[1] - 1) / 2.0,
            estimated_epipole[1] + (center.shape[0] - 1) / 2.0,
            c="yellow",
            edgecolors="black",
            s=45,
            label="Epipole",
        )
        plt.legend(loc="best")
        _finish_figure(args.output_dir / "epipole.png", args.no_show)
        np.savetxt(
            args.output_dir / "epipole.txt",
            estimated_epipole,
            header="Centered homogeneous epipole",
        )

    if args.depth:
        depth_map = depth(
            flow,
            confidence,
            estimated_epipole,
            calibration,
            thres=args.threshmin,
        )
        plt.figure()
        sns.heatmap(depth_map, square=True, cmap="mako")
        _finish_figure(args.output_dir / "depth.png", args.no_show)
        np.save(args.output_dir / "depth.npy", depth_map)

    if args.planar_flow:
        params = compute_planar_params(
            flow[..., 0],
            flow[..., 1],
            calibration,
            up=[312, 0],
            down=[512, 200],
        )
        np.savetxt(
            args.output_dir / "planar_params.txt",
            params,
            header="Eight planar-flow parameters",
        )
        print("Eight planar-flow parameters:", params)

    total_seconds = perf_counter() - started
    timing_report = (
        f"flow_mode=dense_vectorized_pyramid\n"
        f"flow_seconds={flow_seconds:.6f}\n"
        f"total_seconds={total_seconds:.6f}\n"
        f"pyramid_levels={args.levels}\n"
        f"iterations_per_level={args.iterations}\n"
        f"window_size={args.window_size}\n"
    )
    (args.output_dir / "timings.txt").write_text(timing_report, encoding="utf-8")
    print(timing_report, end="")


if __name__ == "__main__":
    main()
