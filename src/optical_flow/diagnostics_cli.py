"""Generate visual diagnostics for consecutive image frames."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from textwrap import fill

import cv2
import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize, PowerNorm, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PIL import Image, ImageDraw

from .benchmark_data import read_rgb
from .flow_diagnostics import FlowDiagnostics, diagnose_flow, flow_to_rgb
from .improved_compute_flow import flow_lk_pyramid

UNAVAILABLE = "#3b4454"
STATUS_COLORS = ("#43b49b", "#f09b48", UNAVAILABLE)
STATUS_NAMES = ("Consistent", "Inconsistent", "Outside frame")


def _write_pair(
    output: Path,
    title: str,
    reference: np.ndarray,
    next_frame: np.ndarray,
    forward: np.ndarray,
    confidence: np.ndarray,
    diagnostics: FlowDiagnostics,
    flow_max: float,
    residual_max: float,
) -> np.ndarray:
    output.mkdir(parents=True, exist_ok=True)
    warped = np.rint(diagnostics.warped_next).astype(np.uint8)
    status = np.full(diagnostics.in_bounds.shape, 2, dtype=np.uint8)
    status[diagnostics.in_bounds] = 1
    status[diagnostics.consistent] = 0
    palette = np.rint(
        255 * np.array([to_rgb(color) for color in STATUS_COLORS])
    ).astype(np.uint8)
    warped[~diagnostics.in_bounds] = palette[2]
    mask = palette[status]
    colored_flow = flow_to_rgb(forward, flow_max)

    residual_colors = colormaps["inferno"].with_extremes(bad=UNAVAILABLE)
    residual_scale = Normalize(0, residual_max, clip=True)
    confidence_colors = colormaps["viridis"]
    confidence_cap = (
        float(np.percentile(confidence, 99)) or float(confidence.max()) or 1.0
    )
    confidence_scale = PowerNorm(0.5, vmin=0, vmax=confidence_cap, clip=True)
    unwarped_heatmap = residual_colors(
        residual_scale(diagnostics.unwarped_residual), bytes=True
    )[..., :3]
    residual_heatmap = residual_colors(
        residual_scale(diagnostics.warp_residual), bytes=True
    )[..., :3]
    confidence_heatmap = confidence_colors(confidence_scale(confidence), bytes=True)[
        ..., :3
    ]
    for filename, image in (
        ("flow.png", colored_flow),
        ("warped_next.png", warped),
        ("residual.png", residual_heatmap),
        ("confidence.png", confidence_heatmap),
        ("consistency.png", mask),
    ):
        Image.fromarray(image).save(output / filename)

    within = diagnostics.in_bounds
    before = (
        f"MAE {diagnostics.unwarped_residual[within].mean():.2f}"
        if within.any()
        else "No valid samples"
    )
    after = (
        f"MAE {diagnostics.warp_residual[within].mean():.2f}"
        if within.any()
        else "No valid samples"
    )
    aspect = np.clip(reference.shape[0] / reference.shape[1], 0.5, 1.4)
    figure = Figure(
        figsize=(13.5, 2.4 + 5.7 * aspect), layout="constrained", facecolor="#f3f5f8"
    )
    FigureCanvasAgg(figure)
    figure.get_layout_engine().set(w_pad=0.12, h_pad=0.12)
    figure.suptitle(
        fill(title, width=78), fontsize=18, fontweight="bold", color="#172133"
    )
    axes = figure.subplots(2, 4)
    panels = (
        ("Reference frame", reference),
        ("Next frame", next_frame),
        ("Warped next frame", warped),
        (f"HSV flow · brightness cap {flow_max:g} px", colored_flow),
        (f"Unwarped residual · {before}", unwarped_heatmap),
        (f"Warp residual · {after}", residual_heatmap),
        ("Lucas–Kanade confidence", confidence_heatmap),
        (f"Consistency · {diagnostics.consistent.mean():.1%} agree", mask),
    )
    for axis, (label, image) in zip(axes.flat, panels):
        axis.imshow(image, interpolation="nearest")
        axis.set_anchor("N")
        axis.set_title(label, fontsize=11, color="#172133", loc="left")
        axis.set_axis_off()

    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    unit_flow = np.stack((np.cos(angles), np.sin(angles)), axis=-1)[None]
    direction_colors = ListedColormap(flow_to_rgb(unit_flow, max_motion=1)[0] / 255)
    direction_key = figure.colorbar(
        ScalarMappable(norm=Normalize(0, 360), cmap=direction_colors),
        ax=axes[0, 3],
        location="bottom",
        shrink=0.85,
        pad=0.04,
        ticks=(0, 90, 180, 270, 360),
    )
    direction_key.ax.set_xticklabels(("→", "↓", "←", "↑", "→"))
    figure.colorbar(
        ScalarMappable(norm=residual_scale, cmap=residual_colors),
        ax=axes[1, :2],
        location="bottom",
        shrink=0.78,
        pad=0.06,
        extend="max",
        label="Mean absolute RGB difference (0–255)",
    )
    figure.colorbar(
        ScalarMappable(norm=confidence_scale, cmap=confidence_colors),
        ax=axes[1, 2],
        location="bottom",
        shrink=0.85,
        pad=0.06,
        extend="max",
        label="Minimum structure tensor eigenvalue",
        ticks=(0, confidence_cap / 4, confidence_cap),
        format="%.3g",
    )
    axes[1, 3].legend(
        handles=[
            Patch(color=color, label=label)
            for color, label in zip(STATUS_COLORS, STATUS_NAMES)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        frameon=False,
        fontsize=8,
        ncols=3,
        handlelength=0.8,
        columnspacing=0.8,
    )
    figure.savefig(output / "overview.png", dpi=135)
    figure.clear()
    return warped


def _gif_frame(image: np.ndarray, label: str) -> Image.Image:
    preview = Image.fromarray(image)
    preview.thumbnail((720, 540), Image.Resampling.LANCZOS)
    frame = Image.new("RGB", (preview.width, preview.height + 28), "#172133")
    frame.paste(preview, (0, 28))
    ImageDraw.Draw(frame).text((7, 7), label, fill="white")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create dense optical-flow diagnostics for ordered image frames."
    )
    parser.add_argument(
        "frames", nargs="+", type=Path, help="two or more frames in chronological order"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics"))
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument(
        "--flow-max",
        type=float,
        default=10,
        help="motion in pixels at full HSV brightness",
    )
    parser.add_argument(
        "--residual-max",
        type=float,
        default=32,
        help="RGB error at the top of the heatmap",
    )
    parser.add_argument(
        "--fb-threshold",
        type=float,
        default=1.0,
        help="maximum consistent round-trip error in pixels",
    )
    parser.add_argument(
        "--no-gif", action="store_true", help="save only the diagnostic PNGs"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.frames) < 2:
        parser.error("provide at least two frames in chronological order")
    if any(
        not math.isfinite(limit) or limit <= 0
        for limit in (args.flow_max, args.residual_max)
    ):
        parser.error("--flow-max and --residual-max must be finite and positive")
    if not math.isfinite(args.fb_threshold) or args.fb_threshold < 0:
        parser.error("--fb-threshold must be finite and non-negative")

    settings = {
        "levels": args.levels,
        "iterations": args.iterations,
        "window_size": args.window_size,
    }
    animation = []
    try:
        previous = read_rgb(args.frames[0])
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
        for index, (first_path, second_path) in enumerate(
            zip(args.frames, args.frames[1:]), start=1
        ):
            following = read_rgb(second_path)
            if previous.shape != following.shape:
                raise ValueError(
                    f"Frame dimensions differ for {first_path} and {second_path}"
                )
            following_gray = cv2.cvtColor(following, cv2.COLOR_RGB2GRAY)
            forward, confidence = flow_lk_pyramid(
                previous_gray, following_gray, **settings
            )
            backward, _ = flow_lk_pyramid(following_gray, previous_gray, **settings)
            diagnostics = diagnose_flow(
                previous, following, forward, backward, args.fb_threshold
            )
            folder = args.output_dir / f"pair_{index:03d}"
            warped = _write_pair(
                folder,
                f"{first_path.name} → {second_path.name}",
                previous,
                following,
                forward,
                confidence,
                diagnostics,
                args.flow_max,
                args.residual_max,
            )
            if not args.no_gif:
                animation.append(_gif_frame(previous, f"Pair {index} / Reference"))
                animation.append(
                    _gif_frame(warped, f"Pair {index} / Warped next frame")
                )
            previous, previous_gray = following, following_gray

        if animation:
            animation[0].save(
                args.output_dir / "reconstruction.gif",
                save_all=True,
                append_images=animation[1:],
                duration=650,
                loop=0,
            )
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"Diagnostics failed: {error}\n")
    print(f"Diagnostics saved to {args.output_dir}")


if __name__ == "__main__":
    main()
