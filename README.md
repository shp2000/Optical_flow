# Optical Flow

An educational implementation of Lucas–Kanade optical flow with epipole
estimation, relative depth recovery, and planar-flow parameter estimation.

## Repository layout

```text
.
├── data/frames/          # Input image sequence
├── examples/results/     # Example visualizations at several thresholds
├── src/optical_flow/     # Importable Python package and CLI
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Setup

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the project:

```bash
python -m pip install -e .
```

## Usage

Run one or more analyses from the repository root:

```bash
python main.py --plot-flow
python main.py --epipole
python main.py --depth --threshmin 5
python main.py --planar-flow
```

The installed command and equivalent module command are:

```bash
optical-flow --plot-flow
python -m optical_flow --plot-flow
```

By default, frames are read from `data/frames` and generated images are written
to `outputs`. Override those locations with `--data-dir` and `--output-dir`.
Use `--no-show` in headless environments.

```bash
optical-flow --plot-flow --depth --threshmin 10 --no-show
```

Run `optical-flow --help` for the complete command reference.

## Optimized implementation

The original implementation is preserved. A separate optimized pipeline adds
vectorized box-filter window sums, direct batched 2×2 solves, coarse-to-fine
pyramidal refinement, subsampled plotting, and quality-guided RANSAC:

```bash
python improved_main.py --all --no-show
```

Its results are written to `outputs/improved` by default. Useful tuning options
include `--levels`, `--iterations`, `--window-size`, `--plot-step`, and
`--max-ransac-points`. After installation, the same pipeline is available as
`optical-flow-improved`.

## Quantitative benchmark

Download `other-color-twoframes.zip` (or `other-gray-twoframes.zip`) and
`other-gt-flow.zip` from the [Middlebury training downloads](https://vision.middlebury.edu/flow/data/).
Extract both into one directory containing `other-data/` and `other-gt-flow/`.
The runner finds the eight labeled `frame10.png` → `frame11.png` pairs using the
`flow10.flo` files.

```bash
python -m optical_flow.benchmark --data-dir downloads/middlebury
```

The installed command is `optical-flow-benchmark`. The default run compares
single-scale Lucas–Kanade (`lk-single`, one refinement), pyramidal
Lucas–Kanade (`lk-pyramid`), and OpenCV Farnebäck (`farneback`). Use `--methods`
to select methods, `--levels`, `--iterations`, and `--window-size` to tune
Lucas–Kanade, or `--limit` for a quick run on the first few pairs.

The output directory defaults to `outputs/benchmark` and contains:

- `summary.json` with aggregate results and run settings.
- `per_pair.csv` with accuracy, runtime, and FPS for each pair and method.
- `accuracy_vs_speed.png` with EPE, angular error, and bad-pixel rate plotted
  against FPS for each method.

Accuracy is averaged over valid ground-truth pixels across all selected pairs.
Endpoint error (EPE) is the Euclidean distance between the predicted and true
flow vectors in pixels. Angular error is the angle in degrees between the
vectors `(u, v, 1)`, as defined in the [Middlebury evaluation paper](https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf).
A pixel is bad when its EPE exceeds `--bad-threshold` (3 pixels by default).

Runtime is the sum of the mean estimate time for each pair (`--repeats`, default
3), after one untimed warm-up per method. FPS is the number of flow fields (one
per pair) divided by that runtime. Timing excludes file decoding, scoring, and
report generation. Peak RSS in MiB is the highest resident memory of each method's
dedicated process. It includes the runtime libraries, decoded frames, and
estimator allocations. OpenCV uses one thread per estimator process.

For [MPI Sintel](https://sintel.is.tue.mpg.de/downloads), extract the training
images and ground-truth archives to a common root. The runner reads the selected
rendering pass, `flow/`, and `invalid/` under `training/`. Nonzero invalid-mask
pixels are excluded from accuracy metrics.

```bash
python -m optical_flow.benchmark --dataset sintel --sintel-pass final --data-dir downloads/sintel --output-dir outputs/sintel-final
```

## Example results

Previously generated flow, epipole, and depth visualizations are kept in
[`examples/results`](examples/results).

## Notes

This repository contains course-style implementations optimized for clarity,
not real-time performance. The input sequence is retained so the results can be
reproduced directly after installation.
