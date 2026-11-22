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

## Video and webcam

Pass a video file to the optimized entry point. It processes consecutive frames
in order and writes a side-by-side MP4 showing the input, HSV flow, and processing FPS.
The first frame has a blank flow panel. Each subsequent panel shows motion from
the previous frame to the displayed frame.

```bash
python improved_main.py --video input.mp4 --output-video outputs/flow.mp4
```

Use `--preview` to open a window for a video file. `--webcam` opens camera `0`
with a preview, and `--webcam 1` selects camera `1`. Press `q` or `Esc` to
finish recording and save the processed frames. Use `--no-show` for headless
capture, `Ctrl+C` to finish from the terminal, or `--max-frames` to stop automatically.

```bash
python improved_main.py --webcam --width 480 --max-frames 300 --output-video outputs/webcam-flow.mp4
```

- `--width` or `--height` preserves the input aspect ratio when used alone.
  Specify both for an exact processing size.
- `--device auto` selects CUDA RAFT Small when CUDA-enabled PyTorch and
  Torchvision are available, with pyramidal Lucas–Kanade as the CPU fallback.
  Use `--device cpu` to select Lucas–Kanade or `--device cuda` to require CUDA.
  Install the optional `raft` extra for the RAFT backend. The first CUDA run
  downloads the pretrained model weights. [PyTorch installation](https://pytorch.org/get-started/locally/)
- The MP4 contains the annotated video frames. It uses the source playback rate
  when available and 30 FPS otherwise. Use `--output-fps` to override playback
  speed. The FPS overlay measures processing throughput.

## Visual diagnostics

Pass two or more images in chronological order to compare each consecutive pair
using dense pyramidal Lucas–Kanade. The command also works as
`optical-flow-diagnostics` after installation.

```bash
python -m optical_flow.diagnostics_cli data/frames/insight20.png data/frames/insight21.png data/frames/insight22.png
```

Each `outputs/diagnostics/pair_*/overview.png` puts the inputs, reconstructed
reference, flow, residuals, confidence, and consistency mask side by side.
The pair directory also contains individual PNGs for flow, reconstruction,
warp residual, confidence, and consistency. `reconstruction.gif` alternates
the reference and reconstruction for each pair. Use `--no-gif` for stills only.

- **HSV flow** uses hue for direction and brightness for magnitude. Set
  `--flow-max` to change the motion magnitude at full brightness (10 pixels by default).
- **Warp residual** shows the mean absolute RGB difference per pixel. Reconstruction
  samples the next frame at each reference pixel plus its estimated displacement.
  Both residual panels use the same color scale (`--residual-max`, default 32 intensity levels)
  and report MAE over the same in-bounds pixels. Slate gray marks unavailable samples.
- **Confidence** shows the minimum eigenvalue of the local structure tensor.
  Each pair uses its own 99th-percentile display maximum and a square-root scale.
- **Consistency** flags pixels where following the forward and backward flow
  misses the starting point by more than `--fb-threshold` (1 pixel by default).
  The mask distinguishes inconsistent motion from coordinates outside the next frame.

## Quantitative benchmark

Download `other-color-twoframes.zip` (or `other-gray-twoframes.zip`) and
`other-gt-flow.zip` from the [Middlebury training downloads](https://vision.middlebury.edu/flow/data/).
Extract both into one directory containing `other-data/` and `other-gt-flow/`.
The runner finds the eight labeled `frame10.png` → `frame11.png` pairs using the
`flow10.flo` files.

```bash
python -m optical_flow.benchmark --data-dir downloads/middlebury
```

The installed command is `optical-flow-benchmark`. The default comparison runs
original Lucas–Kanade (`lk-original`), optimized pyramidal Lucas–Kanade
(`lk-pyramid`), and OpenCV Farnebäck (`farneback`). [OpenCV optical flow](https://docs.opencv.org/4.10.0/d4/dee/tutorial_optical_flow.html)

Original Lucas–Kanade uses the per-pixel least-squares solver and a 5×5 window.
For two-frame input, it evaluates the original spatial derivative kernels at
the frame midpoint and uses a spatially smoothed frame difference for the
temporal derivative. Use `--methods` to select methods, including the optional
vectorized single-scale Lucas–Kanade (`lk-single`). The `--levels`,
`--iterations`, and `--window-size` flags tune the optimized solver.
Use `--limit 1 --repeats 1` for a quick comparison.

The output directory defaults to `outputs/benchmark` and contains:

- `summary.json` with aggregate results and run settings.
- `per_pair.csv` with accuracy, runtime, and FPS for each pair and method.
- `comparison.md` with a Markdown table of EPE, runtime, FPS, and dense output.
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
estimator allocations. All methods run on CPU. OpenCV and PyTorch each use one
thread per estimator process.

For [MPI Sintel](https://sintel.is.tue.mpg.de/downloads), extract the training
images and ground-truth archives to a common root. The runner reads the selected
rendering pass, `flow/`, and `invalid/` under `training/`. Nonzero invalid-mask
pixels are excluded from accuracy metrics.

```bash
python -m optical_flow.benchmark --dataset sintel --sintel-pass final --data-dir downloads/sintel --output-dir outputs/sintel-final
```

### Optional pretrained RAFT

Install the `raft` extra and add `raft-small` or `raft-large` to `--methods`.
Torchvision downloads the selected `C_T_V2` weights on first use. Both variants
were pretrained on FlyingChairs and FlyingThings3D. Model loading and weight
downloads are outside timed runs. Use the Middlebury color archive for RGB input.
[Torchvision RAFT](https://docs.pytorch.org/vision/stable/models/raft.html)

```bash
python -m pip install -e ".[raft]"
python -m optical_flow.benchmark --data-dir downloads/middlebury --methods lk-original lk-pyramid farneback raft-small --limit 1 --repeats 1
```

## Example results

Previously generated flow, epipole, and depth visualizations are kept in
[`examples/results`](examples/results).

## Notes

This repository contains course-style implementations optimized for clarity,
not real-time performance. The input sequence is retained so the results can be
reproduced directly after installation.
