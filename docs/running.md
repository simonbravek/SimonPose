# Running SimonPose projects

All scripts assume you run them from the repository root.

If imports like `from config import *` fail, you are almost certainly not running from the root, or you are not using module execution.

Use:

```bash
python -m projects.<script_name>
```

## Where to tweak parameters

There are two ways to control experiments:

1) Command-line flags (recommended for reproducibility)

Each script defines a small CLI via `argparse`. You can always inspect it with:

```bash
python -m projects.euclidean_fitter --help
```

2) In-file constants (quick iteration)

Most scripts have a `# GLOBALS` section near the top with all-caps constants such as `ITERATIONS`.
These are intended as experiment knobs.

Some constants are overridden by CLI flags (for example, `ITERATIONS` is overridden by `-i/--iterations`).

## `projects.euclidean_fitter`

Status: usable end-to-end (the "working" fitter described in `documentation/simon-bravek-mp-2026.pdf`).

What it does:

- Runs DensePose v2 / CSE on a COCO val2014 image.
- Converts per-pixel embeddings to SMPL vertex IDs (nearest embedding).
- Optimizes SMPL parameters to minimize a 2D euclidean reprojection loss.
- Writes visualizations and loss diagnostics to `output/euclidean_fitter_*`.

Run:

```bash
python -m projects.euclidean_fitter -n 1 -i 300
```

Arguments:

- `-n/--number`: number of COCO images to process (default: 1)
- `-i/--iterations`: optimizer iterations per image (default: 300)

Experiment knobs in `projects/euclidean_fitter.py` (edit under `# GLOBALS`):

- `ITERATIONS` (CLI: `-i/--iterations`): number of Adam steps per fitted instance; higher can converge better but is slower.
- `LEARNING_RATE`: Adam learning rate; too high can destabilize fitting.
- `LOSS_AREA`: target area used to downscale the bounding box for loss computation; larger means more points (slower, potentially more stable).
- `FOV`: field-of-view (degrees) for the synthetic camera intrinsics; affects the projection geometry used by the loss.
- `TORSO_MASK`: if enabled, restricts the loss to a torso vertex subset (experimental; requires an external vertex-to-body-part mapping file).
- `OFFSET` / `NUMBER_OF_IMAGES` (CLI: `-n/--number`): which COCO image IDs are processed.

Outputs:

- `output/euclidean_fitter_*/<image_id>_<person_index>_overview.png`
- `output/euclidean_fitter_*/loss_histogram.png`
- `output/euclidean_fitter_*/loss_remaining_histogram.png`

Implementation notes (useful when reading results):

- The script currently processes only the first detected person per image (there is a `break` in the person loop).
- The "loss" is measured in image-space pixels after scaling to a fixed `LOSS_AREA`.

## `projects.precission_fitter`

Status: work in progress (analysis prototype).

What it does:

- Runs DensePose v2 / CSE on COCO minival annotations.
- Computes per-vertex dispersion (stddev) of all pixels that map to the same SMPL vertex.
- Visualizes correspondence stability on the SMPL mesh and in the image.

It does not run a full SMPL fitting/optimization loop.

Run:

```bash
python -m projects.precission_fitter -n 10
```

Arguments:

- `-n/--number`: number of COCO images to process (default: 1)

Experiment knobs in `projects/precission_fitter.py`:

- Under `# GLOBALS`:
  - `NUMBER_OF_IMAGES` (CLI: `-n/--number`): how many images to process.
  - `OFFSET`: where to start in the COCO image ID list (useful for sampling different images without changing code).
- Inside the script body:
  - `cap`: limits the number of SMPL vertices considered (default: 6890).

Outputs:

- Images like `output/precission_fitter_*/test_<ann_id>.png`

## Reproducibility

This repository is not a benchmark harness.

If you want to reproduce figures/numbers from the report:

- Use the exact dataset/weights described in `docs/data.md`
- Record your environment (PyTorch, CUDA, Detectron2 commit)
- Keep `config.py` paths consistent across runs
