[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# SimonPose
SimonPose is a research prototype exploring how to improve monocular human pose estimation by fitting a parametric body model (SMPL) to DensePose v2 / CSE outputs.

DensePose provides dense 2D-to-body correspondences (per-pixel embeddings). SimonPose uses those correspondences to optimize SMPL parameters and produce a globally consistent 3D human mesh.

This repository accompanies the write-up for maturita exam in [`documentation/simon-bravek-mp-2026.pdf`](documentation/simon-bravek-mp-2026.pdf).

The rest of 

## Requirements (minimal)

Tested stack (recommended):

- Python 3.11
- Linux + NVIDIA GPU
- PyTorch 2.4.x + CUDA (tested on CUDA 12.4 with `cu121` PyTorch wheels)

It may run on CPU, but DensePose + fitting will be very slow.

## Project status

- Research code (not a polished library); scripts are the primary interface.
- `projects/euclidean_fitter.py`: usable end-to-end experiment script (SMPL fitting driven by a 2D euclidean loss).
- `projects/precission_fitter.py`: work-in-progress analysis prototype; computes/visualizes DensePose correspondence stability, but does not run a full SMPL fitting loop.

## Projects

- `projects.euclidean_fitter` (ready): fits SMPL to DensePose CSE correspondences and saves per-iteration visualizations + loss plots.
- `projects.precission_fitter` (WIP): analyzes DensePose-to-SMPL correspondence stability (per-vertex dispersion) and visualizes it; no fitting/optimization yet.

The report also discusses an embedding-based fitter prototype; it was evaluated as a dead end and is not maintained as a runnable project in this repository.

## Inputs / outputs

- Input: RGB images (COCO 2014 val by default) + DensePose CSE weights + SMPL model file.
- Output: visualizations and diagnostics in `output/` (loss plots, overlays of the fitted mesh on the image).

## Quickstart

You run everything from the repo root.

Full documentation index: [`docs/README.md`](docs/README.md)

1) Setup environment and dependencies:

- Local machine: [`docs/setup-local.md`](docs/setup-local.md)
- CTU cluster: [`docs/setup-ctu.md`](docs/setup-ctu.md)

2) Download datasets and model files:

- [`docs/data.md`](docs/data.md)

3) Run the working project:

```bash
python -m projects.euclidean_fitter -n 1 -i 200
```

Expected output:

- A new run directory `output/euclidean_fitter_0/` (or the next available suffix).
- Per-image overview PNGs like `output/euclidean_fitter_0/<image_id>_<person_index>_overview.png`.
- Aggregate plots: `output/euclidean_fitter_0/loss_histogram.png` and `output/euclidean_fitter_0/loss_remaining_histogram.png`.

## Running experiments

- How to run each script, what it produces, and what is WIP: [`docs/running.md`](docs/running.md)

## Repository layout

- `projects/`: runnable experiment scripts.
- `common/`: shared utilities (projection, loss functions, DensePose overrides, optional depth experiments).
- `config.py`: central paths; assumes this directory structure:
  - `data/` (ignored by git): datasets and annotations
  - `models/` (ignored by git): DensePose + SMPL model files
  - `external/` (ignored by git): third-party code such as Detectron2
  - `output/` (ignored by git): generated results

If you want a different layout, edit `config.py`.

Configuration details: [`docs/configuration.md`](docs/configuration.md)

## Troubleshooting

- Common install/runtime issues (Detectron2 builds, CUDA mismatch, missing models/data): [`docs/troubleshooting.md`](docs/troubleshooting.md)

## License

The code in this repository is licensed under Apache 2.0 (see [`LICENSE`](LICENSE)).

Note: datasets and model files (COCO, DensePose weights, SMPL/SMPL-X) are not redistributed and have their own licenses/terms.

## Credits

This project relies on:

- Detectron2 + DensePose (Apache 2.0)
- COCO API (Apache 2.0)
- SMPL/SMPL-X model files (separate license/terms from MPI/MPG; see [`docs/data.md`](docs/data.md))

## Citation

If you use this code in academic work, please cite the accompanying report:

```bibtex
@misc{bravek2026simonpose,
  title        = {SimonPose},
  author       = {Šimon Brávek},
  year         = {2026},
  institution  = {FEL CTU (fel.cvut.cz) and Johannes Kepler Grammar School (gjk.cz)},
  howpublished = {[simon-bravek-mp-2026.pdf](https://github.com/simonbravek/SimonPose/blob/main/documentation/simon-bravek-mp-2026.pdf)}
}
```