# Troubleshooting

This repo sits on top of a sensitive stack (CUDA + PyTorch + Detectron2). Most issues come from version/toolchain mismatches or missing files.

## "No module named densepose"

Cause: Detectron2 is installed but the DensePose project is not.

Fix:

```bash
pip install -e external/detectron2/projects/DensePose
```

## Detectron2 build fails

Common causes:

- You installed a PyTorch build for a different CUDA version than your system/runtime.
- Missing compiler toolchain (gcc/g++, nvcc, ninja).
- Incompatible gcc/nvcc combination on clusters.

Useful checks:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version || true
nvidia-smi || true
```

Practical fixes:

- Install the correct PyTorch wheel/conda package for your CUDA.
- Install build helpers: `pip install ninja`
- Rebuild detectron2 inside the activated venv.

## DensePose config path not found

Error looks like: missing `densepose_rcnn_R_101_FPN_DL_s1x.yaml`.

Cause: `external/detectron2/` is missing or in a different location.

Fix:

- Clone Detectron2 into `external/detectron2/`, or update `DENSEPOSE_CONFIG` in `config.py`.

## SMPL model file not found

Error looks like: missing `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`.

Fix:

- Download SMPL manually and place files as described in `docs/data.md`.
- If the filename differs in your download, update `SMPL_MODEL` in `config.py`.

## COCO images not found

Cause: `data/val2014/` is missing or empty.

Fix:

- Download and extract `val2014.zip` (see `docs/data.md`).

## Script crashes with import errors when run as a file

If you run:

```bash
python projects/euclidean_fitter.py
```

you may get import/path issues.

Fix: run as a module from the repo root:

```bash
python -m projects.euclidean_fitter
```
