# Local setup

This repository is a research prototype. The scripts are tested primarily on Linux with an NVIDIA GPU.

## Requirements

Minimum (to run `projects/euclidean_fitter.py`):

- Python: 3.11
- OS: Linux recommended (Windows via WSL may work; macOS is likely CPU-only)
- GPU: NVIDIA GPU with CUDA-capable driver (strongly recommended)
- Disk: ~10+ GB for COCO val2014 + models

Build requirements for Detectron2-from-source:

- A C++ toolchain (gcc/g++), and a CUDA toolkit if building with CUDA
- `pip`, `setuptools`, `wheel` (recent versions recommended)

## 1) Create a virtual environment

From the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

## 2) Install PyTorch

Pick a PyTorch build that matches your CUDA runtime.

Example (PyTorch 2.4.0, CUDA 12.1 wheels):

```bash
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

If you are unsure which one to pick, use the official selector: https://pytorch.org/get-started/locally/

## 3) Install Python dependencies

```bash
pip install -r requirements.txt
```

Note: Detectron2/DensePose are not installed from `requirements.txt`.

## 4) Install Detectron2 + DensePose (from source)

SimonPose expects Detectron2 to live under `external/detectron2/` (see `config.py`).

```bash
mkdir -p external
git clone https://github.com/facebookresearch/detectron2.git external/detectron2

pip install --no-build-isolation -e external/detectron2
pip install --no-build-isolation -e external/detectron2/projects/DensePose
```

If you prefer to prevent `pip` from changing already-installed dependencies (common in CUDA setups), add `--no-deps` to the two `pip install` commands.

## 5) Verify imports

```bash
python -c "import torch; import detectron2; import densepose; import smplx; print('imports: ok')"
```

## 6) Download data + model files

Follow `docs/data.md`, then run to check:

```bash
python -m projects.euclidean_fitter -n 1 -i 10
```
