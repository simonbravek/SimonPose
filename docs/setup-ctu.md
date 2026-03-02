# CTU cluster setup

This is a practical setup guide for the CTU/FEL cluster environment described in the original README.

More information about the CTU cluster can be found on the [Visual Vecognition Group docs](https://klarajanouskova.github.io/VRG/).

The core idea:

- Let the cluster modules provide heavyweight GPU packages (PyTorch, OpenCV, etc.).
- Install the remaining Python dependencies into a local venv.
- Build/install Detectron2 + DensePose from source into that venv.

## 0) Pick a GPU

Inspect availability:

```bash
nvidia-smi
```

When running a script, restrict to a free device:

```bash
CUDA_VISIBLE_DEVICES=8 python -m projects.euclidean_fitter -n 1 -i 50
```

## 1) Load modules

The exact module names may change; this is a known-good starting point:

```bash
ml PyTorch/2.4.0-foss-2023b-CUDA-12.4.0 torchvision/0.19.0-foss-2023b-CUDA-12.4.0 OpenCV/4.10.0-foss-2023b-CUDA-12.4.0-contrib Albumentations/1.4.4-foss-2023b-CUDA-12.4.0 pycocotools/2.0.7-foss-2023b matplotlib/3.8.2-gfbf-2023b
```

If you need alternative module combinations, see [`docs/dependencies.md`](dependencies.md).

## 2) Create a venv

From the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

## 3) Install remaining Python packages

On the cluster, avoid re-installing packages already provided by modules (especially `torch`, `torchvision`, `opencv`, `pycocotools`, `matplotlib`).

Minimum for `projects/euclidean_fitter.py`:

```bash
pip install scipy smplx tqdm
```

If you plan to run additional experiments/utilities, install the rest as needed (e.g. `scikit-image`, `scikit-learn`, `pandas`, ...).

## 4) Install Detectron2 + DensePose

```bash
mkdir -p external
git clone https://github.com/facebookresearch/detectron2.git external/detectron2

pip install --no-build-isolation --no-deps -e external/detectron2
pip install --no-build-isolation --no-deps -e external/detectron2/projects/DensePose
```

If the build fails, see `docs/troubleshooting.md` (toolchain/CUDA mismatch is the most common cause).

## 5) Data + models

Follow `docs/data.md`.
