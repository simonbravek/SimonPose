# Data and model files

SimonPose does not ship datasets or model weights in this repository. You need to download them separately and place them into the expected paths (see `config.py`).

## Expected directory layout

From the repo root:

```text
data/
  val2014/
    COCO_val2014_*.jpg
  densepose_minival2014_cse.json

models/
  densepose/
    model_final_1d3314.pkl
  smpl/
    models/
      basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl

external/
  detectron2/
    projects/DensePose/...
```

If your layout differs, update `config.py`.

## COCO 2014 val images

The default scripts use COCO 2014 val images (the DensePose minival subset is a subset of these).

```bash
mkdir -p data
curl -L -o data/val2014.zip http://images.cocodataset.org/zips/val2014.zip
unzip -q data/val2014.zip -d data
```

This should create `data/val2014/`.

On clusters, COCO is often available in shared storage. In that case, create a symlink instead of downloading:

```bash
mkdir -p data
ln -s /path/to/val2014 data/val2014
```

## DensePose CSE annotations (minival)

```bash
mkdir -p data
curl -L -o data/densepose_minival2014_cse.json \
  "https://dl.fbaipublicfiles.com/densepose/annotations/coco_cse/densepose_minival2014_cse.json"
```

## SMPL body part mapping

```bash
mkdir -p data
curl -L -o data/smpl_vert_segmentation.json \
  "https://github.com/Meshcapade/wiki/blob/main/assets/SMPL_body_segmentation/smpl/smpl_vert_segmentation.json?raw=true"
```




## DensePose model weights

This repo uses the R101 CSE checkpoint referenced in the Detectron2 DensePose docs.

```bash
mkdir -p models/densepose
curl -L -o models/densepose/model_final_1d3314.pkl \
  "https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x/250713061/model_final_1d3314.pkl"
```

## SMPL model files (manual download)

SMPL model files are distributed under separate terms and typically require registration + license acceptance. Direct `curl` downloads frequently do not work.

Steps:

1) Register and download **SMPL Python v1.1.0** from https://smpl.is.tue.mpg.de/
2) Extract the archive
3) Place the `smpl/` folder into `models/` so that this file exists:

```text
models/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
```

If your download uses a slightly different filename (e.g. `basicModel_...` vs `basicmodel_...`), update `SMPL_MODEL` in `config.py` to match your actual file.

## Optional: SMPL-X

The write-up discusses SMPL-X, but the current `projects/*` scripts use `smplx.SMPL` and do not require SMPL-X model files.

If you want SMPL-X anyway, see https://smpl-x.is.tue.mpg.de/ and update your code/config accordingly.

## Notes on licenses

- COCO images/annotations: separate terms from the COCO project
- DensePose weights/annotations: separate terms from Meta/Detectron2
- SMPL/SMPL-X model files: separate license/terms from MPI/MPG

## Safety note

Model files and datasets are large. Prefer official sources, and verify checksums when they are provided by the upstream project.
