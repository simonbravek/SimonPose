from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
DATA_DIR = REPO_ROOT / "data"
EXTERNAL_DIR = REPO_ROOT / "external"
COCO_DIR = DATA_DIR / "val2014"
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_ROOT = REPO_ROOT / "output"


# Concrete paths
SMPL_MODEL = MODELS_DIR / "smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
SMPL_VERT_SEGMENTATION = DATA_DIR / "smpl_vert_segmentation.json"
DENSEPOSE_CONFIG = EXTERNAL_DIR / "detectron2/projects/DensePose/configs/cse/densepose_rcnn_R_101_FPN_DL_s1x.yaml"
DENSEPOSE_WEIGHTS = MODELS_DIR / "densepose/model_final_1d3314.pkl"
