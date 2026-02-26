import os
import sys
import argparse

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch
from pycocotools.coco import COCO
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.boxes import BoxMode
from densepose import add_densepose_config
from densepose.engine import Trainer
from densepose.modeling.cse.utils import get_closest_vertices_mask_from_ES
from densepose.structures.mesh import create_mesh
from densepose.vis.densepose_outputs_vertex import get_xyz_vertex_embedding
from densepose_overrides import CustomBoxesPredictor



# GLOBALS
ISATTY = sys.stdout.isatty()
OFFSET = 0
NUMBER_OF_IMAGES = 1
LOSS_AREA = 300 * 300
FOV = 60

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
DATA_DIR = os.path.join(REPO_ROOT, "data")
EXTERNAL_DIR = os.path.join(REPO_ROOT, "external")
IMAGE_ROOT = os.path.join(DATA_DIR, "val2014")
SMPL_MODEL = os.path.join(REPO_ROOT, "models/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")
DENSEPOSE_CONFIG = os.path.join(EXTERNAL_DIR, "detectron2/projects/DensePose/configs/cse/densepose_rcnn_R_101_FPN_DL_s1x.yaml")
DENSEPOSE_WEIGHTS = os.path.join(REPO_ROOT, "models/densepose/model_final_1d3314.pkl")

# INITIAL SETUP
parser = argparse.ArgumentParser(description="My program with output and number arguments")
parser.add_argument('-o', '--output', type=str, required=False, help="Output file name in the version root or absolute path")
parser.add_argument('-n', '--number', type=int, required=False, help="Number of images to process", default=NUMBER_OF_IMAGES)
args = parser.parse_args()

for i in range(10):
    if not os.path.exists(os.path.join(PROJECT_ROOT, f"output/output_{i}")):
        OUTPUT_DIR = os.path.join(PROJECT_ROOT, f"output/output_{i}")
        break
    elif i == 9:
        print("No output directory found, please remove the existing ones or change the output name.")
        exit()

# change if modified by arguments
if args.output:
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, args.output)
print(f"Output directory: {OUTPUT_DIR}")
NUMBER_OF_IMAGES = args.number

os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.set_grad_enabled(False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file(DENSEPOSE_CONFIG)
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg.MODEL.DEVICE = device

predictor = CustomBoxesPredictor(cfg)
#predictor = DefaultPredictor(cfg)
model = predictor.model.eval()
embedder = Trainer.extract_embedder_from_model(model)
mesh_vertex_embeddings = embedder('smpl_27554').to(device)

embed_map = get_xyz_vertex_embedding('smpl_27554', 'cpu')
color_map = (embed_map * 255).clip(0, 255).to(device, torch.uint8)

mesh = create_mesh('smpl_27554', 'cpu')
geodists = mesh.geodists.numpy()
create_mesh.cache_clear()

smpl_object = smplx.SMPL(model_path=SMPL_MODEL)
smpl_output = smpl_object.forward()

#joints = smpl_output.joints.squeeze(0)
vertices = smpl_output.vertices.squeeze(0)
bounds = np.stack([vertices.amin(0).numpy(), vertices.amax(0).numpy()], 1)
mean = bounds.mean(1, keepdims=True)
bounds = (bounds - mean) * (2/3) + mean

cap = 6890
#cap = 27_554

cocoGt = COCO(os.path.join(DATA_DIR, 'densepose_minival2014_cse.json'))

for image_id in tqdm(sorted(cocoGt.getImgIds())[:NUMBER_OF_IMAGES]):
    im_data, = cocoGt.loadImgs(image_id)
    ann_ids = cocoGt.getAnnIds(imgIds=[image_id])
    annotations = [ann for ann in cocoGt.loadAnns(ann_ids)
                   if ann.get('dp_vertex')]
    if not annotations:
        continue

    image = cv2.imread(
        os.path.join(IMAGE_ROOT, im_data['file_name'])
    )
    H, W = image.shape[:2]

    dp_output = predictor.inference_detections(image, annotations)
    instances = dp_output['instances']
    boxes_xywh_abs = BoxMode.convert(
        instances.pred_boxes.tensor.clone(),
        BoxMode.XYXY_ABS,
        BoxMode.XYWH_ABS
    ).to(int)

    for di, ann in enumerate(annotations):
        gt_box = np.array([ann['bbox'][:2], ann['bbox'][2:]])
        gt_box[1] += gt_box[0]
        x, y, w, h = boxes_xywh_abs[di].tolist()
        embedding = instances.pred_densepose.embedding[di]
        coarse_segm = instances.pred_densepose.coarse_segm[di]

        closest, inst_mask = get_closest_vertices_mask_from_ES(
            embedding.unsqueeze(0),
            coarse_segm.unsqueeze(0),
            h, w,
            mesh_vertex_embeddings[:cap],
            device
        )
        foreground = closest[inst_mask]

        yrange = torch.arange(h, device=device)
        xrange = torch.arange(w, device=device)
        yi, xi = torch.meshgrid(yrange, xrange, indexing='ij')

        counter = torch.bincount(foreground, minlength=cap)
        ymean = torch.bincount(foreground, yi[inst_mask], minlength=cap) \
            .div(counter) \
            .nan_to_num(0)
        xmean = torch.bincount(foreground, xi[inst_mask], minlength=cap) \
            .div(counter) \
            .nan_to_num(0)

        dist_sq = torch.square(yi[inst_mask] - ymean.take(foreground)) \
            + torch.square(xi[inst_mask] - xmean.take(foreground))

        std = torch.bincount(foreground, dist_sq, minlength=cap) \
            .div(counter) \
            .nan_to_num(0) \
            .sqrt()

        mask = inst_mask.to('cpu', non_blocking=True)
        vis = np.full_like(image, 240)  # f0f0f0
        temp = cv2.applyColorMap(
            color_map.take(closest[inst_mask]).numpy(force=True),
            cv2.COLORMAP_JET
        )
        vis[y:y+h, x:x+w][mask] = temp[:, 0, :]
        del temp

        demo = image.copy()
        cv2.rectangle(demo, *np.int32(gt_box), (0, 255, 0), 2)
        cv2.rectangle(vis, *np.int32(gt_box), (0, 255, 0), 2)

        fig = plt.figure(tight_layout=True, figsize=(12, 4))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

        ax = fig.add_subplot(gs[0])
        ax.imshow(demo[..., ::-1])
        ax.set_box_aspect(1)
        ax.set(xticks=[], yticks=[])

        ax = fig.add_subplot(gs[1])
        ax.imshow(vis[..., ::-1])
        ax.set_box_aspect(1)
        ax.set(xticks=[], yticks=[])

        unseen = counter.eq(0).numpy(force=True)
        color = std[counter > 0].numpy(force=True)

        ax = fig.add_subplot(gs[2], projection='3d')
        ax.scatter(*vertices[unseen].unbind(-1), s=1, c='0.8')  # e0e0e0
        sm = ax.scatter(*vertices[~unseen].unbind(-1), s=3, c=color, vmin=0, cmap='plasma')
        ax.set_axis_off()
        ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2])
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(azim=-90, elev=90)
        plt.colorbar(sm, cax=fig.add_subplot(gs[3]))

        plt.savefig(os.path.join(OUTPUT_DIR, f"test_{ann['id']}"))

        # breakpoint()


