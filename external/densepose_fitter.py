import os
import sys

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Boxes, BoxMode, Instances
from densepose import add_densepose_config
from densepose.engine import Trainer
from densepose.modeling.cse.utils import normalize_embeddings, get_closest_vertices_mask_from_ES
from densepose.structures.mesh import create_mesh
from densepose.vis.densepose_outputs_vertex import get_xyz_vertex_embedding


class CustomBoxesPredictor(DefaultPredictor):

    def inference_detections(self, original_image, detections):
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        boxes = torch.tensor([det['bbox'] for det in detections])
        boxes = Boxes(BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS))
        boxes.scale(image.shape[2] / width, image.shape[1] / height)

        scores = torch.tensor([det.get('score', 0.999) for det in detections])
        labels = torch.tensor([det.get('category_id', 0) for det in detections], dtype=int)

        instances = Instances(
            image.shape[1:],
            pred_boxes=boxes,
            scores=scores,
            pred_classes=labels
        )

        return self.model.inference([inputs], [instances])[0]


torch.set_grad_enabled(False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file('config_cse.yaml')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg.MODEL.DEVICE = device

#predictor = DefaultPredictor(cfg)
predictor = CustomBoxesPredictor(cfg)
model = predictor.model.eval()
embedder = Trainer.extract_embedder_from_model(model)
mesh_vertex_embeddings = embedder('smpl_27554').to(device)

embed_map = get_xyz_vertex_embedding('smpl_27554', 'cpu')
color_map = (embed_map * 255).clip(0, 255).to(device, torch.uint8)

mesh = create_mesh('smpl_27554', 'cpu')
geodists = mesh.geodists.numpy()
create_mesh.cache_clear()

smpl_object = smplx.SMPL(model_path='../repos/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
smpl_output = smpl_object.forward()

#joints = smpl_output.joints.squeeze(0)
vertices = smpl_output.vertices.squeeze(0)
bounds = np.stack([vertices.amin(0).numpy(), vertices.amax(0).numpy()], 1)
mean = bounds.mean(1, keepdims=True)
bounds = (bounds - mean) * (2/3) + mean

cap = 6890
#cap = 27_554

cocoGt = COCO('datasets/coco_cse/densepose_minival2014_cse.json')

for image_id in sorted(cocoGt.getImgIds())[:10]:
    im_data, = cocoGt.loadImgs(image_id)
    ann_ids = cocoGt.getAnnIds(imgIds=[image_id])
    annotations = [ann for ann in cocoGt.loadAnns(ann_ids)
                   if ann.get('dp_vertex') and not ann.get('iscrowd')]
    if not annotations:
        continue

    image = cv2.imread(
        os.path.join('datasets/coco/val2014', im_data['file_name'])
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

        dist_sq = torch.square(yi[inst_mask] - ymean.take(yi[inst_mask])) \
            + torch.square(xi[inst_mask] - xmean.take(xi[inst_mask]))

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

        cv2.rectangle(image, *np.int32(gt_box), (0, 255, 0), 2)
        cv2.rectangle(vis, *np.int32(gt_box), (0, 255, 0), 2)

        fig = plt.figure(tight_layout=True, figsize=(12, 4))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

        ax = fig.add_subplot(gs[0])
        ax.imshow(image[..., ::-1])
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

        plt.savefig(f"test_{ann['id']}")

        #breakpoint()

