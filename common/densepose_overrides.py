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