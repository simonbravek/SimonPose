# pyhon
import json
import shutil
import os
import subprocess
import math
import sys
import argparse

# packages
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import smplx
import torch
import torch.optim as optim
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from densepose import add_densepose_config
from densepose.engine import Trainer
from densepose.modeling.cse.utils import get_closest_vertices_mask_from_ES
from densepose.vis.densepose_outputs_vertex import get_xyz_vertex_embedding
from pycocotools.coco import COCO
from scipy.spatial.transform import Rotation
from tqdm import trange
from tqdm import tqdm

# modules
from utils import (get_translation,
                    get_camera_intrinsics,
                    interpolate,
                    get_inside_box,
                    get_inside_mask,
                    get_closest_vertices,
                    visible_vertices_gpu,
                    visible_vertices_gpu,
                    euclid_loss_gpu
)

#TODO: get them directly with subpixel accuracy
# bigger resolution brings more points but that is not worth the resources
# for some reason it detects around 1200 points on the whole body

device = "cuda" if torch.cuda.is_available() else "cpu"

# GLOBALS
ISATTY = sys.stdout.isatty()
OFFSET = 0
LEARNING_RATE = 0.001
ITERATIONS = 300
NUMBER_OF_IMAGES = 1
LOSS_AREA = 300 * 300
FOV = 60
CONSTANT = 1000000000
TORSO_MASK = False # if False, the whole mesh is used

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(PROJECT_ROOT))
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
parser.add_argument('-i', '--iterations', type=int, required=False, help="Number of iterations to run for each image", default=ITERATIONS)
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
ITERATIONS = args.iterations

os.makedirs(OUTPUT_DIR, exist_ok=True)


beginning_losses = []
final_losses = []

if TORSO_MASK:
    torso_mask = torch.zeros(6890, dtype=torch.bool, device=device) # mask S torso vertices in the SMPL mesh

    with open(os.path.join(DATA_DIR, "body_parts_to_6889_points.json"), "r") as f:
        body_parts_to_points = json.load(f)

    for key in ['spine', 'spine1', 'spine2', 'hips', 'leftShoulder', 'rightShoulder']:
        torso_mask[body_parts_to_points[key]] = True
        
else:
    torso_mask = torch.ones(6890, dtype=torch.bool, device=device)

visualization_timetable = tuple(range(0, ITERATIONS, 50)) + (ITERATIONS - 1,)

# INITIALIZAITON

# detectron2 configuration
config = get_cfg()
add_densepose_config(config)

# the architecture of densepose model
config.merge_from_file(DENSEPOSE_CONFIG)
config.MODEL.WEIGHTS = DENSEPOSE_WEIGHTS
config.MODEL.DEVICE = device
predictor = DefaultPredictor(config) 
model = predictor.model.eval()
embedder = Trainer.extract_embedder_from_model(model)

# matice co spojuje 27554 vertex indexu a 112x112 embeddingu (vektor <-> index)
mesh_vertex_embeddings = embedder('smpl_27554')[:6890].to(device) 
embed_map = get_xyz_vertex_embedding('smpl_27554', device=device)[:6890].to(device)
color_map = (embed_map * 255).to(torch.uint8)

cocoGt = COCO(os.path.join(DATA_DIR, "densepose_minival2014_cse.json"))

# GLoBAL VARIABLES
all_indices = torch.arange(6890, device=device)
torso_indices = all_indices[torso_mask]

all_indices = torch.arange(6890, device=device)
vertices_idx = (torch.arange(6890, device=device))[torso_mask]

r = Rotation.from_euler('xyz', [0, 180, 0], degrees=True)
rotation_matrix = r.as_matrix()
rodrigues, _ = cv2.Rodrigues(rotation_matrix)

selected_indices = all_indices[torso_mask] # these are all the indices projected (even occluded ones)

n_cols = 2
n_rows = math.ceil((6 + len(visualization_timetable)) / n_cols)
plot_dpi = 100
plot_scale = 1.3


image_ids = cocoGt.getImgIds()[OFFSET : NUMBER_OF_IMAGES + OFFSET]
for image_id in tqdm(image_ids, position=0, desc="Images", disable=not ISATTY):
    image_info = cocoGt.loadImgs(image_id)[0]
    original_image = cv2.imread(os.path.join(IMAGE_ROOT, image_info['file_name']))

    plot_h, plot_w = original_image.shape[:2]

    figsize = (
        n_cols * plot_w / plot_dpi * plot_scale,
        n_rows * plot_h / plot_dpi * plot_scale,
    )  # Inches = pixels / dpi

    predictions = predictor(original_image)
    instances = predictions["instances"]

    # get the bounding box xywh for each person detected in the original_image
    number_of_people_detected = len(instances)
    boxes_xywh_abs = BoxMode.convert(
        instances.pred_boxes.tensor.clone(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
    ).to(int)
    K_flipped = get_camera_intrinsics(fov_deg=FOV, width=original_image.shape[1], height=original_image.shape[0], device=device)


    for person_index in range(number_of_people_detected):

        # get the embedding and coarse segmentation for the person
        embedding = instances.pred_densepose.embedding[person_index]
        embedding /= embedding.square().sum(0, keepdim=True).sqrt() # square root normalization
        coarse_segm = instances.pred_densepose.coarse_segm[person_index]
        x, y, w, h = boxes_xywh_abs[person_index].tolist() # BB levy horni roh, sirka, vyska

        # get the initial SMPL, resource intensive, dont duplicate
        # TODO load it just once and then reset the values
        smpl_object = smplx.SMPL(model_path=SMPL_MODEL).to(device)
        smpl_output = smpl_object.forward()

        # rotation
        global_orient = torch.tensor(rodrigues, device=device, dtype=torch.float32).t()
        global_orient.requires_grad = True

        transl = get_translation(
            K_flipped,
            smpl_output.joints[0, 12],
            smpl_output.joints[0, 0],
            torch.tensor([x + w * .5, y + h * .2, 1.0], dtype=torch.float32, device=device),
            torch.tensor([x + w * .5, y + h * .8, 1.0], dtype=torch.float32, device=device),
            device=device
        )
        transl.requires_grad = True

        betas = torch.zeros((1, 10), device=device, requires_grad=True)
        pose = torch.zeros((1, 23*3), device=device, requires_grad=True)

        optimizer = optim.Adam([global_orient, transl, betas, pose], lr=LEARNING_RATE)
        
        progress = []

        # for visualisation
        embedding_points_bb_resolution, segmentation_bb_resolution = get_closest_vertices_mask_from_ES(
            embedding.unsqueeze(0),
            coarse_segm.unsqueeze(0),
            h, w,
            mesh_vertex_embeddings,
            embedding.device
        )

        scale = math.sqrt(LOSS_AREA) / math.sqrt(h * w) 
        h_loss = round(h * scale)
        w_loss = round(w * scale)


        # for loss calculaiton
        embedding_points_loss_resolution, segmentation_loss_resolution = get_closest_vertices_mask_from_ES(
            embedding.unsqueeze(0),
            coarse_segm.unsqueeze(0),
            h_loss, w_loss,
            mesh_vertex_embeddings,
            embedding.device
        )

        embedding_points_loss_resolution = embedding_points_loss_resolution.t()
        segmentation_loss_resolution = segmentation_loss_resolution.t()


        # LOSS_AREA[segmentation] -> E
        E_indices_whole = embedding_points_loss_resolution[segmentation_loss_resolution]

        E_indices = E_indices_whole[torso_mask[E_indices_whole]] # here I have the points' indices
        E_coordinates = segmentation_loss_resolution.nonzero()[torso_mask[E_indices_whole]] # here I have their coordinates
        
        plt.figure(figsize=figsize, dpi=plot_dpi, tight_layout=True)

        for iteration in trange(ITERATIONS, position=1, desc="Instance", leave=False, disable=not ISATTY):
            smpl_output = smpl_object.forward(
                transl=transl.unsqueeze(0),
                global_orient=global_orient,
                betas=betas,
                body_pose=pose
            )

            # 6890 . 3
            vertices = smpl_output.vertices[0]
            
            # TODO project just the vertices in front of the camera
            
            # 6890 . 3
            projection = vertices @ K_flipped.T

            # 6890 . 2
            SMPL_coordinates_IMG = (projection[:, :2] / projection[:, 2, None])[torso_mask]

            # inside_box = get_inside_box([x, y, x + w, y + h], SMPL_coordinates_IMG)
            
            # S . 2
            # SMPL_coordinates_IMG = SMPL_coordinates_IMG[inside_box]

            # S
            # selected_indices = all_indices # these are all the indices projected (even occluded ones)
            # selected_indices = selected_indices[inside_box]

            # to get the E coordinates I need: 
            # subtract x, y
            # multiply by the scale each dimension

            SMPL_coordinates_LOSS = (SMPL_coordinates_IMG - torch.tensor([x, y], device=device)) * scale

            # regloss = betas.square().sum() * CONSTANT
            loss = euclid_loss_gpu(selected_indices, SMPL_coordinates_LOSS, E_indices, E_coordinates, device)
            


            # visualisation
            if iteration in visualization_timetable:
                # 1766
                u, v = SMPL_coordinates_IMG.unbind(-1)

                inside = (u >= 0) & (u < original_image.shape[1]) & (v >= 0) & (v < original_image.shape[0])
                v_vis = v[inside].long().detach().cpu().numpy()
                u_vis = u[inside].long().detach().cpu().numpy()
                temp = cv2.applyColorMap(
                    color_map[torso_mask][inside].cpu().numpy(),
                    cv2.COLORMAP_JET
                )

                vis = original_image.copy() // 4
                for ux, vx, cx in zip(u_vis, v_vis, temp[:, 0, :].tolist()):
                    cv2.circle(vis, (ux, vx), 2, tuple(cx), -1)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(vis, (int(x + w * .5), int(y + h * .2)), 5, (255, 0, 0), -1)
                cv2.circle(vis, (int(x + w * .5), int(y + h * .8)), 5, (0, 255, 0), -1)

                indices = []
                if iteration == 0:
                    indices.append(4)
                elif iteration == ITERATIONS - 1:
                    indices.append(6)
                indices.append(7 + (iteration + 1) // 50)

                for idx in indices:
                    plt.subplot(n_rows, n_cols, idx)
                    plt.imshow(vis[..., ::-1])
                    plt.title(f"The {iteration}th iteration")
                    plt.grid(False) 
                    plt.axis('on')

                if iteration == 0:
                    beginning_losses.append(loss.item())
                elif iteration == ITERATIONS - 1:
                    final_losses.append(loss.item())

            # The vertices should have shape w, h
            loss.backward(retain_graph=True)
            progress.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()


        mask = segmentation_bb_resolution.cpu().numpy()
        temp = cv2.applyColorMap(color_map.take(embedding_points_bb_resolution).cpu().numpy(), cv2.COLORMAP_JET)

        vis = original_image.copy()
        vis[y:y+h, x:x+w][mask] = temp[mask]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        densepose_output_image = vis

        plt.subplot(n_rows, n_cols, 1)
        plt.plot(progress)
        plt.title(f"Loss over time (Euclidian distance between the SMPL mesh and the DensePose mesh.)")
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.ylim(bottom=0.)
        plt.grid()

        color_mesh = cv2.imread(os.path.join(REPO_ROOT, "visualisations/color_mesh_single_shaved.png"))
        
        plt.subplot(n_rows, n_cols, 2)
        plt.imshow(color_mesh[..., ::-1])
        plt.title(f"Color mesh")
        plt.grid(False) 
        plt.axis('on')
        
        plt.subplot(n_rows, n_cols, 3)
        plt.imshow(original_image[..., ::-1])
        plt.title(f"The original image {image_id}")
        plt.grid(False) 
        plt.axis('on')

        plt.subplot(n_rows, n_cols, 5)
        plt.imshow(densepose_output_image[..., ::-1])
        plt.title(f"Densepose output")
        plt.grid(False) 
        plt.axis('on')

        plt.savefig(os.path.join(OUTPUT_DIR, f"{image_id}_{person_index}_overview.png"))
        plt.close()

        # TODO filter out poor detections
        break

    loss_improvement = np.array(final_losses) / np.array(beginning_losses)

    plt.figure(tight_layout=True)
    plt.hist(final_losses, bins=10)
    plt.xlabel('Final loss')
    plt.ylabel('Number of images')
    plt.title('Distribution of final losses across images')
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_histogram.png'))
    plt.close()

    plt.figure(tight_layout=True)
    plt.hist(loss_improvement, bins=10)
    plt.xlabel('Improvement from heuristic guess (%)')
    plt.ylabel('Number of images')
    plt.title('Distribution of loss improvement across images')
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_improvement_histogram.png'))
    plt.close()


    