import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms.functional as F
#import torchvision.transforms.v2.functional as F

model = None


@torch.no_grad()
def init():
    global model
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    model.cuda().eval()


@torch.no_grad()
def run_monodepth(path, vis=False):
    #intrinsic = np.array([707.0493, 707.0493, 604.0814, 180.5066])
    #intrinsic = np.array([1961.85286, 1969.23077, 540, 960])

    # keep ratio resize
    input_size = (616, 1064) # for vit model
    # input_size = (544, 1216) # for convnext model

    image = torchvision.io.read_image(path).cuda()
    h, w = image.shape[1:]
    scale = min(input_size[0] / h, input_size[1] / w)
    rescale_size = (int(h * scale), int(w * scale))
    # remember to scale intrinsic, hold depth
    #intrinsic = intrinsic * scale

    # padding to input_size
    pad_h = input_size[0] - rescale_size[0]
    pad_w = input_size[1] - rescale_size[1]
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2

    image = image.to(torch.float32)
    #image = F.to_dtype(image, torch.float32, scale=False)
    image = F.resize(image, rescale_size)
    image = F.normalize(image, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    image = F.pad(image, [pad_w_half, pad_h_half, pad_w - pad_w_half, pad_h - pad_h_half], fill=0)

    rgb = image.unsqueeze(0)

    # inference
    if model is None:
        init()
    pred_depth, confidence, output_dict = model.inference({'input': rgb})

    # un pad
    pred_depth = F.center_crop(pred_depth, rescale_size)

    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth, (h, w), mode='bilinear').squeeze()
    ###################### canonical camera space ######################

    #### de-canonical transform
    # 1000.0 is the focal length of canonical camera
    # pred_depth = pred_depth * (intrinsic[0] / 1000.0)  # now the depth is metric

    if vis:
        # Choose max depth for your data (e.g., 80 for KITTI-like outdoor)
        max_vis_depth = 20  # adjust as appropriate for your dataset

        # Clamp and scale
        pred_depth_np = pred_depth.clamp(0, max_vis_depth).mul(255 / max_vis_depth).to('cpu', torch.uint8).numpy()

        # Optionally invert (remove inversion if you prefer brighter=near)
        depth_colored = cv2.applyColorMap(pred_depth_np, cv2.COLORMAP_INFERNO)  # Try INFERNO for high contrast
        return pred_depth, depth_colored

    """
    #### normal are also available
    pred_normal = output_dict['prediction_normal'][:, :3, :, :]
    normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
    # un pad and resize to some size if needed
    pred_normal = pred_normal.squeeze(0)
    pred_normal = F.center_crop(pred_normal, rescale_size)
    # you can now do anything with the normal
    # such as visualize pred_normal
    pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
    pred_normal_vis = (pred_normal_vis + 1) / 2
    cv2.imwrite('normal_vis.png', (pred_normal_vis * 255).astype(np.uint8))
    """
    return pred_depth


if __name__ == '__main__':
    direct = sys.argv[1]
    base = os.path.basename(direct.rstrip('/'))
    os.makedirs(os.path.join('depths', base), exist_ok=True)
    init()
    for entry in os.scandir(direct):
        if entry.is_file():
            depth, vis = run_monodepth(entry.path, vis=True)
            np.save(os.path.join('depths', base, entry.name), depth.numpy(force=True))
            cv2.imwrite(os.path.join('depths', base, entry.name), vis)

