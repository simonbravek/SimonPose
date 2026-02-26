import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cv2


def interpolate(data, points):
    max_y, max_x = data.shape[-2:]
    points = points * torch.tensor([[max_x, max_y]], device=points.device)

    x_lo = points[:, 0].floor().clamp(0, max_x - 1).long()
    x_hi = x_lo.add(1).clamp(max=max_x - 1)
    x_w = points[:, 0] - x_lo

    y_lo = points[:, 1].floor().clamp(0, max_y - 1).long()
    y_hi = y_lo.add(1).clamp(max=max_y - 1)
    y_w = points[:, 1] - y_lo

    w_ylo_xlo = (1.0 - x_w) * (1.0 - y_w)
    w_ylo_xhi = x_w * (1.0 - y_w)
    w_yhi_xlo = (1.0 - x_w) * y_w
    w_yhi_xhi = x_w * y_w

    return (data[:, y_lo, x_lo] * w_ylo_xlo
        + data[:, y_lo, x_hi] * w_ylo_xhi
        + data[:, y_hi, x_lo] * w_yhi_xlo
        + data[:, y_hi, x_hi] * w_yhi_xhi)


def get_inside_box(bbox, pixels):
    return (bbox[0] <= pixels[:, 0]) & (pixels[:, 0] < bbox[2]) \
        & (bbox[1] <= pixels[:, 1]) & (pixels[:, 1] < bbox[3])


def get_inside_mask(mask, points):
    X = interpolate(mask, points)
    return X.argmax(0) == 1


def get_closest_vertices(embeddings, mesh_vertex_embeddings):
    return torch.argmax(
        embeddings.t() @ mesh_vertex_embeddings.t(),
        dim=1
    )



def get_camera_intrinsics(fov_deg, width, height, device):
    """
    Generate camera intrinsic matrix from FoV and image dimensions.

    Args:
        fov_deg (float): Horizontal field of view in degrees.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        torch.Tensor: 3x3 camera intrinsic matrix.
    """
    fov_rad = math.radians(fov_deg)
    fx = fy = width / (2 * math.tan(fov_rad / 2))
    cx = width / 2
    cy = height / 2

    # Fy @ K
    K_flipped = torch.tensor([
        [fx, 0,  cx],
        [0,  -fy, height - cy],
        [0,  0,  1]
    ], dtype=torch.float32, device=device)

    return K_flipped



def get_translation(K, p1_smpl, p2_smpl, p1_image, p2_image, device):
    x1, y1, z1 = K @ p1_smpl
    x2, y2, z2 = K @ p2_smpl # E carou

    u1, v1, _ = p1_image
    u2, v2, _ = p2_image

    tz = (y1 - y2 - v1 * z1 + v2 * z2)/(v1 - v2)
    tx = -(x1 - u1 * z1 - u1 * tz)
    ty = -(y1 - v1 * z1 - v1 * tz)


    cx = K[0, 2]
    cy = K[1, 2]

    return torch.tensor([
        (tx - cx * tz)/(K[0, 0]),
        (ty - cy * tz)/(K[1, 1]),
        tz
    ], device=device)



# used to compute mask
def visible_vertices_cpu(vertices_3d, vertices_2d, faces, query_pixels, min_depth=0):
    """
    Computes closest vertex index only for the given pixels.

    Parameters:
    - vertices_3d: (N, 3) array of 3D vertex coordinates.
    - vertices_2d: (N, 2) array of 2D projected pixel coordinates.
    - faces: (M, 3) array of triangle face indices.
    - query_pixels: List of (x, y) pixel coordinates to compute depth for.

    Returns:
    - result: 
    """
    result = np.full((len(query_pixels),), -1, dtype=int)
    V = vertices_3d[faces.flat].reshape(-1, 3, 3)
    P = vertices_2d[faces.flat].reshape(-1, 3, 2)

    x_min = P[..., 0].min(-1)
    x_max = P[..., 0].max(-1)
    y_min = P[..., 1].min(-1)
    y_max = P[..., 1].max(-1)

    for S, pair in enumerate(query_pixels):
        x, y = pair

        mask = (x_min <= x) & (x <= x_max) & (y_min <= y) & (y <= y_max)
        if not np.any(mask):
            #breakpoint()
            continue
        # F . 3 . 2
        Pmask = P[mask]
        # from compute_barycentric_coords

        # F . 3
        x0, y0 = Pmask[:, 0].T
        x1, y1 = Pmask[:, 1].T
        x2, y2 = Pmask[:, 2].T

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        valid = np.abs(denom) > 1e-6
        if not np.any(valid):
            #breakpoint()
            continue

        a_nom = (y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)
        b_nom = (y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)

        alpha = a_nom[valid] / denom[valid]
        beta = b_nom[valid] / denom[valid]
        gamma = 1 - alpha - beta

        within = (0 <= alpha) & (alpha <= 1) \
            & (0 <= beta) & (beta <= 1) \
            & (0 <= gamma) & (gamma <= 1)
        if not np.any(within):
            #breakpoint()
            continue

        mask[mask] = valid
        mask[mask] = within

        # F . 3
        Vmask = V[mask]

        z_interpolated = alpha[within] * Vmask[:, 0, 2] \
            + beta[within] * Vmask[:, 1, 2] \
            + gamma[within] * Vmask[:, 2, 2]
        if np.all(z_interpolated <= min_depth):
            #breakpoint()
            continue
        z_interpolated[z_interpolated <= min_depth] = np.inf

        # discriminates all the points behind the camera
        # for behind add np.abs()
        argmin = np.argmin((z_interpolated))
        # F values that come from 0 to Q
        nonzero, = np.nonzero(mask)
        face_idx = nonzero[argmin]
        idx = np.argmax([alpha[argmin], beta[argmin], gamma[argmin]])
        result[S] = faces[face_idx, idx]
    breakpoint()
    return result


    import torch






def visible_vertices_gpu(vertices, points, faces, query_pixels, device, min_depth=0):
    '''
    query points are assumed to be in front of the camera

    V . 3 - vertices_3d
    V . 2 - vertices_2d
    F . 3 - faces
    Q . 2 - query_pixels
    
    TODO instead of allocating large spaces use indexing to spare 99% of the space
    '''

    Q = query_pixels.shape[0]
    F = faces.shape[0]

    # F . 3 . 3
    vertices_per_face = vertices[faces.flatten()].reshape(-1, 3, 3)
    # F . 3 . 2
    points_per_face = points[faces.flatten()].reshape(-1, 3, 2)

    # F
    x_min = points_per_face[..., 0].amin(dim=-1)
    x_max = points_per_face[..., 0].amax(dim=-1)
    y_min = points_per_face[..., 1].amin(dim=-1)
    y_max = points_per_face[..., 1].amax(dim=-1)

    # Q
    x, y = query_pixels.unbind(-1)

    # Broadcast properly for Q × F table:
    in_x = (x.unsqueeze(1) >= x_min.unsqueeze(0)) & (x.unsqueeze(1) <= x_max.unsqueeze(0))
    in_y = (y.unsqueeze(1) >= y_min.unsqueeze(0)) & (y.unsqueeze(1) <= y_max.unsqueeze(0))

    # Q . F
    mask = in_x & in_y


    # P
    query_idx, face_idx = mask.nonzero(as_tuple=True)
    P = query_idx.shape[0]

    # P . 3 . 2
    PPF_masked = points_per_face[face_idx] 
    # P . 3 . 3
    VPF_masked = vertices_per_face[face_idx]  
    # P . 2
    QP_masked = query_pixels[query_idx]


    # P
    x0 = PPF_masked[..., 0, 0]
    y0 = PPF_masked[..., 0, 1]
    x1 = PPF_masked[..., 1, 0]
    y1 = PPF_masked[..., 1, 1]
    x2 = PPF_masked[..., 2, 0]
    y2 = PPF_masked[..., 2, 1]

    # P
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    valid = denom.abs() > 1e-6

    denom_valid = denom[valid]
    x0_valid, y0_valid, x1_valid, y1_valid, x2_valid, y2_valid = x0[valid], y0[valid], x1[valid], y1[valid], x2[valid], y2[valid]
    QP_valid = QP_masked[valid]
    PPF_valid = PPF_masked[valid]
    VPF_valid = VPF_masked[valid]
    QI_valid = query_idx[valid]
    FI_valid = face_idx[valid]
    x_valid, y_valid = QP_valid.unbind(-1)
    
    # P
    a_nom_valid = (y1_valid - y2_valid) * (x_valid - x2_valid) + (x2_valid - x1_valid) * (y_valid - y2_valid)
    b_nom_valid = (y2_valid - y0_valid) * (x_valid - x2_valid) + (x0_valid - x2_valid) * (y_valid - y2_valid)

    # Va
    alpha_valid = a_nom_valid / denom_valid
    beta_valid = b_nom_valid / denom_valid
    gamma_valid = 1 - alpha_valid - beta_valid

    # Va
    within = (0 <= alpha_valid) & (alpha_valid <= 1) \
            & (0 <= beta_valid) & (beta_valid <= 1) \
            & (0 <= gamma_valid) & (gamma_valid <= 1)

    # W
    alpha_within = alpha_valid[within]
    beta_within = beta_valid[within]
    gamma_within = gamma_valid[within]
    PPF_within = PPF_valid[within]
    VPF_within = VPF_valid[within]
    QI_within = QI_valid[within]
    FI_within = FI_valid[within]

    # W
    z_interpolated = alpha_within * VPF_within[..., 0, 2] \
        + beta_within * VPF_within[..., 1, 2] \
        + gamma_within * VPF_within[..., 2, 2]

    # W
    z_interpolated[z_interpolated <= min_depth] = math.inf


    # (Q . F)
    valid_mask = mask.clone()
    valid_mask[mask] = valid
    within_mask = valid_mask.clone()
    within_mask[valid_mask] = within

    # Q . F
    intersect_depths = torch.full((Q, F), math.inf, device=device)
    intersect_depths[within_mask] = z_interpolated

    # Q
    FI_min = intersect_depths.argmin(dim=1)
    del intersect_depths

    aux = torch.zeros_like(within_mask, device=device)
    aux[torch.arange(Q, device=device), FI_min] = True
    min_mask = aux[within_mask]
    del aux

    # Q . 3 . 3
    face_vertices_min = vertices_per_face[FI_min]

    v0, v1, v2 = face_vertices_min.unbind(1)

    alpha_min = alpha_within[min_mask]
    beta_min = beta_within[min_mask]
    gamma_min = gamma_within[min_mask]

    # Point in 3D using barycentric weights
    p = alpha_min.unsqueeze(1) * v0 + beta_min.unsqueeze(1) * v1 + gamma_min.unsqueeze(1) * v2

    # Compute squared distances to each vertex
    d0 = (p - v0).square().sum(dim=1)
    d1 = (p - v1).square().sum(dim=1)
    d2 = (p - v2).square().sum(dim=1)

    # Q
    closest_idx = torch.stack([d0, d1, d2], dim=1).argmin(dim=1)
    closest_vertex = faces[FI_min, closest_idx]

    return closest_vertex





def euclid_loss_cpu(SMPL_indices, SMPL_coordinates, E_indices, E_coordinates, device):
    loss = 0
    for SMPL_position, SMPL_index in enumerate(SMPL_indices):
        candidate_E_positions = (E_indices == SMPL_index).nonzero()
        for candidate_E_position in candidate_E_positions:
            candidate_E_index = E_indices[candidate_E_position]
            candidates.cat(candidates, torch.sqrt(torch.square(E_coordinates[E_position, 0] - SMPL_coordinates[SMPL_position, 0]) + torch.square(E_coordinates[E_position, 1] - SMPL_coordinates[SMPL_position, 1])))
        best_candidate = candidates.amax()
        loss += best_candidate

    return loss




def euclid_loss_gpu(SMPL_indices, SMPL_coordinates, E_indices, E_coordinates, device):
    '''
    E_indices - E
    E_coordinates - E . 2
    SMPL_indices - S
    SMPL_coordinates - S . 2
    '''
    S = SMPL_indices.shape[0]
    E = E_indices.shape[0]

    # E . S mask
    index_matches = SMPL_indices[None, :] == E_indices[:, None]
    
    # M . 2
    SMPL_coordinates_matches = SMPL_coordinates[None, ...].expand(E, S, 2)[index_matches]
    E_coordinates_matches = E_coordinates[:, None, :].expand(E, S, 2)[index_matches]

    euclid = torch.sqrt(
        torch.square(E_coordinates_matches[:, 0] - SMPL_coordinates_matches[:, 0])
        + torch.square(E_coordinates_matches[:, 1] - SMPL_coordinates_matches[:, 1])
    )
    
    # E . S
    # could be changed to penalize being outside of bb but that should be penalized enough already
    plane = torch.full((E, S), torch.inf, device=device)
    plane[index_matches] = euclid

    # S
    best_candidates = plane.amin(dim=0)
    inf_mask = best_candidates.isinf()
    best_candidates = best_candidates[~inf_mask]
    # breakpoint()
    return best_candidates.sum() / best_candidates.shape[0]



def resize_with_padding(image, target_size, pad_color=(255, 255, 255)):
    if image.shape[:2] == target_size:
        return image

    target_h, target_w = target_size
    h, w = image.shape[:2]

    # Compute scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank canvas with padding color
    result = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)

    # Compute top-left corner to center the image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Paste resized image into result
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    return result




def populate_overview(*images, target_resolution, margin=0):
    """
    Arranges images into a grid with 2 images per row.
    All images must have the same resolution and type.
    """
    if len(images) == 0:
        raise ValueError("No images were provided.")
    
    img_h, img_w = target_resolution
    num_rows = (len(images) + 1) // 2
    num_cols = 2

    # Determine final image size
    overview_height = num_rows * img_h + (num_rows - 1) * margin + 2 * margin
    overview_width = num_cols * img_w + margin + 2 * margin

    # Allocate the output canvas
    overview = np.ones((overview_height, overview_width, 3), dtype=np.uint8) * 255  # white background

    for i, image in enumerate(images):
        # image = resize_with_padding(image, target_resolution)
        row = i // 2
        col = i % 2

        y_offset = row * (img_h + margin) + margin
        x_offset = col * (img_w + margin) + margin

        overview[y_offset:y_offset+img_h, x_offset:x_offset+img_w] = image

    return overview





    
def get_colormap_preview(smpl_object, original_image, device):
        vertices = smpl_output.vertices[0]
        vertices_idx = torch.arange(vertices.shape[0], device=device)
        projection = vertices @ K_flipped.T
        projected_points = projection[:, :2] / projection[:, 2, None]
        SMPL_coordinates_IMG = projected_points

        u, v = SMPL_coordinates_IMG.unbind(-1)
        
        v_vis = v.long().cpu().numpy()
        u_vis = u.long().cpu().numpy()
        
        vis = np.zeros_like(original_image)
        breakpoint()
        vis[v_vis, u_vis, 2] = 255
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(vis, (int(x + w * .5), int(y + h * .2)), 5, (255, 0, 0), -1)
        cv2.circle(vis, (int(x + w * .5), int(y + h * .8)), 5, (0, 255, 0), -1)

        colormap_preview = vis
        # mask = segmentation_bb_resolution.cpu().numpy()
        # temp = cv2.applyColorMap(color_map.take(embedding_points_bb_resolution).numpy(force=True), cv2.COLORMAP_JET)
