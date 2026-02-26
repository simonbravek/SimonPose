from collections import defaultdict

import numpy as np
import torch


def tiled_rasterize_depth(
    vertices3d: torch.Tensor,        # (N, 3)
    vertices2d: torch.Tensor,        # (N, 2)
    faces: torch.Tensor,             # (F, 3), on same device
    image_size: tuple[int, int],
    *,
    min_depth: float = 0.0,
    tile_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-tiled rasterizer for depth with closest-vertex map.
    """
    device = vertices3d.device
    H, W = image_size
    F = faces.shape[0]

    # triangle corners
    tri_verts2d = vertices2d[faces]           # (F, 3, 2)
    tri_verts_z = vertices3d[:, 2][faces]     # (F, 3)

    x = tri_verts2d[..., 0]
    y = tri_verts2d[..., 1]

    minx = x.amin(dim=1).floor().clamp(0, W - 1).int()
    maxx = x.amax(dim=1).ceil().clamp(0, W - 1).int()
    miny = y.amin(dim=1).floor().clamp(0, H - 1).int()
    maxy = y.amax(dim=1).ceil().clamp(0, H - 1).int()

    # bin triangles to tiles
    tile_minx = (minx // tile_size).int()
    tile_maxx = (maxx // tile_size).int()
    tile_miny = (miny // tile_size).int()
    tile_maxy = (maxy // tile_size).int()

    tiles_x = (W + tile_size - 1) // tile_size  # ceil
    tiles_y = (H + tile_size - 1) // tile_size  # ceil

    # build tile/triangle occupancy mask
    tile_map = torch.empty((tiles_y, tiles_x, F), dtype=torch.bool, device=device)
    for ty in range(tiles_y):
        ymask = (tile_miny <= ty) & (ty <= tile_maxy)
        for tx in range(tiles_x):
            mask = ymask & (tile_minx <= tx) & (tx <= tile_maxx)
            tile_map[ty, tx] = mask

    # global maps
    depth_map = torch.full((H, W), float("inf"), device=device)
    vertex_index_map = torch.full((H, W), -1, dtype=torch.long, device=device)

    V0 = tri_verts2d[:, 1] - tri_verts2d[:, 0]  # (F, 2)
    V1 = tri_verts2d[:, 2] - tri_verts2d[:, 0]  # (F, 2)
    D00 = (V0 * V0).sum(dim=1)
    D01 = (V0 * V1).sum(dim=1)
    D11 = (V1 * V1).sum(dim=1)
    Denom = D00 * D11 - D01 * D01

    arng = torch.arange(F, device=device)

    ignore_block = (~tile_map.any(dim=2)).tolist()  # sync!

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            if ignore_block[ty][tx]:
                continue
            tile_mask = tile_map[ty, tx]
            tri_indices = arng[tile_mask]

            # pull out triangle data in a batched way
            p0 = tri_verts2d[tri_indices, 0]
            p1 = tri_verts2d[tri_indices, 1]
            #p2 = tri_verts2d[tri_indices, 2]

            # barycentric precomputations
            #v0 = p1 - p0            # (T,2)
            #v1 = p2 - p0
            v0 = V0[tri_indices]
            v1 = V1[tri_indices]
            d00 = D00[tri_indices]
            d01 = D01[tri_indices]
            d11 = D11[tri_indices]
            denom = Denom[tri_indices]
            safe = denom > 1e-6

            # pixel coordinates for this tile
            y0 = ty * tile_size
            y1 = min((ty+1)*tile_size, H)
            x0 = tx * tile_size
            x1 = min((tx+1)*tile_size, W)

            yy, xx = torch.meshgrid(
                torch.arange(y0,y1,device=device),
                torch.arange(x0,x1,device=device),
                indexing="ij"
            )
            pts = torch.stack([xx, yy], dim=-1).reshape(-1, 2)   # (P,2)

            # all triangle tests at once
            v2 = pts[:, None] - p0[None, :]  # (P,T,2)
            d20 = (v2 * v0[None]).sum(dim=2)
            d21 = (v2 * v1[None]).sum(dim=2)

            v = (d11[None] * d20 - d01[None] * d21) / denom[None].clamp(1e-6)
            w = (d00[None] * d21 - d01[None] * d20) / denom[None].clamp(1e-6)
            u = 1.0 - v - w

            bc = torch.stack([u,v,w], dim=2)   # (P,T,3)
            inside = (bc >= 0).all(dim=2)       # (P,T)
            inside &= safe[None]

            # depth per pixel per triangle
            z = (bc * tri_verts_z[tri_indices][None]).sum(dim=2)

            # invalidate outside
            z[~inside] = float("inf")
            z[z < min_depth] = float("inf")

            # find best triangle per pixel
            z_min, tri_idx_min = z.min(dim=1)    # (P,)

            local_depth = depth_map[y0:y1,x0:x1].clone().view(-1)
            local_idxmap = vertex_index_map[y0:y1,x0:x1].clone().view(-1)

            update_mask = z_min < local_depth
            local_depth[update_mask] = z_min[update_mask]

            winner_triangles = tri_indices[tri_idx_min[update_mask]]
            winner_bc        = bc[update_mask, tri_idx_min[update_mask]]
            winner_local     = winner_bc.argmax(dim=1)
            winner_vertex    = faces[winner_triangles, winner_local]

            local_idxmap[update_mask] = winner_vertex

            # store
            depth_map[y0:y1,x0:x1] = local_depth.view(y1-y0, x1-x0)
            vertex_index_map[y0:y1,x0:x1] = local_idxmap.view(y1-y0, x1-x0)

    return depth_map, vertex_index_map


def compute_depth_map_vect(vertices_3d, vertices_2d, faces, img_size, *, min_depth=0):
    H, W = img_size
    depth_map = np.full((H, W), np.inf)
    vertex_map = np.full((H, W), -1, dtype=int)

    # Get the 3D coordinates of the triangle's vertices
    V = vertices_3d[faces.flat].reshape(-1, 3, 3)
    # Get the corresponding 2D projected positions (in pixel space)
    P = vertices_2d[faces.flat].reshape(-1, 3, 2)

    # Compute the bounding box of the triangle
    x_min = P[..., 0].min(-1).astype(np.int32).clip(0, W)
    y_min = P[..., 1].min(-1).astype(np.int32).clip(0, H)
    x_max = np.ceil(P[..., 0].max(-1)).astype(np.int32).clip(0, W)
    y_max = np.ceil(P[..., 1].max(-1)).astype(np.int32).clip(0, H)

    for i in range(faces.shape[0]):
        if y_min[i] >= y_max[i] or x_min[i] >= x_max[i]:
            continue

        x0, y0, x1, y1, x2, y2 = P[i].flat
        #x0, y0 = P[i, 0]
        #x1, y1 = P[i, 1]
        #x2, y2 = P[i, 2]

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-6:
            continue

        x_slice = slice(x_min[i], x_max[i])
        y_slice = slice(y_min[i], y_max[i])
        y, x = np.mgrid[y_slice, x_slice].reshape(2, -1)

        alpha = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
        beta = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
        gamma = 1 - alpha - beta

        within = (0 <= alpha) & (alpha <= 1) \
            & (0 <= beta) & (beta <= 1) \
            & (0 <= gamma) & (gamma <= 1)
        if not np.any(within):
            continue

        alpha = alpha[within]
        beta = beta[within]
        gamma = gamma[within]
        xi = x[within]
        yi = y[within]
        z_interpolated = alpha * V[i, 0, 2] \
            + beta * V[i, 1, 2] \
            + gamma * V[i, 2, 2]

        update = (min_depth < z_interpolated) & (z_interpolated < depth_map[yi, xi])
        depth_map[yi[update], xi[update]] = z_interpolated[update]
        vertex_map[yi[update], xi[update]] = faces[
            i,
            np.argmax(np.stack([alpha[update], beta[update], gamma[update]]), 0)
        ]

    return depth_map, vertex_map


def compute_depth_map(vertices_3d, vertices_2d, faces, img_size, *, min_depth=0):
    """
    Compute a depth map from a given mesh using barycentric rasterization.

    Parameters:
    - vertices_3d: (N, 3) array of 3D vertex coordinates
    - vertices_2d: (N, 2) array of 2D projected pixel coordinates
    - faces: (M, 3) array of triangle face indices
    - img_size: (H, W) tuple defining the output image size

    Returns:
    - depth_map: (H, W) depth buffer storing per-pixel depth values
    """
    H, W = img_size
    depth_map = np.full((H, W), np.inf)  # Initialize depth buffer with 'inf'

    for face in faces:
        # Get the 3D coordinates of the triangle's vertices
        v0, v1, v2 = vertices_3d[face]

        # Get the corresponding 2D projected positions (in pixel space)
        p0, p1, p2 = vertices_2d[face].astype(np.int32)  # Convert to integer pixel positions

        # Compute the bounding box of the triangle
        x_min = max(0, min(p0[0], p1[0], p2[0]))
        x_max = min(W - 1, max(p0[0], p1[0], p2[0]))
        y_min = max(0, min(p0[1], p1[1], p2[1]))
        y_max = min(H - 1, max(p0[1], p1[1], p2[1]))

        # Iterate over all pixels within the bounding box
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                # Compute barycentric coordinates for the pixel (x, y)
                bcoords = compute_barycentric_coords(p0, p1, p2, (x, y))

                if bcoords is not None:  # If inside the triangle
                    alpha, beta, gamma = bcoords
                    # Interpolate depth using barycentric coordinates
                    z_interpolated = alpha * v0[2] + beta * v1[2] + gamma * v2[2]

                    # Depth test: Only update if the new depth is closer
                    if min_depth < z_interpolated < depth_map[y, x]:
                        depth_map[y, x] = z_interpolated

    return depth_map

# used to compute mask
def compute_depth_for_pixels_vect(vertices_3d, vertices_2d, faces, query_pixels, *, min_depth=0):
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

    for i, pair in enumerate(query_pixels):
        x, y = pair

        mask = (x_min <= x) & (x <= x_max) & (y_min <= y) & (y <= y_max)
        if not np.any(mask):
            #breakpoint()
            continue
        Pmask = P[mask]

        # from compute_barycentric_coords
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

        nonzero, = np.nonzero(mask)
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
        face_idx = nonzero[argmin]
        idx = np.argmax([alpha[argmin], beta[argmin], gamma[argmin]])
        result[i] = faces[face_idx, idx]

    return result


def compute_depth_for_pixels(vertices_3d, vertices_2d, faces, query_pixels, *, min_depth=0):
    """
    Computes depth and closest vertex index only for the given pixels.

    Parameters:
    - vertices_3d: (N, 3) array of 3D vertex coordinates.
    - vertices_2d: (N, 2) array of 2D projected pixel coordinates.
    - faces: (M, 3) array of triangle face indices.
    - query_pixels: List of (x, y) pixel coordinates to compute depth for.

    Returns:
    - results: Dict mapping (x, y) -> closest_vertex_index
    """
    results = {}
    depth = defaultdict(lambda: float("inf"))

    for face in faces:
        # Get triangle's 3D and 2D vertices
        v0, v1, v2 = vertices_3d[face]
        p0, p1, p2 = vertices_2d[face]  # .astype(np.int32)

        # Compute bounding box of triangle
        x_min, x_max = min(p0[0], p1[0], p2[0]), max(p0[0], p1[0], p2[0])
        y_min, y_max = min(p0[1], p1[1], p2[1]), max(p0[1], p1[1], p2[1])

        # Filter query pixels that are inside the bounding box
        for pair in query_pixels:
            (x, y) = pair
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue  # Skip pixels outside bounding box

            # Compute barycentric coordinates
            bcoords = compute_barycentric_coords(p0, p1, p2, pair)
            if bcoords is None:
                continue  # Skip pixels outside the triangle

            alpha, beta, gamma = bcoords
            z_interpolated = alpha * v0[2] + beta * v1[2] + gamma * v2[2]

            if min_depth < z_interpolated < depth[pair]:
                # Find the closest vertex (highest barycentric weight)
                if alpha > beta and alpha > gamma:
                    closest_vertex = face[0]
                elif beta > gamma:
                    closest_vertex = face[1]
                else:
                    closest_vertex = face[2]

                depth[pair] = z_interpolated
                results[pair] = closest_vertex

    return results


def compute_barycentric_coords(p0, p1, p2, p):
    """
    Compute barycentric coordinates for point p within triangle (p0, p1, p2).
    """
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    x, y = p

    # Compute determinant
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-6:
        return None  # Degenerate triangle

    # Compute barycentric coordinates
    alpha = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
    beta = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
    gamma = 1 - alpha - beta

    # Return only if inside triangle
    if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
        return alpha, beta, gamma
    return None  # Outside triangle

