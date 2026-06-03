from typing import Union

from torchvision.transforms import GaussianBlur
import torch
import numpy as np


def gaussian_blur(img: Union[torch.Tensor, np.ndarray], kernel_size=(5, 5), sigma=1.5):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img = img.float()
    # Create a Gaussian kernel
    gaussian_kernel = GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    # Apply the Gaussian kernel to the image
    blurred_img = gaussian_kernel(img)

    return blurred_img.to(img.device)

def cloud2sideViews_torch_with_reference_cube(
    points: torch.Tensor,
    reference_points: torch.Tensor,
    resolution_xy: int,
    margin_ratio: float = 0.05,
) -> torch.Tensor:
    if resolution_xy is None:
        raise ValueError("resolution_xy must not be None")

    if reference_points.ndim != 2 or reference_points.shape[1] < 3:
        raise ValueError(
            f"reference_points must have shape (N, >=3), got {tuple(reference_points.shape)}"
        )

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {tuple(points.shape)}")

    if points.shape[0] == 0:
        return torch.zeros(
            (5, resolution_xy, resolution_xy),
            dtype=torch.float32,
            device=reference_points.device,
        )

    points = points[:, :3].to(
        device=reference_points.device,
        dtype=torch.float64,
    )

    reference_points = reference_points[:, :3].to(
        device=reference_points.device,
        dtype=torch.float64,
    )

    min_xyz = reference_points.min(dim=0).values
    max_xyz = reference_points.max(dim=0).values

    center = (min_xyz + max_xyz) / 2.0
    max_range = (max_xyz - min_xyz).max()
    cube_half = max_range / 2.0 * (1.0 + 2.0 * margin_ratio)

    cube_min = center - cube_half
    cube_max = center + cube_half

    def to_grid(val, min_val, max_val):
        return torch.clamp(
            ((val - min_val) / (max_val - min_val + 1e-8) * (resolution_xy - 1)).long(),
            0,
            resolution_xy - 1,
        )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    gx = to_grid(x, cube_min[0], cube_max[0])
    gy = to_grid(y, cube_min[1], cube_max[1])
    gz = to_grid(z, cube_min[2], cube_max[2])

    views = []

    def build_depth_map(indices_2d, distances, flip_y=False, flip_x=False):
        y_idx, x_idx = indices_2d

        if flip_y:
            y_idx = resolution_xy - 1 - y_idx

        if flip_x:
            x_idx = resolution_xy - 1 - x_idx

        flat_indices = y_idx * resolution_xy + x_idx

        depth_map = torch.full(
            (resolution_xy * resolution_xy,),
            float("inf"),
            dtype=torch.float64,
            device=distances.device,
        )

        depth_map = torch.scatter_reduce(
            depth_map,
            0,
            flat_indices,
            distances,
            reduce="amin",
            include_self=True,
        )

        img = depth_map.view(resolution_xy, resolution_xy)
        valid_mask = torch.isfinite(img)

        if torch.any(valid_mask):
            values = img[valid_mask]

            min_val = values.min()
            max_val = values.max()

            # Same convention as cloud2sideViews_torch:
            # smaller distance to viewer -> brighter.
            normalised = (max_val - values) / (max_val - min_val + 1e-8)
            normalised = normalised * (1.0 - 1.0 / 255.0) + (1.0 / 255.0)

            img = img.clone()
            img[valid_mask] = normalised
            img[~valid_mask] = 0.0
        else:
            img = torch.zeros_like(img)

        return img.to(dtype=torch.float32)

    dist_top = cube_max[2] - z
    views.append(build_depth_map((gy, gx), dist_top))

    dist_front = cube_max[1] - y
    views.append(build_depth_map((gz, gx), dist_front, flip_y=True))

    dist_back = y - cube_min[1]
    views.append(build_depth_map((gz, gx), dist_back, flip_y=True, flip_x=True))

    dist_left = cube_max[0] - x
    views.append(build_depth_map((gz, gy), dist_left, flip_y=True))

    dist_right = x - cube_min[0]
    views.append(build_depth_map((gz, gy), dist_right, flip_y=True, flip_x=True))

    return torch.stack(views, dim=0).to(dtype=torch.float32)

def cloud2sideViews_torch(points: torch.Tensor,
                       resolution_xy: int | None = None,
                       margin_ratio: float = 0.05) -> torch.Tensor:
 
        points = points.type(torch.float64)

        min_xyz = points.min(dim=0).values
        max_xyz = points.max(dim=0).values

        center = (min_xyz + max_xyz) / 2
        max_range = (max_xyz - min_xyz).max()
        cube_half = max_range / 2 * (1 + 2 * margin_ratio)

        cube_min = center - cube_half
        cube_max = center + cube_half

        def to_grid(val, min_val, max_val):
            return torch.clamp(
                ((val - min_val) / (max_val - min_val + 1e-8) * (resolution_xy - 1)).long(),
                0, resolution_xy - 1
            )

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        gx = to_grid(x, cube_min[0], cube_max[0])
        gy = to_grid(y, cube_min[1], cube_max[1])
        gz = to_grid(z, cube_min[2], cube_max[2])

        views = []

        def build_depth_map(indices_2d, distances, flip_y=False, flip_x=False):
            y_idx, x_idx = indices_2d
            if flip_y:
                y_idx = resolution_xy - 1 - y_idx
            if flip_x:
                x_idx = resolution_xy - 1 - x_idx

            flat_indices = y_idx * resolution_xy + x_idx
            depth_map = torch.full((resolution_xy * resolution_xy,), float('inf'),
                                   dtype=torch.float64, device=distances.device)
            depth_map = torch.scatter_reduce(depth_map, 0, flat_indices, distances,
                                             reduce='amin', include_self=True)

            img = depth_map.view(resolution_xy, resolution_xy)
            valid_mask = torch.isfinite(img)

            if torch.any(valid_mask):
                values = img[valid_mask]
                min_val = values.min()
                max_val = values.max()
                normalised = (max_val - values) / (max_val - min_val + 1e-8)
                normalised = normalised * (1.0 - 1.0 / 255.0) + (1.0 / 255.0)
                img = img.clone()
                img[valid_mask] = normalised
                img[~valid_mask] = 0.0
            else:
                img = torch.zeros_like(img)

            return img.type(torch.float32)

        dist_top = cube_max[2] - z
        views.append(build_depth_map((gy, gx), dist_top))

        dist_front = cube_max[1] - y
        views.append(build_depth_map((gz, gx), dist_front, flip_y=True))

        dist_back = y - cube_min[1]
        views.append(build_depth_map((gz, gx), dist_back, flip_y=True, flip_x=True))

        dist_left = cube_max[0] - x
        views.append(build_depth_map((gz, gy), dist_left, flip_y=True))

        dist_right = x - cube_min[0]
        views.append(build_depth_map((gz, gy), dist_right, flip_y=True, flip_x=True))

        return torch.stack(views, dim=0).type(torch.float32)

import torch


def voxel_tree_skeleton_torch(
    xyz: torch.Tensor,
    voxel_size: float = 1.5,
    connect_radius: float | None = None,
    min_branch_length: float = 4.0,
    simplify_spacing: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sparse tree-like skeleton from a point cloud.

    Torch version. Returns tensors on the same device as xyz.

    Returns
    -------
    points:
        Skeleton points, shape (M, 3), same dtype/device as xyz.
    edges:
        Skeleton edges, shape (E, 2), dtype torch.long, same device as xyz.
    """
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"xyz must have shape (N, >=3), got {tuple(xyz.shape)}")

    device = xyz.device
    dtype = xyz.dtype

    xyz = xyz[:, :3]

    if xyz.shape[0] == 0:
        return (
            torch.empty((0, 3), dtype=dtype, device=device),
            torch.empty((0, 2), dtype=torch.long, device=device),
        )

    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")

    if connect_radius is None:
        connect_radius = 3.0 * voxel_size

    if connect_radius <= 0.0:
        raise ValueError("connect_radius must be positive")

    if min_branch_length < 0.0:
        raise ValueError("min_branch_length must be non-negative")

    if simplify_spacing < 0.0:
        raise ValueError("simplify_spacing must be non-negative")

    points, voxel_keys = _voxel_means_torch(xyz, voxel_size)

    if points.shape[0] <= 1:
        return points, torch.empty((0, 2), dtype=torch.long, device=device)

    candidate_edges, candidate_d2 = _radius_edges_from_voxels_torch(
        points=points,
        voxel_keys=voxel_keys,
        voxel_size=voxel_size,
        connect_radius=connect_radius,
    )

    if candidate_edges.shape[0] == 0:
        return points, torch.empty((0, 2), dtype=torch.long, device=device)

    edges = _minimum_spanning_forest_torch(
        edges=candidate_edges,
        d2=candidate_d2,
        node_count=points.shape[0],
    )

    if edges.shape[0] == 0:
        return points, edges

    if min_branch_length > 0.0:
        edges = _prune_short_leaf_branches_torch(
            points=points,
            edges=edges,
            min_branch_length=min_branch_length,
        )

    if edges.shape[0] == 0:
        return points, edges

    if simplify_spacing > 0.0:
        points, edges = _simplify_tree_chains_torch(
            points=points,
            edges=edges,
            spacing=simplify_spacing,
        )

    return points, edges

def _half_neighbor_offsets_torch(
    radius_voxels: int,
    voxel_size: float,
    radius2: float,
) -> list[tuple[int, int, int]]:
    offsets = []

    for dx in range(-radius_voxels, radius_voxels + 1):
        for dy in range(-radius_voxels, radius_voxels + 1):
            for dz in range(-radius_voxels, radius_voxels + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                if (dx, dy, dz) <= (0, 0, 0):
                    continue

                offset_d2 = (dx * dx + dy * dy + dz * dz) * voxel_size * voxel_size

                if offset_d2 > radius2:
                    continue

                offsets.append((dx, dy, dz))

    return offsets

def _voxel_means_torch(
    xyz: torch.Tensor,
    voxel_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys_abs = torch.floor(xyz / voxel_size).to(dtype=torch.long)

    keys_min = keys_abs.amin(dim=0)
    keys = keys_abs - keys_min

    key_range = keys.amax(dim=0) + 1
    rx = int(key_range[0].item())
    ry = int(key_range[1].item())
    rz = int(key_range[2].item())

    if rx * ry * rz >= torch.iinfo(torch.long).max:
        raise OverflowError("voxel key encoding overflow")

    sx = ry * rz
    sy = rz

    encoded = keys[:, 0] * sx + keys[:, 1] * sy + keys[:, 2]

    order = torch.argsort(encoded, stable=True)
    encoded_sorted = encoded[order]
    xyz_sorted = xyz[order]
    keys_sorted = keys[order]

    _, inverse, counts = torch.unique_consecutive(
        encoded_sorted,
        return_inverse=True,
        return_counts=True,
    )

    sums = torch.zeros(
        (counts.shape[0], 3),
        dtype=xyz.dtype,
        device=xyz.device,
    )

    sums.scatter_add_(
        dim=0,
        index=inverse[:, None].expand(-1, 3),
        src=xyz_sorted,
    )

    means = sums / counts.to(dtype=xyz.dtype)[:, None]

    first = torch.cumsum(counts, dim=0) - counts
    voxel_keys = keys_sorted[first]

    return means, voxel_keys

def _radius_edges_from_voxels_torch(
    points: torch.Tensor,
    voxel_keys: torch.Tensor,
    voxel_size: float,
    connect_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_count = points.shape[0]
    device = points.device

    key_range = voxel_keys.amax(dim=0) + 1
    rx = int(key_range[0].item())
    ry = int(key_range[1].item())
    rz = int(key_range[2].item())

    sx = ry * rz
    sy = rz

    encoded = voxel_keys[:, 0] * sx + voxel_keys[:, 1] * sy + voxel_keys[:, 2]

    order = torch.argsort(encoded, stable=True)
    encoded_sorted = encoded[order]
    points_sorted = points[order]
    voxel_keys_sorted = voxel_keys[order]

    sorted_to_original = order

    radius_voxels = int(torch.ceil(torch.tensor(connect_radius / voxel_size)).item())
    radius2 = connect_radius * connect_radius

    offsets = _half_neighbor_offsets_torch(
        radius_voxels=radius_voxels,
        voxel_size=voxel_size,
        radius2=radius2,
    )

    edges_parts = []
    d2_parts = []

    for dx, dy, dz in offsets:
        valid = (
            (voxel_keys_sorted[:, 0] + dx >= 0)
            & (voxel_keys_sorted[:, 0] + dx < rx)
            & (voxel_keys_sorted[:, 1] + dy >= 0)
            & (voxel_keys_sorted[:, 1] + dy < ry)
            & (voxel_keys_sorted[:, 2] + dz >= 0)
            & (voxel_keys_sorted[:, 2] + dz < rz)
        )

        src = torch.nonzero(valid, as_tuple=False).flatten()

        if src.numel() == 0:
            continue

        delta = dx * sx + dy * sy + dz
        target_encoded = encoded_sorted[src] + delta

        dst = torch.searchsorted(encoded_sorted, target_encoded)

        inside = dst < node_count

        if not torch.any(inside):
            continue

        src = src[inside]
        dst = dst[inside]
        target_encoded = target_encoded[inside]

        hit = encoded_sorted[dst] == target_encoded

        if not torch.any(hit):
            continue

        src = src[hit]
        dst = dst[hit]

        diff = points_sorted[src] - points_sorted[dst]
        dist2 = torch.sum(diff * diff, dim=1)

        close = dist2 <= radius2

        if not torch.any(close):
            continue

        src_original = sorted_to_original[src[close]]
        dst_original = sorted_to_original[dst[close]]

        edges_parts.append(torch.stack((src_original, dst_original), dim=1))
        d2_parts.append(dist2[close])

    if not edges_parts:
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.empty((0,), dtype=points.dtype, device=device),
        )

    return torch.cat(edges_parts, dim=0), torch.cat(d2_parts, dim=0)


def _sample_skeleton_edges_torch(
    points: torch.Tensor,
    edges: torch.Tensor,
    spacing: float,
) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {tuple(points.shape)}")

    points = points[:, :3]

    if points.shape[0] == 0:
        return points

    if edges.numel() == 0:
        return points

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (E, 2), got {tuple(edges.shape)}")

    valid = (
        (edges[:, 0] >= 0)
        & (edges[:, 0] < points.shape[0])
        & (edges[:, 1] >= 0)
        & (edges[:, 1] < points.shape[0])
    )

    edges = edges[valid]

    if edges.shape[0] == 0:
        return points

    if spacing <= 0.0:
        return points

    sampled_parts = [points]

    p0 = points[edges[:, 0]]
    p1 = points[edges[:, 1]]

    lengths = torch.linalg.norm(p1 - p0, dim=1)
    counts = torch.ceil(lengths / spacing).long() + 1
    counts = torch.clamp(counts, min=2)

    for a, b, count in zip(p0, p1, counts):
        t = torch.linspace(
            0.0,
            1.0,
            int(count.item()),
            device=points.device,
            dtype=points.dtype,
        )

        segment_points = a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None]
        sampled_parts.append(segment_points)

    return torch.cat(sampled_parts, dim=0)

def _minimum_spanning_forest_torch(
    edges: torch.Tensor,
    d2: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    if edges.shape[0] == 0:
        return edges

    order = torch.argsort(d2, stable=True)

    parent = list(range(node_count))
    rank = [0] * node_count
    keep = torch.zeros(edges.shape[0], dtype=torch.bool, device=edges.device)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    for edge_id_tensor in order:
        edge_id = int(edge_id_tensor.item())

        a = int(edges[edge_id, 0].item())
        b = int(edges[edge_id, 1].item())

        root_a = find(a)
        root_b = find(b)

        if root_a == root_b:
            continue

        keep[edge_id] = True

        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    return edges[keep]

def _prune_short_leaf_branches_torch(
    points: torch.Tensor,
    edges: torch.Tensor,
    min_branch_length: float,
) -> torch.Tensor:
    node_count = points.shape[0]
    adjacency = _build_adjacency_torch(edges, node_count)

    alive = [True] * node_count

    changed = True

    while changed:
        changed = False

        degree = _alive_degrees_torch(adjacency, alive)
        leaves = [idx for idx, deg in enumerate(degree) if deg == 1]

        for leaf in leaves:
            if not alive[leaf]:
                continue

            path = [leaf]
            length = 0.0

            prev = -1
            cur = leaf

            while True:
                next_nodes = [
                    n for n in adjacency[cur]
                    if alive[n] and n != prev
                ]

                if not next_nodes:
                    break

                nxt = next_nodes[0]

                length += float(torch.linalg.norm(points[cur] - points[nxt]).item())

                prev = cur
                cur = nxt
                path.append(cur)

                cur_degree = 0

                for n in adjacency[cur]:
                    if alive[n]:
                        cur_degree += 1

                if cur_degree != 2:
                    break

                if length >= min_branch_length:
                    break

            if length < min_branch_length:
                for node in path[:-1]:
                    alive[node] = False

                changed = True

    alive_tensor = torch.tensor(alive, dtype=torch.bool, device=edges.device)
    keep_edge = alive_tensor[edges[:, 0]] & alive_tensor[edges[:, 1]]

    return edges[keep_edge]

def _simplify_tree_chains_torch(
    points: torch.Tensor,
    edges: torch.Tensor,
    spacing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_count = points.shape[0]
    adjacency = _build_adjacency_torch(edges, node_count)

    degree = [len(neigh) for neigh in adjacency]
    important = [deg != 2 for deg in degree]

    if not any(important):
        important[0] = True

    visited_edges = set()

    out_points = []
    out_edges = []
    old_to_new = {}

    def get_output_node(old_idx: int) -> int:
        new_idx = old_to_new.get(old_idx)

        if new_idx is not None:
            return new_idx

        new_idx = len(out_points)
        old_to_new[old_idx] = new_idx
        out_points.append(points[old_idx])

        return new_idx

    for start in range(node_count):
        if not important[start]:
            continue

        for nxt in adjacency[start]:
            edge_id = _edge_key_torch(start, nxt)

            if edge_id in visited_edges:
                continue

            chain = [start, nxt]
            visited_edges.add(edge_id)

            prev = start
            cur = nxt

            while not important[cur]:
                candidates = [n for n in adjacency[cur] if n != prev]

                if not candidates:
                    break

                nxt2 = candidates[0]
                visited_edges.add(_edge_key_torch(cur, nxt2))

                prev = cur
                cur = nxt2
                chain.append(cur)

            simplified_chain = _resample_chain_by_spacing_torch(
                points=points,
                chain=chain,
                spacing=spacing,
            )

            for a, b in zip(simplified_chain[:-1], simplified_chain[1:]):
                out_a = get_output_node(a)
                out_b = get_output_node(b)

                if out_a != out_b:
                    out_edges.append((out_a, out_b))

    if not out_points:
        return (
            torch.empty((0, 3), dtype=points.dtype, device=points.device),
            torch.empty((0, 2), dtype=torch.long, device=points.device),
        )

    out_points_tensor = torch.stack(out_points, dim=0)

    if out_edges:
        out_edges_tensor = torch.tensor(
            out_edges,
            dtype=torch.long,
            device=points.device,
        )
    else:
        out_edges_tensor = torch.empty(
            (0, 2),
            dtype=torch.long,
            device=points.device,
        )

    return out_points_tensor, out_edges_tensor

def _resample_chain_by_spacing_torch(
    points: torch.Tensor,
    chain: list[int],
    spacing: float,
) -> list[int]:
    if len(chain) <= 2 or spacing <= 0.0:
        return chain

    kept = [chain[0]]
    accumulated = 0.0

    for i in range(1, len(chain) - 1):
        prev = chain[i - 1]
        cur = chain[i]

        accumulated += float(torch.linalg.norm(points[prev] - points[cur]).item())

        if accumulated >= spacing:
            kept.append(cur)
            accumulated = 0.0

    if kept[-1] != chain[-1]:
        kept.append(chain[-1])

    return kept

def _build_adjacency_torch(
    edges: torch.Tensor,
    node_count: int,
) -> list[list[int]]:
    adjacency = [[] for _ in range(node_count)]

    for edge in edges:
        a = int(edge[0].item())
        b = int(edge[1].item())

        adjacency[a].append(b)
        adjacency[b].append(a)

    return adjacency

def _alive_degrees_torch(
    adjacency: list[list[int]],
    alive: list[bool],
) -> list[int]:
    degree = [0] * len(adjacency)

    for i, neighbors in enumerate(adjacency):
        if not alive[i]:
            continue

        count = 0

        for n in neighbors:
            if alive[n]:
                count += 1

        degree[i] = count

    return degree

def _edge_key_torch(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)

    return (a, b) if a < b else (b, a)