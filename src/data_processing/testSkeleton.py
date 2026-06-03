import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def voxel_tree_skeleton_numpy(
    xyz: np.ndarray,
    voxel_size: float = 1.5,
    connect_radius: float | None = None,
    min_branch_length: float = 4.0,
    simplify_spacing: float = 2.5,
):
    """
    Sparse tree-like skeleton from a point cloud.

    NumPy only.

    Pipeline:
        1. Voxelize point cloud using voxel means.
        2. Build local Euclidean candidate edges.
        3. Compute a geometric minimum spanning forest.
        4. Prune short dangling branches.
        5. Simplify degree-2 chains.

    Parameters
    ----------
    xyz:
        Point cloud, shape (N, 3) or (N, >=3).
    voxel_size:
        Voxel size for sparse representative points.
    connect_radius:
        Maximum candidate edge distance.
        If None, uses 3.0 * voxel_size.
    min_branch_length:
        Dangling leaf branches shorter than this are removed.
        Set to 0.0 to disable pruning.
    simplify_spacing:
        Approximate spacing between kept points along simple chains.
        Set to 0.0 to disable simplification.

    Returns
    -------
    points:
        Skeleton points, shape (M, 3).
    edges:
        Skeleton edges, shape (E, 2), dtype int64.
    """
    xyz = np.asarray(xyz)

    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"xyz must have shape (N, >=3), got {xyz.shape}")

    xyz = xyz[:, :3]

    if xyz.shape[0] == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.int64),
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

    points, voxel_keys = _voxel_means(xyz, voxel_size)

    if points.shape[0] <= 1:
        return points, np.empty((0, 2), dtype=np.int64)

    candidate_edges, candidate_d2 = _radius_edges_from_voxels(
        points=points,
        voxel_keys=voxel_keys,
        voxel_size=voxel_size,
        connect_radius=connect_radius,
    )

    if candidate_edges.shape[0] == 0:
        return points, np.empty((0, 2), dtype=np.int64)

    edges = _minimum_spanning_forest(
        edges=candidate_edges,
        d2=candidate_d2,
        node_count=points.shape[0],
    )

    if edges.shape[0] == 0:
        return points, edges

    if min_branch_length > 0.0:
        edges = _prune_short_leaf_branches(
            points=points,
            edges=edges,
            min_branch_length=min_branch_length,
        )

    if edges.shape[0] == 0:
        return points, edges

    if simplify_spacing > 0.0:
        points, edges = _simplify_tree_chains(
            points=points,
            edges=edges,
            spacing=simplify_spacing,
        )

    return points, edges


def plot_skeleton_matplotlib(
    points: np.ndarray,
    edges: np.ndarray,
    point_size: float = 6.0,
    edge_width: float = 1.2,
    show_points: bool = True,
    ax=None,
):
    """
    Plot skeleton points and edges in matplotlib.

    Uses Line3DCollection, which is much faster than plotting edges one by one.
    """
    points = np.asarray(points)
    edges = np.asarray(edges)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (E, 2), got {edges.shape}")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if edges.shape[0] > 0:
        segments = points[edges]
        collection = Line3DCollection(
            segments,
            linewidths=edge_width,
            alpha=0.85,
        )
        ax.add_collection3d(collection)

    if show_points and points.shape[0] > 0:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=point_size,
            depthshade=False,
        )

    _set_axes_equal_3d(ax, points)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig, ax


def _voxel_means(xyz: np.ndarray, voxel_size: float):
    """
    Return one representative point per occupied voxel.

    The representative is the mean of real points inside the voxel, not the
    geometric center of the voxel cell. This avoids hard grid/lattice artifacts.
    """
    keys_abs = np.floor(xyz / voxel_size).astype(np.int64)

    keys_min = keys_abs.min(axis=0)
    keys = keys_abs - keys_min

    key_range = keys.max(axis=0) + 1
    rx, ry, rz = map(int, key_range)

    if rx * ry * rz >= np.iinfo(np.int64).max:
        raise OverflowError("voxel key encoding overflow")

    sx = ry * rz
    sy = rz

    encoded = keys[:, 0] * sx + keys[:, 1] * sy + keys[:, 2]

    order = np.argsort(encoded, kind="mergesort")
    encoded_sorted = encoded[order]
    xyz_sorted = xyz[order]
    keys_sorted = keys[order]

    _, first, counts = np.unique(
        encoded_sorted,
        return_index=True,
        return_counts=True,
    )

    sums = np.add.reduceat(xyz_sorted, first, axis=0)
    means = sums / counts[:, None]
    voxel_keys = keys_sorted[first]

    return means.astype(xyz.dtype, copy=False), voxel_keys


def _radius_edges_from_voxels(
    points: np.ndarray,
    voxel_keys: np.ndarray,
    voxel_size: float,
    connect_radius: float,
):
    """
    Build local Euclidean candidate edges using voxel hashing.

    Each occupied voxel has one point, so candidate lookup can be done by
    encoded voxel key search instead of kNN.
    """
    node_count = points.shape[0]

    key_range = voxel_keys.max(axis=0) + 1
    rx, ry, rz = map(int, key_range)

    sx = ry * rz
    sy = rz

    encoded = voxel_keys[:, 0] * sx + voxel_keys[:, 1] * sy + voxel_keys[:, 2]

    order = np.argsort(encoded, kind="mergesort")
    encoded_sorted = encoded[order]
    points_sorted = points[order]
    voxel_keys_sorted = voxel_keys[order]

    sorted_to_original = order

    radius_voxels = int(np.ceil(connect_radius / voxel_size))
    radius2 = connect_radius * connect_radius

    offsets = _half_neighbor_offsets(radius_voxels, voxel_size, radius2)

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

        src = np.flatnonzero(valid)
        if src.size == 0:
            continue

        delta = dx * sx + dy * sy + dz
        target_encoded = encoded_sorted[src] + delta

        dst = np.searchsorted(encoded_sorted, target_encoded)

        inside = dst < node_count
        if not np.any(inside):
            continue

        src = src[inside]
        dst = dst[inside]
        target_encoded = target_encoded[inside]

        hit = encoded_sorted[dst] == target_encoded
        if not np.any(hit):
            continue

        src = src[hit]
        dst = dst[hit]

        diff = points_sorted[src] - points_sorted[dst]
        dist2 = np.einsum("ij,ij->i", diff, diff)

        close = dist2 <= radius2
        if not np.any(close):
            continue

        src_original = sorted_to_original[src[close]]
        dst_original = sorted_to_original[dst[close]]

        edges_parts.append(np.column_stack((src_original, dst_original)))
        d2_parts.append(dist2[close])

    if not edges_parts:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=points.dtype),
        )

    edges = np.vstack(edges_parts).astype(np.int64, copy=False)
    d2 = np.concatenate(d2_parts)

    return edges, d2


def _half_neighbor_offsets(
    radius_voxels: int,
    voxel_size: float,
    radius2: float,
):
    offsets = []

    for dx in range(-radius_voxels, radius_voxels + 1):
        for dy in range(-radius_voxels, radius_voxels + 1):
            for dz in range(-radius_voxels, radius_voxels + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                # Only one half of the symmetric neighborhood.
                # This avoids duplicate undirected candidate edges.
                if (dx, dy, dz) <= (0, 0, 0):
                    continue

                # Conservative offset rejection.
                # Exact rejection is still done using Euclidean point distance.
                offset_d2 = (dx * dx + dy * dy + dz * dz) * voxel_size * voxel_size
                if offset_d2 > radius2:
                    continue

                offsets.append((dx, dy, dz))

    return offsets


def _minimum_spanning_forest(
    edges: np.ndarray,
    d2: np.ndarray,
    node_count: int,
):
    """
    Kruskal minimum spanning forest.

    This removes local overconnections and cycles while keeping short geometric
    links. If the candidate graph has multiple disconnected components, the
    result is one tree per component.
    """
    order = np.argsort(d2, kind="mergesort")

    parent = np.arange(node_count, dtype=np.int64)
    rank = np.zeros(node_count, dtype=np.uint8)
    keep = np.zeros(edges.shape[0], dtype=bool)

    def find(x: int):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for edge_id in order:
        a = int(edges[edge_id, 0])
        b = int(edges[edge_id, 1])

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


def _prune_short_leaf_branches(
    points: np.ndarray,
    edges: np.ndarray,
    min_branch_length: float,
):
    """
    Remove dangling chains shorter than min_branch_length.

    This removes small noisy twigs while preserving junctions.
    """
    node_count = points.shape[0]
    adjacency = _build_adjacency(edges, node_count)
    alive = np.ones(node_count, dtype=bool)

    changed = True
    while changed:
        changed = False

        degree = _alive_degrees(adjacency, alive)
        leaves = np.flatnonzero(degree == 1)

        for leaf in leaves:
            if not alive[leaf]:
                continue

            path = [int(leaf)]
            length = 0.0

            prev = -1
            cur = int(leaf)

            while True:
                next_nodes = [
                    n for n in adjacency[cur]
                    if alive[n] and n != prev
                ]

                if not next_nodes:
                    break

                nxt = int(next_nodes[0])
                length += float(np.linalg.norm(points[cur] - points[nxt]))

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
                # Keep final node. It is usually the junction or continuation.
                for node in path[:-1]:
                    alive[node] = False
                changed = True

    keep_edge = alive[edges[:, 0]] & alive[edges[:, 1]]
    return edges[keep_edge]


def _simplify_tree_chains(
    points: np.ndarray,
    edges: np.ndarray,
    spacing: float,
):
    """
    Collapse simple degree-2 chains into sparse polylines.

    Junctions and leaves are preserved. Intermediate chain points are kept only
    every roughly `spacing` distance.
    """
    node_count = points.shape[0]
    adjacency = _build_adjacency(edges, node_count)

    degree = np.array([len(neigh) for neigh in adjacency], dtype=np.int32)
    important = degree != 2

    if not np.any(important):
        important[0] = True

    visited_edges = set()

    out_points = []
    out_edges = []
    old_to_new = {}

    def get_output_node(old_idx: int):
        old_idx = int(old_idx)

        new_idx = old_to_new.get(old_idx)
        if new_idx is not None:
            return new_idx

        new_idx = len(out_points)
        old_to_new[old_idx] = new_idx
        out_points.append(points[old_idx])
        return new_idx

    for start in np.flatnonzero(important):
        start = int(start)

        for nxt in adjacency[start]:
            edge_id = _edge_key(start, nxt)

            if edge_id in visited_edges:
                continue

            chain = [start, int(nxt)]
            visited_edges.add(edge_id)

            prev = start
            cur = int(nxt)

            while not important[cur]:
                candidates = [n for n in adjacency[cur] if n != prev]

                if not candidates:
                    break

                nxt2 = int(candidates[0])
                visited_edges.add(_edge_key(cur, nxt2))

                prev = cur
                cur = nxt2
                chain.append(cur)

            simplified_chain = _resample_chain_by_spacing(
                points=points,
                chain=chain,
                spacing=spacing,
            )

            for a, b in zip(simplified_chain[:-1], simplified_chain[1:]):
                out_a = get_output_node(a)
                out_b = get_output_node(b)

                if out_a != out_b:
                    out_edges.append((out_a, out_b))

    if len(out_points) == 0:
        return (
            np.empty((0, 3), dtype=points.dtype),
            np.empty((0, 2), dtype=np.int64),
        )

    return (
        np.asarray(out_points, dtype=points.dtype),
        np.asarray(out_edges, dtype=np.int64),
    )


def _resample_chain_by_spacing(
    points: np.ndarray,
    chain: list[int],
    spacing: float,
):
    if len(chain) <= 2 or spacing <= 0.0:
        return chain

    kept = [chain[0]]
    accumulated = 0.0

    for i in range(1, len(chain) - 1):
        prev = chain[i - 1]
        cur = chain[i]

        accumulated += float(np.linalg.norm(points[prev] - points[cur]))

        if accumulated >= spacing:
            kept.append(cur)
            accumulated = 0.0

    if kept[-1] != chain[-1]:
        kept.append(chain[-1])

    return kept


def _build_adjacency(edges: np.ndarray, node_count: int):
    adjacency = [[] for _ in range(node_count)]

    for a, b in edges:
        a = int(a)
        b = int(b)

        adjacency[a].append(b)
        adjacency[b].append(a)

    return adjacency


def _alive_degrees(adjacency: list[list[int]], alive: np.ndarray):
    degree = np.zeros(len(adjacency), dtype=np.int32)

    for i, neighbors in enumerate(adjacency):
        if not alive[i]:
            continue

        count = 0
        for n in neighbors:
            if alive[n]:
                count += 1

        degree[i] = count

    return degree


def _edge_key(a: int, b: int):
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def _set_axes_equal_3d(ax, points: np.ndarray):
    if points.shape[0] == 0:
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))

    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_side_views_with_skeleton(
    pcd: torch.Tensor,
    skeleton_points,
    skeleton_edges,
    resolution_xy: int = 512,
    margin_ratio: float = 0.05,
    skeleton_gray: float = 0.55,
    skeleton_linewidth: float = 0.8,
    skeleton_point_size: float = 2.0,
    show_skeleton_points: bool = True,
):
    """
    Plot 5 flat depth-map side views with skeleton overlaid in grayscale.

    Views are:
        0: top
        1: front
        2: back
        3: left
        4: right

    Parameters
    ----------
    pcd:
        Point cloud tensor, shape (N, 3) or (N, >=3).
    skeleton_points:
        Skeleton node coordinates, NumPy array or torch tensor, shape (M, 3).
    skeleton_edges:
        Skeleton edges, shape (E, 2).
    resolution_xy:
        Output image resolution.
    margin_ratio:
        Same meaning as in cloud2sideViews_torch.
    skeleton_gray:
        Grayscale value for skeleton lines/points.
        0.0 = black, 1.0 = white.
    skeleton_linewidth:
        Skeleton edge width.
    skeleton_point_size:
        Skeleton node marker size.
    show_skeleton_points:
        Whether to draw skeleton nodes.
    """
    if resolution_xy is None:
        raise ValueError("resolution_xy must not be None")

    pcd = torch.as_tensor(pcd)
    pcd = pcd[:, :3].to(dtype=torch.float64)

    views = cloud2sideViews_torch(
        pcd,
        resolution_xy=resolution_xy,
        margin_ratio=margin_ratio,
    ).detach().cpu().numpy()

    skel_points = _as_numpy_xyz(skeleton_points)
    skel_edges = np.asarray(skeleton_edges, dtype=np.int64)

    if skel_points.shape[0] == 0:
        skel_pixels = [np.empty((0, 2), dtype=np.float64) for _ in range(5)]
    else:
        skel_pixels = project_skeleton_to_side_views(
            pcd=pcd,
            skeleton_points=skel_points,
            resolution_xy=resolution_xy,
            margin_ratio=margin_ratio,
        )

    titles = ["top", "front", "back", "left", "right"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)

    for i, ax in enumerate(axes):
        ax.imshow(
            views[i],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
        )

        _draw_projected_skeleton(
            ax=ax,
            points_2d=skel_pixels[i],
            edges=skel_edges,
            gray=skeleton_gray,
            linewidth=skeleton_linewidth,
            point_size=skeleton_point_size,
            show_points=show_skeleton_points,
        )

        ax.set_title(titles[i])
        ax.set_axis_off()

    return fig, axes


def project_skeleton_to_side_views(
    pcd: torch.Tensor,
    skeleton_points,
    resolution_xy: int,
    margin_ratio: float = 0.05,
):
    """
    Project 3D skeleton points into the same 5 image planes used by
    cloud2sideViews_torch.

    Returns
    -------
    projected:
        List of 5 arrays, each shape (M, 2), with columns:
            x_pixel, y_pixel
    """
    pcd = torch.as_tensor(pcd)
    pcd = pcd[:, :3].to(dtype=torch.float64)

    skel = torch.as_tensor(skeleton_points, dtype=torch.float64, device=pcd.device)
    skel = skel[:, :3]

    min_xyz = pcd.min(dim=0).values
    max_xyz = pcd.max(dim=0).values

    center = (min_xyz + max_xyz) / 2.0
    max_range = (max_xyz - min_xyz).max()
    cube_half = max_range / 2.0 * (1.0 + 2.0 * margin_ratio)

    cube_min = center - cube_half
    cube_max = center + cube_half

    def to_grid_float(val, min_val, max_val):
        grid = (val - min_val) / (max_val - min_val + 1e-8)
        grid = grid * float(resolution_xy - 1)
        return torch.clamp(grid, 0.0, float(resolution_xy - 1))

    x = skel[:, 0]
    y = skel[:, 1]
    z = skel[:, 2]

    gx = to_grid_float(x, cube_min[0], cube_max[0])
    gy = to_grid_float(y, cube_min[1], cube_max[1])
    gz = to_grid_float(z, cube_min[2], cube_max[2])

    last = float(resolution_xy - 1)

    # Each output is x_pixel, y_pixel for matplotlib imshow coordinates.
    top = torch.stack([gx, gy], dim=1)

    front = torch.stack([gx, last - gz], dim=1)

    back = torch.stack([last - gx, last - gz], dim=1)

    left = torch.stack([gy, last - gz], dim=1)

    right = torch.stack([last - gy, last - gz], dim=1)

    return [
        top.detach().cpu().numpy(),
        front.detach().cpu().numpy(),
        back.detach().cpu().numpy(),
        left.detach().cpu().numpy(),
        right.detach().cpu().numpy(),
    ]


def _draw_projected_skeleton(
    ax,
    points_2d: np.ndarray,
    edges: np.ndarray,
    gray: float = 0.55,
    linewidth: float = 0.8,
    point_size: float = 2.0,
    show_points: bool = True,
):
    if points_2d.shape[0] == 0:
        return

    gray = float(np.clip(gray, 0.0, 1.0))
    color = str(gray)

    if edges.shape[0] > 0:
        valid = (
            (edges[:, 0] >= 0)
            & (edges[:, 0] < points_2d.shape[0])
            & (edges[:, 1] >= 0)
            & (edges[:, 1] < points_2d.shape[0])
        )

        edges = edges[valid]

        if edges.shape[0] > 0:
            segments = points_2d[edges]
            collection = LineCollection(
                segments,
                colors=color,
                linewidths=linewidth,
                alpha=1.0,
            )
            ax.add_collection(collection)

    if show_points:
        ax.scatter(
            points_2d[:, 0],
            points_2d[:, 1],
            s=point_size,
            c=color,
            linewidths=0,
        )


def _as_numpy_xyz(points):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"skeleton_points must have shape (N, >=3), got {points.shape}")

    return points[:, :3]

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

def main():
    import pathlib as pth
    path = pth.Path("/mnt/SSD_EXT4_1TB/DATA/tree_data/FULL_LAZ/cut/07-45_tile_000_010.npy")

    pcd = np.load(path)[: , :3]

    points, edges = voxel_tree_skeleton_numpy(
        pcd,
        voxel_size=1.,
        connect_radius=5.0,
        min_branch_length=2.0,
        simplify_spacing=3.,
    )

    fig, axes = plot_side_views_with_skeleton(
        pcd=torch.from_numpy(pcd),
        skeleton_points=points,
        skeleton_edges=edges,
        resolution_xy=128,
        skeleton_gray=0.0,
        skeleton_linewidth=2.,
        skeleton_point_size=2.0,
    )

    plt.show()

if __name__ == "__main__":
    main()