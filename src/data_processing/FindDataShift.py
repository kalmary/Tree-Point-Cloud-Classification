#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter


VIEW_NAMES = ("top", "front", "back", "left", "right")
DATASET_ORIGINAL = "original"
DATASET_TARGET = "grajewo"


@dataclass(frozen=True)
class TreeRecord:
    path: Path
    label: str
    dataset: str


@dataclass(frozen=True)
class TreeStats:
    path: str
    label: str
    dataset: str
    n_points: int
    width_x: float
    width_y: float
    height_z: float
    xy_width: float
    bbox_volume: float
    density: float
    z_q05: float
    z_q25: float
    z_q50: float
    z_q75: float
    z_q95: float
    voxel_occupancy_32: float
    top_fg_ratio: float
    front_fg_ratio: float
    back_fg_ratio: float
    left_fg_ratio: float
    right_fg_ratio: float


def cloud2sideViews_torch(
    points: torch.Tensor,
    resolution_xy: int | None = None,
    margin_ratio: float = 0.05,
) -> torch.Tensor:
    if resolution_xy is None:
        raise ValueError("resolution_xy must be provided")

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points shaped [N, >=3], got {tuple(points.shape)}")

    points = points[:, :3].type(torch.float64)

    min_xyz = points.min(dim=0).values
    max_xyz = points.max(dim=0).values

    center = (min_xyz + max_xyz) / 2
    max_range = (max_xyz - min_xyz).max()
    cube_half = max_range / 2 * (1 + 2 * margin_ratio)

    cube_min = center - cube_half
    cube_max = center + cube_half

    def to_grid(
        val: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
    ) -> torch.Tensor:
        return torch.clamp(
            ((val - min_val) / (max_val - min_val + 1e-8) * (resolution_xy - 1)).long(),
            0,
            resolution_xy - 1,
        )

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    gx = to_grid(x, cube_min[0], cube_max[0])
    gy = to_grid(y, cube_min[1], cube_max[1])
    gz = to_grid(z, cube_min[2], cube_max[2])

    views = []

    def build_depth_map(
        indices_2d: tuple[torch.Tensor, torch.Tensor],
        distances: torch.Tensor,
        flip_y: bool = False,
        flip_x: bool = False,
    ) -> torch.Tensor:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)

    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--seed", default=13, type=int)

    parser.add_argument("--max-pdf-trees-per-label", default=100, type=int)

    return parser.parse_args()


def label_from_path(path: Path) -> str:
    stem = path.stem

    if "_" not in stem:
        raise ValueError(f"Cannot parse label from filename without underscore: {path}")

    return stem.rsplit("_", 1)[1]


def collect_records(root: Path, dataset: str) -> list[TreeRecord]:
    if not root.exists():
        raise FileNotFoundError(root)

    records = []

    for path in sorted(root.rglob("*.npy")):
        label = label_from_path(path)
        records.append(TreeRecord(path=path, label=label, dataset=dataset))

    if not records:
        raise RuntimeError(f"No .npy files found under {root}")

    return records


def load_points(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected [N, >=3] point array in {path}, got {arr.shape}")

    points = arr[:, :3].astype(np.float64, copy=False)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]

    if len(points) < 8:
        raise ValueError(f"Too few valid points in {path}: {len(points)}")

    return points


def voxel_occupancy(points: np.ndarray, grid_size: int = 32) -> float:
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    span = max_xyz - min_xyz

    if float(span.max()) <= 1e-12:
        return 0.0

    scaled = (points - min_xyz) / (span + 1e-12)
    idx = np.floor(scaled * (grid_size - 1)).astype(np.int64)
    idx = np.clip(idx, 0, grid_size - 1)

    flat = idx[:, 0] * grid_size * grid_size + idx[:, 1] * grid_size + idx[:, 2]
    occupied = len(np.unique(flat))

    return occupied / float(grid_size**3)


def side_views(points: np.ndarray, resolution: int) -> np.ndarray:
    tensor = torch.from_numpy(points[:, :3])

    with torch.no_grad():
        views = cloud2sideViews_torch(tensor, resolution_xy=resolution)

    return views.cpu().numpy()


def compute_stats(record: TreeRecord, resolution: int) -> TreeStats:
    points = load_points(record.path)

    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    span = max_xyz - min_xyz

    width_x = float(span[0])
    width_y = float(span[1])
    height_z = float(span[2])
    xy_width = float(max(width_x, width_y))

    bbox_volume = float(max(width_x * width_y * height_z, 0.0))
    density = float(len(points) / (bbox_volume + 1e-12))

    z = points[:, 2]
    z_q05, z_q25, z_q50, z_q75, z_q95 = np.quantile(
        z,
        [0.05, 0.25, 0.50, 0.75, 0.95],
    )

    views = side_views(points, resolution)
    fg_ratios = [(view > 0.0).mean() for view in views]

    return TreeStats(
        path=str(record.path),
        label=record.label,
        dataset=record.dataset,
        n_points=int(len(points)),
        width_x=width_x,
        width_y=width_y,
        height_z=height_z,
        xy_width=xy_width,
        bbox_volume=bbox_volume,
        density=density,
        z_q05=float(z_q05),
        z_q25=float(z_q25),
        z_q50=float(z_q50),
        z_q75=float(z_q75),
        z_q95=float(z_q95),
        voxel_occupancy_32=float(voxel_occupancy(points)),
        top_fg_ratio=float(fg_ratios[0]),
        front_fg_ratio=float(fg_ratios[1]),
        back_fg_ratio=float(fg_ratios[2]),
        left_fg_ratio=float(fg_ratios[3]),
        right_fg_ratio=float(fg_ratios[4]),
    )


def stats_to_dict(stats: TreeStats) -> dict[str, object]:
    return {
        "path": stats.path,
        "label": stats.label,
        "dataset": stats.dataset,
        "n_points": stats.n_points,
        "width_x": stats.width_x,
        "width_y": stats.width_y,
        "height_z": stats.height_z,
        "xy_width": stats.xy_width,
        "bbox_volume": stats.bbox_volume,
        "density": stats.density,
        "z_q05": stats.z_q05,
        "z_q25": stats.z_q25,
        "z_q50": stats.z_q50,
        "z_q75": stats.z_q75,
        "z_q95": stats.z_q95,
        "voxel_occupancy_32": stats.voxel_occupancy_32,
        "top_fg_ratio": stats.top_fg_ratio,
        "front_fg_ratio": stats.front_fg_ratio,
        "back_fg_ratio": stats.back_fg_ratio,
        "left_fg_ratio": stats.left_fg_ratio,
        "right_fg_ratio": stats.right_fg_ratio,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_records(records: list[TreeRecord]) -> dict[tuple[str, str], list[TreeRecord]]:
    grouped: dict[tuple[str, str], list[TreeRecord]] = {}

    for record in records:
        key = (record.label, record.dataset)
        grouped.setdefault(key, []).append(record)

    return grouped


def group_stats(stats: list[TreeStats]) -> dict[tuple[str, str], list[TreeStats]]:
    grouped: dict[tuple[str, str], list[TreeStats]] = {}

    for item in stats:
        key = (item.label, item.dataset)
        grouped.setdefault(key, []).append(item)

    return grouped


def numeric_fields() -> list[str]:
    return [
        "n_points",
        "width_x",
        "width_y",
        "height_z",
        "xy_width",
        "bbox_volume",
        "density",
        "z_q05",
        "z_q25",
        "z_q50",
        "z_q75",
        "z_q95",
        "voxel_occupancy_32",
        "top_fg_ratio",
        "front_fg_ratio",
        "back_fg_ratio",
        "left_fg_ratio",
        "right_fg_ratio",
    ]


def get_field(stats: TreeStats, field: str) -> float:
    return float(getattr(stats, field))


def median_iqr(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)

    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")

    q25, med, q75 = np.quantile(arr, [0.25, 0.50, 0.75])
    return float(med), float(q25), float(q75)


def distribution_distance(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")

    qa = np.quantile(np.asarray(a, dtype=np.float64), np.linspace(0.05, 0.95, 19))
    qb = np.quantile(np.asarray(b, dtype=np.float64), np.linspace(0.05, 0.95, 19))

    scale = np.median(np.abs(qa)) + np.median(np.abs(qb)) + 1e-12
    return float(np.mean(np.abs(qa - qb)) / scale)


def make_label_summary(stats: list[TreeStats]) -> list[dict[str, object]]:
    rows = []
    grouped = group_stats(stats)

    for (label, dataset), items in sorted(grouped.items()):
        row: dict[str, object] = {
            "label": label,
            "dataset": dataset,
            "count": len(items),
        }

        for field in numeric_fields():
            values = [get_field(item, field) for item in items]
            med, q25, q75 = median_iqr(values)

            row[f"{field}_median"] = med
            row[f"{field}_q25"] = q25
            row[f"{field}_q75"] = q75

        rows.append(row)

    return rows


def suggest_augmentation(field: str, target_median: float, original_median: float) -> str:
    if not np.isfinite(target_median) or not np.isfinite(original_median):
        return ""

    ratio = target_median / (original_median + 1e-12)

    if field == "n_points" and ratio < 0.75:
        return "Add random point dropout / stronger downsampling to original data."

    if field == "density" and ratio < 0.75:
        return "Add thinning, missing branches, and lower local density."

    if field == "density" and ratio > 1.25:
        return "Add local duplicate density variation or non-uniform sampling."

    if field in {"front_fg_ratio", "back_fg_ratio", "left_fg_ratio", "right_fg_ratio"}:
        if ratio < 0.80:
            return "Add branch dropout, occlusion, reduced branch visibility, and sparse side projections."
        if ratio > 1.25:
            return "Add segmentation clutter / neighboring-tree fragments."

    if field == "top_fg_ratio":
        if ratio < 0.80:
            return "Add crown sparsification and crown-hole dropout."
        if ratio > 1.25:
            return "Add crown leakage and neighboring-tree fragments."

    if field == "xy_width" and ratio > 1.20:
        return "Add horizontal clutter, crown leakage, and imperfect segmentation fragments."

    if field == "xy_width" and ratio < 0.85:
        return "Add XY cropping or tighter segmentation simulation."

    if field == "height_z" and ratio < 0.85:
        return "Add vertical cropping or Z-scale jitter."

    if field == "height_z" and ratio > 1.15:
        return "Add tall outlier fragments or imperfect ground/top segmentation."

    if field == "voxel_occupancy_32" and ratio < 0.75:
        return "Add sparsification and remove small branch structures."

    if field == "voxel_occupancy_32" and ratio > 1.25:
        return "Add spatial noise and extra segmented fragments."

    return ""


def make_comparison_rows(stats: list[TreeStats]) -> list[dict[str, object]]:
    rows = []
    grouped = group_stats(stats)
    labels = sorted({item.label for item in stats})

    for label in labels:
        original = grouped.get((label, DATASET_ORIGINAL), [])
        target = grouped.get((label, DATASET_TARGET), [])

        if not original or not target:
            continue

        for field in numeric_fields():
            original_values = [get_field(item, field) for item in original]
            target_values = [get_field(item, field) for item in target]

            original_median, original_q25, original_q75 = median_iqr(original_values)
            target_median, target_q25, target_q75 = median_iqr(target_values)

            ratio = target_median / (original_median + 1e-12)
            distance = distribution_distance(original_values, target_values)

            rows.append(
                {
                    "label": label,
                    "field": field,
                    "original_count": len(original),
                    "target_count": len(target),
                    "original_median": original_median,
                    "target_median": target_median,
                    "target_over_original_median": ratio,
                    "original_q25": original_q25,
                    "original_q75": original_q75,
                    "target_q25": target_q25,
                    "target_q75": target_q75,
                    "quantile_distance": distance,
                    "augmentation_hint": suggest_augmentation(
                        field,
                        target_median,
                        original_median,
                    ),
                }
            )

    rows.sort(key=lambda row: (row["label"], -float(row["quantile_distance"])))
    return rows


def render_tree_page(
    pdf: PdfPages,
    record: TreeRecord,
    resolution: int,
) -> None:
    points = load_points(record.path)
    views = side_views(points, resolution)

    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    span = max_xyz - min_xyz

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))

    for ax, view, name in zip(axes, views, VIEW_NAMES):
        ax.imshow(view, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(
        (
            f"dataset={record.dataset} | label={record.label} | file={record.path.name}\n"
            f"points={len(points)} | bbox=({span[0]:.2f}, {span[1]:.2f}, {span[2]:.2f})"
        ),
        fontsize=10,
    )

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def make_per_label_pdfs(
    records: list[TreeRecord],
    out_dir: Path,
    resolution: int,
    max_trees_per_label: int,
    rng: random.Random,
) -> list[Path]:
    grouped = group_records(records)
    groups = sorted(grouped.items())
    pdf_paths = []

    for group_idx, ((label, dataset), items) in enumerate(groups, start=1):
        selected = list(items)

        if len(selected) > max_trees_per_label:
            selected = rng.sample(selected, max_trees_per_label)

        pdf_path = out_dir / "by_label" / f"{label}_{dataset}_side_views.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        print(
            f"[PDF {group_idx}/{len(groups)}] "
            f"label={label} dataset={dataset} trees={len(selected)} -> {pdf_path}",
            flush=True,
        )

        with PdfPages(pdf_path) as pdf:
            for tree_idx, record in enumerate(selected, start=1):
                try:
                    render_tree_page(
                        pdf=pdf,
                        record=record,
                        resolution=resolution,
                    )
                except Exception as exc:
                    print(f"  SKIP PDF page {record.path}: {exc}", flush=True)
                    continue

                if tree_idx % 10 == 0 or tree_idx == len(selected):
                    print(
                        f"  rendered {tree_idx}/{len(selected)} trees "
                        f"for label={label} dataset={dataset}",
                        flush=True,
                    )

        pdf_paths.append(pdf_path)

    return pdf_paths


def merge_pdfs(input_paths: list[Path], output_path: Path) -> None:
    writer = PdfWriter()

    for path in input_paths:
        reader = PdfReader(str(path))

        for page in reader.pages:
            writer.add_page(page)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        writer.write(f)


def merge_dataset_pdfs(pdf_paths: list[Path], out_dir: Path) -> None:
    original_paths = [path for path in pdf_paths if f"_{DATASET_ORIGINAL}_side_views.pdf" in path.name]
    target_paths = [path for path in pdf_paths if f"_{DATASET_TARGET}_side_views.pdf" in path.name]

    if original_paths:
        print("Merging original side-view PDFs...", flush=True)
        merge_pdfs(
            input_paths=sorted(original_paths),
            output_path=out_dir / "all_original_side_views.pdf",
        )

    if target_paths:
        print("Merging Grajewo side-view PDFs...", flush=True)
        merge_pdfs(
            input_paths=sorted(target_paths),
            output_path=out_dir / "all_grajewo_side_views.pdf",
        )

    if pdf_paths:
        print("Merging all side-view PDFs...", flush=True)
        merge_pdfs(
            input_paths=sorted(pdf_paths),
            output_path=out_dir / "all_by_label_side_views.pdf",
        )


def write_llm_prompt(out_path: Path) -> None:
    text = """# Tree species domain-shift review prompt

You are comparing two LiDAR tree datasets.

Dataset A: original GitHub validation dataset.
Dataset B: Grajewo real terrain dataset, automatically segmented into individual trees.

The PDFs do NOT show paired trees. They show independent examples grouped by species and dataset.

For each species label, inspect:
- label_original_side_views.pdf
- label_grajewo_side_views.pdf

Each page shows depth-map side views:
1. top
2. front
3. back
4. left
5. right

Describe the domain shift between original and Grajewo for the same species.

Focus on:
1. Crown shape differences.
2. Trunk visibility differences.
3. Branch visibility differences.
4. Point sparsity / density differences.
5. Noise and outlier differences.
6. Segmentation leakage: neighboring trees, ground, cut-off crowns, extra fragments.
7. Scale or aspect-ratio differences.
8. Whether Grajewo examples are consistently harder or only occasionally corrupted.
9. Which augmentations should be added to original training data.

Use the CSV files as quantitative support:
- stats_per_tree.csv
- label_summary.csv
- species_comparisons.csv

Prefer concrete augmentation suggestions:
- random point dropout
- branch dropout
- local cuboid dropout
- crown dropout
- reduced side-view contrast
- added outlier clusters
- horizontal crown leakage
- neighboring-tree fragments
- Z cropping
- XY cropping
- XY/Z scale jitter
- random rotations
- partial occlusion
- non-uniform thinning
- density-dependent resampling
"""

    out_path.write_text(text)


def print_top_hints(comparison_rows: list[dict[str, object]], max_rows: int = 40) -> None:
    printed = 0

    for row in comparison_rows:
        hint = str(row["augmentation_hint"])
        if not hint:
            continue

        print(
            f"{str(row['label']):>12s} | {str(row['field']):<20s} | "
            f"ratio={float(row['target_over_original_median']):.3f} | {hint}",
            flush=True,
        )

        printed += 1
        if printed >= max_rows:
            break


def main() -> None:
    args = parse_args()

    if args.resolution <= 8:
        raise ValueError("--resolution must be greater than 8")

    if args.max_pdf_trees_per_label <= 0:
        raise ValueError("--max-pdf-trees-per-label must be positive")

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    original_records = collect_records(args.original, DATASET_ORIGINAL)
    target_records = collect_records(args.target, DATASET_TARGET)
    records = original_records + target_records

    print(f"Found {len(original_records)} original trees", flush=True)
    print(f"Found {len(target_records)} Grajewo trees", flush=True)

    labels_original = {record.label for record in original_records}
    labels_target = {record.label for record in target_records}
    common_labels = sorted(labels_original & labels_target)

    print(f"Original labels: {len(labels_original)}", flush=True)
    print(f"Grajewo labels: {len(labels_target)}", flush=True)
    print(f"Common labels: {len(common_labels)}", flush=True)

    if not common_labels:
        raise RuntimeError("No common labels found between datasets")

    stats: list[TreeStats] = []

    print("Computing per-tree statistics...", flush=True)

    for idx, record in enumerate(records, start=1):
        try:
            stats.append(compute_stats(record, args.resolution))
        except Exception as exc:
            print(f"SKIP stats {record.path}: {exc}", flush=True)

        if idx % 100 == 0 or idx == len(records):
            print(f"Processed {idx}/{len(records)} files", flush=True)

    if not stats:
        raise RuntimeError("No valid tree statistics computed")

    print("Finished statistics. Writing CSV files...", flush=True)

    stats_rows = [stats_to_dict(item) for item in stats]
    summary_rows = make_label_summary(stats)
    comparison_rows = make_comparison_rows(stats)

    write_csv(args.out / "stats_per_tree.csv", stats_rows)
    write_csv(args.out / "label_summary.csv", summary_rows)
    write_csv(args.out / "species_comparisons.csv", comparison_rows)

    print("Creating per-label side-view PDFs...", flush=True)

    pdf_paths = make_per_label_pdfs(
        records=records,
        out_dir=args.out / "pdfs",
        resolution=args.resolution,
        max_trees_per_label=args.max_pdf_trees_per_label,
        rng=rng,
    )

    merge_dataset_pdfs(
        pdf_paths=pdf_paths,
        out_dir=args.out / "pdfs",
    )

    write_llm_prompt(args.out / "llm_review_prompt.md")

    print("", flush=True)
    print("Most obvious augmentation hints:", flush=True)
    print_top_hints(comparison_rows)
    print("", flush=True)
    print(f"Wrote analysis to: {args.out}", flush=True)


if __name__ == "__main__":
    main()