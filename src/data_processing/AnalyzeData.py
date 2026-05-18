#!/usr/bin/env python3
"""Audit 5-view tree depth-map datasets for redundancy, leakage, and class ambiguity.

Input layout expected by default:

    dataset_root/
      train/*.npy
      test/*.npy
      val/*.npy

Each .npy file must contain at least XYZ in columns 0:3. Labels are parsed from the
filename suffix: "anything_<label>.npy".

The tool generates the same 5 side/top depth maps used by the classifier, extracts
fixed DINOv2 descriptors per view, aggregates them into one ordered multi-view
embedding per file, and produces:

    output_dir/files_audit.csv
    output_dir/species_split_summary.csv
    output_dir/species_pair_confusion.csv
    output_dir/split_leakage_summary.csv
    output_dir/removal_impact_split_species.csv
    output_dir/removal_impact_split_summary.csv
    output_dir/report.txt
    output_dir/embeddings.npy

The tool never deletes files. It only assigns a conservative per-file utility score:
  * 0: keep; no strong signal of harmful redundancy
  * 1: boundary/review case; inspect manually
  * 2: removal candidate; obvious redundancy or likely evaluation-side leakage
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


SPLITS = ("train", "test", "val")
VIEW_NAMES = ("top", "front", "back", "left", "right")


@dataclass(frozen=True)
class Sample:
    path: Path
    relpath: str
    split: str
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit 5-view tree depth-map datasets for redundancy and ambiguity."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=350)
    parser.add_argument("--margin-ratio", type=float, default=0.05)
    parser.add_argument("--dinov2-model", type=str, default="dinov2_vits14")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--neighbor-k", type=int, default=64)
    parser.add_argument("--redundancy-quantile", type=float, default=0.995)
    parser.add_argument("--leakage-quantile", type=float, default=0.995)
    parser.add_argument("--min-redundancy-sim", type=float, default=0.985)
    parser.add_argument("--min-leakage-sim", type=float, default=0.985)
    parser.add_argument("--ambiguity-quantile", type=float, default=0.95)
    parser.add_argument("--ambiguity-margin", type=float, default=0.015)
    parser.add_argument("--min-ambiguity-sim", type=float, default=0.90)
    parser.add_argument("--report-top-n", type=int, default=40)
    return parser.parse_args()


def parse_label(path: Path) -> int:
    suffix = path.stem.rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse integer label from filename suffix: {path.name!r}. "
            "Expected pattern like 'tree_12.npy'."
        ) from exc


def load_samples(dataset_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for split in SPLITS:
        split_root = dataset_root / split
        if not split_root.exists():
            continue
        for path in sorted(split_root.rglob("*.npy")):
            samples.append(
                Sample(
                    path=path,
                    relpath=str(path.relative_to(dataset_root)),
                    split=split,
                    label=parse_label(path),
                )
            )
    if not samples:
        raise RuntimeError(
            f"No .npy samples found below {dataset_root}. Expected train/test/val folders."
        )
    return samples


def cloud2sideviews_torch(
    points: torch.Tensor,
    resolution_xy: int,
    margin_ratio: float,
) -> torch.Tensor:
    points = points.to(dtype=torch.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points shaped [N, >=3], got {tuple(points.shape)}")
    if points.shape[0] == 0:
        raise ValueError("Cannot build depth maps from an empty point cloud")

    min_xyz = points[:, :3].min(dim=0).values
    max_xyz = points[:, :3].max(dim=0).values

    center = (min_xyz + max_xyz) / 2
    max_range = (max_xyz - min_xyz).max()
    cube_half = max_range / 2 * (1 + 2 * margin_ratio)

    cube_min = center - cube_half
    cube_max = center + cube_half

    def to_grid(val: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
        scaled = (val - min_val) / (max_val - min_val + 1e-8)
        return torch.clamp((scaled * (resolution_xy - 1)).long(), 0, resolution_xy - 1)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    gx = to_grid(x, cube_min[0], cube_max[0])
    gy = to_grid(y, cube_min[1], cube_max[1])
    gz = to_grid(z, cube_min[2], cube_max[2])

    def build_depth_map(
        y_idx: torch.Tensor,
        x_idx: torch.Tensor,
        distances: torch.Tensor,
        *,
        flip_y: bool = False,
        flip_x: bool = False,
    ) -> torch.Tensor:
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

        image = depth_map.view(resolution_xy, resolution_xy)
        valid = torch.isfinite(image)
        if not torch.any(valid):
            return torch.zeros_like(image, dtype=torch.float32)

        values = image[valid]
        min_val = values.min()
        max_val = values.max()
        normalised = (max_val - values) / (max_val - min_val + 1e-8)
        normalised = normalised * (1.0 - 1.0 / 255.0) + (1.0 / 255.0)

        image = image.clone()
        image[valid] = normalised
        image[~valid] = 0.0
        return image.to(dtype=torch.float32)

    views = [
        build_depth_map(gy, gx, cube_max[2] - z),
        build_depth_map(gz, gx, cube_max[1] - y, flip_y=True),
        build_depth_map(gz, gx, y - cube_min[1], flip_y=True, flip_x=True),
        build_depth_map(gz, gy, cube_max[0] - x, flip_y=True),
        build_depth_map(gz, gy, x - cube_min[0], flip_y=True, flip_x=True),
    ]
    return torch.stack(views, dim=0)


def load_views(sample: Sample, resolution: int, margin_ratio: float, device: torch.device) -> torch.Tensor:
    arr = np.load(sample.path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{sample.path}: expected array shaped [N, >=3], got {arr.shape}")
    xyz = torch.as_tensor(arr[:, :3], dtype=torch.float32, device=device)
    return cloud2sideviews_torch(xyz, resolution_xy=resolution, margin_ratio=margin_ratio)


def load_dinov2(model_name: str, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    model.to(device)
    return model


def preprocess_views_for_dinov2(views: torch.Tensor) -> torch.Tensor:
    if views.shape[0] != 5:
        raise ValueError(f"Expected exactly 5 views, got {views.shape[0]}")
    images = views[:, None, :, :].repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=images.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def l2_normalise(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True), min=eps)


def aggregate_view_embeddings(view_embeddings: torch.Tensor) -> torch.Tensor:
    """Build an ordered multi-view descriptor from [B, 5, D] view features."""
    view_embeddings = l2_normalise(view_embeddings)
    mean = view_embeddings.mean(dim=1)
    std = view_embeddings.std(dim=1, unbiased=False)
    ordered = view_embeddings.flatten(start_dim=1)

    pairwise_parts: list[torch.Tensor] = []
    for left in range(5):
        for right in range(left + 1, 5):
            pairwise_parts.append((view_embeddings[:, left] * view_embeddings[:, right]).sum(dim=1, keepdim=True))
    pairwise = torch.cat(pairwise_parts, dim=1)

    fused = torch.cat((mean, std, ordered, pairwise), dim=1)
    return l2_normalise(fused)


def batched(items: list[Sample], batch_size: int) -> Iterable[list[Sample]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def extract_embeddings(
    samples: list[Sample],
    model: torch.nn.Module,
    *,
    resolution: int,
    margin_ratio: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for batch in tqdm(list(batched(samples, batch_size)), desc="Embedding depth maps"):
        prepared: list[torch.Tensor] = []
        for sample in batch:
            views = load_views(sample, resolution=resolution, margin_ratio=margin_ratio, device=device)
            prepared.append(preprocess_views_for_dinov2(views))
        model_input = torch.cat(prepared, dim=0)
        with torch.inference_mode():
            features = model(model_input)
        if features.ndim != 2:
            raise RuntimeError(f"Unexpected DINOv2 output shape: {tuple(features.shape)}")
        features = features.view(len(batch), 5, -1)
        fused = aggregate_view_embeddings(features)
        outputs.append(fused.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def fit_neighbors(embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if len(embeddings) < 2:
        raise RuntimeError("Need at least two samples for similarity analysis")
    n_neighbors = min(k + 1, len(embeddings))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    knn.fit(embeddings)
    distances, indices = knn.kneighbors(embeddings, return_distance=True)
    similarities = 1.0 - distances
    return similarities, indices


def exact_same_group_neighbor(
    embeddings: np.ndarray,
    groups: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    best_sim = np.full(len(embeddings), np.nan, dtype=np.float32)
    best_idx = np.full(len(embeddings), -1, dtype=np.int64)

    group_to_indices: dict[object, np.ndarray] = {}
    for group, frame in groups.groupby(groups).groups.items():
        group_to_indices[group] = np.asarray(list(frame), dtype=np.int64)

    for idxs in group_to_indices.values():
        if len(idxs) < 2:
            continue
        subset = embeddings[idxs]
        model = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute")
        model.fit(subset)
        distances, local_indices = model.kneighbors(subset, return_distance=True)
        best_sim[idxs] = (1.0 - distances[:, 1]).astype(np.float32)
        best_idx[idxs] = idxs[local_indices[:, 1]]

    return best_sim, best_idx


def first_matching_neighbor(
    similarities: np.ndarray,
    indices: np.ndarray,
    mask_fn,
) -> tuple[np.ndarray, np.ndarray]:
    out_sim = np.full(len(indices), np.nan, dtype=np.float32)
    out_idx = np.full(len(indices), -1, dtype=np.int64)
    for row in range(len(indices)):
        for pos in range(1, indices.shape[1]):
            candidate = int(indices[row, pos])
            if mask_fn(row, candidate):
                out_sim[row] = float(similarities[row, pos])
                out_idx[row] = candidate
                break
    return out_sim, out_idx


def finite_quantile(values: np.ndarray, quantile: float, floor: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return floor
    return max(float(np.quantile(finite, quantile)), floor)


def build_file_audit(
    samples: list[Sample],
    embeddings: np.ndarray,
    *,
    neighbor_k: int,
    redundancy_quantile: float,
    leakage_quantile: float,
    min_redundancy_sim: float,
    min_leakage_sim: float,
    ambiguity_quantile: float,
    ambiguity_margin: float,
    min_ambiguity_sim: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = pd.DataFrame(
        {
            "file": [sample.relpath for sample in samples],
            "split": [sample.split for sample in samples],
            "label": [sample.label for sample in samples],
        }
    )

    global_sims, global_indices = fit_neighbors(embeddings, k=neighbor_k)

    same_label_sim, same_label_idx = exact_same_group_neighbor(embeddings, frame["label"])
    same_split_label_group = frame["split"].astype(str) + "::" + frame["label"].astype(str)
    same_split_label_sim, same_split_label_idx = exact_same_group_neighbor(embeddings, same_split_label_group)

    other_label_sim, other_label_idx = first_matching_neighbor(
        global_sims,
        global_indices,
        lambda row, candidate: frame.at[row, "label"] != frame.at[candidate, "label"],
    )
    cross_split_sim, cross_split_idx = first_matching_neighbor(
        global_sims,
        global_indices,
        lambda row, candidate: frame.at[row, "split"] != frame.at[candidate, "split"],
    )

    redundancy_threshold = finite_quantile(
        same_split_label_sim,
        quantile=redundancy_quantile,
        floor=min_redundancy_sim,
    )
    leakage_threshold = finite_quantile(
        cross_split_sim,
        quantile=leakage_quantile,
        floor=min_leakage_sim,
    )
    ambiguity_threshold = finite_quantile(
        other_label_sim,
        quantile=ambiguity_quantile,
        floor=min_ambiguity_sim,
    )

    frame["same_label_similarity"] = same_label_sim
    frame["same_label_neighbor"] = index_to_file(frame, same_label_idx)
    frame["same_label_neighbor_label"] = index_to_label(frame, same_label_idx)
    frame["same_label_neighbor_split"] = index_to_split(frame, same_label_idx)

    frame["same_split_same_label_similarity"] = same_split_label_sim
    frame["same_split_same_label_neighbor"] = index_to_file(frame, same_split_label_idx)

    frame["other_label_similarity"] = other_label_sim
    frame["other_label_neighbor"] = index_to_file(frame, other_label_idx)
    frame["other_label_neighbor_label"] = index_to_label(frame, other_label_idx)
    frame["other_label_neighbor_split"] = index_to_split(frame, other_label_idx)

    frame["cross_split_similarity"] = cross_split_sim
    frame["cross_split_neighbor"] = index_to_file(frame, cross_split_idx)
    frame["cross_split_neighbor_label"] = index_to_label(frame, cross_split_idx)
    frame["cross_split_neighbor_split"] = index_to_split(frame, cross_split_idx)

    frame["redundancy_score"] = np.nan_to_num(same_split_label_sim, nan=0.0)
    frame["ambiguity_margin_observed"] = same_label_sim - other_label_sim
    frame["is_extremely_redundant"] = frame["same_split_same_label_similarity"] >= redundancy_threshold
    frame["is_cross_split_near_duplicate"] = frame["cross_split_similarity"] >= leakage_threshold
    frame["is_ambiguous_against_other_species"] = (
        (frame["other_label_similarity"] >= ambiguity_threshold)
        & (
            frame["same_label_similarity"].isna()
            | (frame["other_label_similarity"] + ambiguity_margin >= frame["same_label_similarity"])
        )
    )

    utility_score: list[int] = []
    utility_reason: list[str] = []
    for row in frame.itertuples(index=False):
        if bool(row.is_cross_split_near_duplicate) and row.split in {"test", "val"}:
            utility_score.append(2)
            utility_reason.append("likely_eval_side_cross_split_near_duplicate")
        elif bool(row.is_extremely_redundant):
            utility_score.append(2)
            utility_reason.append("extreme_same_species_redundancy_same_split")
        elif bool(row.is_cross_split_near_duplicate):
            utility_score.append(1)
            utility_reason.append("train_side_cross_split_near_duplicate_review")
        elif bool(row.is_ambiguous_against_other_species):
            utility_score.append(1)
            utility_reason.append("cross_species_boundary_case_review")
        else:
            utility_score.append(0)
            utility_reason.append("no_strong_redundancy_signal")
    frame["utility_score"] = utility_score
    frame["utility_reason"] = utility_reason
    frame["utility_description"] = frame["utility_score"].map({
        0: "keep",
        1: "boundary_case_review",
        2: "removal_candidate_after_manual_verification",
    })

    thresholds = {
        "redundancy_threshold": redundancy_threshold,
        "leakage_threshold": leakage_threshold,
        "ambiguity_threshold": ambiguity_threshold,
        "ambiguity_margin": ambiguity_margin,
    }
    return frame, thresholds


def index_to_file(frame: pd.DataFrame, indices: np.ndarray) -> list[str | None]:
    out: list[str | None] = []
    for idx in indices:
        out.append(None if idx < 0 else str(frame.at[int(idx), "file"]))
    return out


def index_to_label(frame: pd.DataFrame, indices: np.ndarray) -> list[int | None]:
    out: list[int | None] = []
    for idx in indices:
        out.append(None if idx < 0 else int(frame.at[int(idx), "label"]))
    return out


def index_to_split(frame: pd.DataFrame, indices: np.ndarray) -> list[str | None]:
    out: list[str | None] = []
    for idx in indices:
        out.append(None if idx < 0 else str(frame.at[int(idx), "split"]))
    return out


def species_split_summary(files: pd.DataFrame) -> pd.DataFrame:
    grouped = files.groupby(["split", "label"], dropna=False)
    summary = grouped.agg(
        n_files=("file", "count"),
        n_score_2=("utility_score", lambda s: int((s == 2).sum())),
        n_score_1=("utility_score", lambda s: int((s == 1).sum())),
        n_score_0=("utility_score", lambda s: int((s == 0).sum())),
        mean_same_species_similarity=("same_label_similarity", "mean"),
        median_same_species_similarity=("same_label_similarity", "median"),
        mean_other_species_similarity=("other_label_similarity", "mean"),
        redundancy_rate=("is_extremely_redundant", "mean"),
        leakage_rate=("is_cross_split_near_duplicate", "mean"),
        ambiguity_rate=("is_ambiguous_against_other_species", "mean"),
    ).reset_index()
    return summary.sort_values(
        ["redundancy_rate", "ambiguity_rate", "n_files"],
        ascending=[False, False, False],
    )


def species_pair_confusion(files: pd.DataFrame) -> pd.DataFrame:
    valid = files.dropna(subset=["other_label_neighbor_label", "other_label_similarity"]).copy()
    valid["other_label_neighbor_label"] = valid["other_label_neighbor_label"].astype(int)
    pairs = valid.groupby(["label", "other_label_neighbor_label"], dropna=False).agg(
        n_files_pointing_to_pair=("file", "count"),
        mean_similarity=("other_label_similarity", "mean"),
        max_similarity=("other_label_similarity", "max"),
        n_score_1=("utility_score", lambda s: int((s == 1).sum())), 
    ).reset_index()
    return pairs.sort_values(
        ["n_files_pointing_to_pair", "mean_similarity", "max_similarity"],
        ascending=[False, False, False],
    )


def split_leakage_summary(files: pd.DataFrame) -> pd.DataFrame:
    valid = files.dropna(subset=["cross_split_neighbor_split", "cross_split_similarity"]).copy()
    valid["split_pair"] = valid["split"] + " -> " + valid["cross_split_neighbor_split"].astype(str)
    summary = valid.groupby("split_pair", dropna=False).agg(
        n_files=("file", "count"),
        n_flagged=("is_cross_split_near_duplicate", "sum"),
        mean_cross_split_similarity=("cross_split_similarity", "mean"),
        max_cross_split_similarity=("cross_split_similarity", "max"),
    ).reset_index()
    return summary.sort_values(["n_flagged", "max_cross_split_similarity"], ascending=[False, False])


def removal_impact_tables(files: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = {
        "baseline_no_removal": files,
        "after_manual_removal_score_2": files[files["utility_score"] != 2],
        "after_manual_removal_score_2_and_1": files[files["utility_score"] == 0],
    }

    detail_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    baseline_split_counts = files.groupby("split", dropna=False).size().rename("baseline_n_files")

    for scenario, subset in scenarios.items():
        split_label = subset.groupby(["split", "label"], dropna=False).size().rename("n_files").reset_index()
        split_totals = subset.groupby("split", dropna=False).size().rename("split_total_files").reset_index()
        detail = split_label.merge(split_totals, on="split", how="left")
        detail["label_share_within_split"] = detail["n_files"] / detail["split_total_files"].clip(lower=1)
        detail.insert(0, "scenario", scenario)
        detail_frames.append(detail)

        split_summary = split_label.groupby("split", dropna=False)["n_files"].agg(
            n_labels="count",
            min_label_files="min",
            max_label_files="max",
            mean_label_files="mean",
            std_label_files="std",
        ).reset_index()
        split_summary = split_summary.merge(split_totals, on="split", how="outer")
        split_summary = split_summary.merge(baseline_split_counts.reset_index(), on="split", how="left")
        split_summary["scenario"] = scenario
        split_summary["files_removed_vs_baseline"] = split_summary["baseline_n_files"].fillna(0) - split_summary["split_total_files"].fillna(0)
        split_summary["removal_rate_vs_baseline"] = split_summary["files_removed_vs_baseline"] / split_summary["baseline_n_files"].replace(0, np.nan)
        split_summary["imbalance_ratio_max_to_min_nonzero"] = split_summary["max_label_files"] / split_summary["min_label_files"].replace(0, np.nan)
        summary_frames.append(split_summary)

    detail_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary_all = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    summary_all = summary_all[[
        "scenario",
        "split",
        "split_total_files",
        "baseline_n_files",
        "files_removed_vs_baseline",
        "removal_rate_vs_baseline",
        "n_labels",
        "min_label_files",
        "max_label_files",
        "mean_label_files",
        "std_label_files",
        "imbalance_ratio_max_to_min_nonzero",
    ]]
    return detail_all, summary_all


def save_utility_plot(files: pd.DataFrame, output_dir: Path) -> str:
    counts = files["utility_score"].value_counts().reindex([0, 1, 2], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["0 - keep", "1 - review", "2 - removal candidate"], counts.values)
    ax.set_title("Dataset audit utility scores")
    ax.set_ylabel("Number of files")
    ax.set_xlabel("Utility score")
    fig.tight_layout()
    path = output_dir / "utility_scores.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def save_top_redundant_species_plot(summary: pd.DataFrame, output_dir: Path, top_n: int = 15) -> str | None:
    data = summary.sort_values("redundancy_rate", ascending=False).head(top_n)
    if data.empty:
        return None
    labels = data["split"].astype(str) + "/" + data["label"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, data["redundancy_rate"].to_numpy())
    ax.set_title("Highest same-split redundancy rates")
    ax.set_ylabel("Rate")
    ax.set_xlabel("split/label")
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    path = output_dir / "top_redundancy_rates.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def dataframe_text(df: pd.DataFrame, max_rows: int) -> str:
    if df.empty:
        return "No rows."
    preview = df.head(max_rows).copy()
    return preview.to_string(index=False) + ""


def write_report(
    *,
    files: pd.DataFrame,
    summary: pd.DataFrame,
    pair_confusion: pd.DataFrame,
    leakage: pd.DataFrame,
    removal_impact_detail: pd.DataFrame,
    removal_impact_summary: pd.DataFrame,
    thresholds: dict[str, float],
    output_dir: Path,
    top_n: int,
) -> None:
    action_plot = save_utility_plot(files, output_dir)
    redundancy_plot = save_top_redundant_species_plot(summary, output_dir)

    counts = files["utility_score"].value_counts().to_dict()
    remove_count = int(counts.get(2, 0))
    review_count = int(counts.get(1, 0))
    keep_count = int(counts.get(0, 0))

    most_redundant = summary.sort_values("redundancy_rate", ascending=False)
    strongest_confusions = pair_confusion.sort_values(
        ["n_files_pointing_to_pair", "mean_similarity"], ascending=[False, False]
    )
    flagged_leakage = leakage[leakage["n_flagged"] > 0]

    threshold_lines = "".join(
        f"- {name}: {value:.6f}"
        for name, value in thresholds.items()
    )

    plot_lines = [f"Utility score chart: {action_plot}"]
    if redundancy_plot:
        plot_lines.append(f"Top redundancy chart: {redundancy_plot}")
    plots_section = "".join(f"- {line}" for line in plot_lines)

    report = f"""5-VIEW DEPTH-MAP DATASET AUDIT
    {'=' * 36}

    PURPOSE
    -------
    This report detects three different issues:
    1. Extreme same-species redundancy.
    2. Likely near-duplicates across train/test/val.
    3. Cross-species ambiguity.

    The tool never deletes files. It only assigns a per-file utility score:
    - 0 = keep
    - 1 = boundary/review case
    - 2 = removal candidate after manual verification

    DATASET-LEVEL UTILITY SUMMARY
    -----------------------------
    Score 0 - keep: {keep_count}
    Score 1 - review: {review_count}
    Score 2 - removal candidate: {remove_count}

    GENERATED PLOTS
    ---------------
    {plots_section}

    ADAPTIVE THRESHOLDS
    -------------------
    {threshold_lines}

    MOST REDUNDANT SPLIT/SPECIES GROUPS
    ------------------------------------
    {dataframe_text(most_redundant, top_n)}
    STRONGEST APPARENT SPECIES CONFUSIONS
    -------------------------------------
    {dataframe_text(strongest_confusions, top_n)}
    POTENTIAL TRAIN/TEST/VAL LEAKAGE
    --------------------------------
    {dataframe_text(flagged_leakage, top_n)}
    HIGHEST-PRIORITY SCORE-2 CANDIDATES
    -----------------------------------
    {dataframe_text(files[files['utility_score'] == 2].sort_values(['is_cross_split_near_duplicate', 'redundancy_score', 'cross_split_similarity'], ascending=[False, False, False]), top_n)}
    HIGHEST-PRIORITY SCORE-1 MANUAL REVIEWS
    ---------------------------------------
    {dataframe_text(files[files['utility_score'] == 1].sort_values(['is_cross_split_near_duplicate', 'other_label_similarity'], ascending=[False, False]), top_n)}
    FOLDER BALANCE AFTER MANUAL REMOVAL SCENARIOS
    ---------------------------------------------
    Scenario 1: manual removal of score-2 files only.
    Scenario 2: manual removal of both score-2 and score-1 files.
    These are simulations only; the tool performs no deletions.

    {dataframe_text(removal_impact_summary, top_n)}
    SPECIES-LEVEL BALANCE AFTER MANUAL REMOVAL SCENARIOS
    -----------------------------------------------------
    {dataframe_text(removal_impact_detail, top_n)}
    INTERPRETATION RULES
    --------------------
    - Score 2: removal candidate after manual verification; usually obvious redundancy or likely evaluation-side leakage.
    - Score 1: boundary/review case; potentially ambiguous, mislabeled, or train-side leakage pair.
    - Score 0: keep; no strong redundancy signal was found under the current embedding model and thresholds.
    """
    (output_dir / "report.txt").write_text(report, encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.neighbor_k < 2:
        raise ValueError("--neighbor-k must be >= 2")
    for name in ("redundancy_quantile", "leakage_quantile", "ambiguity_quantile"):
        value = getattr(args, name)
        if not 0.0 < value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1)")
    for name in ("min_redundancy_sim", "min_leakage_sim", "min_ambiguity_sim"):
        value = getattr(args, name)
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [-1, 1]")
    if args.resolution < 8:
        raise ValueError("--resolution is unrealistically small")


def main() -> int:
    args = parse_args()
    validate_args(args)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.dataset_root)
    device = torch.device(args.device)
    model = load_dinov2(args.dinov2_model, device=device)

    embeddings = extract_embeddings(
        samples,
        model,
        resolution=args.resolution,
        margin_ratio=args.margin_ratio,
        batch_size=args.batch_size,
        device=device,
    )
    np.save(output_dir / "embeddings.npy", embeddings)

    files, thresholds = build_file_audit(
        samples,
        embeddings,
        neighbor_k=args.neighbor_k,
        redundancy_quantile=args.redundancy_quantile,
        leakage_quantile=args.leakage_quantile,
        min_redundancy_sim=args.min_redundancy_sim,
        min_leakage_sim=args.min_leakage_sim,
        ambiguity_quantile=args.ambiguity_quantile,
        ambiguity_margin=args.ambiguity_margin,
        min_ambiguity_sim=args.min_ambiguity_sim,
    )
    summary = species_split_summary(files)
    pairs = species_pair_confusion(files)
    leakage = split_leakage_summary(files)
    removal_impact_detail, removal_impact_summary = removal_impact_tables(files)

    files.to_csv(output_dir / "files_audit.csv", index=False)
    summary.to_csv(output_dir / "species_split_summary.csv", index=False)
    pairs.to_csv(output_dir / "species_pair_confusion.csv", index=False)
    leakage.to_csv(output_dir / "split_leakage_summary.csv", index=False)
    removal_impact_detail.to_csv(output_dir / "removal_impact_split_species.csv", index=False)
    removal_impact_summary.to_csv(output_dir / "removal_impact_split_summary.csv", index=False)

    write_report(
        files=files,
        summary=summary,
        pair_confusion=pairs,
        leakage=leakage,
        removal_impact_detail=removal_impact_detail,
        removal_impact_summary=removal_impact_summary,
        thresholds=thresholds,
        output_dir=output_dir,
        top_n=args.report_top_n,
    )

    print(f"Wrote audit outputs to: {output_dir}")
    print(f"Files audited: {len(files)}")
    print(files["utility_score"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
