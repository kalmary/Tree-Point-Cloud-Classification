#!/usr/bin/env python3
"""Inject controlled eval-to-train leakage for diagnostic experiments only.

This script copies stratified subsets of ``test`` and ``val`` into ``train``.
It is intentionally explicit about what it does because contaminated evaluation
metrics are not valid model-quality measurements.

Leakage semantics
-----------------
``--leak-fraction 0.40`` means:

    40% of existing test files are copied into train
    40% of existing val files are copied into train

Because the exact eval samples are now present in the training set, test and
validation metrics become intentionally optimistic.

Stratification
--------------
Sampling is stratified within each source split. The copied files preserve the
existing label proportions of ``test`` and ``val`` respectively, rather than
copying mostly from the dominant class.

Safety
------
Dry-run is the default. Physical copies require ``--apply``.
A CSV plan is always written before copying.
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_SPLITS = ("test", "val")
SUPPORTED_SUFFIXES = (".npy",)


@dataclass(frozen=True)
class Sample:
    path: Path
    split: str
    label: int
    relpath_from_dataset: str
    relpath_inside_split: Path


def parse_label(path: Path) -> int:
    """Parse label from the final underscore-separated filename token."""
    token = path.stem.rsplit("_", 1)[-1]
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse integer label from filename {path.name!r}. "
            "Expected a final suffix like '_12.npy'."
        ) from exc


def collect_split_samples(dataset_root: Path, split: str) -> list[Sample]:
    split_root = dataset_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Missing split directory: {split_root}")

    samples: list[Sample] = []
    for suffix in SUPPORTED_SUFFIXES:
        for path in sorted(split_root.rglob(f"*{suffix}")):
            rel_inside_split = path.relative_to(split_root)
            samples.append(
                Sample(
                    path=path,
                    split=split,
                    label=parse_label(path),
                    relpath_from_dataset=str(path.relative_to(dataset_root)),
                    relpath_inside_split=rel_inside_split,
                )
            )
    return samples


def allocate_label_quotas(label_counts: pd.Series, total_to_copy: int) -> dict[int, int]:
    """Largest-remainder quota allocation preserving the source split label mix."""
    if total_to_copy < 0:
        raise ValueError("total_to_copy must be non-negative")
    if total_to_copy == 0:
        return {int(label): 0 for label in label_counts.index}

    total_existing = int(label_counts.sum())
    if total_existing <= 0:
        raise ValueError("Cannot allocate quotas from an empty source split")

    exact = label_counts.astype(float) / total_existing * total_to_copy
    floor = np.floor(exact).astype(int)
    remaining = total_to_copy - int(floor.sum())

    quotas = {int(label): int(count) for label, count in floor.items()}
    if remaining == 0:
        return quotas

    remainders = (exact - floor).sort_values(ascending=False)
    for label in remainders.index[:remaining]:
        quotas[int(label)] += 1
    return quotas


def compute_copy_count(source_split_count: int, leak_fraction: float) -> int:
    if source_split_count < 0:
        raise ValueError("source_split_count must be non-negative")
    if not 0.0 <= leak_fraction <= 1.0:
        raise ValueError("leak_fraction must be in [0, 1]")
    return math.floor(source_split_count * leak_fraction)


def build_leakage_plan(
    *,
    dataset_root: Path,
    leak_fraction: float = 0.40,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a stratified test/val-to-train copy plan without changing files."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not 0.0 <= leak_fraction <= 1.0:
        raise ValueError("leak_fraction must be in [0, 1]")

    train_root = dataset_root / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"Missing train directory: {train_root}")

    rng = np.random.default_rng(random_state)
    plan_parts: list[pd.DataFrame] = []

    for source_split in SOURCE_SPLITS:
        source_samples = collect_split_samples(dataset_root, source_split)
        source_df = pd.DataFrame(
            {
                "source_path": [str(sample.path) for sample in source_samples],
                "source_file": [sample.relpath_from_dataset for sample in source_samples],
                "source_split": [sample.split for sample in source_samples],
                "source_rel_inside_split": [str(sample.relpath_inside_split) for sample in source_samples],
                "label": [sample.label for sample in source_samples],
            }
        )
        if source_df.empty:
            raise RuntimeError(f"No .npy files found in {source_split!r} split")

        source_count = int(len(source_df))
        total_to_copy = compute_copy_count(source_count, leak_fraction)
        if total_to_copy == 0:
            continue

        label_counts = source_df["label"].value_counts().sort_index()
        quotas = allocate_label_quotas(label_counts, total_to_copy)

        selected_parts: list[pd.DataFrame] = []
        for label, quota in quotas.items():
            if quota == 0:
                continue

            available = source_df[source_df["label"] == label].copy()
            if len(available) < quota:
                raise RuntimeError(
                    f"Not enough {source_split!r} files for label {label}. "
                    f"Need {quota}, have {len(available)}."
                )

            sampled_indices = rng.choice(available.index.to_numpy(), size=quota, replace=False)
            sampled = available.loc[sampled_indices].copy()
            sampled["quota_for_label"] = quota
            sampled["source_split_count_before"] = source_count
            sampled["planned_total_copies_from_split"] = total_to_copy
            sampled["target_leak_fraction"] = leak_fraction
            selected_parts.append(sampled)

        if not selected_parts:
            continue

        selected = pd.concat(selected_parts, ignore_index=True)
        selected = selected.sort_values(["label", "source_file"]).reset_index(drop=True)
        selected["copy_index_from_source_split"] = np.arange(1, len(selected) + 1)
        selected["destination_file"] = selected.apply(
            lambda row: build_destination_relpath(
                source_rel_inside_split=Path(row["source_rel_inside_split"]),
                source_split=str(row["source_split"]),
                label=int(row["label"]),
                copy_index=int(row["copy_index_from_source_split"]),
            ),
            axis=1,
        )
        selected["destination_path"] = selected["destination_file"].map(
            lambda relpath: str(dataset_root / relpath)
        )
        selected["planned_action"] = "would_copy"
        plan_parts.append(selected)

    if not plan_parts:
        return pd.DataFrame(
            columns=[
                "source_path",
                "source_file",
                "source_split",
                "label",
                "destination_file",
                "destination_path",
                "target_leak_fraction",
                "planned_action",
            ]
        )

    return pd.concat(plan_parts, ignore_index=True)


def build_destination_relpath(
    *,
    source_rel_inside_split: Path,
    source_split: str,
    label: int,
    copy_index: int,
) -> str:
    """Create a collision-resistant train path that preserves label parsing."""
    parent = source_rel_inside_split.parent
    stem = source_rel_inside_split.stem
    label_suffix = f"_{label}"
    base_stem = stem[:-len(label_suffix)] if stem.endswith(label_suffix) else stem
    filename = f"{base_stem}_leak_from_{source_split}_{copy_index:06d}_{label}.npy"
    return str(Path("train") / parent / filename)


def apply_leakage_plan(plan: pd.DataFrame) -> None:
    """Execute a copy plan produced by ``build_leakage_plan``."""
    if plan.empty:
        return

    required = {"source_path", "destination_path"}
    missing = required.difference(plan.columns)
    if missing:
        raise ValueError(f"Plan is missing required columns: {sorted(missing)}")

    for row in plan.itertuples(index=False):
        source = Path(row.source_path)
        destination = Path(row.destination_path)
        if not source.exists():
            raise FileNotFoundError(f"Missing planned source file: {source}")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def summarize_plan(plan: pd.DataFrame) -> pd.DataFrame:
    if plan.empty:
        return pd.DataFrame(
            columns=[
                "source_split",
                "source_split_count_before",
                "planned_copies_into_train",
                "target_leak_fraction",
                "achieved_source_fraction_copied",
            ]
        )

    summary = (
        plan.groupby("source_split", dropna=False)
        .agg(
            source_split_count_before=("source_split_count_before", "first"),
            planned_copies_into_train=("source_file", "count"),
            target_leak_fraction=("target_leak_fraction", "first"),
        )
        .reset_index()
    )
    summary["achieved_source_fraction_copied"] = (
        summary["planned_copies_into_train"] / summary["source_split_count_before"]
    )
    return summary


def summarize_plan_by_label(plan: pd.DataFrame) -> pd.DataFrame:
    if plan.empty:
        return pd.DataFrame(columns=["source_split", "label", "planned_copies_into_train"])
    return (
        plan.groupby(["source_split", "label"], dropna=False)
        .size()
        .rename("planned_copies_into_train")
        .reset_index()
        .sort_values(["source_split", "label"])
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy stratified subsets of test and val into train for leakage diagnostics only."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--leak-fraction",
        type=float,
        default=0.40,
        help=(
            "Fraction of each source eval split to copy into train. "
            "Example: 0.40 means copy 40%% of test and 40%% of val into train."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--plan-csv", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute the copy plan. Without this flag, only write the CSV plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    plan = build_leakage_plan(
        dataset_root=args.dataset_root,
        leak_fraction=args.leak_fraction,
        random_state=args.random_state,
    )

    args.plan_csv.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(args.plan_csv, index=False)

    print("Leakage plan written to:", args.plan_csv)
    print()
    print(summarize_plan(plan).to_string(index=False))
    print()
    print(summarize_plan_by_label(plan).to_string(index=False))

    if args.apply:
        apply_leakage_plan(plan)
        print()
        print(f"Copied {len(plan)} test/val files into train.")
    else:
        print()
        print("Dry-run only. No files were copied. Use --apply to execute the plan.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
