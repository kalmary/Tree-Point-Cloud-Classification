#!/usr/bin/env python3
"""Downsample one label by a fixed per-split keep fraction.

Current intended use case:
- target label: 15
- keep fraction: 0.70
- audit file: files_audit.csv produced by the dataset audit script

The script preserves train/test/val proportions by operating split-by-split.
It removes the least useful target-label files first according to the audit CSV:

  1. utility_score == 2
  2. score-1 redundancy/leakage review cases
  3. remaining score-1 cases
  4. random score-0 files only if further downsampling is required

Dry-run is the default. Physical deletion requires --apply.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "test", "val")
REQUIRED_AUDIT_COLUMNS = {
    "file",
    "split",
    "label",
    "utility_score",
    "utility_reason",
    "is_extremely_redundant",
    "is_cross_split_near_duplicate",
}


def build_downsample_plan(
    *,
    dataset_root: Path,
    audit_csv: Path,
    target_label: int = 15,
    keep_fraction: float = 0.70,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a deletion plan without modifying the dataset."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not audit_csv.exists():
        raise FileNotFoundError(f"Audit CSV does not exist: {audit_csv}")
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in the interval (0, 1].")

    audit = pd.read_csv(audit_csv)
    missing_columns = REQUIRED_AUDIT_COLUMNS.difference(audit.columns)
    if missing_columns:
        raise ValueError(
            f"Audit CSV is missing required columns: {sorted(missing_columns)}"
        )

    audit = audit.copy()
    audit["label"] = audit["label"].astype(int)
    audit["utility_score"] = audit["utility_score"].astype(int)
    audit["absolute_path"] = audit["file"].map(
        lambda relpath: dataset_root / str(relpath)
    )
    audit["file_exists"] = audit["absolute_path"].map(Path.exists)

    missing_files = audit.loc[~audit["file_exists"], "file"].tolist()
    if missing_files:
        preview = missing_files[:10]
        suffix = (
            ""
            if len(missing_files) <= 10
            else f" ... and {len(missing_files) - 10} more"
        )
        raise FileNotFoundError(
            "Audit CSV references files that do not exist under dataset_root: "
            f"{preview}{suffix}"
        )

    rng = np.random.default_rng(random_state)
    plans: list[pd.DataFrame] = []

    for split in SPLITS:
        target_rows = audit[
            (audit["split"] == split) & (audit["label"] == target_label)
        ].copy()

        current_target_count = int(len(target_rows))
        target_keep_count = math.ceil(current_target_count * keep_fraction)
        remove_count = current_target_count - target_keep_count

        if remove_count == 0:
            continue

        target_rows["removal_priority"] = _removal_priority(target_rows)
        target_rows["random_tiebreaker"] = rng.random(len(target_rows))
        target_rows = target_rows.sort_values(
            ["removal_priority", "random_tiebreaker", "file"],
            ascending=[True, True, True],
        )

        selected = target_rows.head(remove_count).copy()
        selected["keep_fraction"] = keep_fraction
        selected["current_target_count_in_split"] = current_target_count
        selected["target_keep_count_in_split"] = target_keep_count
        selected["planned_removals_in_split"] = remove_count
        selected["planned_action"] = "would_delete"
        plans.append(selected)

    if not plans:
        return pd.DataFrame(
            columns=[
                "file",
                "split",
                "label",
                "utility_score",
                "utility_reason",
                "absolute_path",
                "removal_priority",
                "keep_fraction",
                "current_target_count_in_split",
                "target_keep_count_in_split",
                "planned_removals_in_split",
                "planned_action",
            ]
        )

    return pd.concat(plans, ignore_index=True)


def apply_downsample_plan(plan: pd.DataFrame) -> None:
    """Delete files listed in a previously generated plan."""
    if plan.empty:
        return
    if "absolute_path" not in plan.columns:
        raise ValueError("Plan is missing the 'absolute_path' column")

    missing: list[str] = []
    for raw_path in plan["absolute_path"]:
        path = Path(raw_path)
        if not path.exists():
            missing.append(str(path))
            continue
        path.unlink()

    if missing:
        preview = missing[:10]
        suffix = (
            ""
            if len(missing) <= 10
            else f" ... and {len(missing) - 10} more"
        )
        raise FileNotFoundError(
            f"Some planned files were missing during deletion: {preview}{suffix}"
        )


def summarize_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """Return a concise split-level summary of the deletion plan."""
    if plan.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "planned_removals",
                "current_target_count_in_split",
                "target_keep_count_in_split",
                "keep_fraction",
            ]
        )

    columns = [
        "split",
        "current_target_count_in_split",
        "target_keep_count_in_split",
        "planned_removals_in_split",
        "keep_fraction",
    ]
    summary = plan[columns].drop_duplicates().copy()
    summary = summary.rename(
        columns={"planned_removals_in_split": "planned_removals"}
    )
    return summary.sort_values("split").reset_index(drop=True)


def _removal_priority(rows: pd.DataFrame) -> np.ndarray:
    utility_score = rows["utility_score"]
    utility_reason = rows["utility_reason"].astype(str)

    redundancy_like_reason = (
        utility_reason.str.contains("redund", case=False, na=False)
        | utility_reason.str.contains("duplicate", case=False, na=False)
        | utility_reason.str.contains("leakage", case=False, na=False)
    )
    redundancy_like_flag = (
        rows["is_extremely_redundant"].astype(bool)
        | rows["is_cross_split_near_duplicate"].astype(bool)
    )

    return np.select(
        [
            utility_score.eq(2),
            utility_score.eq(1)
            & (redundancy_like_reason | redundancy_like_flag),
            utility_score.eq(1),
        ],
        [0, 1, 2],
        default=3,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Downsample one label by a fixed keep fraction, split-by-split."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--target-label", type=int, default=15)
    parser.add_argument(
        "--keep-fraction",
        type=float,
        default=0.70,
        help=(
            "Fraction of target-label files to keep in each split. "
            "Default: 0.70."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--plan-csv", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Delete files in the generated plan. "
            "Without this flag, the script is dry-run only."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    plan = build_downsample_plan(
        dataset_root=args.dataset_root,
        audit_csv=args.audit_csv,
        target_label=args.target_label,
        keep_fraction=args.keep_fraction,
        random_state=args.random_state,
    )

    args.plan_csv.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(args.plan_csv, index=False)

    print("Deletion plan written to:", args.plan_csv)
    print()
    print(summarize_plan(plan).to_string(index=False))

    if args.apply:
        apply_downsample_plan(plan)
        print()
        print(
            f"Deleted {len(plan)} files from label {args.target_label}."
        )
    else:
        print()
        print(
            "Dry-run only. No files were deleted. "
            "Use --apply to execute the plan."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())