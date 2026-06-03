#!/usr/bin/env python3
"""Remove tree point-cloud .npy files whose vertical extent is below a threshold.

Usage example:
    python src/data_processing/remove_low_trees.py --source /path/to/npy/dir

The script computes the tree height as the z-axis range of the point cloud
(i.e. max(z) - min(z)) and removes any .npy file with height < min_height.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def tree_height_from_npy(path: Path) -> float:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected point cloud with at least 3 columns, got {arr.shape} from {path}")
    z = arr[:, 2]
    if z.size == 0:
        return 0.0
    return float(np.max(z) - np.min(z))


def remove_short_trees(source: Path, min_height: float, dry_run: bool = False) -> int:
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    removed_count = 0
    all_paths = sorted(source.rglob("*.npy"))
    if not all_paths:
        raise RuntimeError(f"No .npy files found under {source}")

    for path in tqdm(all_paths, desc="Scanning .npy files", unit="file"):
        try:
            height = tree_height_from_npy(path)
        except Exception as exc:
            tqdm.write(f"[WARN] Skipping {path}: {exc}")
            continue

        if height < min_height:
            if dry_run:
                tqdm.write(f"[DRY-RUN] Would remove {path} (height={height:.3f} m)")
            else:
                path.unlink()
                tqdm.write(f"Removed {path} (height={height:.3f} m)")
            removed_count += 1

    return removed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove .npy tree point-cloud files with vertical height below a threshold."
    )
    parser.add_argument("--source", type=Path, required=True, help="Root directory containing .npy tree point-cloud files")
    parser.add_argument("--min-height", type=float, default=1.5, help="Minimum tree height in meters; files below this are removed")
    parser.add_argument("--dry-run", action="store_true", help="Only print files that would be removed, do not delete them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    removed_count = remove_short_trees(args.source, args.min_height, dry_run=args.dry_run)
    print(f"Checked {args.source}. Removed {removed_count} file(s) below {args.min_height} m.")


if __name__ == "__main__":
    main()
