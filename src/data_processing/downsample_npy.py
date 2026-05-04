import argparse
import pathlib as pth
from typing import Union

import fpsample
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm



def fps_sample(xyz: np.ndarray, n_points) -> np.ndarray:
    n = xyz.shape[0]
    if n > n_points:
        return fpsample.bucket_fps_kdline_sampling(xyz, n_points, h=7)
    if 0 < n < n_points:
        return np.random.choice(n, n_points, replace=True)
    if n == n_points:
        return np.arange(n)
    return None

def train_test_split_safe(paths, labels, train_size, random_state=42):
    label_counts = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1
    stratify = labels if all(c >= 2 for c in label_counts.values()) else None
    return train_test_split(
        paths, labels,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )


def split_and_copy(
    source: pth.Path,
    dest: pth.Path,
    train_ratio: float,
    test_ratio: float,
    val_ratio: float,
    n_points: int
) -> None:
    all_paths = list(source.rglob('*.npy'))
    if not all_paths:
        raise RuntimeError(f'No .npy files in {source}')
    
    dest.mkdir(parents=True, exist_ok=True)

    labels = [int(p.stem.rsplit('_', 1)[-1]) for p in all_paths]

    train_paths, rest_paths, _, rest_labels = train_test_split_safe(
        all_paths, labels, train_size=train_ratio,
    )
    test_paths, val_paths, _, __ = train_test_split_safe(
            rest_paths, rest_labels,
            train_size=test_ratio / (test_ratio + val_ratio),
    )

    splits = [('train', train_paths), ('test', test_paths), ('val', val_paths)]
    for split_name, paths in splits:
        out_dir = dest / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in tqdm(paths, desc=split_name, leave=True):
            arr = np.load(p).astype(np.float32)
            xyz = arr[:, :3]
            xyz -= xyz.mean(axis=0)

            idx = fps_sample(xyz, n_points)
            if idx is None:
                continue

            out = np.concatenate([xyz[idx], arr[idx, 3:]], axis=1)
            np.save(out_dir / p.name, out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',      type=str,   default='')
    parser.add_argument('--dest',        type=str,   default='')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--test_ratio',  type=float, default=0.2)
    parser.add_argument('--val_ratio',   type=float, default=0.1)
    parser.add_argument('--n_points',    type=int,   default=16384)
    args = parser.parse_args()

    source = pth.Path(args.source) if args.source else pth.Path(__file__).parent.parent / 'data/raw'
    dest   = pth.Path(args.dest)   if args.dest   else pth.Path(__file__).parent.parent / 'data/split'

    split_and_copy(source, dest, args.train_ratio, args.test_ratio, args.val_ratio, args.n_points)


if __name__ == '__main__':
    main()