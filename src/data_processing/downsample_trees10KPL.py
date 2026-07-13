import argparse
from typing import Union
import pathlib as pth
import h5py
import shutil
from tqdm import tqdm
import sys
import fpsample

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import laspy
import numpy as np

import sys
main_dir = pth.Path(__file__).parent.parent
sys.path.append(str(main_dir))

from utils import convert_str_values, load_json, save2json
import pandas as pd

from typing import Optional


def decimate_chunk_laz(work_dir: pth.Path, goal_dir: pth.Path, n_points: int = 16384) -> None:
    """
    Reads .laz files, looks up per-file labels from metadata, performs FPS subsampling,
    and saves (N, 4) arrays [X, Y, Z, label] into train/test/val subfolders.
    Files are split per source filename to avoid data leakage.
    Files not found in metadata are assigned label 19 ("other").
    """
    # OTHER_LABEL = 15

    if not work_dir.exists():
        raise ValueError('Incorrect path:', work_dir)

    goal_dir.mkdir(parents=True, exist_ok=True)

    # train_pth = goal_dir.joinpath('train')
    # train_pth.mkdir(exist_ok=True, parents=True)
    # test_pth = goal_dir.joinpath('test')
    # test_pth.mkdir(exist_ok=True, parents=True)
    # val_pth = goal_dir.joinpath('val')
    # val_pth.mkdir(exist_ok=True, parents=True)

    all_paths = list(work_dir.rglob('*.laz'))

    # # All files are included; missing metadata entries get label OTHER_LABEL

    # # Stratified split per file (not per point) to avoid leakage
    # labels_for_split = [tree_id_to_label[p.stem] for p in labeled_paths]

    # train_paths, test_paths = train_test_split(
    #     labeled_paths,
    #     train_size=folder_split['train_ratio'],
    #     random_state=42,
    #     shuffle=True,
    #     stratify=labels_for_split,
    # )

    # test_labels_for_split = [tree_id_to_label[p.stem] for p in test_paths]
    # test_paths, val_paths = train_test_split(
    #     test_paths,
    #     train_size=folder_split['test_ratio'] / (folder_split['test_ratio'] + folder_split['val_ratio']),
    #     random_state=42,
    #     shuffle=True,
    #     stratify=test_labels_for_split,
    # )

    # def decimate_folder(paths: list[pth.Path], goal: pth.Path, desc: str, n_points: int = 16384):
    for path in tqdm(all_paths, desc='Raw files - Tree extraction and decimation', total=len(all_paths)):

        try:
            las = laspy.read(path)
        except Exception as e:
            print(f'[ERROR] Could not read {path.name}: {e}')
            continue

        full_xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
        full_xyz -= full_xyz.mean(axis=0)  # center

        full_treeID = np.asarray(las.treeID)
        full_treeSP = np.asarray(las.treeSP)
        full_completelyInside = np.asarray(las.completelyInside)

        for tree_id in np.unique(full_treeID).flatten():
            if tree_id == 0:
                continue  # Skip points with treeID 0 (not part of any tree)
            
            tree_mask = full_treeID == tree_id
            xyz = full_xyz[tree_mask]
            tree_height = xyz[:, 2].max() - xyz[:, 2].min()
            if tree_height < 1.4:
                continue

            label = full_treeSP[tree_mask][0]  # Assuming all points of the same tree have the same species label
            if label == 0:
                continue  # Skip trees with species label 0 (not part of any species)

            completely_inside = full_completelyInside[tree_mask][0]  # Assuming all points

            n = xyz.shape[0]

            if n > n_points:
                sampled_idx = fpsample.bucket_fps_kdline_sampling(xyz, n_points, h=7)
            elif 0 < n < n_points:
                sampled_idx = np.random.choice(n, n_points, replace=True)
            elif n == n_points:
                sampled_idx = np.arange(n)

            xyz = xyz[sampled_idx]  # (N, 3)

            labels = np.full((n_points, 1), fill_value=label, dtype=np.int64)  # (N, 1)
            points_with_label = np.concatenate([xyz, labels], axis=1)  # (N, 4)

            out_path = goal_dir / f'{path.stem}_{completely_inside}_{label}.npy'
            np.save(out_path, points_with_label)


def split_data(work_dir: pth.Path, goal_dir: pth.Path, folder_split: dict):
    """
    Splits .npy files into train/test/val folders based on the provided ratios.
    """
    if not work_dir.exists():
        raise ValueError('Incorrect path:', work_dir)

    goal_dir.mkdir(parents=True, exist_ok=True)

    train_pth = goal_dir.joinpath('train')
    train_pth.mkdir(exist_ok=True, parents=True)
    test_pth = goal_dir.joinpath('test')
    test_pth.mkdir(exist_ok=True, parents=True)
    val_pth = goal_dir.joinpath('val')
    val_pth.mkdir(exist_ok=True, parents=True)

    all_files = list(work_dir.rglob('*.npy'))
    labels = [f.stem.split('_')[-1] for f in all_files]  # Extract labels from filenames
    encoder = LabelEncoder()  # Encode labels so they start from 0 and are consecutive integers
    labels_org = np.array(labels).copy()
    labels = encoder.fit_transform(labels)

    total_files = len(all_files)

    if total_files == 0:
        raise RuntimeError("No .npy files found in the source directory.")

    train_ratio = folder_split['train_ratio']
    test_ratio = folder_split['test_ratio']
    val_ratio = folder_split['val_ratio']

    train_files, temp_files = train_test_split(
        all_files,
        train_size=train_ratio,
        random_state=42,
        stratify=labels,
        shuffle=True
    )

    test_size_adjusted = test_ratio / (test_ratio + val_ratio)
    test_files, val_files = train_test_split(
        temp_files,
        train_size=test_size_adjusted,
        random_state=42,
        stratify=[labels[all_files.index(f)] for f in temp_files],
        shuffle=True
    )
    #edit f.name - use new label
    def copy_files(files, dest_pth):
        for f in files:
            f_name = f.stem.split('_')[:-1]  # get all elements but the last one (label)
            f_name.append(str(encoder.transform([f.stem.split('_')[-1]])[0]))  # append new label
            f_name_new = '_'.join(f_name) + '.npy'
            shutil.copy(str(f), str(dest_pth / f_name_new))

    copy_files(train_files, train_pth)
    copy_files(test_files, test_pth)
    copy_files(val_files, val_pth)

    return labels_org



def rebalance_dataset(work_dir: pth.Path, folder_split: dict, tolerance=0.03):
    work_train = work_dir.joinpath('train')
    work_test  = work_dir.joinpath('test')
    work_val   = work_dir.joinpath('val')

    work_pths  = [work_train, work_test, work_val]
    split_keys = ['train_ratio', 'test_ratio', 'val_ratio']
    folder_ratio = [folder_split[k] for k in split_keys]

    if not work_dir.exists():
        raise ValueError('Incorrect path:', work_dir)

    counts = np.array([len(list(p.rglob('*.npy'))) for p in work_pths])
    total  = counts.sum()

    print(f"File counts before balancing: train={counts[0]}, test={counts[1]}, val={counts[2]}")

    if total == 0:
        raise RuntimeError("No .npy files found in dataset folders.")

    current_ratio = counts / total
    desired_ratio = np.array(folder_ratio)

    if np.all(np.abs(current_ratio - desired_ratio) <= tolerance):
        print("No rebalancing needed. Folders are within desired ratio.")
        return

    target_counts = np.round(desired_ratio * total).astype(int)
    diffs = counts - target_counts

    surplus = np.where(diffs > 0)[0]
    deficit = np.where(diffs < 0)[0]

    move_log = {}

    for s in surplus:
        files = list(work_pths[s].rglob('*.npy'))
        np.random.shuffle(files)
        for d in deficit:
            move_n = min(diffs[s], -diffs[d])
            if move_n <= 0:
                continue

            to_move = files[:move_n]
            for f in to_move:
                dest = work_pths[d] / f.name
                if dest.exists():
                    raise FileExistsError(f"Destination file already exists: {dest}")
                shutil.move(str(f), str(dest))

            move_log[f"{split_keys[s]} -> {split_keys[d]}"] = move_n
            diffs[s] -= move_n
            diffs[d] += move_n
            files = files[move_n:]

            if np.all(diffs == 0):
                break

    new_counts = [len(list(p.rglob('*.npy'))) for p in work_pths]
    print(f"File counts after balancing: train={new_counts[0]}, test={new_counts[1]}, val={new_counts[2]}")
    print("Move summary:", move_log)


def argparser():
    parser = argparse.ArgumentParser(
        description="Script for preprocessing .LAZ files. Each point cloud is fragmented into voxels, decimated and stored in:\n"
                    "1. .npy files - checkpoint part, files cut, but not splitted to faster format\n"
                    "2. .hdf5 files - files used during training/ validation/ testing.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--source_path',    type=str, default="")
    parser.add_argument('--decimated_path', type=str, default="")
    parser.add_argument('--splitted_path', type=str, default="")

    parser.add_argument(
        '--folder_split',
        type=Union[list[int], list[str]],
        default=[0.7, 0.2, 0.1],
        help="Folder split ratios for train, test, validation."
    )

    parser.add_argument('--metadata_path', type=str, default="")

    return parser.parse_args()

def find_metadata(work_dir: pth.Path, labels_org: np.ndarray, species: pd.DataFrame, dir_save: Optional[pth.Path]=None, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    args:
        work_dir: directory containing the dataset
        labels_org: original labels array
        species: DataFrame containing species metadata
        dir_save: optional argument, if given metadata will be saved to this directory
        verbose: if True, will print metadata
    returns:
        [metadata A, metadata B]: list of dataframes, where species info is stored

    1. creates table A with columns: label, species_name, count_train, count_test, count_val
    2. creates table B with columns: file_num, file_name, label, species_name, folder_name (train/test/val)
    """

    files = list(work_dir.rglob('*.npy'))
    if not files:
        raise ValueError(f"No .npy files found in {work_dir}")

    required_columns = {'species_code', 'species'}
    missing_columns = required_columns - set(species.columns)
    if missing_columns:
        raise ValueError(f"Missing species metadata columns: {sorted(missing_columns)}")

    encoder = LabelEncoder()
    encoder.fit(labels_org)
    species_names = (
        species
        .assign(species_code=lambda df: df['species_code'].astype(str))
        .drop_duplicates('species_code')
        .set_index('species_code')['species']
        .to_dict()
    )

    rows = []
    for file_num, file in enumerate(sorted(files)):
        label = int(file.stem.rsplit('_', 1)[-1])
        original_label = encoder.inverse_transform([label])[0]
        folder_name = file.parent.name
        species_name = species_names.get(str(original_label), f'species_{original_label}')
        rows.append({
            'file_num': file_num,
            'file_name': file.name,
            'label': label,
            'species_name': species_name,
            'folder_name': folder_name,
        })

    metadata_B = pd.DataFrame(
        rows,
        columns=['file_num', 'file_name', 'label', 'species_name', 'folder_name'],
    )

    counts = (
        metadata_B
        .groupby(['label', 'species_name', 'folder_name'])
        .size()
        .unstack(fill_value=0)
    )

    metadata_A = counts.rename(columns={
        'train': 'count_train',
        'test': 'count_test',
        'val': 'count_val',
    }).reset_index()

    for column in ['count_train', 'count_test', 'count_val']:
        if column not in metadata_A:
            metadata_A[column] = 0

    metadata_A = metadata_A[
        ['label', 'species_name', 'count_train', 'count_test', 'count_val']
    ].sort_values('label').reset_index(drop=True)

    if dir_save is not None:
        dir_save.mkdir(parents=True, exist_ok=True)
        metadata_A.to_csv(dir_save / 'metadata_A.csv', index=False)
        metadata_B.to_csv(dir_save / 'metadata_B.csv', index=False)

    if verbose:
        print(metadata_A)
        print(metadata_B)

    return metadata_A, metadata_B


def update_paths_config(path2train: pth.Path, path2test: pth.Path, path2val: pth.Path):

    def _update_path(path2dataset: Union[str, pth.Path], dataset_name: str):
        config_dir = pth.Path(__file__).parent.parent.joinpath('model_pipeline/training_configs')
        path2config_single = config_dir / 'config_train_single.json'
        path2config        = config_dir / 'config_train.json'

        config_single = load_json(path2config_single)
        config        = load_json(path2config)

        config_single[dataset_name] = str(path2dataset)
        config[dataset_name]        = str(path2dataset)

        save2json(config_single, path2config_single)
        save2json(config,        path2config)

    _update_path(path2train, 'data_path_train')
    _update_path(path2test,  'data_path_test')
    _update_path(path2val,   'data_path_val')


def main():
    parser = argparser()

    source = pth.Path(parser.source_path) if parser.source_path else \
             pth.Path(__file__).parent.parent.parent / 'data/raw'

    decimated = pth.Path(parser.decimated_path) if parser.decimated_path else \
                pth.Path(__file__).parent.parent.parent / 'data/decimated'

    splitted = pth.Path(parser.splitted_path) if parser.splitted_path else decimated

    metadata_path = pth.Path(parser.metadata_path) if parser.metadata_path else \
                    pth.Path(__file__).parent.parent.parent / 'data/species_id_names.csv'

    species = pd.read_csv(metadata_path)

    folder_split = convert_str_values({
        'train_ratio': parser.folder_split[0],
        'test_ratio':  parser.folder_split[1],
        'val_ratio':   parser.folder_split[2],
    })

    decimate_chunk_laz(source, decimated)
    labels_org = split_data(decimated, splitted, folder_split)

    find_metadata(work_dir=splitted,
                  species=species,
                  labels_org=labels_org,
                  dir_save=splitted.parent,
                  verbose=True)


if __name__ == '__main__':
    main()
