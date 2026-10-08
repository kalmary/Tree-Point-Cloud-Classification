from __future__ import annotations

import argparse
from typing import Union
import pathlib as pth
import shutil
import sys

import random

_preprocessing_dependencies_loaded = False


def _load_preprocessing_dependencies():
    global _preprocessing_dependencies_loaded
    global LabelEncoder, convert_str_values, fpsample, h5py, laspy, load_json
    global np, pd, save_to_json, tqdm, train_test_split

    if _preprocessing_dependencies_loaded:
        return

    import fpsample as fpsample_module
    import h5py as h5py_module
    import laspy as laspy_module
    import numpy as numpy_module
    import pandas as pandas_module
    from sklearn.model_selection import train_test_split as split_dataset
    from sklearn.preprocessing import LabelEncoder as label_encoder
    from tqdm import tqdm as progress_bar

    if __package__:
        from ..utils.nn_utils import (
            convert_str_values as convert_values,
            load_json as load_config,
            save_to_json as save_config,
        )
    else:
        main_dir = pth.Path(__file__).parent.parent
        sys.path.insert(0, str(main_dir))
        from utils.nn_utils import (
            convert_str_values as convert_values,
            load_json as load_config,
            save_to_json as save_config,
        )

    fpsample = fpsample_module
    h5py = h5py_module
    laspy = laspy_module
    np = numpy_module
    pd = pandas_module
    train_test_split = split_dataset
    LabelEncoder = label_encoder
    tqdm = progress_bar
    convert_str_values = convert_values
    load_json = load_config
    save_to_json = save_config
    _preprocessing_dependencies_loaded = True


def decimate_chunk_laz(work_dir: pth.Path, goal_dir: pth.Path, folder_split: dict, metadata: pd.DataFrame) -> None:
    _load_preprocessing_dependencies()
    """
    Reads .laz files, looks up per-file labels from metadata, performs FPS subsampling,
    and saves (N, 4) arrays [X, Y, Z, label] into train/test/val subfolders.
    Files are split per source filename to avoid data leakage.
    Files not found in metadata are assigned label 19 ("other").
    """
    OTHER_LABEL = 15

    if not work_dir.exists():
        raise ValueError('Incorrect path:', work_dir)

    goal_dir.mkdir(parents=True, exist_ok=True)

    train_pth = goal_dir.joinpath('train')
    train_pth.mkdir(exist_ok=True, parents=True)
    test_pth = goal_dir.joinpath('test')
    test_pth.mkdir(exist_ok=True, parents=True)
    val_pth = goal_dir.joinpath('val')
    val_pth.mkdir(exist_ok=True, parents=True)

    all_paths = list(work_dir.rglob('*.laz'))

    # Build treeID -> label lookup from metadata
    tree_id_to_label: dict[str, int] = dict(
        zip(
            metadata['filename'].apply(lambda x: pth.Path(x).stem),
            metadata['species_number'].astype(int)
        )
    )


    # All files are included; missing metadata entries get label OTHER_LABEL
    labeled_paths = []
    for p in all_paths:
        tree_id = p.stem
        if tree_id not in tree_id_to_label:
            # print(f'[WARN] No metadata entry for {p.name}, assigning to "other" ({OTHER_LABEL}).')
            tree_id_to_label[tree_id] = OTHER_LABEL
        labeled_paths.append(p)

    if len(labeled_paths) == 0:
        raise RuntimeError('No .laz files found in source directory.')

    # Stratified split per file (not per point) to avoid leakage
    labels_for_split = [tree_id_to_label[p.stem] for p in labeled_paths]

    train_paths, test_paths = train_test_split(
        labeled_paths,
        train_size=folder_split['train_ratio'],
        random_state=42,
        shuffle=True,
        stratify=labels_for_split,
    )

    test_labels_for_split = [tree_id_to_label[p.stem] for p in test_paths]
    test_paths, val_paths = train_test_split(
        test_paths,
        train_size=folder_split['test_ratio'] / (folder_split['test_ratio'] + folder_split['val_ratio']),
        random_state=42,
        shuffle=True,
        stratify=test_labels_for_split,
    )

    def decimate_folder(paths: list[pth.Path], goal: pth.Path, desc: str, n_points: int = 16384):
        for path in tqdm(paths, desc=desc, total=len(paths)):
            tree_id = path.stem
            label = tree_id_to_label[tree_id]

            try:
                las = laspy.read(path)
            except Exception as e:
                print(f'[ERROR] Could not read {path.name}: {e}')
                continue

            xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
            xyz -= xyz.mean(axis=0)  # center

            n = xyz.shape[0]

            if n > n_points:
                sampled_idx = fpsample.bucket_fps_kdline_sampling(xyz, n_points, h=7)
            elif 0 < n < n_points:
                sampled_idx = np.random.choice(n, n_points, replace=True)
            elif n == n_points:
                sampled_idx = np.arange(n)
            else:
                # print(f'[SKIP] {path.name} has only {n} points (< {n_points // 2}), skipping.')
                continue

            xyz = xyz[sampled_idx]  # (N, 3)

            labels = np.full((n_points, 1), fill_value=label, dtype=np.int64)  # (N, 1)
            points_with_label = np.concatenate([xyz, labels], axis=1)  # (N, 4)

            out_path = goal / f'{path.stem}_{label}.npy'
            np.save(out_path, points_with_label)

    decimate_folder(train_paths, train_pth, desc=f'Decimating train  [{work_dir.name}]')
    decimate_folder(test_paths,  test_pth,  desc=f'Decimating test   [{work_dir.name}]')
    decimate_folder(val_paths,   val_pth,   desc=f'Decimating val    [{work_dir.name}]')


def convert_dataset(work_dir: pth.Path, goal_dir: pth.Path) -> tuple[pth.Path, pth.Path, pth.Path]:
    _load_preprocessing_dependencies()
    """
    Chunks .npy files (N, 4) = [X, Y, Z, label] into HDF5 datasets.
    Each HDF5 dataset has shape (chunk_h5_shape, N, 4).
    """
    work_train = work_dir.joinpath('train')
    work_test  = work_dir.joinpath('test')
    work_val   = work_dir.joinpath('val')

    if work_dir == goal_dir:
        return work_train, work_test, work_val

    n_points       = 16384
    chunk_h5_shape = 30
    n_features     = 4  # X, Y, Z, label

    if not work_dir.exists():
        raise ValueError('Incorrect path:', work_dir)
    goal_dir.mkdir(parents=True, exist_ok=True)

    train_paths      = list(work_train.rglob('*.npy'))
    test_paths       = list(work_test.rglob('*.npy'))
    validation_paths = list(work_val.rglob('*.npy'))

    def convert2_h5(path_list: list[pth.Path], h5_path: pth.Path):
        if len(path_list) == 0:
            print(f'[WARN] No .npy files found for {h5_path.name}, skipping.')
            return

        with h5py.File(h5_path, 'w') as h5_file:
            chunk_buf = np.zeros((0, n_points, n_features), dtype=np.float32)
            chunk_num = 0

            for path in tqdm(path_list, desc=f'Converting -> {h5_path.name}', total=len(path_list)):
                arr = np.load(path).astype(np.float32)  # (N, 4)

                if arr.shape != (n_points, n_features):
                    print(f'[WARN] Unexpected shape {arr.shape} in {path.name}, skipping.')
                    continue

                arr = arr[np.newaxis, ...]  # (1, N, 4)
                chunk_buf = np.concatenate([chunk_buf, arr], axis=0)

                if chunk_buf.shape[0] >= chunk_h5_shape:
                    h5_file.create_dataset(str(chunk_num), data=chunk_buf)
                    chunk_buf = np.zeros((0, n_points, n_features), dtype=np.float32)
                    chunk_num += 1

            # flush remaining
            if chunk_buf.shape[0] > 0:
                h5_file.create_dataset(str(chunk_num), data=chunk_buf)

    convert2_h5(train_paths,      goal_dir / 'train.h5')
    convert2_h5(test_paths,       goal_dir / 'test.h5')
    convert2_h5(validation_paths, goal_dir / 'validation.h5')

    return goal_dir / 'train.h5', goal_dir / 'test.h5', goal_dir / 'validation.h5'


def rebalance_dataset(work_dir: pth.Path, folder_split: dict, tolerance=0.03):
    _load_preprocessing_dependencies()
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
                    "1. .npy files - checkpoint part, files cut, but not converted to faster format\n"
                    "2. .hdf5 files - files used during training/ validation/ testing.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--source-path',    type=str, default="")
    parser.add_argument('--decimated-path', type=str, default="")
    parser.add_argument('--converted-path', type=str, default="")

    parser.add_argument(
        '--folder-split',
        type=Union[list[int], list[str]],
        default=[0.7, 0.2, 0.1],
        help="Folder split ratios for train, test, validation."
    )

    parser.add_argument('--metadata-path', type=Union[str, pth.Path], default=None)
    parser.add_argument('--species-path',  type=Union[str, pth.Path], default=None)

    return parser.parse_args()


def get_metadata(
    path2csv: Union[str, pth.Path],
    species: list,
    merge_map: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _load_preprocessing_dependencies()
    
    df = pd.read_csv(path2csv)

    # Apply merges before anything else
    if merge_map:
        for new_name, old_names in merge_map.items():
            df.loc[df['species'].isin(old_names), 'species'] = new_name
        # Update species list too: replace merged origins with new name
        species = [s for s in species if s not in {n for v in merge_map.values() for n in v}]
        species += [k for k in merge_map if k not in species]

    specified_species = df[df['species'].isin(species)].copy()
    other_species     = df[~df['species'].isin(species)].copy()

    le = LabelEncoder()
    specified_species['species_number'] = le.fit_transform(specified_species['species'])

    if len(other_species) > 0:
        other_species['species']        = 'others'
        other_species['species_number'] = 15

    df = pd.concat([specified_species, other_species], ignore_index=True)
    df = df[['treeID', 'species', 'species_number', 'filename']].copy()

    species_number_counts = df.groupby(['species', 'species_number']).size().reset_index(name='count')
    species_number_counts.to_csv('output.csv', index=False)

    final_metadata = df[['treeID', 'species_number', 'filename']].copy()
    return final_metadata, species_number_counts


def update_paths_config(path2train: pth.Path, path2test: pth.Path, path2val: pth.Path):
    _load_preprocessing_dependencies()

    def _update_path(path2dataset: Union[str, pth.Path], dataset_name: str):
        config_dir = pth.Path(__file__).parent.parent.joinpath('model_pipeline/training_configs')
        path2config_single = config_dir / 'config_train_single.json'
        path2config        = config_dir / 'config_train.json'

        config_single = load_json(path2config_single)
        config        = load_json(path2config)

        config_single[dataset_name] = str(path2dataset)
        config[dataset_name]        = str(path2dataset)

        save_to_json(config_single, path2config_single)
        save_to_json(config,        path2config)

    _update_path(path2train, 'data_path_train')
    _update_path(path2test,  'data_path_test')
    _update_path(path2val,   'data_path_val')


def main():
    parser = argparser()
    _load_preprocessing_dependencies()

    source = pth.Path(parser.source_path) if parser.source_path else \
             pth.Path(__file__).parent.parent.parent / 'data/raw'

    decimated = pth.Path(parser.decimated_path) if parser.decimated_path else \
                pth.Path(__file__).parent.parent.parent / 'data/decimated'

    converted = pth.Path(parser.converted_path) if parser.converted_path else decimated

    metadata_path = pth.Path(parser.metadata_path) if parser.metadata_path else \
                    pth.Path(__file__).parent.parent.parent / 'data/tree_metadata_dev.csv'

    species_path = pth.Path(parser.species_path) if parser.species_path else \
                   pth.Path(__file__).parent.parent.parent / 'data/species.txt'

    with species_path.open('r') as f:
        species = f.read().splitlines()

    folder_split = convert_str_values({
        'train_ratio': parser.folder_split[0],
        'test_ratio':  parser.folder_split[1],
        'val_ratio':   parser.folder_split[2],
    })
    merge_map = {
        'Pinus':        ['Pinus_sylvestris'],
        'Picea_abies':  ['Picea_abies'],
        'Abies':        ['Abies_alba'],
        'Larix':        ['Larix_decidua'],
        'Pseudotsuga':  ['Pseudotsuga_menziesii'],
        'Quercus':      ['Quercus_petraea', 'Quercus_rubra', 'Quercus_robur'],
        'Ulmus':        ['Ulmus_laevis'],
        'Fagus':        ['Fagus_sylvatica'],
        'Tilia':        ['Tilia_cordata'],
        'Carpinus':     ['Carpinus_betulus'],
        'Acer':         ['Acer_pseudoplatanus', 'Acer_campestre'],
        'Fraxinus':     ['Fraxinus_excelsior'],
        'Betula':       ['Betula_pendula'],
        'Corylus':      ['Corylus_avellana'],
        'Crataegus':    ['Crataegus_monogyna'],
    }

    metadata, species_counts = get_metadata(metadata_path, species, merge_map=merge_map)

    
    print(f"Loaded metadata: {len(metadata)} entries, {species_counts.shape[0]} species groups.")
    print(metadata, species_counts)
    decimate_chunk_laz(source, decimated, folder_split, metadata)
    rebalance_dataset(decimated, folder_split)

    path2train, path2test, path2val = convert_dataset(decimated, converted)
    update_paths_config(path2train, path2test, path2val)


if __name__ == '__main__':
    main()
