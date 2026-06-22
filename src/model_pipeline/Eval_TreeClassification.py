import sys
import os
import argparse
import numpy as np
import pathlib as pth
from tqdm import tqdm


import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader


from _data_loader import *
from model_pipeline.model import CNN2D_Residual
from model_pipeline.model_en import EfficientNetClassifier

current_dir = pth.Path(__file__).parent.parent
sys.path.append(str(current_dir.parent))

from utils import load_json, load_model, convert_str_values
from utils import calculate_accuracy, get_intLabels, get_Probabilities,get_dataset_len, compute_pos_weights, FocalLoss
from utils import Plotter, ClassificationReport

OTHERS = None
OTHERS = 15

def voxel_subsample_vectorized(xyz, voxel_size=0.25):
    if xyz.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    xyz -= xyz.mean(axis =0)

    keys     = np.floor(xyz / voxel_size).astype(np.int32)
    centers  = (keys + 0.5) * voxel_size
    dists_sq = np.sum((xyz - centers) ** 2, axis=1)

    keys_min  = keys.min(axis=0)
    keys      = keys - keys_min
    key_range = keys.max(axis=0) + 1

    key_range = key_range.astype(np.int64)
    assert np.prod(key_range) < np.iinfo(np.int64).max, "key encoding overflow"
    strides = np.cumprod(np.r_[1, key_range[:0:-1]], dtype=np.int64)[::-1]
    key_enc = keys.astype(np.int64) @ strides
    
    order      = np.lexsort((dists_sq, key_enc))
    key_sorted = key_enc[order]
    _, first   = np.unique(key_sorted, return_index=True)
    chosen     = order[first]

    mask = np.zeros(xyz.shape[0], dtype=bool)
    mask[chosen] = True
    return mask

def _eval_model(config_dict: dict,
               model: nn.Module) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    device_gpu = torch.device('cuda')
    device_cpu = torch.device('cpu')

    num_classes = config_dict['num_classes']
    ignore_index = OTHERS  # None or int index to ignore in loss/accuracy

    test_dataset = NpyDatasetAug(path_dir=config_dict['data_path_test'],
                                 resolution_xy=config_dict['input_dim'],
                                 training=False,
                                 ignore_index=ignore_index,
                                 device=device_gpu,
                                 n_points=16384,
                                 use_domain_aug=True)
    
    testLoader = DataLoader(
        test_dataset,
        batch_size=config_dict["batch_size"],
        num_workers=15,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

    total = get_dataset_len(testLoader, verbose=False)
    weights, _ = compute_pos_weights(data_dir=config_dict['data_path_train'],
                                    num_classes=num_classes,
                                    power=0.35,
                                    ignore_index=ignore_index)

    criterion = FocalLoss(alpha=weights.cpu(), gamma=config_dict['focal_loss_gamma']).cpu()

    loss_per_epoch = 0.
    accuracy_per_epoch = 0.0
    epoch_samples = 0

    all_predictions = []
    all_probs = np.zeros((0, num_classes))
    all_labels = []

    pbar = tqdm(testLoader, total=total, desc="Testing", unit="batch")
    with torch.no_grad():
        for batch_x, batch_y in pbar:
            model.eval()
            batch_x = batch_x.to(config_dict['device'])
            outputs = model(batch_x)

            if ignore_index is not None:
                mask = batch_y < ignore_index
                if mask.any():
                    loss = criterion(outputs.cpu()[mask], batch_y.cpu()[mask])
                    accuracy = calculate_accuracy(outputs.cpu()[mask], batch_y.cpu()[mask])
                    loss_per_epoch += loss.item() * mask.sum().item()
                    accuracy_per_epoch += accuracy * mask.sum().item()
                    epoch_samples += mask.sum().item()
            else:
                loss = criterion(outputs.cpu(), batch_y.cpu())
                accuracy = calculate_accuracy(outputs.cpu(), batch_y.cpu())
                loss_per_epoch += loss.item() * batch_y.size(0)
                accuracy_per_epoch += accuracy * batch_y.size(0)
                epoch_samples += batch_y.size(0)

            total_loss = loss_per_epoch / epoch_samples if epoch_samples > 0 else 0.
            total_accuracy = accuracy_per_epoch / epoch_samples if epoch_samples > 0 else 0.

            all_labels.extend(batch_y.cpu().tolist())

            probs = get_Probabilities(outputs.cpu())
            int_preds = get_intLabels(probs)

            all_probs = np.concatenate([all_probs, probs.numpy()], axis=0)
            all_predictions.extend(int_preds.numpy())

    return (
        total_loss,
        total_accuracy,
        np.asarray(all_labels),
        all_probs,
        np.asarray(all_predictions)
    )


def prediction_accuracy(predictions: np.ndarray,
                        labels: np.ndarray,
                        ignore_index: int | None = None) -> float:
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if predictions.shape != labels.shape:
        raise ValueError(f"predictions shape {predictions.shape} does not match labels shape {labels.shape}")

    if ignore_index is not None:
        mask = labels < ignore_index
        predictions = predictions[mask]
        labels = labels[mask]

    if labels.size == 0:
        return 0.0

    return float(np.mean(predictions == labels))


def bdl_species_to_model_label(species_label: int) -> int:
    from src.BDL_api import SPECIES_DBL, SPECIES_MODEL

    model_label_by_latin_name = {value[0]: label for label, value in SPECIES_MODEL.items()}

    species_model_label_by_bdl_label = {}
    for species_data in SPECIES_DBL.values():
        species_name = species_data[0]
        genus = species_name.split('_', 1)[0]
        species_model_label_by_bdl_label[species_data[2]] = model_label_by_latin_name.get(
            species_name,
            model_label_by_latin_name.get(genus, model_label_by_latin_name['Others']),
        )

    return species_model_label_by_bdl_label[int(species_label)]


def load_laz_points_and_crs(laz_path: pth.Path) -> tuple[np.ndarray, object]:
    import laspy

    laz = laspy.read(laz_path)
    crs = laz.header.parse_crs()
    if crs is None:
        raise ValueError(f"Could not read CRS from {laz_path}")

    points = np.vstack((laz.x, laz.y, laz.z)).transpose()
    return points, crs


def bdl_refined_predictions(predictions: np.ndarray,
                            map_points: np.ndarray,
                            crs,
                            size_m: int = 5000,
                            model_based: bool = False) -> np.ndarray:
    from src.BDL_api import BDLCall

    tree_bdl = BDLCall(size_m=size_m, model_based=model_based)
    tree_bdl.build_data_map(points=map_points, crs=crs)

    refined = np.asarray(predictions).copy()
    pbar = tqdm(enumerate(predictions), total=predictions.shape[0], desc="Refining predictions with BDL")

    for index, prediction in pbar:
        species_prediction = tree_bdl.predict(
            pcd=map_points,
            tree_label=int(prediction),
            crs=crs,
        )
        refined[index] = bdl_species_to_model_label(species_prediction)

    return refined

def eval_model_front(config_dict: dict,
        model: nn.Module,
        paths: list[pth.Path]):
    
    model_path = paths[0]
    model_name = model_path.stem

    plot_dir = paths[1]

    total_loss, total_accuracy, all_labels, all_probs, all_predictions = _eval_model(
        config_dict=config_dict,
        model=model,
    )
    print('MODEL TESTED')
    print('Model path', model_path)
    print('Loss: ', total_loss)
    print('ACCURACY: ', total_accuracy)
    print('Plots saved to:', plot_dir)
    print('='*20)
    
    plotter = Plotter(class_num=config_dict['num_classes'], plots_dir=plot_dir)
    
    plotter.roc_curve(f'roc_{model_name}.png', all_labels, all_probs)
    plotter.prc_curve(f'prc_{model_name}.png', all_labels, all_probs)
    plotter.cnf_matrix(f'cnf_{model_name}.png', all_labels, all_predictions)
    plotter.threshold_hist(f'threshold_hist_{model_name}.pdf', all_probs)

    ClassificationReport(file_path=plot_dir.joinpath(f'classification_report_{model_name}.txt'),
                        pred=all_predictions,
                        target=all_labels)

    # laz_path = '/mnt/DATA_SSD/BRIK/GRAJEWO_NOWE/RAW/Grajewo_2026_2.laz'
    # if laz_path:
    #     map_points, crs = load_laz_points_and_crs(pth.Path(laz_path))
    #     mask = voxel_subsample_vectorized(map_points.copy(), voxel_size=1.)
    #     map_points = map_points[mask]

    #     bdl_predictions = bdl_refined_predictions(
    #         predictions=all_predictions,
    #         map_points=map_points,
    #         crs=crs,
    #     )
    #     bdl_accuracy = prediction_accuracy(
    #         predictions=bdl_predictions,
    #         labels=all_labels,
    #         ignore_index=OTHERS,
    #     )

    #     print('BDL CLASSIFIER ENABLED')
    #     print('BDL accuracy: ', bdl_accuracy)
    #     print('='*20)

    #     plotter.cnf_matrix(f'cnf_bdl_{model_name}.png', all_labels, bdl_predictions)
    #     ClassificationReport(file_path=plot_dir.joinpath(f'classification_report_bdl_{model_name}.txt'),
    #                         pred=bdl_predictions,
    #                         target=all_labels)
    # else:
    #     print("BDL classifier skipped: set laz_path in eval_model_front to enable it.")


def test_function(config_dict: dict,
                model):
    
    device_gpu = torch.device('cuda')
    device_cpu = torch.device('cpu')

    device_loader = device_gpu
    device_loss = device_gpu
    
    test_dataset = NpyDatasetAug(path_dir=config_dict['data_path_test'],
                                 resolution_xy=config_dict['input_dim'],
                                 training=False,
                                 device=device_loader,
                                 n_points=16384,
                                 use_domain_aug=True)
    
    testLoader = DataLoader(
        test_dataset,
        batch_size=config_dict["batch_size"],
        num_workers=15,
        pin_memory=True,          # faster CPU->GPU transfers
        persistent_workers=False,  # keep workers alive between epochs
        prefetch_factor=2,
    )
    
    batch_x, _ = next(iter(testLoader))
    batch_x = batch_x.to(config_dict['device'])
    model.eval()
    outputs = model(batch_x)

    if outputs.shape == (batch_x.shape[0], config_dict['num_classes']):
        print('Model works as expected')
    else:
        print(f'Model does not work as expected\n')
        print(f'Expected output shape: (batch_x.shape[0], {config_dict["num_classes"]})\nReceived: {outputs.shape}')

def parser():
    """
    Parse command-line arguments for automated CNN training pipeline configuration.
    Accepts model naming, computational device selection (CPU/CUDA/GPU), and optional test mode activation.
    Returns parsed arguments with validation for device choices and formatted help text display.
    """
    
    parser = argparse.ArgumentParser(
        description="Script for testing the choosen model",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help=(
            "Base of the model's name.\n"
            "When iterating, name also gets an ID."
        )
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'gpu'], # choice limit
        help=(
            "Device for tensor based computation.\n"
            "Pick 'cpu' or 'cuda'/ 'gpu'.\n"
        )
    )

    parser.add_argument(
        '--mode',
        type=int,
        default=0,
        choices=[0, 1], # choice limit
        help=(
            "Device for tensor based computation.\n"
            'Pick:\n'
            '0: testing mode - check if model compiles and works as expected\n'
            '1: evaluate trained model'
        )
    )

    return parser.parse_args()

def main():
    args = parser()
    base_path = pth.Path(__file__).parent
    device_name = args.device
    device = torch.device('cuda') if (('cuda' in device_name.lower() or 'gpu' in device_name.lower()) and torch.cuda.is_available()) else torch.device('cpu')

    model_name = args.model_name
    model_name_no_num = model_name.rsplit('_', 1)[0]

    config_dir = base_path.joinpath('config_files')
    model_dir = base_path.joinpath(f'training_results/{model_name_no_num}')
    config_trained_dir = model_dir.joinpath('dict_files')
    model_path = config_trained_dir.joinpath(f'{model_name}_config.json')
    plot_dir = model_dir.joinpath('plots')

    config_dict = load_json(model_path)
    config_dict = convert_str_values(config_dict)
    config_dict['device'] = device
    
    model = EfficientNetClassifier(config=config_dict['model_config'],
                            num_classes=config_dict['num_classes'])
    
    model = load_model(file_path=model_dir.joinpath(f'{model_name}.pt'),
                       model=model,
                       device=device)
    model = model.to(device)
    model.eval()

    if args.mode == 0:
        test_function(config_dict, model)
    elif args.mode == 1:
        eval_model_front(config_dict=config_dict,
                         model=model,
                         paths=[model_path,
                                plot_dir])


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
