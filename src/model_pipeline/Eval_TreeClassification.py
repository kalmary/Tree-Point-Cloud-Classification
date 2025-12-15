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
from model import *

current_dir = pth.Path(__file__).parent.parent
sys.path.append(str(current_dir.parent))

from utils import load_json, load_model, convert_str_values
from utils import calculate_accuracy, get_intLabels, get_Probabilities,get_dataset_len, calculate_class_weights, FocalLoss_CLASS
from utils import Plotter, ClassificationReport


def _eval_model(config_dict: dict,
               model: nn.Module) -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray]:##TODO spectrogram_params: dict,filtration_params: dict,

    test_dataset = Dataset(path_dir = config_dict['data_path_val'],
                                resolution_xy=config_dict['input_dim'],
                                batch_size = config_dict['batch_size'],
                                shuffle = False,
                                device = torch.device('cpu'))
    
    testLoader = DataLoader(test_dataset,
                             batch_size=None,
                             num_workers = 15,
                             pin_memory=True) ## TODO RANDLANET valLoader i num_workesr = 14 / pin_memeory = False

    total = get_dataset_len(testLoader, verbose=False)
    weights = calculate_class_weights(testLoader, 
                                      config_dict['num_classes'], 
                                      total=total,
                                      verbose=False)

    criterion = FocalLoss_CLASS(alpha= weights.cpu(), gamma=config_dict['focal_loss_gamma']).cpu() ## TODO Inne lossfunction

    loss_per_epoch = 0.
    accuracy_per_epoch = 0.0
    epoch_samples = 0

    all_predictions = []
    all_probs = np.zeros((0, config_dict['num_classes']))
    all_labels = []

    pbar = tqdm(testLoader, total=total, desc="Testing", unit="batch")
    with torch.no_grad():
        for batch_x, batch_y in pbar:

            model.eval()
            batch_x = batch_x.to(config_dict['device'])

            outputs = model(batch_x)
            loss = criterion(outputs.cpu(), batch_y.cpu())

            accuracy= calculate_accuracy(outputs.cpu(), batch_y.cpu()) ##TODO accuracy = calculate_weighted_accuracy(outputs, batch_y, weights=weights)

            loss_per_epoch += loss.item()*batch_y.size(0)
            accuracy_per_epoch += accuracy * batch_y.size(0)

            epoch_samples += batch_y.size(0)

            total_loss = loss_per_epoch / epoch_samples
            total_accuracy = accuracy_per_epoch / epoch_samples


            all_labels.extend(batch_y.cpu().tolist())

            probs = get_Probabilities(outputs.cpu())
            int_preds = get_intLabels(probs)

            probs = probs.numpy()
            int_preds = int_preds.numpy()

            all_probs = np.concatenate([all_probs, probs.reshape(-1, config_dict['num_classes'])], axis=0)
            all_predictions.extend(int_preds)

    return total_loss, total_accuracy, np.asarray(all_labels), all_probs, np.asarray(all_predictions)

def eval_model_front(config_dict: dict,
        model: nn.Module,
        paths: list[pth.Path]):    ##TODO spectrogram_params: dict,filtration_params: dict
    
    model_path = paths[0]
    model_name = model_path.stem

    plot_dir = paths[1]

    total_loss, total_accuracy, all_labels, all_probs, all_predictions  = _eval_model(config_dict=config_dict,
                                                                                        model=model)##TODO spectrogram_params: dict,filtration_params: dict,
    ##TODO miou, avg_iou_pc = compute_mIoU(torch.asarray(all_predictions), torch.asarray(all_labels), config_dict['num_classes'])
    print('='*20)
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

    ClassificationReport(file_path=plot_dir.joinpath(f'classification_report_{model_name}.txt'),
                        pred=all_predictions,
                        target=all_labels)

def test_function(config_dict: dict,
                model):
    
    test_dataset = Dataset(path_dir = config_dict['data_path_val'], ##TODO val_dataset
                                resolution_xy=config_dict['input_dim'],
                                batch_size = config_dict['batch_size'],
                                shuffle = False,
                                device = torch.device('cpu'))
    
    testLoader = DataLoader(test_dataset,
                            batch_size=None,
                            num_workers = 15,
                            pin_memory=True)
    
    batch_x, batch_y = next(iter(testLoader))
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
    
    model = YOLOClassifier(model_version=config_dict['model_config']['model_version'],
                            num_classes=config_dict['num_classes'],
                            in_channels=config_dict['num_channels'])
    
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
    main()





