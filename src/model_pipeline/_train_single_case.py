import torch
import torch.nn as nn
import os
import sys
import warnings
from typing import Union, Generator
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR


from model import CNN2D_Residual
from model_test import ResNetClassifier

from _data_loader import *
from utils import compute_pos_weights, get_dataset_len, calculate_accuracy, FocalLoss
from utils import wrap_hist

warning_to_filter = "Attempting to run cuBLAS, but there was no current CUDA context!"

# Adding a filter to ignore warnings from PyTorch
warnings.filterwarnings(
    "ignore", 
    message=warning_to_filter, 
    category=UserWarning
)


nerual_net_dir = os.path.dirname(__file__)
sys.path.append(nerual_net_dir)



def train_model(training_dict: dict, num_workers = 20) -> Union[Generator[tuple[nn.Module, dict], None, None],
                                                Generator[tuple[None, dict], None, None]]:      
    
    device_gpu = torch.device('cuda')
    device_cpu = torch.device('cpu')

    device_loader = device_cpu
    device_loss = device_gpu


    # train_dataset = Dataset(path_dir = training_dict['data_path_train'],
    #                              resolution_xy = training_dict['input_dim'],
    #                              num_classes=training_dict['num_classes'],
    #                              batch_size = training_dict['batch_size'],
    #                              shuffle = True,
    #                              device = device_loader)
    
    # val_dataset = Dataset(path_dir = training_dict['data_path_val'],
    #                            resolution_xy=training_dict['input_dim'],
    #                            num_classes=training_dict['num_classes'],
    #                            batch_size = training_dict['batch_size'],
    #                            shuffle = False,
    #                            device = device_loader)
    
    # trainLoader = DataLoader(train_dataset,
    #                          batch_size=None,
    #                          num_workers = num_workers,
    #                          pin_memory=False)
    
    # valLoader = DataLoader(val_dataset,
    #                        batch_size=None,
    #                        num_workers = num_workers,
    #                        pin_memory=False)
    

    train_dataset = NpyDataset(path_dir=training_dict['data_path_train'],
                               resolution_xy=training_dict['input_dim'],
                               training=True,
                               device=device_loader)
    
    val_dataset = NpyDataset(path_dir=training_dict['data_path_val'],
                            resolution_xy=training_dict['input_dim'],
                            training=False,
                            device=device_loader)
    
    trainLoader = DataLoader(
        train_dataset,
        batch_size=training_dict["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,          # faster CPU->GPU transfers
        persistent_workers=False,  # keep workers alive between epochs
        prefetch_factor=2,
    )

    valLoader = DataLoader(
        val_dataset,
        batch_size=training_dict["batch_size"],
        num_workers=num_workers,
        pin_memory=False,          # faster CPU->GPU transfers
        persistent_workers=False,  # keep workers alive between epochs
        prefetch_factor=2,
    )


    weights_t, _ = compute_pos_weights(data_dir=training_dict['data_path_train'],
                                    num_classes=training_dict["num_classes"],
                                    power=0.35,
                                    ignore_index=18)
    
    # weights_t_samples = weights_t[labels]

    weights_v, _ = compute_pos_weights(data_dir=training_dict['data_path_val'],
                                    num_classes=training_dict["num_classes"],
                                    power=0.35,
                                    ignore_index=18)

    # sampler_t = WeightedRandomSampler(
    #     weights=weights_t_samples,
    #     num_samples=len(weights_t_samples),
    #     replacement=True
    # )

    # trainLoader = DataLoader(
    #     train_dataset,
    #     batch_size=training_dict["batch_size"],
    #     sampler=sampler_t,          # mutually exclusive with shuffle=True
    #     num_workers=num_workers,
    #     pin_memory=True,          # faster CPU->GPU transfers
    #     persistent_workers=False,  # keep workers alive between epochs
    #     prefetch_factor=2,
    # )

    total_t = get_dataset_len(trainLoader)
    total_v = get_dataset_len(valLoader)

    try:
        model = ResNetClassifier(config=training_dict['model_config'], num_classes=training_dict['num_classes']).to(training_dict['device'])
    except Exception as e:
        print(f"Error initializing model: {e}")
        yield None, {}

    criterion_t = FocalLoss(alpha= weights_t.to(device_loss), gamma=training_dict['focal_loss_gamma']).to(device_loss) 
    criterion_v = FocalLoss(alpha= weights_v.to(device_loss), gamma=training_dict['focal_loss_gamma']).to(device_loss)

    optimizer = optim.AdamW(model.parameters(), lr = training_dict['learning_rate'], weight_decay=training_dict['weight_decay'])


    scheduler = OneCycleLR(
        optimizer,
        max_lr=training_dict['learning_rate'],  # Example: Adjust based on your LR range test
        total_steps=total_t*training_dict['epochs'],
        pct_start=training_dict['pc_start'],  # % of steps for warm-up
        anneal_strategy='cos',  # Cosine annealing
        div_factor=training_dict['div_factor'],  # Initial LR will be max_lr / x
        final_div_factor=training_dict['final_div_factor'],  # Final LR
        cycle_momentum=True  # Cycle momentum as well
    )

    loss_hist = []
    acc_hist = []

    loss_v_hist = []
    acc_v_hist = []

    # try:
    repeat_pbar = tqdm(range(training_dict['train_repeat']), 
                    desc="Training Repetition", 
                    unit="repeat",
                    position=1, 
                    leave=False) 

    for _ in repeat_pbar:

        epoch_pbar = tqdm(range(training_dict['epochs']), 
                            desc="Epoch Progress", 
                            unit="epoch",
                            position=2, 
                            leave=False) 

        for epoch in epoch_pbar:

            epoch_loss_t = 0.
            epoch_loss_v = 0.

            epoch_accuracy_v = 0.

            epoch_samples_t = 0
            epoch_samples_v = 0


            progressbar_t = tqdm(trainLoader, 
                                desc=f"Epoch training {epoch+1}/ {training_dict['epochs']}", 
                                total=total_t, 
                                position=3,
                                leave=False)
            
            for batch_x, batch_y in progressbar_t:
                model.train(True)
                batch_x = batch_x.to(training_dict['device'])

                optimizer.zero_grad()
                outputs = model(batch_x)

                outputs = outputs.to(device_loss)
                batch_y = batch_y.to(device_loss)

                loss_t = criterion_t(outputs, batch_y)

                loss_t.backward()
                optimizer.step()

                try:
                    scheduler.step()
                except Exception:
                    pass


                current_lr = optimizer.param_groups[0]['lr']

                epoch_loss_t += loss_t.item() * batch_y.size(0)

                epoch_samples_t += batch_y.size(0)

                avg_loss_t = epoch_loss_t / epoch_samples_t


                progressbar_t.set_postfix({
                    "Loss_train": f"{avg_loss_t:.6f}",

                    "learning_rate": f"{current_lr:.10f}"
                })

            loss_hist.append(avg_loss_t)
            acc_hist.append(-1.)

            progressbar_v = tqdm(valLoader, desc=f"Epoch validation {epoch + 1}/ {training_dict['epochs']}", total=total_v, position=3, leave=False)
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in progressbar_v:
                    
                    batch_x = batch_x.to(training_dict['device'])
                    outputs = model(batch_x)

                    outputs = outputs.to(device_loss)
                    batch_y = batch_y.to(device_loss)


                    loss_v = criterion_v(outputs, batch_y)

                    accuracy_v = calculate_accuracy(outputs.cpu(), 
                                                            batch_y.cpu())

                    epoch_loss_v += loss_v.item() * batch_y.size(0)
                    epoch_accuracy_v += accuracy_v * batch_y.size(0)
                    epoch_samples_v += batch_y.size(0)

                    avg_loss_v = epoch_loss_v / epoch_samples_v
                    avg_accuracy_v = epoch_accuracy_v / epoch_samples_v

                    progressbar_v.set_postfix({
                        "Loss_val": f"{avg_loss_v:.6f}",
                        "Acc_val": f"{avg_accuracy_v:.6f}"
                    })

            loss_v_hist.append(avg_loss_v)
            acc_v_hist.append(avg_accuracy_v)


            hist_dict = wrap_hist(acc_hist = acc_hist, loss_hist = loss_hist, acc_hist_val = acc_v_hist, loss_hist_val = loss_v_hist)

            yield model, hist_dict


            epoch_pbar.set_postfix({
                "Epoch": epoch + 1,
                "Loss_train": f"{avg_loss_t:.6f}",
                # "Acc_train": f"{avg_accuracy_t:.6f}",
                "Loss_val": f"{avg_loss_v:.6f}",
                "Acc_val": f"{avg_accuracy_v:.6f}",
                "learning_rate_max": f"{training_dict['learning_rate']:.10f}"
            })

    # except Exception as e:
    #     print(f"Error during training: {e}")
    #     yield None, {}