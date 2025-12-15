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
from torchinfo import summary
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR


from model import CNN2D_Residual
from _data_loader import *
from utils import calculate_class_weights, get_dataset_len, EarlyStopping, calculate_accuracy_weighted
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



def train_model(training_dict: dict) -> Union[Generator[tuple[nn.Module, dict], None, None],
                                                Generator[tuple[None, dict], None, None]]:      
    
    device_gpu = torch.device('cuda')
    device_cpu = torch.device('cpu')

    device_loader = device_gpu
    device_loss = device_gpu


    train_dataset = Dataset(path_dir = training_dict['data_path_train'],
                                 resolution_xy = training_dict['input_dim'],
                                 batch_size = training_dict['batch_size'],
                                 shuffle = True,
                                 device = device_loader)
    
    val_dataset = Dataset(path_dir = training_dict['data_path_val'],
                               resolution_xy=training_dict['input_dim'],
                               batch_size = training_dict['batch_size'],
                               shuffle = False,
                               device = device_loader)
    
    trainLoader = DataLoader(train_dataset,
                             batch_size=None,
                             num_workers = 15,
                             pin_memory=True)
    
    valLoader = DataLoader(val_dataset,
                           batch_size=None,
                           num_workers = 15,
                           pin_memory=True)
    
    total_t = get_dataset_len(trainLoader)
    total_v = get_dataset_len(valLoader)
# START
    weights_t = calculate_class_weights(trainLoader,
                                        training_dict['num_classes'],
                                        total = total_t,
                                        device=training_dict['device'],
                                        verbose=False)
    weights_v = calculate_class_weights(valLoader,
                                        training_dict['num_classes'],
                                        total = total_v,
                                        device=training_dict['device'], ##TODO Changed
                                        verbose=False)
    train_dataset = Dataset(path_dir = training_dict['data_path_train'],
                                resolution_xy = training_dict['input_dim'],
                                batch_size = training_dict['batch_size'],
                                shuffle = True,
                                weights=weights_t,
                                device = torch.device('cpu'))
    
    trainLoader = DataLoader(train_dataset,
                             batch_size=None,
                             num_workers = 15,          
                             pin_memory=True)
    
    total_t = get_dataset_len(trainLoader)

    weights_t = calculate_class_weights(trainLoader,
                                    training_dict['num_classes'],
                                    total = total_t,
                                    device=training_dict['device'],
                                    verbose=False)
#STOP
    try:
        model = CNN2D_Residual(training_dict['num_classes'], config_data=training_dict['model_config']).to(training_dict['device'])
    except Exception as e:
        print(f"Error initializing model: {e}")
        yield None, {}

    criterion_t = FocalLoss_CLASS(alpha= weights_t, gamma=training_dict['focal_loss_gamma']).cpu()  ##TODO Changed  
    criterion_v = FocalLoss_CLASS(alpha= weights_v, gamma=training_dict['focal_loss_gamma']).cpu()

    optimizer = optim.Adam(model.parameters(), lr = training_dict['learning_rate'])

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

    early_stopping = EarlyStopping(patience=40, delta=0.0001, verbose=False)

    loss_hist = []
    acc_hist = []

    loss_v_hist = []
    acc_v_hist = []

    try:
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

                epoch_accuracy_t = 0.
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
                    outputs = model(batch_x)

                    outputs = outputs.cpu()
                    batch_y = batch_y.cpu()

                    loss_t = criterion_t(outputs, batch_y)

                    optimizer.zero_grad()
                    loss_t.backward()
                    optimizer.step()

                    try:
                        scheduler.step()
                    except Exception:
                        pass

                    accuracy_t = calculate_accuracy_weighted(outputs, batch_y, num_classes=training_dict['num_classes'])
                    current_lr = optimizer.param_groups[0]['lr']

                    epoch_loss_t += loss_t.item() * batch_y.size(0)
                    epoch_accuracy_t += accuracy_t * batch_y.size(0)
                    epoch_samples_t += batch_y.size(0)

                    avg_loss_t = epoch_loss_t / epoch_samples_t
                    avg_accuracy_t = epoch_accuracy_t / epoch_samples_t

                    progressbar_t.set_postfix({
                        "Loss_train": f"{avg_loss_t:.6f}",
                        "Acc_train": f"{avg_accuracy_t:.6f}",
                        "learning_rate": f"{current_lr:.10f}"
                    })

                loss_hist.append(avg_loss_t)
                acc_hist.append(avg_accuracy_t)

                progressbar_v = tqdm(valLoader, desc=f"Epoch validation {epoch + 1}/ {training_dict['epochs']}", total=total_v, position=3, leave=False)
                with torch.no_grad():
                    for batch_x, batch_y in progressbar_v:
                        model.eval()
                        batch_x = batch_x.to(training_dict['device'])
                        outputs = model(batch_x)

                        outputs = outputs.cpu()
                        batch_y = batch_y.cpu()


                        loss_v = criterion_v(outputs, batch_y)

                        accuracy_v = calculate_accuracy_weighted(outputs, 
                                                                 batch_y, 
                                                                 num_classes=training_dict['num_classes']) ##TODO Changed 

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


                hist_dict = wrap_hist(acc_hist, loss_hist, acc_v_hist, loss_v_hist)

                yield model, hist_dict


                epoch_pbar.set_postfix({
                    "Epoch": epoch + 1,
                    "Loss_train": f"{avg_loss_t:.6f}",
                    "Acc_train": f"{avg_accuracy_t:.6f}",
                    "Loss_val": f"{avg_loss_v:.6f}",
                    "Acc_val": f"{avg_accuracy_v:.6f}",
                    "learning_rate_max": f"{training_dict['learning_rate']:.10f}"
                })

                if early_stopping.stop_training:
                    break

    except Exception as e:
        print(f"Error during training: {e}")
        yield None, {}

if __name__ == '__main__':

    mp.set_start_method('spawn')  # Set the start method to 'spawn'
    main()