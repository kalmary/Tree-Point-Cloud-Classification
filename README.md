![BANNER](https://github.com/kalmary/Tree-Point-Cloud-Classification/blob/main/img/Banner_tree_point.png)
# Table of contents
1. [Overview](#overview)
2. [Repository Structure](#fstructure)
3. [Installation](#installation)
4. [Usage](#usage)
    1. [Preprocessing](#preprocessing)
    2. [Training](#training)
    3. [Evaluation](#evaluation)

--- 
# 1. Overview <a name="overview"></a>

**TREE-POINT-CLOUD-CLASSIFICATION** is a specialized framework for tree species classification that bridges 3D point cloud processing with 2D deep learning. By projecting complex 3D point cloud data into multi-channel 2D representations, the system leverages a dynamically configurable Residual CNN (ResNet) architecture for high-accuracy feature extraction.

Key Features:

- Data preprocessing: Features a specialized data_loader for multi-channel projections,
- Model definition: necessary code to define and build the ResNet model architecture, allows for scalability and adjustment for hardware-specific needs,
- Training & Evaluation: Tools for training the model on custom datasets and evaluating its performance,
---
# 2. Repository structure: <a name="fstructure"></a>

```
.
├── data_processing
│   └── downsample_trees.py        #Program for preprocessing point cloud files. Creates hdf5 files for training, validation and testing                    
├── model_pipeline
│   ├── Train_Automated.py          #Main Python Program for training 
│   ├── _data_loader.py
│   ├── Eval_TreeClassification.py  #Main Python Program for evaluating outputs
│   ├── model.py                    #ResNet model, configurable from dictionary
│   ├── model_configs               #Folder containing all architecture config files
│   ├── config_files                #Folder containing all parameters config files
│   └── _train_single_case.py
└── utils
    ├── nn_utils                 #nn_utils (external link)   
    ├── pcd_manipulation.py      #Point clound manipulations    
    ├── __init__.py
    └── data_augmentation.py     #Data augmentation 

```
---

# 3. Instalation: <a name="installation"></a>

Clone the repository to your local machine:
```bash

git clone https://github.com/kalmary/Tree-Point-Cloud-Classification/tree/main

cd Tree-Point-Cloud-Classification
```

Create and activate a Virtual Environment and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate

# Install all requirements, without pytorch and cuda
pip install requirements.txt

# Tested on this, but should work with any other version
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# update git submodules
git submodule update --init --recursive
```

---

# 4. Usage <a name="usage"></a>
### 1. Preprocessing <a name="preprocessing"></a>

Before training a model, data preprocessing must be done. To do so run:
```bash
python src/data_processing/downsample_trees.py --source_path path/to/raw/data --decimated_path path/to/decimated/pcds --converted_path path/to/final/processed/files
```
Paths used when processing:
- source_path: directory with raw (.LAZ by default) point clouds,
- decimated_path: serves as a checkpoint. Files are cut, decimated and saved as .npy files,
- converted_path: final directory with processed files, distributed into training/ validation/ testing datasets, chunked into .h5 files,

For more guidance/ guidance when running code, run it with ``--help`` flag.

### 2. Training <a name="training"></a>
Examine contents of:
- ``src/model_pipeline/model_configs`` - .json files with model architectures,
- ``src/model_pipeline/training_configs`` - .json files with training configs.
Pay attention to above files and adjust them to ensure the fit with your available resources.
  
Files with `_single` suffix are meant for single training without any optimizations. Others are for multi-hyperparameter optimization.

To start training run:
```bash
cd src/model_pipeline
python src/model_pipeline/Train_Automated.py --model_name MODEL_NAME --device cuda --mode 3
```
Available flags:
- ``model_name`` - name of your model. Results are stored in ``src/model_pipeline/training_results/MODEL_NAME``,
- ``device`` - we recommend training on strong GPUs (tested on RTX 5090), but any CUDA enabled one with at least 16 GB of VRAM (smaller models), will work fine,
- ``mode`` - you can choose following:
    - `0` - testing mode: ,
    - `1` - single training without optimizations: good choice if fast/ based on known optimal parameters training is required.,
    - `2` - grid based optimal hyperparameters search: only with optimization case is really simple/ configs combination number is low,,
    - `3` - Optuna library based hyperparameter search: for high dimensional and complex cases. If you have special needs in terms of time computation,
    - `4` - check models - run this to get a rough idea of model resource demands.
 

For most optimal results we recommend using options based on multidimensional space of hyperparameters. If you choose Optuna based option you can change ``n_trials`` in ``main`` function of ``Train_Automated.py``:
```python
optuna_based_training(exp_config=exp_configs,
                      model_name=model_name,
                      n_trials=80)

```
``n_trials=80`` is what we found to be giving good, repeatable results, but lower numbers where also acceptable. Even if the optimization process takes too long, 
best model and its config are saved and overwritten if a better model is found. Alongside them, plots with metrics history (loss, accuracy, mIoU) are also saved.

For more guidance/ guidance when running code, run it with ``--help`` flag.

### 3. Evaluation <a name="evaluation"></a>

To evaluate trained model, run Eval_TreeClassification.py with proper flags:

```bash
cd src/model_pipeline
python src/model_pipeline/Eval_TreeClassification.py --model_name MODEL_NAME --device cuda --mode 1
```
``model_name`` flag must be the exact name of model you got from training, but without extension name. For example:
```
ResNetTest_123.pt -> ResNetTest_123
```
Available modes are:
- 0: testing mode:  check if model compiles and works as expected
- 1: evaluation mode
  
Evaluation mode output are plots:
- Precision recall curve,
- Receiver operating characteristic curve,
- Confusion matrix

As previously, you can run this script with ``--help`` flag.


## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.






















