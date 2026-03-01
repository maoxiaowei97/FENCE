# Spatial-Temporal Feedback Diffusion Guidance for Controlled Traffic Imputation


This is an official implementation of FENCE. We provided the codes about the experiments on [PEMS](https://github.com/Davidham3/ASTGCN-2019-mxnet/tree/master/data) traffic flow datasets.

## Code Structure Description

Below is a brief description of the project's main files and directories:

```
FENCE/
├── config/              # Contains configuration files for the main models (e.g., PEMS04.conf), defining hyperparameters and data paths.
├── data/                # Contains the datasets used for training and evaluation.
│   └── miss_data/       # Stores the generated true and missing data (e.g., PEMS04/).
├── logs/                # Used to store the running logs output during training and evaluation.
├── params/              # Used to store the trained model weights (.pth files for conditional and unconditional models).
├── results/             # Used to store the evaluation results in CSV files.
├── args.py              # Parses command-line arguments and configuration files.
├── dataset_traffic.py   # Data loading and preprocessing, defines the Dataset and DataLoader.
├── diff_models.py       # Core network structure of the diffusion model (ResidualBlock, Attention, etc.).
├── eval_SC-TC.sh        # Batch evaluation script for the SC-TC missing pattern.
├── eval_SR-TC.sh        # Batch evaluation script for the SR-TC missing pattern.
├── main_model.py        # Overall architecture of the FENCE model, integrating the diffusion process, guidance strategies (CFG/FBG), and loss calculation.
├── run.py               # Main entry point of the project, coordinating data loading, model creation, training, and evaluation processes.
└── utils.py             # Contains utility functions, such as training loop, evaluation metric calculation, early stopping strategy, etc.
```

## Requirement

We recommend using Python 3.8. See `requirements.txt` for the list of packages.

## Quick Start Workflow

The project workflow consists of two main steps: **1. Data Generation** and **2. Model Training/Evaluation**.

### 1. Data Generation

All traffic flow datasets can be used in the experiments, which can be downloaded from this [link](https://github.com/guoshnBJTU/ASTGNN/tree/main/data). Before training, you need to generate datasets with specific missing patterns and rates from the original traffic datasets.

1.  **Prepare Original Data**: Place the downloaded original dataset files (e.g., `PEMS08.npz`) in the `./data/` directory.
2.  **Run Generation Script**: Use the `data/generator.py` script to create missing data. This script reads the source files from `./data/` and generates `.npz` files containing complete and missing data according to your settings, saving them to the directory specified in the configuration file.

**Example Command**:
```bash
# Generate data with SC-TC missing pattern and 80% missing rate for the PEMS dataset
python data/generator.py --dataset PEMS04 --misstype SC-TC --missrate 0.8
```

-   `--misstype`: Missing type, options are:
    -   `SR-TR`: Spatially Random, Temporally Random
    -   `SR-TC`: Spatially Random, Temporally Contiguous
    -   `SC-TR`: Spatially Contiguous, Temporally Random
    -   `SC-TC`: Spatially Contiguous, Temporally Contiguous
-   `--missrate`: Missing rate, e.g., `0.8`.

### 2. Model Training and Evaluation

After generating the data, you can use the `run.py` script to train the FENCE model, or use the provided bash scripts to quickly evaluate pre-trained models.

**Training FENCE**:

You can train models with different configurations by modifying the command-line arguments. The settings in the configuration file (`config/*.conf`) can be overridden by command-line arguments.

```bash
# Train on the PEMS08 dataset, specifying a missing rate of 0.8 and a missing type of SC-TC
python run.py --device cuda:0 --mode train --dataset PEMS04 --miss_rate 0.8 --miss_type SC-TC
```

### Inference by the trained FENCE

**Pre-constructed Data:** You can directly use our already constructed missing datasets for evaluation. Please download the data from this [Google Drive link](https://drive.google.com/drive/folders/1Z9Gsyo9l-hrS2VVFNZWSWs9-RyXbqkF1?usp=sharing) and place them in your data directory (e.g., `./data/miss_data/PEMS04/`).

We provide pre-trained model parameters in the `./params/` directory. You can directly run the provided bash scripts to load these parameters, evaluate the models, and get the test results for different missing patterns.

To evaluate the model on the **SC-TC** (Spatially Contiguous, Temporally Contiguous) missing pattern:

```bash
bash eval_SC-TC.sh
```
To evaluate the model on the **SR-TC** (Spatially Random, Temporally Contiguous) missing pattern:

```bash
bash eval_SR-TC.sh 
```

The evaluation results (e.g., RMSE, MAE, MAPE, CRPS) will be printed to the console and appended to a CSV file in the ./results/ directory.

