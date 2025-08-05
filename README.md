# Spatial-Temporal Feedback Diffusion Guidance for Controlled Traffic Imputation

This is an official implementation of FENCE. We provided the codes about the experiments on traffic flow datasets PMES04, PMES07 and PEMS08.

## Code Structure Description

Below is a brief description of the project's main files and directories:

```
FENCE/
├── config/              # Contains configuration files for the main models (e.g., PEMS08.conf), defining hyperparameters and data paths.
├── data/                # Contains the original datasets and the generated datasets.
│   ├── PEMS08/          # Example: Stores the generated data for PEMS08.
│   ├── config/          # Configuration files used by the data generator.
│   └── generator.py     # Script for generating data with different missing patterns.
├── params/              # Used to store the trained model weights (.pth files).
├── results/             # Used to store the evaluation results in CSV files.
├── args.py              # Parses command-line arguments and configuration files.
├── dataset_traffic.py   # Data loading and preprocessing, defines the Dataset and DataLoader.
├── diff_models.py       # Core network structure of the diffusion model (ResidualBlock, Attention, etc.).
├── main_model.py        # Overall architecture of the FENCE model, integrating the diffusion process, guidance strategies (CFG/FBG), and loss calculation.
├── run.py               # Main entry point of the project, coordinating data loading, model creation, training, and evaluation processes.
└── utils.py             # Contains utility functions, such as training loop, evaluation metric calculation, early stopping strategy, etc.
```

## Requirement

We recommend using Python 3.8. See `requirements.txt` for the list of packages.

## Quick Start Workflow

The project workflow consists of two main steps: **1. Data Generation** and **2. Model Training/Evaluation**.

### 1. Data Generation

All traffice flow datasets can be used in the experiments, which can be downloaded from this [link](https://github.com/guoshnBJTU/ASTGNN/tree/main/data).Before training, you need to generate datasets with specific missing patterns and rates from the original traffic datasets (e.g., `PEMS08.npz`).

1.  **Prepare Original Data**: Place the downloaded original dataset files (e.g., `PEMS08.npz`) in the `./data/` directory.

2.  **Run Generation Script**: Use the `data/generator.py` script to create missing data. This script reads the source files from `./data/` and generates `.npz` files containing complete and missing data according to your settings, saving them to the directory specified in the configuration file (defaulting to `./data/PEMS08/`, etc.).

**Example Command**:
```bash
# Generate data with SC-TC missing pattern and 80% missing rate for the PEMS08 dataset
python data/generator.py --dataset PEMS08 --misstype SC-TC --missrate 0.8
```

-   `--misstype`: Missing type, options are:
    -   `SR-TR`: Spatially Random, Temporally Random
    -   `SR-TC`: Spatially Random, Temporally Contiguous
    -   `SC-TR`: Spatially Contiguous, Temporally Random
    -   `SC-TC`: Spatially Contiguous, Temporally Contiguous
-   `--missrate`: Missing rate, e.g., `0.2`, `0.5`, `0.8`.

### 2. Model Training and Evaluation

After generating the data, you can use the `run.py` script to train or evaluate the FENCE model.

**Trainging FENCE**:

You can train models with different configurations by modifying the command-line arguments. The settings in the configuration file (`config/*.conf`) can be overridden by command-line arguments.

```bash
# Train on the PEMS08 dataset, specifying a missing rate of 0.8 and a missing type of SC-TC
python run.py --device cuda:0 --mode train --dataset PEMS08 --miss_rate 0.8 --miss_type SC-TC
```

**Inference by the trained FENCE**:

Use `--mode eval` to evaluate a trained model. The script automatically loads the model weights saved in the `./params/` directory.

```bash
# Evaluate the performance of the model on PEMS08, specifying CFG guidance with a scale of 2.5
python run.py --device cuda:0 --mode eval --dataset PEMS08 --guidance cfg --cfg_scale 2.5

# Evaluate the model, specifying the FBG mode as 'cluster'
python run.py --device cuda:0 --mode eval --dataset PEMS08 --guidance fbg --fbg_mode cluster
```

The evaluation results (e.g., RMSE, MAE, MAPE, CRPS) will be printed to the console and appended to a CSV file in the `./results/` directory.
