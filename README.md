# Hierarchical Knowledge Distillation for Fair Federated Learning

Official implementation of the paper:

**Hierarchical Knowledge Distillation for Fair Federated Learning**
Submitted to **ECML-PKDD 2026**

## Overview

This repository provides an experimental framework for evaluating fairness-aware federated learning methods.
The framework supports multiple federated optimization strategies and datasets, with configurable experiments using **Hydra**.

Implemented strategies include:

* FedAvg
* FedProx
* q-FedAvg
* DEFFT

Supported datasets include:

* FEMNIST
* MNIST
* CIFAR-10

## Repository Structure

```
.
├── clients/                # Client implementations
├── server/                 # Server implementations
├── models/                 # Neural network architectures
├── datasets/               # Dataset preprocessing pipelines
├── evals/                  # Loss functions and evaluation utilities
├── conf/                   # Hydra configuration files
├── └──dataset
    └── mnist.yaml
    └── mnist.yaml
    └── femnist.yaml
│   └── config.yaml
├── utils/                  # Utility functions
├── checkpt/                # Saved checkpoints
├── .logs/                  # Experiment logs
└── main.py                 # Training entry point
```

## Configuration

Experiments are configured using **Hydra**.

The main configuration file is located at:

```
conf/config.yaml
```

Dataset-specific settings are defined in:

```
conf/dataset/
```

## Running an Experiment

Basic training run:

```
python main.py
```

Example with explicit parameters:

```
python main.py dataset=femnist stratergy=defft
```


## Checkpoints

Model checkpoints are saved to:

```
checkpt/<strategy>/<dataset>/<version>/
```

Each experiment directory contains:

* `global_model.pth` — trained global model
* `experiment_info.txt` — experiment metadata
* training logs

## Logging

Logs are stored in:

```
.logs/
```

Each run generates a timestamped log file containing:

* training configuration
* round statistics
* evaluation metrics

## Reproducibility

The framework enforces deterministic execution by fixing random seeds for:

* Python
* NumPy
* PyTorch

## License

This project is released under the **GNU General Public License v3.0**.

## Citation

If you use this code in your research, please cite:

```
Anonymous Author.
Hierarchical Knowledge Distillation for Fair Federated Learning.
ECML-PKDD 2026 (under review).
```
