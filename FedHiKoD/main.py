"""
Copyright (C) [2025] [Tharuka Kasthuriarachchige]

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Published in: 
"""
import os
import json
import time
import logging
from datetime import datetime
from enum import Enum
import torch
import copy
import random
from omegaconf import OmegaConf

import numpy as np

from clients import Client, BoostingClient, DittoClient, FedHiKoDClient, FedProxClient, QFFedAvgClient
from server import FedAvgServer, BoostingServer, DittoServer, FedHiKoDServer, FedProxServer, QFedAvgServer
from utils import get_device, get_client_ids

from models.kv import ShallowNN
from models.femnist import FEMNISTNet
from models.mnist import MNISTNet
from models.celeba import CELEBANet
from models.cifar10 import CIFAR10Net, CIFAR10ResNet18, CIFAR10ResNet18Lite
from evals import FocalLoss, HybridLoss

from datasets.kv.preprocess import KVDataSet
from datasets.femnist.preprocess import FEMNISTDataset
from datasets.mnist.preprocess import MNISTDataset
from datasets.celeba.preprocess import CELEBADataset
from datasets.cifar10.preprocess import CIFARDataset

import hydra
from omegaconf import OmegaConf


def setup_logging(strategy, dataset, timestamp, console: bool = True) -> str:
    log_dir = ".logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"federation__{dataset}_{strategy}_{timestamp}.log")

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers (Hydra or previous runs might add duplicates)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # --- File handler ---
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # --- Console handler ---
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return log_filename


class Federation:
    """
    Class for federated learning.

    Parameters:
    ----------------
    client_ids: list;
        List of client ids.
    model: torch.nn.Module;
        Model to be trained.
    loss_fn: torch.nn.Module;
        Loss function.
    train_data_dir: str;
        Path to the training data directory.
    test_data_dir: str;
        Path to the testing data directory.
    num_classes: int;
        Number of classes in the dataset.
    global_rounds: int;
        Number of global rounds.
    stratergy: callable;
        Federated learning stratergy.
    learning_rate: float;
        Learning rate.
    train_batch_size: int;
        Batch size for training.
    test_batch_size: int;
        Batch size for testing.
    weight_decay: float;    
    """

    def __init__(
        self,
        client_ids: list,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        train_data_dir: str,
        test_data_dir: str,
        num_classes: int,
        global_rounds: int,
        stratergy: callable,
        learning_rate: float,
        train_batch_size: int,
        test_batch_size: int,
        weight_decay: float,
        eta: float,
        error_threshold: float,
        checkpt_path: str,
    ) -> None:
        
        self.client_ids = client_ids
        self.num_classes = num_classes
        self.model = model
        self.loss_fn = loss_fn
        self.global_rounds = global_rounds
        self.stratergy = stratergy
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.weight_decay = weight_decay
        self.eta = eta
        self.error_threshold = error_threshold

        if stratergy == "fedaboost":
            self.server = BoostingServer(global_rounds, stratergy, checkpt_path=checkpt_path)
            self.server.init_model(model)

            # Set up the clients for fedaboost server
            for id in client_ids:
                self.server.connect_client(BoostingClient(
                    id,
                    torch.load(f"{train_data_dir}/{id}.pt", weights_only=False),
                    torch.load(f"{test_data_dir}/{id}.pt", weights_only=False),
                    self.loss_fn,
                    self.train_batch_size,
                    self.test_batch_size,
                    self.learning_rate,
                    self.weight_decay,
                    local_model= copy.deepcopy(self.model),
                    num_classes=self.num_classes,
                    eta=self.eta,
                    error_threshold=self.error_threshold,
                ))

        elif stratergy == "fedavg":
            self.server = FedAvgServer(global_rounds,checkpt_path=checkpt_path)
            self.server.init_model(model)

            # Set up the clients for fedavg server
            for id in client_ids:
                self.server.connect_client(Client(
                    id,
                    torch.load(f"{train_data_dir}/{id}.pt", weights_only=False),
                    torch.load(f"{test_data_dir}/{id}.pt", weights_only=False),
                    self.loss_fn,
                    self.train_batch_size,
                    self.test_batch_size,
                    self.learning_rate,
                    self.weight_decay,
                    local_model= copy.deepcopy(self.model),
                ))

        elif stratergy == "fedprox":
            self.server = FedProxServer(global_rounds, checkpt_path=checkpt_path)
            self.server.init_model(model)

            for id in client_ids:
                self.server.connect_client(FedProxClient(
                    id,
                    torch.load(f"{train_data_dir}/{id}.pt", weights_only=False),
                    torch.load(f"{test_data_dir}/{id}.pt", weights_only=False),
                    self.loss_fn,
                    self.train_batch_size,
                    self.test_batch_size,
                    self.learning_rate,
                    self.weight_decay,
                    local_model= copy.deepcopy(self.model),
                ))

        elif stratergy == "fedhikod":
            self.server = FedHiKoDServer(global_rounds,checkpt_path=checkpt_path)
            self.server.init_model(model)

            # Set up the clients for fedhikod server
            for id in client_ids:
                self.server.connect_client(FedHiKoDClient(
                    id,
                    torch.load(f"{train_data_dir}/{id}.pt", weights_only=False),
                    torch.load(f"{test_data_dir}/{id}.pt", weights_only=False),
                    self.loss_fn,
                    self.train_batch_size,
                    self.test_batch_size,
                    self.learning_rate,
                    self.weight_decay,
                    local_model= copy.deepcopy(self.model),
                ))

        elif stratergy == "ditto":
            self.server = FedAvgServer(global_rounds, checkpt_path=checkpt_path)
            self.server.init_model(model)

            for id in client_ids:
                self.server.connect_client(
                    DittoClient(
                        client_id=id,
                        train_dataset=torch.load(f"{train_data_dir}/{id}.pt", weights_only=False),
                        test_dataset=torch.load(f"{test_data_dir}/{id}.pt", weights_only=False),
                        loss_fn=self.loss_fn,
                        train_batch_size=self.train_batch_size,
                        test_batch_size=self.test_batch_size,
                        learning_rate=self.learning_rate,
                        weight_decay=self.weight_decay,
                        local_model=copy.deepcopy(self.model),
                        personal_learning_rate=0.001,
                        ditto_lambda=0.1,
                        personalized=True,
                        checkpt_path=checkpt_path,

                    )
                )

        elif stratergy == "qfedavg":
            self.server = QFedAvgServer(global_rounds,checkpt_path=checkpt_path)
            self.server.init_model(model)

            for id in client_ids:
                self.server.connect_client(
                    QFFedAvgClient(
                    id,
                    torch.load(f"{train_data_dir}/{id}.pt", weights_only=False),
                    torch.load(f"{test_data_dir}/{id}.pt", weights_only=False),
                    self.loss_fn,
                    self.train_batch_size,
                    self.test_batch_size,
                    self.learning_rate,
                    self.weight_decay,
                    local_model= copy.deepcopy(self.model),
                    )
                )
        else:
                raise ValueError(f"Invalid stratergy. Choose from: {', '.join([stratergy.value for stratergy in Stratergy])}")

    def train(self, training_samples, max_local_round:int = 10, threshold:float = 0.01, patience = 2) -> tuple:
        """
        Training the federated learning model.

        Returns:
        ----------------
        trained_model: torch.nn.Module;
            Trained model.
        """
        print(threshold)
        print(type(threshold))
        if self.stratergy == "fedaboost":
            trained_model = self.server.train(training_samples, max_local_round, threshold=0.05, patience=5)

        elif self.stratergy == "fedprox":
            trained_model = self.server.train(training_samples, max_local_round,mu=5)
        else:
            trained_model = self.server.train(training_samples, max_local_round, threshold, patience)
        return trained_model

    def save_models(self, model: torch.nn.Module, ckptpath: str) -> None:
        """
        Saving the training stats and the model.

        Parameters:
        ----------------
        model:
            Trained model.
        ckptpath: str;
            Path to save the model and the training stats. Default is None.
        """
        if os.path.exists(ckptpath):
            torch.save(
                model.state_dict(),
                ckptpath,
            )
        else:
            os.makedirs(os.path.dirname(ckptpath), exist_ok=True)
            torch.save(
                model.state_dict(),
                ckptpath,
            )

class Dataset(Enum):
    CELEBA = "celeba"
    FEMNIST = "femnist"
    MNIST = "mnist"
    KV = "kv"
    CIFAR10 = "cifar10"

class Stratergy(Enum):
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDABOOST = "fedaboost" 
    DITTO = "ditto"

def dataset_enum(dataset_str: str) -> str:
    """
    Returns the dataset enum.
    
    Parameters:
    ----------------
    dataset_str: str;
        Dataset string.
    """
    try:
        return Dataset(dataset_str.lower())
    except ValueError:
        raise ValueError("Invalid dataset. Choose from: femnist, mnist, kv, celeba, cifar10")

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))  # Shows full merged config

    seed = cfg.get("seed", 42)  # default if not specified in config
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed Control] Using seed: {seed}")

    # General
    loss_threshold = cfg.loss_threshold
    patience = cfg.patience

    # Training setup
    global_rounds = cfg.dataset.global_rounds
    local_rounds = cfg.dataset.local_rounds
    data_dir = cfg.dataset.data_dir

    strartegy = cfg.stratergy.lower()

    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = setup_logging(strartegy, cfg.dataset.name, timestamp)

    # Dataset
    train_data_dir = f"{data_dir}/trainpt"
    test_data_dir = f"{data_dir}/testpt"
    client_ids = get_client_ids(train_data_dir)
    training_samples = json.load(open(f"{data_dir}/{cfg.dataset.train_samples_file}"))

    # Model
    model_class = globals()[cfg.dataset.model]
    model = model_class(cfg.dataset.num_classes) if cfg.dataset.name == "femnist" else model_class()

    if cfg.loss_function == "FocalLoss":
        loss_fn = FocalLoss(alpha=1, gamma=0, reduction='mean')
    else:
        loss_fn = getattr(torch.nn, cfg.loss_function)()

    epochs = global_rounds * local_rounds

    checkpt_path = os.path.join(
        "checkpt",
        str(strartegy).lower(),          # argparse or Hydra string
        cfg.dataset.name,  # works for Enum or str
        cfg.version,
        f"epoch_{epochs}",
        f"{global_rounds}_rounds_{local_rounds}_epochs_per_round"
    )

    federation = Federation(
        client_ids=client_ids,
        model=model,
        loss_fn=loss_fn,
        train_data_dir=train_data_dir,
        test_data_dir=test_data_dir,
        num_classes=cfg.dataset.num_classes,
        global_rounds=global_rounds,
        stratergy=strartegy,
        learning_rate=cfg.dataset.learning_rate,
        train_batch_size=cfg.dataset.train_batch_size,
        test_batch_size=cfg.dataset.test_batch_size,
        weight_decay=cfg.dataset.weight_decay,
        eta=cfg.dataset.eta,
        error_threshold=cfg.dataset.error_threshold,
        checkpt_path=checkpt_path,
    )

    os.makedirs(checkpt_path, exist_ok=True)
    info_file = os.path.join(checkpt_path, "experiment_info.txt")

    # Prepare all info lines
    info_lines = [
        "Federation initialized.",
        f"Dataset: {cfg.dataset.name}",
        f"Strategy: {cfg.stratergy}",
        f"Loss function: {cfg.loss_function}",
        f"Number of clients: {len(client_ids)}",
        f"Client IDs: {', '.join(client_ids)}",
        f"train_samples_file: {training_samples}",
        f"Global rounds: {cfg.dataset.global_rounds}",
        f"Local rounds: {cfg.dataset.local_rounds}",
        f"Total epochs: {epochs}",
        f"Learning rate: {cfg.dataset.learning_rate}",
        f"Train batch size: {cfg.dataset.train_batch_size}",
        f"Test batch size: {cfg.dataset.test_batch_size}",
        f"Weight decay: {cfg.dataset.weight_decay}",
        f"Eta: {cfg.dataset.eta}",
        f"Patience: {cfg.patience}",
        f"Checkpoint path: {checkpt_path}",
        f"Log file: {log_filename}",
        "Special notes: random client selection"
    ]

    # Write to file
    with open(info_file, "w") as f:
        for line in info_lines:
            f.write(line + "\n")

    print(f"Experiment info written to: {info_file}")

    print("Server type:", type(federation.server))
    print("Instance has 'train' attr?", 'train' in federation.server.__dict__)  # should be False
    attr = getattr(federation.server, 'train', None)
    print("callable?", callable(attr))
    print("repr:", attr)
    try:
        print("qualname:", getattr(attr, '__qualname__', None))
        print("bound to:", getattr(attr, '__self__', None))  # bound method should show the instance
    except Exception as e:
        print("introspection error:", e)

    print("Federation with clients " + ", ".join(client_ids))
    epochs = cfg.dataset.global_rounds * cfg.dataset.local_rounds

    logging.info("Federation initialized.")
    logging.info(f"Dataset: {cfg.dataset.name}")
    logging.info(f"Strategy: {cfg.stratergy}")
    logging.info(f"Loss function: {cfg.loss_function}")
    logging.info(f"Number of clients: {len(client_ids)}")
    logging.info(f"Client IDs: {', '.join(client_ids)}")
    logging.info(f"Global rounds: {cfg.dataset.global_rounds}")
    logging.info(f"Local rounds: {cfg.dataset.local_rounds}")
    logging.info(f"Total epochs: {epochs}")
    logging.info(f"Learning rate: {cfg.dataset.learning_rate}")
    logging.info(f"Train batch size: {cfg.dataset.train_batch_size}")
    logging.info(f"Test batch size: {cfg.dataset.test_batch_size}")
    logging.info(f"Weight decay: {cfg.dataset.weight_decay}")
    logging.info(f"Eta: {cfg.dataset.eta}")
    logging.info(f"Patience: {cfg.patience}")
    logging.info(f"Checkpoint path: {os.path.join(cfg.checkpt_root, cfg.dataset.name)}")
    logging.info(f"Log file: {log_filename}")
    logging.info("Special notes: Controlled Experiment - All clients in 1st round is used for all global epochs")

    start = time.time()
    # Train
    trained_model = federation.train(
        training_samples,
        max_local_round=cfg.dataset.local_rounds,
        threshold=cfg.loss_threshold,
        patience=cfg.patience,
    )

    # Save global model
    model_path = os.path.join(checkpt_path, "global_model.pth")
    federation.save_models(trained_model, model_path)

    logging.info(f"Model saved to {model_path}")
    print(f"Model saved to {model_path}")
    print(f"Training completed in {time.time() - start:.2f} seconds.")
    elapsed = time.time() - start
    print(f"Training completed in {elapsed / 60:.2f} minutes.")

if __name__ == "__main__":
    main()