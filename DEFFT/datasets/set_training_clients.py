"""
Copyright (C) [2026] Annonymous Author

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Paper: Hierarchical Knowledge Distillation for Fair Federated Learning
Submitted to: ECML-PKDD 2026 
"""
import os
import json
import random
import argparse
import configparser
import numpy as np

def sample_clients(client_ids: list, client_fraction:float) -> dict:
    """
    Sample clients from the client dictionary.

    Parameters:
    ----------------
    num_clients: int;
        Number of clients to sample

    Returns:
    ----------------
    list
        Dict of sampled clients
    """
    num_clients = int(len(client_ids) * client_fraction)
    sampled_client_ids = np.random.choice(client_ids, num_clients, replace=False)

    return sampled_client_ids

def get_client_ids(folder_path):
    client_ids = []
    try:
        files = os.listdir(folder_path)
        for file in files:
            if file.endswith('.pt'):
                client_id = file.split('.')[0]
                client_ids.append(client_id)
        return client_ids
    except Exception as e:
        raise e

def parse_arguments():
    parser = argparse.ArgumentParser(description="Federated training parameters")
    parser.add_argument("--dir", type=str, default="femnist", help="Choose a dataset from the available options; femnist, mnist, kv, celeba, cifar10")
    parser.add_argument("--client_fraction", type=float, default=0.1)
    return parser.parse_args()

def main():
    args = parse_arguments()
    folder_path = args.dir
    client_fraction = args.client_fraction

    client_ids = get_client_ids(f"{folder_path}/trainpt")

    training_samples = {i: sample_clients(client_ids, client_fraction) for i in range(1, 501)}
    training_samples = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in training_samples.items()}
    output_file = f"{folder_path}/training_samples_v1.json"

    with open(output_file, 'w') as f:
        json.dump(training_samples, f, indent=4)

if __name__ == "__main__":
    main()