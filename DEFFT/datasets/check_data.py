import os
import torch
import argparse
import pickle
import numpy as np
from collections import Counter

from femnist.preprocess import FEMNISTDataset
from mnist.preprocess import MNISTDataset
from cifar10.preprocess import CIFARDataset

def verify_clients(clients: dict) -> bool:
    """
    Verify that each client in the given dictionary has at least two different classes.
    Also prints the class distribution for each client to assess skewness.
    
    Parameters:
    -----------
    clients : dict
        A dictionary where each key is a client name and the value is a list of (image, label) tuples.
        The label is assumed to be either an integer or a one-hot encoded vector.
        
    Returns:
    --------
    bool
        True if all clients have at least two different classes, False otherwise.
    """
    all_clients_valid = True
    for client, data in clients.items():
        class_counts = {}
        for _, label in data:
            # If label is one-hot (list or numpy array), convert to class index
            if isinstance(label, (list, np.ndarray)):
                label_arr = np.array(label)
                cls = int(np.argmax(label_arr))
            else:
                cls = label
            class_counts[cls] = class_counts.get(cls, 0) + 1

        classes = list(class_counts.keys())
        print(f"Client '{client}' has classes: {classes}")
        print(f"Class distribution: {class_counts}\n")
        
        if len(classes) < 2:
            print(f"Error: Client '{client}' does not have at least two different classes!\n")
            all_clients_valid = False
    return all_clients_valid


def dataset_summary(clients: dict) -> dict:
    """
    Analyzes the dataset distribution to assess the level of non-IID-ness.

    Parameters:
    -----------
    clients : dict
        A dictionary where each key is a client name and the value is a list of (image, label) tuples.

    Returns:
    --------
    dict
        A dictionary containing dataset statistics, including non-IID analysis.
    """
    total_samples = 0
    overall_class_counts = {}
    client_class_distribution = {}
    client_data_counts = {}
    client_class_variance = {}

    # Count class distribution and data points per client
    for client, data in clients.items():
        total_samples += len(data)
        client_data_counts[client] = len(data)
        class_counts = {}

        for _, label in data:
            if isinstance(label, (list, np.ndarray)):
                label = int(np.argmax(label))
            class_counts[label] = class_counts.get(label, 0) + 1
            overall_class_counts[label] = overall_class_counts.get(label, 0) + 1

        num_classes = len(class_counts)
        client_class_distribution[client] = num_classes

        # Compute variance of class distribution for non-IID analysis
        class_sizes = np.array(list(class_counts.values()))
        variance = np.var(class_sizes) if len(class_sizes) > 1 else 0
        client_class_variance[client] = variance

    # Count number of clients with 2, 3, 4, etc., classes
    class_count_distribution = {}
    for num_classes in client_class_distribution.values():
        class_count_distribution[num_classes] = class_count_distribution.get(num_classes, 0) + 1

    # Generate readable class count summary
    class_count_summary = [
        f"{count} clients have {num_classes} classes"
        for num_classes, count in sorted(class_count_distribution.items())
    ]

    # Compute average variance of class distributions
    avg_variance = np.mean(list(client_class_variance.values()))

    summary = {
        "num_clients": len(clients),
        "total_samples": total_samples,
        "class_distribution": overall_class_counts,
        "client_class_summary": class_count_summary,
        "client_data_counts": client_data_counts,
        "client_class_variance": client_class_variance,
        "avg_class_variance": avg_variance  # Higher variance = more non-IID
    }

    return summary


def train_dataset_statistics(train_dir: str) -> dict:
    """
    Compute statistics of the train datasets saved as .pt files.

    Parameters
    ----------
    train_dir : str
        Directory containing train datasets (.pt files).

    Returns
    -------
    dict
        Statistics of the train dataset.
    """

    files = [f for f in os.listdir(train_dir) if f.endswith(".pt")]

    total_samples = 0
    overall_class_counts = Counter()
    client_data_counts = {}
    client_class_variance = {}

    for file in files:
        client_id = file.replace(".pt", "")
        dataset = torch.load(os.path.join(train_dir, file), weights_only=False)

        if hasattr(dataset, "data"):
            labels = [int(label) for _, label in dataset.data]
        elif hasattr(dataset, "y"):
            labels = [int(label) for label in dataset.y]
        else:
            raise AttributeError("Dataset has neither 'data' nor 'y' attribute")

        num_samples = len(labels)
        total_samples += num_samples
        client_data_counts[client_id] = num_samples

        counts = Counter(labels)

        for k, v in counts.items():
            overall_class_counts[k] += v

        class_sizes = np.array(list(counts.values()))
        variance = np.var(class_sizes) if len(class_sizes) > 1 else 0
        client_class_variance[client_id] = variance

    num_clients = len(files)

    avg_samples = total_samples / num_clients if num_clients > 0 else 0
    min_samples = min(client_data_counts.values()) if client_data_counts else 0
    max_samples = max(client_data_counts.values()) if client_data_counts else 0

    avg_variance = np.mean(list(client_class_variance.values())) if client_class_variance else 0

    summary = {
        "total_samples": total_samples,
        "num_clients": num_clients,
        "avg_samples_per_client": avg_samples,
        "min_samples_per_client": min_samples,
        "max_samples_per_client": max_samples,
        "avg_class_variance": avg_variance,
        "global_class_distribution": dict(overall_class_counts),
    }

    print("\n===== Train Dataset Statistics =====")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Number of clients: {summary['num_clients']}")
    print(f"Average samples per client: {summary['avg_samples_per_client']:.2f}")
    print(f"Min samples per client: {summary['min_samples_per_client']}")
    print(f"Max samples per client: {summary['max_samples_per_client']}")
    print(f"Average class variance: {summary['avg_class_variance']:.4f}")
    print(f"Global class distribution: {summary['global_class_distribution']}")

    return summary


def load_clients(data_dir):
    """
    Load client data from the specified directory.
    """
    files = os.listdir(data_dir)

    clients = {}
    for file in files:
        client_id = file.split(".")[0]
        file_path = os.path.join(data_dir, file)

        with open(file_path, "rb") as f:
            clients[client_id] = pickle.load(f)

    return clients


def process_dataset(data_dir, train_dir):
    """
    Process the dataset by loading clients, verifying their class distributions, and summarizing the dataset.
    """
    
    clients = load_clients(data_dir)

    valid = verify_clients(clients)
    if valid:
        print("All clients have at least two different classes.")
    else:
        print("Some clients do not have at least two different classes.")

    summary = dataset_summary(clients)

    print("\nDataset summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    train_dataset_statistics(train_dir)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset name (femnist, mnist, cifar10)",
    )
    args = parser.parse_args()

    dataset = args.dataset.lower()

    dataset_paths = {
        "mnist": {
            "data_dir": "mnist/client_data/alpha_0_1",
            "train_dir": "mnist/alpha_0_1/trainpt",
        },
        "cifar10": {
            "data_dir": "cifar10/client_data/alpha_0_1",
            "train_dir": "cifar10/alpha_0_1/trainpt",
        },
    }

    if dataset == "femnist":
        train_dir = os.path.join("femnist", "trainpt")
        train_dataset_statistics(train_dir)

    elif dataset in dataset_paths:
        paths = dataset_paths[dataset]
        process_dataset(paths["data_dir"], paths["train_dir"])

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

if __name__ == "__main__":
    main()