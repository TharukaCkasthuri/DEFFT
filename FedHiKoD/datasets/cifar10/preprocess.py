import numpy as np
import pickle
import cv2
import os
import argparse
import random


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from imutils import paths

import torch
from torch.utils.data import Dataset

import hydra
from omegaconf import DictConfig

class_mapping = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}

""" def load(paths, verbose=-1) -> tuple:
    Loads the images and labels from disk (CIFAR10 version but MNIST-style output)

    Returns:
        image_list: list of flattened RGB images normalized to [0,1]
        label_list: list of integer labels
    data = []
    labels = []

    for (i, imgpath) in enumerate(paths):
        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Flatten the 32x32x3 image
        image = img_rgb.astype("float32").flatten() / 255.0

        label_str = imgpath.split(os.path.sep)[-2]
        label = class_mapping[label_str]  # INTEGER label

        data.append(image)
        labels.append(label)

        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print(f"[INFO] processed {i+1}/{len(paths)}")

    return data, labels """

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
CIFAR10_STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)

def load(paths, verbose=-1):
    data = []
    labels = []

    for i, imgpath in enumerate(paths):
        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # HWC → CHW
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        # [0,255] → [0,1]
        img = img / 255.0

        # Per-channel standardization
        img = (img - CIFAR10_MEAN) / CIFAR10_STD

        label_str = imgpath.split(os.path.sep)[-2]
        label = int(class_mapping[label_str])

        data.append(img)
        labels.append(label)

        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print(f"[INFO] processed {i+1}/{len(paths)}")

    return data, labels



def create_clients(image_list: list, label_list: list, num_clients: int, initial: str, save_dir: str, batch_size: int) -> dict:
    """
    Create clients using the given images and labels, and save each client's data into a directory.
    This version splits the sorted data (skewed by class) into many shards, assigns several shards per client,
    and then ensures that each client has at least two different classes by swapping shards if needed.
    
    Parameters:
    ------------
    image_list: list of numpy arrays.
    label_list: list of binary labels (assumed one-hot).
    num_clients: number of clients to create.
    initial: client name prefix (e.g., 'client').
    save_dir: the base directory to save client data.
    batch_size: minimum batch size requirement per client.
    
    Returns:
    ------------
    clients: dictionary of client data.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Attempting to create {num_clients} clients from {len(image_list)} samples.")

    # Compute the class index for each sample
    max_y = np.argmax(label_list, axis=-1)
    sorted_zip = sorted(zip(max_y, label_list, image_list), key=lambda x: x[0])
    # Create tuples (image, label, class) for later processing.
    data = [(x, y, cls) for cls, y, x in sorted_zip]

    # Minimum number of samples each client 
    min_client_size = max(batch_size*3, 1)
    max_clients_possible = len(data) // min_client_size
    if num_clients > max_clients_possible:
        print(f"Reducing clients from {num_clients} to {max_clients_possible} to ensure batch-size allocation.")
        num_clients = max_clients_possible

    client_names = [f"{initial}_{i+1}" for i in range(num_clients)]
    print("Client names:", client_names)

    num_shards = 4 * num_clients
    shard_size = len(data) // num_shards
    shards = [data[i * shard_size: (i + 1) * shard_size] for i in range(num_shards)]

    random.shuffle(shards)

    # Assign shards to clients in a round-robin fashion.
    clients_shards = {client: [] for client in client_names}
    for i, shard in enumerate(shards):
        client = client_names[i % num_clients]
        clients_shards[client].append(shard)

    def get_client_classes(shard_list):
        """
        Get set of classes for a client's assigned shards.
        """
        return set(shard[0][2] for shard in shard_list if shard)

    # Check each client and ensure it has at least two different classes.
    for client in client_names:
        client_classes = get_client_classes(clients_shards[client])
        if len(client_classes) < 2:
            # Check for shard in another client to swap that brings in a new class.
            for other in client_names:
                if other == client:
                    continue
                for j, other_shard in enumerate(clients_shards[other]):
                    other_class = other_shard[0][2] if other_shard else None
                    if other_class not in client_classes:
                        # Swap one shard with this shard.
                        clients_shards[client][0], clients_shards[other][j] = clients_shards[other][j], clients_shards[client][0]
                        client_classes = get_client_classes(clients_shards[client])
                        break
                if len(client_classes) >= 2:
                    break
            if len(client_classes) < 2:
                print(f"Warning: Client {client} could not get at least two different classes.")

    # Combine shards for each client.
    clients = {}
    for client in client_names:
        client_data = [item for shard in clients_shards[client] for item in shard]
        client_data = [(img, label) for img, label, cls in client_data]
        clients[client] = client_data

        # Save client data.
        file_name = f"{client}.pkl"
        with open(os.path.join(save_dir, file_name), 'wb') as f:
            pickle.dump(client_data, f)

    print(f"Successfully created {num_clients} clients, each with at least two different classes and a skewed distribution.")
    return clients

def create_clients_dirichlet(
    image_list: list,
    label_list: list,
    num_clients: int,
    initial: str,
    save_dir: str,
    batch_size: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Create clients using Dirichlet-based splitting without data duplication.
    Clients with too few samples are merged, and their data is redistributed to other clients.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    alpha_tag = f"alpha_{alpha}".replace('.', '_')
    output_dir = os.path.join(save_dir, alpha_tag)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Creating {num_clients} clients from {len(image_list)} samples using Dirichlet(alpha={alpha})...")

    # Use integer labels (step 1 already ensures this)
    y = np.array(label_list, dtype=int)

    classes = np.unique(y)
    class_indices = [np.where(y == c)[0] for c in classes]
    client_data_indices = [[] for _ in range(num_clients)]

    np.random.seed(seed)
    random.seed(seed)

    # Distribute data per class
    for c_indices in class_indices:
        n_samples = len(c_indices)
        if n_samples == 0:
            continue

        props = np.random.dirichlet([alpha] * num_clients)
        counts = np.round(props * n_samples).astype(int)

        # Adjust total to match exactly
        diff = n_samples - np.sum(counts)
        while diff > 0:
            counts[np.argmin(counts)] += 1
            diff -= 1
        while diff < 0:
            idx = np.argmax(counts)
            if counts[idx] > 0:
                counts[idx] -= 1
                diff += 1

        np.random.shuffle(c_indices)
        start = 0
        for client_id, count in enumerate(counts):
            if count > 0:
                client_data_indices[client_id].extend(c_indices[start:start + count])
                start += count

    # Minimum samples per client
    min_samples = int(2 * batch_size)
    client_names = [f"{initial}_{i+1}" for i in range(num_clients)]

    clients = {}
    small_clients = []

    for i, cname in enumerate(client_names):
        idxs = client_data_indices[i]
        if len(idxs) < min_samples:
            print(f"Client {cname} too small: {len(idxs)} samples (needs {min_samples}). Marking for merge.")
            small_clients.extend(idxs)
            continue

        data = [(image_list[j], label_list[j]) for j in idxs]
        with open(os.path.join(output_dir, f"{cname}.pkl"), 'wb') as f:
            pickle.dump(data, f)
        clients[cname] = data

    # Redistribute small-client samples
    valid_clients = list(clients.keys())
    np.random.shuffle(valid_clients)

    print("\nRedistributing small-client samples...")

    for idx in small_clients:
        chosen = random.choice(valid_clients)
        clients[chosen].append((image_list[idx], int(y[idx])))

    print(f"\nCreated {len(clients)} clients. Total samples preserved: {len(image_list)}.")
    print(f"Saved client files to: {output_dir}")

    return clients

class CIFARDataset(Dataset):
    """
    Custom dataset class for the training and validation dataset.
    """

    def __init__(self, data) -> None:
        self.data = data

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        --------
        length: int; length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        """
        Returns the item at the given index.

        Parameters:
        ------------
        idx: int; index of the item

        Returns:
        ------------
        x_train: torch.tensor object; input data
        y_train: torch.tensor object; label
        """
        return self.data[idx]
    
    def num_classes(self) -> int:
        """
        Returns the number of unique classes in the dataset.

        Returns:
        ------------
        num_classes: int; number of unique labels in the dataset
        """
        # Extract labels from the dataset
        labels = [label for _, label in self.data]
        return len(torch.unique(torch.tensor(labels)))
    

def build_dataset(data_dir, saving_dir, alpha) -> None:
    """
    Split the pickles into train and test, saving as a PyTorch dataset with stratified split.

    Parameters:
    ------------
    data_dir: str; path to the pickle files
    saving_dir: str; path to save the PyTorch datasets

    Returns:
    ------------
    None
    """

    alpha_tag = f"alpha_{alpha}".replace(".", "_")
    data_dir = os.path.join(data_dir, alpha_tag)

    files = os.listdir(data_dir)
    ids = [file.split(".")[0] for file in files]
    file_paths = [os.path.join(data_dir, f) for f in files]

    os.makedirs(saving_dir, exist_ok=True)

    train_dir = os.path.join(saving_dir, "trainpt")
    test_dir = os.path.join(saving_dir, "testpt")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for cid, pkl_path in zip(ids, file_paths):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        labels = [label for _, label in data]
        integer_labels = [int(l) for l in labels]

        # Skip clients with fewer than 2 classes
        if len(set(integer_labels)) < 2:
            print(f"Skipping client {cid} — only one class: {set(integer_labels)}")
            continue

        try:
            train_data, test_data = train_test_split(
                data,
                test_size=0.2,
                random_state=42,
                stratify=integer_labels
            )
        except ValueError:
            # fallback when stratify breaks
            train_data, test_data = train_test_split(
                data, test_size=0.2, random_state=42
            )

        train_dataset = CIFARDataset(train_data)
        test_dataset  = CIFARDataset(test_data)

        torch.save(train_dataset, os.path.join(train_dir, f"{cid}.pt"))
        torch.save(test_dataset,  os.path.join(test_dir, f"{cid}.pt"))

        print(f"Saved {train_dir}/{cid}.pt and {test_dir}/{cid}.pt")

def main():
    parser = argparse.ArgumentParser(description="Preprocess the CIFAR dataset.")
    parser.add_argument("--num_clients", type=int, default=150)
    parser.add_argument("--image_path", type=str, default="/Users/tak/Documents/BTH/cifar10")
    parser.add_argument("--alpha", type=float, default=0.3                                        )
    args = parser.parse_args()

    image_path = args.image_path
    image_paths = list(paths.list_images(image_path))
    image_list, label_list = load(image_paths, verbose=10000)

    #binarize the labels
    label_list = [int(label) for label in label_list]
    print("Labels:", set(label_list))

    create_clients_dirichlet(image_list, label_list, num_clients=args.num_clients,
               initial='client',
               save_dir='client_data',
               batch_size=32,
               alpha=args.alpha)
    #create_clients(X_train, y_train, num_clients=args.num_clients, initial='client', save_dir='client_data', batch_size=32)
    build_dataset("client_data", f"alpha_{args.alpha}".replace(".", "_"), args.alpha)

if __name__ == "__main__":
    main()