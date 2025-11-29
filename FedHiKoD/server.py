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

Paper: [FedHiKoD: Fairness-Aware Hierarchical Knowledge Distillation for Robust Federated Learning]
Published in: 
"""

import os
import json
import copy
import logging
import math
import statistics
from abc import ABC, abstractmethod

import torch
import hashlib
import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from typing import List
from collections import defaultdict

from concurrent.futures import ThreadPoolExecutor, as_completed

from clients import Client
from aggregators import weighted_avg

from datasets.femnist.preprocess import FEMNISTDataset
from datasets.mnist.preprocess import MNISTDataset
from datasets.cifar10.preprocess import CIFARDataset
from utils import stable_hash
from typing import Tuple, Dict
from config import FitnessCfg

class Server(ABC):
    """
    Abstract Federated Learning Server.
    """

    def __init__(self, rounds: int, checkpt_path: str = None, log_dir: str = 'runs') -> None:
        self.rounds = rounds
        self.client_dict = {}
        self.checkpoint_path = checkpt_path or "./checkpoints"

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logging.info(f"Server initialized on {self.device}")
        #self.writer = SummaryWriter(log_dir=log_dir)
        
    def init_model(self, model: torch.nn.Module) -> None:
        """
        Initialize the model for federated learning.

        Parameters:
        ----------------
        model: torch.nn.Module object;
            The global model.
        """
        self.global_model = model.to(self.device).train()

    def connect_client(self, client) -> None:
        """
        Add a client for federated learning setup.

        Parameters:
        ----------------
        client_id: str;
            Client id
        """
        self.client_dict[client.client_id] = client

    def sample_clients(self, num_clients: int) -> dict:
        """
        Sample clients from the client dictionary.

        Parameters:
        ----------------
        num_clients: int;
            Number of clients to sample

        Returns:
        ----------------
        sampled_clients: dict
            Dict of sampled clients
        """
        num_clients = min(num_clients, len(self.client_dict))
        sampled_client_ids = np.random.choice(list(self.client_dict.keys()), num_clients, replace=False)
        sampled_clients = {client_id: self.client_dict[client_id] for client_id in sampled_client_ids}
  
        return sampled_clients

    def _broadcast(self, model: torch.nn.Module, clients:list = None) -> None:
        """
        Broadcast the model to the clients.

        Parameters:
        ----------------
        model: torch.nn.Module object;
            Model to be broadcasted
        """
        model_state_dict = model.state_dict()

        if clients is None:
            clients = list(self.client_dict.keys())

        for client_id, client in self.client_dict.items():
            if client_id in clients:
                client.set_model(copy.deepcopy(model_state_dict))
                self.client_dict[client_id] = client
                #print(f"Broadcasted model to client {client.client_id}")

    @abstractmethod
    def _aggregate(self, trained_clients: dict, weights: list) -> bool:
        """
        Must be implemented in subclass.
        Should return True if model was updated.
        """
        pass

    def train(self, train_schedule: dict, max_local_round: int, threshold: float, patience: int, multithreading: bool = False) -> torch.nn.Module:
        """
        Train the model using federated learning.
        
        Parameters:
        ----------------
        train_schedule: dict;
            Dictionary containing the training schedule for clients.
        max_local_round: int;
            Maximum number of local rounds for each client.
        threshold: float;
            Threshold for early stopping.
        patience: int;
            Patience for early stopping.
        
        Returns:
        ----------------
        model: torch.nn.Module object;
            Trained model
        """

        for round in range(1, self.rounds + 1):
            logging.info(f"\n=== Global Round {round} ===")
            train_clients_ids = train_schedule.get(str(round), [])

            if not train_clients_ids:
                logging.warning(f"No clients for round {round}. Skipping.")
                continue

            train_clients = {cid: self.client_dict[cid] for cid in train_clients_ids}
            self._broadcast(self.global_model, train_clients)

            num_data_points = {}
            for client in train_clients.values():
                try:
                    client.train(round, max_local_round, threshold, patience)
                    num_data_points[client.client_id] = client.get_num_datapoints()
                    
                except Exception as e:
                    logging.error(f"Client {client.client_id} failed: {e}")
                    raise
                logging.info(f"\n")

            total_points = sum(num_data_points.values())
            if total_points == 0:
                logging.warning("No data points collected this round.")
                continue

            weights = [num_data_points[c.client_id] / total_points for c in train_clients.values()]
            logging.info(f"Global model aggregation weights: {weights}")
            self.global_model = self._aggregate(train_clients, weights)

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")


        return self.global_model

    def save_checkpt(self, checkpoint: torch.nn.Module, ckptpath: str) -> None:
        """
        Saving the checkpoints.

        Parameters:
        ----------------
        checkpoint:
            Model at a specific checkpoint.
        ckptpath: str;
            Path to save the checkpoint. Default is None.
        """
        if os.path.exists(ckptpath):
            torch.save(
                checkpoint.state_dict(),
                ckptpath,
            )
        else:
            os.makedirs(os.path.dirname(ckptpath), exist_ok=True)
            torch.save(
                checkpoint.state_dict(),
                ckptpath,
            )

    def evaluate(self, test_dataset: torch.utils.data.Dataset) -> dict:
        """
        Evaluate the global model on a test dataset.

        Parameters:
        ----------------
        test_dataset: torch.utils.data.Dataset;
            Dataset to evaluate the model on.

        Returns:
        ----------------
        results: dict;
            Dictionary containing evaluation metrics.
        """
        self.global_model.eval()
        results = {}

        # Assuming test_dataset has a method to get data and labels
        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.global_model(data)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='weighted')
        results['f1_score'] = f1

        return results


class FedAvgServer(Server):
    """
    The federated learning server class for FedAvg."""
    def _aggregate(self, trained_clients, weights):
        return weighted_avg(self.global_model, [c.get_model() for c in trained_clients.values()], weights)
    

class QFedAvgServer(Server):
    """
    Federated server implementing q-FedAvg aggregation.
    """

    def __init__(self, rounds: int, q: float = 0.5, L: float = 1.0,
                 checkpt_path: str = None, log_dir: str = 'runs') -> None:
        super().__init__(rounds=rounds, checkpt_path=checkpt_path, log_dir=log_dir)
        self.q = q
        self.L = L
        self._eps = 1e-9  # numerical stability

    def _aggregate(self, trained_clients: dict, weights) -> torch.nn.Module:
        """
        q-FedAvg aggregation.
        trained_clients: dict[client_id -> client]
        We use:
            - client.get_loss_at_global()
            - client.get_model()
        """
        device = self.device

        # snapshot global params w^t
        global_state = {
            name: param.detach().clone().to(device)
            for name, param in self.global_model.state_dict().items()
        }

        # initialize accumulators
        delta_sum = {
            name: torch.zeros_like(param, device=device)
            for name, param in global_state.items()
        }
        h_sum = torch.tensor(0.0, device=device)

        for cid, client in trained_clients.items():

            # local model weights w_k^{t+1}
            local_models_state = {
                name: p.detach().clone().to(device)
                for name, p in client.get_model().state_dict().items()
            }

            # compute Δw and its norm
            delta_norm_sq = torch.tensor(0.0, device=device)
            delta_w = {}

            for name in global_state.keys():
                diff = self.L * (global_state[name] - local_models_state[name])
                delta_w[name] = diff
                delta_norm_sq += torch.sum(diff * diff)

            # retrieve F_k(w^t)
            F_k = client.get_loss_at_global()
            if not torch.is_tensor(F_k):
                F_k = torch.tensor(F_k, dtype=torch.float32, device=device)
            F_k = torch.clamp(F_k.to(device), min=self._eps)
            Fk_q = F_k ** self.q

            # accumulate Δ_k^t
            for name in delta_sum.keys():
                delta_sum[name] += Fk_q * delta_w[name]

            # accumulate h_k^t
            h_k = self.q * (F_k ** (self.q - 1.0)) * delta_norm_sq + self.L * Fk_q
            h_sum += h_k

        if h_sum.item() == 0.0:
            logging.warning("q-FedAvg: h_sum == 0, skipping update")
            return self.global_model

        # new model update
        new_state = {}
        for name, w_t in global_state.items():
            new_state[name] = w_t - delta_sum[name] / h_sum

        self.global_model.load_state_dict(new_state)
        return self.global_model

    
    def train(self, train_schedule: dict, max_local_round: int, threshold: float, patience: int,
          multithreading: bool = False) -> torch.nn.Module:

        for round in range(1, self.rounds + 1):
            logging.info(f"\n=== Global Round {round} ===")
            print(f"\n=== Global Round {round} ===")

            train_clients_ids = train_schedule.get(str(round), [])

            train_clients = {cid: self.client_dict[cid] for cid in train_clients_ids}
            self._broadcast(self.global_model, train_clients)

            num_data_points = {}

            for client in train_clients.values():
                try:
                    client.train(
                        round, max_local_round, threshold, patience
                    )
                    num_data_points[client.client_id] = client.get_num_datapoints()
                except Exception as e:
                    logging.error(f"Client {client.client_id} failed: {e}")
                    raise
                logging.info(f"\n")

            total_points = sum(num_data_points.values())
            if total_points == 0:
                logging.warning("No data points collected this round.")
                continue

            # not used by q-FedAvg but passed to keep API consistent
            weights = [
                num_data_points[cid] / total_points
                for cid in train_clients_ids
            ]

            #   client.get_model()  -> w_k^{t+1}
            #   client.get_loss_at_global() -> F_k(w^t)
            self.global_model = self._aggregate(train_clients, weights)

            self.save_checkpt(
                self.global_model,
                f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt"
            )

        return self.global_model



class FedProxServer(Server):

    def receive_global(self, model_weights: dict) -> None:
        try:
            self.broadcast_model.load_state_dict(model_weights, strict=True)
        except Exception as e:
            logging.error(f"Client {self.client_id} broadcast load error: {e}. Keeping old weights.")
            return
        self.local_model.load_state_dict(self.broadcast_model.state_dict())
        # re-init optimizer here to avoid stale momentum/state
        if isinstance(self.train_dataset, CIFARDataset):
            self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.optimizer.defaults['lr'],
                                            weight_decay=self.optimizer.defaults['weight_decay'], momentum=0.9)
        else:
            self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.optimizer.defaults['lr'],
                                            weight_decay=self.optimizer.defaults['weight_decay'])


    def _aggregate(self, trained_clients, weights):
        return weighted_avg(self.global_model, [c.get_model() for c in trained_clients.values()], weights)
    

    def train(self, train_schedule: dict, max_local_round: int, mu:float, threshold: float=None, patience: int=None,) -> torch.nn.Module:
        """
        
        """

        for round in range(1, self.rounds + 1):
            logging.info(f"\n=== Global Round {round} ===")
            train_clients_ids = train_schedule.get(str(1), [])

            train_clients = {cid: self.client_dict[cid] for cid in train_clients_ids}
            self._broadcast(self.global_model, train_clients)

            num_data_points = {}
            for client in train_clients.values():
                client.train(round, max_local_round, mu, threshold, patience)
                num_data_points[client.client_id] = client.get_num_datapoints()
                logging.info(f"\n")

            total_points = sum(num_data_points.values())
            if total_points == 0:
                logging.warning("No data points collected this round.")
                continue

            weights = [num_data_points[c.client_id] / total_points for c in train_clients.values()]
            self.global_model = self._aggregate(train_clients, weights)

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")
            


        return self.global_model


class DittoServer(Server):
    """
    DittoServer class extends the base Server for Ditto Personalized FL.
    In Ditto, each client trains both a global model copy (for aggregation)
    and a personal model. The server still only manages the global model,
    using standard aggregation. The personal models remain on each client.

    Parameters (inherited from Server):
    -----------------------------------
    rounds: int
        Number of global training rounds.
    stratergy: callable
        Aggregation strategy for federated learning.
    checkpt_path: str, optional
        Path to save checkpoints of the global model.
    log_dir: str
        TensorBoard log directory.
    """

    def __init__(self, rounds: int, stratergy: callable, checkpt_path: str = None, log_dir: str = 'runs'):
        super().__init__(rounds, stratergy, checkpt_path, log_dir)
  
    def train(self, train_samples: dict, max_local_round: int, threshold: float, patience: int) -> torch.nn.Module:
        """
        Overridden train method. The main difference for Ditto is that clients 
        (DittoClient) will internally train both the global model copy and their 
        personal model. The server side remains mostly the same as standard FL.
        """
        self.global_model = super().train(train_samples, max_local_round, threshold, patience)
        return self.global_model


class FedHiKoDServer(Server):

    """
    The federated learning server class for FedHiKoD.
    """

    def __init__(self, *args, fitness_cfg: FitnessCfg = FitnessCfg(), **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness_cfg = fitness_cfg

        self._g_ema = None             # torch.Tensor on CPU or None
        self._g_beta = 0.5             # momentum factor for g_EMA
        self._g_num_final_layers = 2   # must match clients' alignment layers
        self._groups: dict[str, int] = {}              # cid -> group id (fixed after round 1)
        self._client_distributions: dict[str, dict[int, float]] = {}
        self._leader_models_by_group: dict[int, dict] = {}  # gid -> state_dict
  

    def flatten_last_modules(self, model: torch.nn.Module, last_n_modules: int | None = None) -> torch.Tensor:
        """
        Flatten parameters from the last `last_n_modules` child modules of `model`.
        If `last_n_modules` is None, flatten all parameters.
        """
        with torch.no_grad():
            # children keep registration order: fc1, relu, fc2
            modules = [m for _, m in model.named_children()]
            # drop non-param modules like ReLU
            modules = [m for m in modules if any(p.requires_grad for p in m.parameters(recurse=False))]

            if last_n_modules is None:
                selected = modules
            else:
                selected = modules[-last_n_modules:]

            flats = []
            for m in selected:
                for p in m.parameters(recurse=False):
                    flats.append(p.detach().cpu().flatten())
            if not flats:  # fallback to all params if something odd happens
                flats = [p.detach().cpu().flatten() for p in model.parameters()]
            return torch.cat(flats)


    def _global_update_vector(self, old_model: torch.nn.Module, new_model: torch.nn.Module,
                              num_final_layers: int | None = 2) -> torch.Tensor:
        with torch.no_grad():
            v_old = self.flatten_last_modules(old_model)
            v_new = self.flatten_last_modules(new_model)
            dtheta = (v_new - v_old).detach().cpu()
        return dtheta
    

    def _update_global_momentum(self, dtheta: torch.Tensor) -> torch.Tensor:
        # no-op if the update is degenerate
        if dtheta is None or torch.norm(dtheta) == 0:
            return self._g_ema

        if self._g_ema is None or torch.numel(self._g_ema) != torch.numel(dtheta):
            # initialize on first use or if shape changed
            self._g_ema = dtheta.clone().detach().cpu()
        else:
            self._g_ema = (self._g_beta * self._g_ema) + ((1.0 - self._g_beta) * dtheta.detach().cpu())
        return self._g_ema


    def _embed_for_clustering(self, distributions: Dict, total_classes: int, alpha: int = 1)-> Tuple[list, np.ndarray]:
        """
        Create a fixed-size embedding vector from each client's class distribution
        and compute pairwise Jensen-Shannon distances.
        """
        from scipy.spatial.distance import pdist
        from scipy.spatial.distance import jensenshannon

        client_ids = list(distributions.keys())
        vectors = []

        for cid in client_ids:
            counts = np.array([distributions[cid].get(i, 0) for i in range(total_classes)], dtype=float)
            counts += alpha  # Laplace smoothing
            counts /= counts.sum()
            vectors.append(counts)

        vectors = np.vstack(vectors)
        jsd_condensed = pdist(vectors, metric=lambda u, v: jensenshannon(u, v))
        return client_ids, jsd_condensed


    def hierarchical_clustering(self, client_ids, jsd_condensed, num_clusters):
        """
        Perform hierarchical clustering using Jensen–Shannon distances.
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from collections import defaultdict

        # Perform linkage using precomputed distances
        Z = linkage(jsd_condensed, method='average', optimal_ordering=True)
        clusters = fcluster(Z, t=num_clusters, criterion='maxclust')

        client_to_cluster = {}
        cluster_to_clients = defaultdict(list)

        for cid, cluster_id in zip(client_ids, clusters):
            client_to_cluster[cid] = cluster_id
            cluster_to_clients[cluster_id].append(cid)

        return client_to_cluster, cluster_to_clients


    def _collect_self_reports(self, client: Client) -> dict[str, dict]:
        cfg = self.fitness_cfg
        recipe_hash = hashlib.sha256(f"metric:loss|clip:{cfg.clip_range}".encode()).hexdigest()

        if not hasattr(client, "report_fitness"):
            raise RuntimeError(f"Client {client.client_id} lacks report_fitness()")
        
        # these are hyperparams
        rpt = client.report_fitness(
            recipe_hash=recipe_hash,
            clip_range=cfg.clip_range,
        )

        return rpt

 
    def _aggregate(self, trained_clients, weights):
        """
        Aggregate the models of the clients using FedHiKoD aggregation strategy.
        """
        return weighted_avg(self.global_model, trained_clients, weights)


    def _cluster_quality(self, reports: dict[str, dict], c2g: dict[str, int]) -> dict[int, float]:
        """
        Aggregate client 'score' values into per-cluster quality.
        Normalizes to [0.1, 1.0] to avoid zero weights.
        """

        cluster_scores = defaultdict(list)
        for cid, rpt in reports.items():
            gid = c2g.get(cid)
            score = float(rpt.get("score", 0.0))
            cluster_scores[gid].append(score)

        if not cluster_scores:
            return {}

        avg_scores = {g: np.mean(v) for g, v in cluster_scores.items()}
        vals = np.array(list(avg_scores.values()))
        vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        vals = np.clip(vals, 0.1, 1.0)  # keep all clusters active
        return {g: float(vals[i]) for i, g in enumerate(avg_scores.keys())}
    

    def find_nclasses(self, jsd_condensed):
        from scipy.cluster.hierarchy import linkage, fcluster

        Z = linkage(jsd_condensed, method='average', optimal_ordering=True)

        best_k, sil_scores = self.evaluate_silhouette_scores(Z, jsd_condensed, k_min=2, k_max=11)

        return best_k, sil_scores


    def evaluate_silhouette_scores(self, Z, jsd_condensed, k_min=2, k_max=11):
        """
        """
        from scipy.cluster.hierarchy import fcluster
        from scipy.spatial.distance import squareform
        from sklearn.metrics import silhouette_score

        # Convert condensed distances to a full matrix for silhouette_score
        jsd_matrix = squareform(jsd_condensed)
        n_samples = jsd_matrix.shape[0]

        sil_scores = {}

        for k in range(k_min, min(k_max, n_samples - 1) + 1):
            clusters_k = fcluster(Z, k, criterion='maxclust')

            # Skip if clustering degenerates (e.g., all samples in one cluster)
            if len(np.unique(clusters_k)) < 2:
                continue

            score = silhouette_score(jsd_matrix, clusters_k, metric='precomputed')
            sil_scores[k] = score

        best_k = max(sil_scores, key=sil_scores.get)

        return best_k, sil_scores
    

    def train(self, train_schedule: dict, max_local_round: int, threshold: float, patience: int,
              multithreading: bool = False) -> torch.nn.Module:
        
        logging.info("Starting FedHiKoD training.............")
        logging.info(f"Client dict: {self.client_dict}")

        self._client_distributions = {}
        for client in self.client_dict.values():
            self._client_distributions[client.client_id] = client.get_class_distribution()

        logging.info(f"Client distributions: {self._client_distributions}")

        client_ids, jsd_condensed = self._embed_for_clustering(self._client_distributions, total_classes=10, alpha=1)
        num_clusters, _ = self.find_nclasses(jsd_condensed)

        self._c2g, self._g2c = self.hierarchical_clustering(client_ids, jsd_condensed, num_clusters)

        #self._c2g, self._g2c = self.hierarchical_clustering(client_ids, jsd_condensed, 10)

        for round in range(1, self.rounds + 1):
            logging.info(f"=== Global Round {round} ===")
            train_clients_ids = train_schedule.get(str(round), [])
            reports = {}

            train_clients = {cid: self.client_dict[cid] for cid in train_clients_ids}
            #----------------------------------------------------
            # Broadcast the global model to all clients
            #----------------------------------------------------

            self._broadcast(self.global_model, train_clients)

            num_data_points = {}
            for client in train_clients.values():
                
                leader_state = None
                #----------------------------------------------------
                # Send the leader model to the client if applicable
                #----------------------------------------------------
                if round > 1:
                    cid = client.client_id
                    gid = self._c2g[cid]

                    # Retrieve and send the leader model
                    if gid in inactive_clusters:
                        logging.info(f"Client {cid} in inactive cluster {gid}, skipping leader send.")
                    leader_state = self._leader_models_by_group.get(gid)
                    if leader_state is None:
                        logging.info(f"Not Sending leader model to client {cid} from group {gid}.")
                    if leader_state is not None:
                        client.receive_leader(leader_state)

                #----------------------------------------------------
                # Train the client with FedHiKoD local updates
                # receive trained model from client           
                # receive num of data points and class distribution 
                # ----------------------------------------------------
                client.train(round, 
                            max_local_round)
                
                num_data_points[client.client_id] = client.get_num_datapoints()
                self._client_distributions[client.client_id] = client.get_class_distribution()

                try:
                    score = self._collect_self_reports(client)
                    reports[client.client_id] = score
                except Exception as e:
                    logging.error(f"Client {client.client_id} reporting failed: {e}")
                logging.info(f"\n")

            inactive_clusters = [
                gid for gid, client_ids in self._g2c.items()
                if all(cid not in train_clients for cid in client_ids)
            ]

            logging.info(f"Inactive clusters: {inactive_clusters}")

            # ----------------------------------------------------
            # Build group-aggregated leader models
            # ----------------------------------------------------

            self._leader_models_by_group = {}

            for group_id, client_ids in self._g2c.items():
                # Keep only clients that participated in this round
                valid_clients = [cid for cid in client_ids if cid in train_clients]
                if not valid_clients:
                    self._leader_models_by_group[group_id] = None
                    continue

                # Prepare list of models and weights for this group
                group_models = [train_clients[cid].get_model() for cid in valid_clients]
                group_weights_dict = {
                    cid: num_data_points[cid] / sum(num_data_points[c] for c in valid_clients)
                    for cid in valid_clients
                }
                group_weights =[group_weights_dict[cid] for cid in valid_clients]

                # Build the group leader
                logging.info("Subglobal (group:%s) model aggregation weights: %s", group_id, group_weights)
                group_leader = self._aggregate(group_models, group_weights)

                # Store as CPU state_dict (for later KD)
                self._leader_models_by_group[group_id] = {
                    k: v.detach().to('cpu', copy=False) for k, v in group_leader.state_dict().items()
                }

            cluster_q = self._cluster_quality(reports, self._c2g)
            logging.info(f"Cluster Quality: {cluster_q}")
            logging.info(f"Number of data points: {num_data_points}")

            weighted_size = {}
            for cid, n in num_data_points.items():
                gid = self._c2g.get(cid)
                q = cluster_q.get(gid, 1.0)
                weighted_size[cid] = n * q

            total_weight = sum(weighted_size.values())
            weights = {cid: weighted_size[cid] / total_weight for cid in weighted_size}
        
            cid_order = list(train_clients.keys())
            model_list = [train_clients[cid].get_model() for cid in cid_order]
            w_list = [weights[cid] for cid in cid_order]

            logging.info(f"Global model aggregation weights: {w_list}")
            self.global_model = self._aggregate(model_list, w_list)

            # No cluster_q needed anymore unless used elsewhere
            cluster_q = self._cluster_quality(reports, self._c2g)

            #weighted_size = {}
            #for cid, n in num_data_points.items():
            #    score = float(reports.get(cid, {}).get("score", 1.0))
            #    weighted_size[cid] = n * score

            #total_weight = sum(weighted_size.values())
            #weights = {cid: weighted_size[cid] / total_weight for cid in weighted_size}

            #cid_order = list(train_clients.keys())
            #model_list = [train_clients[cid].get_model() for cid in cid_order]
            #w_list = [weights[cid] for cid in cid_order]

            #logging.info(f"Global model aggregation weights: {w_list}")
            #self.global_model = self._aggregate(model_list, w_list)

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")

            self._prev_cluster_q = getattr(self, "_prev_cluster_q", cluster_q)
            cluster_q = {
                g: 0.7 * self._prev_cluster_q.get(g, q) + 0.3 * q
                for g, q in cluster_q.items()
            }
            self._prev_cluster_q = cluster_q

        return self.global_model

"""
    def train(self, train_schedule: dict, max_local_round: int, threshold: float, patience: int,
              multithreading: bool = False) -> torch.nn.Module:
        
        logging.info("Starting FedHiKoD training.............")
        logging.info(f"Client dict: {self.client_dict}")

        self._client_distributions = {}
        #for client in self.client_dict.values():
        #    self._client_distributions[client.client_id] = client.get_class_distribution()

        #logging.info(f"Client distributions: {self._client_distributions}")

        #client_ids, jsd_condensed = self._embed_for_clustering(self._client_distributions, total_classes=10, alpha=1)
        #self._c2g, self._g2c = self.hierarchical_clustering(client_ids, jsd_condensed, 10)

        # Get the client IDs that will participate in round 1
        round1_client_ids = set(train_schedule.get(str(1), []))
        for client in self.client_dict.values():
            if client.client_id in round1_client_ids:
                self._client_distributions[client.client_id] = client.get_class_distribution()

        logging.info(f"Client distributions (Round 1 only): {self._client_distributions}")

        client_ids, jsd_condensed = self._embed_for_clustering(self._client_distributions, total_classes=10, alpha=1)
        #num_clusters, _ = self.find_nclasses(jsd_condensed)

        self._c2g, self._g2c = self.hierarchical_clustering(client_ids, jsd_condensed, 10)

        for round in range(1, self.rounds + 1):
            logging.info(f"=== Global Round {round} ===")
            train_clients_ids = train_schedule.get(str(1), [])
            reports = {}

            train_clients = {cid: self.client_dict[cid] for cid in train_clients_ids}
            #----------------------------------------------------
            # Broadcast the global model to all clients
            #----------------------------------------------------

            self._broadcast(self.global_model, train_clients)

            num_data_points = {}
            for client in train_clients.values():
                
                leader_state = None
                #----------------------------------------------------
                # Send the leader model to the client if applicable
                #----------------------------------------------------
                if round > 1:
                    cid = client.client_id
                    gid = self._c2g[cid]

                    # Retrieve and send the leader model
                    if gid in inactive_clusters:
                        logging.info(f"Client {cid} in inactive cluster {gid}, skipping leader send.")
                    leader_state = self._leader_models_by_group.get(gid)
                    if leader_state is None:
                        logging.info(f"Not Sending leader model to client {cid} from group {gid}.")
                    if leader_state is not None:
                        client.receive_leader(leader_state)

                #----------------------------------------------------
                # Train the client with FedHiKoD local updates
                # receive trained model from client           
                # receive num of data points and class distribution 
                # ----------------------------------------------------
                client.train(round, 
                            max_local_round)
                
                num_data_points[client.client_id] = client.get_num_datapoints()
                self._client_distributions[client.client_id] = client.get_class_distribution()

                try:
                    score = self._collect_self_reports(client)
                    reports[client.client_id] = score
                except Exception as e:
                    logging.error(f"Client {client.client_id} reporting failed: {e}")
                logging.info(f"\n")

            inactive_clusters = [
                gid for gid, client_ids in self._g2c.items()
                if all(cid not in train_clients for cid in client_ids)
            ]

            logging.info(f"Inactive clusters: {inactive_clusters}")

            # ----------------------------------------------------
            # Build group-aggregated leader models
            # ----------------------------------------------------

            self._leader_models_by_group = {}

            for group_id, client_ids in self._g2c.items():
                # Keep only clients that participated in this round
                valid_clients = [cid for cid in client_ids if cid in train_clients]
                if not valid_clients:
                    self._leader_models_by_group[group_id] = None
                    continue

                # Prepare list of models and weights for this group
                group_models = [train_clients[cid].get_model() for cid in valid_clients]
                group_weights_dict = {
                    cid: num_data_points[cid] / sum(num_data_points[c] for c in valid_clients)
                    for cid in valid_clients
                }
                group_weights =[group_weights_dict[cid] for cid in valid_clients]

                # Build the group leader
                logging.info("Subglobal (group:%s) model aggregation weights: %s", group_id, group_weights)
                group_leader = self._aggregate(group_models, group_weights)

                # Store as CPU state_dict (for later KD)
                self._leader_models_by_group[group_id] = {
                    k: v.detach().to('cpu', copy=False) for k, v in group_leader.state_dict().items()
                }

            cluster_q = self._cluster_quality(reports, self._c2g)
            logging.info(f"Cluster Quality: {cluster_q}")
            logging.info(f"Number of data points: {num_data_points}")

            weighted_size = {}
            for cid, n in num_data_points.items():
                gid = self._c2g.get(cid)
                q = cluster_q.get(gid, 1.0)
                weighted_size[cid] = n * q

            total_weight = sum(weighted_size.values())
            weights = {cid: weighted_size[cid] / total_weight for cid in weighted_size}
        
            cid_order = list(train_clients.keys())
            model_list = [train_clients[cid].get_model() for cid in cid_order]
            w_list = [weights[cid] for cid in cid_order]

            logging.info(f"Global model aggregation weights: {w_list}")
            self.global_model = self._aggregate(model_list, w_list)

            # No cluster_q needed anymore unless used elsewhere
            cluster_q = self._cluster_quality(reports, self._c2g)

            #weighted_size = {}
            #for cid, n in num_data_points.items():
            #    score = float(reports.get(cid, {}).get("score", 1.0))
            #    weighted_size[cid] = n * score

            #total_weight = sum(weighted_size.values())
            #weights = {cid: weighted_size[cid] / total_weight for cid in weighted_size}

            #cid_order = list(train_clients.keys())
            #model_list = [train_clients[cid].get_model() for cid in cid_order]
            #w_list = [weights[cid] for cid in cid_order]

            #logging.info(f"Global model aggregation weights: {w_list}")
            #self.global_model = self._aggregate(model_list, w_list)

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")

            self._prev_cluster_q = getattr(self, "_prev_cluster_q", cluster_q)
            cluster_q = {
                g: 0.7 * self._prev_cluster_q.get(g, q) + 0.3 * q
                for g, q in cluster_q.items()
            }
            self._prev_cluster_q = cluster_q

        return self.global_model
"""


class BoostingServer(Server):

    """
    The federated learning server class for fedaboost.
    Parameters:
    ----------------
    rounds: int;
        Number of global rounds
    stratergy: callable;
        Aggregation stratergy for federated learning.
    checkpt_path: str;
        Path to save checkpoints of the global model.
    log_dir: str;
        TensorBoard log directory.
    max_local_round: int;
        Maximum number of local rounds for each client.

    Methods:
    ----------------
    __aggregate(self, weights = []) -> None:
        Aggregate the models of the clients.
    train(self) -> None:
        Train the model using federated fedaboost-optima algorithm.
    
    """

    def __init__(self, rounds: int, strategy: callable, checkpt_path:str=None,  log_dir:str = 'runs', max_local_round = 10) -> None:
        super().__init__(rounds, strategy, checkpt_path, log_dir)
        self.max_local_round = max_local_round
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        logging.info("Boosting Server Initialized")


    def __aggregate(self,trained_clients, weights = None) -> None:
        """
        Aggregate the models of the clients.

        Parameters:
        ----------------
        weights: list or None
            List of weights for the weighted average strategy. 

        Returns:
        ----------------
        model: torch.nn.Module object
            Aggregated model.
        updated: bool
            Indicates whether the global model was updated.
        """
    
        prev_params = [p.clone() for p in self.global_model.parameters()]
        self.global_model = fedaboost_avg(self.global_model, trained_clients, weights)

        updated_params = [p.clone() for p in self.global_model.parameters()]
        updated = self._check_model_update(prev_params, updated_params)

        return self.global_model, updated


    def train(self, train_samples:dict, max_local_round:int, threshold:float, patience:int) -> torch.nn.Module:
        """
        Train the model using federated learning.

        Parameters:
        ----------------
        model: torch.nn.Module object;
            Model to be trained

        Returns:
        ----------------
        model: torch.nn.Module object;
            Trained model
        """
        consecutive_no_update_rounds = 0
        weights_dict = {client: (1 / len(train_samples[str(1)])) for client in self.client_dict}

        for client in self.client_dict.values():
            client.set_weight(1/10) #/len(train_samples[str(1)])

        logging.info(f"Initial Weights: {weights_dict}")

        gamma_history = {}

        for round in range(1,self.rounds+1):
            update_status = False

            print(f"\n | Global Training Round : {round} |\n")
            logging.info(f"\n | Global Training Round : {round} |\n")

            alphas = {}
            train_clients_list= train_samples[str(round)]
            k = len(train_clients_list)
            train_clients = {client: self.client_dict[client] for client in train_clients_list}
            self._broadcast(self.global_model, train_clients.keys())

            for client in train_clients.values():
                _, alpha, gamma = client.train(round, max_local_round, k, threshold, patience)
                alphas[client.client_id] = alpha
                gamma_history[client.client_id] = gamma
                self._receive(client)
            logging.info(f"Alpha values used for aggregation: {alphas}")


            # align the client order
            client_ids = list(train_clients.keys())
            alpha_list = [alphas[cid] for cid in client_ids]  # consistent ordering
            #alpha_values = self.scaler.fit_transform(np.array(alpha_list).reshape(-1, 1)).flatten()

            # Normalize alpha values to sum to 1, comment this later
            alpha_tensor = torch.tensor(alpha_list, dtype=torch.float32)
            alpha_values = torch.softmax(alpha_tensor, dim=0).numpy()  # weights sum to 1

            client_models = [train_clients[cid].get_model() for cid in client_ids]
            self.global_model, update_status = self.__aggregate(client_models, alpha_values)
        
            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")

            print(f"Model Updated: {update_status}")
            logging.info(f"Model Updated Successfully using Fedaboost weighted averaging: {update_status}")
            
            if not update_status:
                consecutive_no_update_rounds += 1
                print("The global model parameters have not been updated, so the training has converged.")
                logging.info("The global model parameters have not been updated, so the training has converged. Might be some issue with the model aggregation.")
            else:
                consecutive_no_update_rounds = 0

            if consecutive_no_update_rounds == 3:
                print("The global model parameters have not been updated for 5 consecutive rounds, so the training has converged.")
                logging.info("The global model parameters have not been updated for 3 consecutive rounds. Hence, stopping the training.")
                break
        
        #pd.DataFrame(gamma_history).to_csv(f"{self.checkpoint_path}/gamma_history.csv", index=False)
        with open(f"{self.checkpoint_path}/clients_gamma.json", 'w') as f:
            json.dump(gamma_history, f, indent=4)

        return self.global_model

