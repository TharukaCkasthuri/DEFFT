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

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


from clients import Client
from aggregators import weighted_avg

from datasets.femnist.preprocess import FEMNISTDataset
from datasets.mnist.preprocess import MNISTDataset
from datasets.cifar10.preprocess import CIFARDataset
from utils import stable_hash, save_client_distributions
from typing import Tuple, Dict
from config import FitnessCfg
from scipy.stats import entropy

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
            new_global = self._aggregate(train_clients, weights)
            self.global_model.load_state_dict(new_global.state_dict())

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

    def _aggregate(self, trained_clients: dict) -> torch.nn.Module:
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
            if torch.is_floating_point(param)
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
                if torch.is_floating_point(p)
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

        new_state = self.global_model.state_dict()

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

            #   client.get_model()  -> w_k^{t+1}
            #   client.get_loss_at_global() -> F_k(w^t)
            new_global = self._aggregate(train_clients)
            self.global_model.load_state_dict(new_global.state_dict())

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
    def __init__(
        self,
        rounds: int,
        beta: float,
        checkpt_path: str = None,
        log_dir: str = "runs",
    ):
        super().__init__(rounds, checkpt_path=checkpt_path, log_dir=log_dir)
        self.beta = beta
        self._groups: dict[str, int] = {}              # cid -> group id (fixed after round 1)
        self._client_distributions: dict[str, dict[int, float]] = {}
        self._leader_models_by_group: dict[int, dict] = {}  # gid -> state_dict


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
            print(f"Client {cid} counts before normalization: {counts}")
            counts /= counts.sum()
            vectors.append(counts)

        vectors = np.vstack(vectors)
        jsd_condensed = pdist(vectors, metric=lambda u, v: jensenshannon(u, v))
        return client_ids, jsd_condensed


    def hierarchical_clustering(self, client_ids, jsd_condensed):
        """
        Perform hierarchical clustering based on JSD distances.
        """
        jsd_threshold, Z = self._find_jsd_threshold(jsd_condensed, method="average")

        logging.info(f"JSD distance threshold for clustering: {jsd_threshold:.4f}")

        clusters = fcluster(Z, t=498, criterion='distance')
        _, counts = np.unique(clusters, return_counts=True)

        singleton_frac = np.mean(counts == 1)

        if singleton_frac > 0.25:
            # gently relax, not jump to the top
            T = np.quantile(Z[:, 2], 0.88)
            clusters = fcluster(Z, t=0.35, criterion="distance")


        client_to_cluster = {}
        cluster_to_clients = defaultdict(list)

        for cid, cluster_id in zip(client_ids, clusters):
            client_to_cluster[cid] = cluster_id
            cluster_to_clients[cluster_id].append(cid)

        return client_to_cluster, cluster_to_clients

    def gmm_clustering(self,
        client_ids,
        X,
        k_min=2,
        k_max=10,
        pca_dim=None,
        random_state=0,
        min_cluster_frac=0.01
    ):
        """
        Perform GMM-based clustering on client embeddings.

        Returns:
            client_to_cluster: dict {client_id -> cluster_id}
            cluster_to_clients: dict {cluster_id -> list(client_ids)}
        """

        # Standardize features
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # PCA (denoising)
        if pca_dim is not None:
            pca = PCA(n_components=pca_dim, random_state=random_state)
            Xs = pca.fit_transform(Xs)

        # Fit GMMs and select K via BIC
        bics = []
        models = []

        for k in range(k_min, k_max + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=10,
                random_state=random_state
            )
            gmm.fit(Xs)
            bics.append(gmm.bic(Xs))
            models.append(gmm)

        best_idx = int(np.argmin(bics))
        best_k = best_idx + k_min
        gmm = models[best_idx]

        logging.info(f"Selected number of clusters (BIC): {best_k}")

        # Predict cluster labels
        labels = gmm.predict(Xs)

        # mappings
        client_to_cluster = {}
        cluster_to_clients = defaultdict(list)

        for cid, cluster_id in zip(client_ids, labels):
            client_to_cluster[cid] = int(cluster_id)
            cluster_to_clients[int(cluster_id)].append(cid)

        #  detect tiny clusters
        n_clients = len(client_ids)
        tiny_clusters = [
            k for k, v in cluster_to_clients.items()
            if len(v) / n_clients < min_cluster_frac
        ]

        if tiny_clusters:
            logging.warning(
                f"Detected very small clusters: "
                f"{ {k: len(cluster_to_clients[k]) for k in tiny_clusters} }"
            )

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

    def __embed_femnist_client(self, dist, num_labels=62):
        """
        dist: dict {label(str or int): count}
        returns: 1D numpy array (client embedding)
        """

        # build full histogram
        counts = np.zeros(num_labels, dtype=float)
        for k, v in dist.items():
            counts[int(k)] = v

        total = counts.sum()
        if total == 0:
            raise ValueError("Client has no samples")

        p = counts / total

        # semantic mass
        digit_mass = p[0:10].sum()
        lower_mass = p[10:36].sum()
        upper_mass = p[36:62].sum()

        # concentration
        ent = entropy(p + 1e-12)
        top1 = np.max(p)
        top5 = np.sort(p)[-5:].sum()

        # scale
        log_n = np.log1p(total)

        return np.array([
            digit_mass,
            lower_mass,
            upper_mass,
            ent,
            top1,
            top5,
            log_n
        ])

    def _femnist_embeddings(self, distributions):
        client_ids = list(distributions.keys())
        X = np.vstack([
            self.__embed_femnist_client(distributions[cid])
            for cid in client_ids
        ])
        return X, client_ids


    def _aggregate(
        self,
        trained_clients: dict,
        weights: dict
    ) -> torch.nn.Module:
        """
        Weighted FedAvg aggregation.

        trained_clients: dict[client_id -> client]
        weights: dict[client_id -> float]
        """
        device = self.device

        # snapshot global params w^t
        global_state = {
            name: param.detach().clone().to(device)
            for name, param in self.global_model.state_dict().items()
            if torch.is_floating_point(param)
        }

        # initialize accumulators
        delta_sum = {
            name: torch.zeros_like(param, device=device)
            for name, param in global_state.items()
        }
        weight_sum = 0.0

        for cid, client in trained_clients.items():
            w_k = weights.get(cid, 0.0)
            if w_k <= 0:
                continue

            weight_sum += w_k

            local_state = {
                name: p.detach().clone().to(device)
                for name, p in client.get_model().state_dict().items()
                if torch.is_floating_point(p)
            }

            for name in global_state.keys():
                delta_sum[name] += w_k * (global_state[name] - local_state[name])

        if weight_sum == 0.0:
            logging.warning("FedAvg: weight_sum == 0, skipping update")
            return self.global_model

        new_state = self.global_model.state_dict()
        for name, w_t in global_state.items():
            new_state[name] = w_t - delta_sum[name] / weight_sum

        self.global_model.load_state_dict(new_state)
        return self.global_model


    def _cluster_quality_old(self, reports: dict[str, dict], c2g: dict[str, int]) -> dict[int, float]:
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
    
    def _cluster_quality(self, reports: dict[str, float],
                     c2g: dict[str, int],
                     q: float = 1.0,
                     eps: float = 1e-6) -> dict[int, float]:
        """
        Compute cluster quality q_g ∝ (ℓ_g + eps)^q
        where ℓ_g is mean post-training loss of clients in cluster g.

        Returns normalized weights in [0.1, 1.0].
        """

        cluster_losses = defaultdict(list)

        for cid, loss in reports.items():
            if cid not in c2g:
                continue
            gid = c2g[cid]
            cluster_losses[gid].append(float(loss))

        if not cluster_losses:
            return {}

        # ℓ_g
        avg_loss = {g: np.mean(v) for g, v in cluster_losses.items()}

        # (ℓ_g + eps)^q
        raw = np.array([(v + eps) ** q for v in avg_loss.values()], dtype=np.float64)

        # normalize to [0.1, 1.0]
        mn, mx = raw.min(), raw.max()
        norm = (raw - mn) / (mx - mn + 1e-12)
        norm = 0.1 + 0.9 * norm   # scale instead of clip

        return {g: float(norm[i]) for i, g in enumerate(avg_loss.keys())}
    
    def _cluster_loss(self, reports: dict[str, float], c2g: dict[str, int]) -> dict[int, float]:
        cluster_losses = defaultdict(list)
        for cid, loss in reports.items():
            if cid not in c2g:
                continue
            cluster_losses[c2g[cid]].append(float(loss))
        return {g: np.mean(v) for g, v in cluster_losses.items()}
    

    def _ema_cluster_loss(self, cluster_loss: dict[int, float], beta: float = 0.2) -> dict[int, float]:
        self._ema_loss = getattr(self, "_ema_loss", cluster_loss)
        self._ema_loss = {
            g: beta * self._ema_loss.get(g, l) + (1 - beta) * l
            for g, l in cluster_loss.items()
        }
        return self._ema_loss
    
    def _cluster_quality_from_loss(self,
                               ema_loss: dict[int, float],
                               q_exp: float = 1,
                               eps: float = 1e-6,
                               alpha: float = 0.2,
                               R: float = 3.0) -> dict[int, float]:

        #raw = {g: (l + eps) ** q_exp for g, l in ema_loss.items()}
        raw = np.array([(v + eps) ** q_exp for v in ema_loss.values()], dtype=np.float64)

        #vals = np.array(list(raw.values()), dtype=np.float64)
        #vals /= vals.mean()  # normalize around 1

        # cap ratio
        #mx, mn = vals.max(), vals.min()
        #if mx / (mn + 1e-12) > R:
        #    vals = np.clip(vals, mx / R, mx)

        # α-mix with FedAvg (weight=1)
        #vals = (1 - alpha) * 1.0 + alpha * vals

        #return {g: float(vals[i]) for i, g in enumerate(raw.keys())}

        mn, mx = raw.min(), raw.max()
        norm = (raw - mn) / (mx - mn + 1e-12)
        norm = 0.1 + 0.9 * norm   # scale instead of clip

        return {g: float(norm[i]) for i, g in enumerate(ema_loss.keys())}


    def _find_jsd_threshold_old(self, jsd_condensed, method="average", trim_frac=0.95):
        """
        Returns a dendrogram distance threshold based on the largest merge gap.

        Parameters
        ----------
        jsd_condensed : ndarray
            Condensed distance matrix (pdist format) of JSD distances.
        method : str
            Linkage method, default 'average'.
        trim_frac : float
            Fraction of smallest merge distances to keep to avoid root outliers.

        Returns
        -------
        jsd_threshold : float
            Distance at which to cut the dendrogram.
        Z : ndarray
            Linkage matrix for reuse.
        """

        Z = linkage(jsd_condensed, method=method)
        merge_dists = np.sort(Z[:, 2])

        # Trim pathological root merges
        cutoff = int(len(merge_dists) * trim_frac)
        trimmed = merge_dists[:cutoff]

        gaps = np.diff(trimmed)
        k = np.argmax(gaps)

        return trimmed[k], Z

    def _find_jsd_threshold(self, jsd_condensed, method="average",
                            q_low=0.70, q_high=0.95):
        Z = linkage(jsd_condensed, method=method)
        m = np.sort(Z[:, 2])

        n = len(m)
        lo = int(n * q_low)
        hi = int(n * q_high)
        hi = max(hi, lo + 2)  # ensure room for diffs

        window = m[lo:hi]
        gaps = np.diff(window)
        k = int(np.argmax(gaps))

        T = window[k]   # cut at the left value of max gap
        return T, Z

    
    
    def train(self, train_schedule: dict, max_local_round: int, threshold: float, patience: int,
              multithreading: bool = False) -> torch.nn.Module:
        
        logging.info("Starting FedHiKoD training.............")
        logging.info(f"Client dict: {self.client_dict}")

        self._client_distributions = {}
        for client in self.client_dict.values():
            self._client_distributions[client.client_id] = client.get_class_distribution()

        logging.info(f"Client distributions: {self._client_distributions}")
        save_client_distributions(self._client_distributions, f"datasets/femnist/client_distributions_round.json")

        if isinstance(next(iter(self.client_dict.values())).train_dataset, FEMNISTDataset):
            X, client_ids = self._femnist_embeddings(self._client_distributions)
            self._c2g, self._g2c = self.gmm_clustering(client_ids,
                        X,
                        k_min=2,
                        k_max=10,
                        pca_dim=3,
                        random_state=0)
        else:
            client_ids, jsd_condensed = self._embed_for_clustering(self._client_distributions, total_classes=62, alpha=0)
            self._c2g, self._g2c = self.hierarchical_clustering(client_ids, jsd_condensed)

        from collections import Counter
        sizes = Counter(self._c2g.values())
        sizes_list = list(sizes.values())

        num_singletons = sum(s == 1 for s in sizes_list)
        singleton_frac = num_singletons / len(sizes_list)

        logging.info(
            f"Group sizes (min/median/max): "
            f"{np.min(sizes_list)}, {np.median(sizes_list)}, {np.max(sizes_list)} | "
            f"Singletons: {num_singletons} ({singleton_frac:.2%})"
        )

        logging.info(f"Client to group mapping: {self._c2g}")
        logging.info(f"Number of groups formed: {len(self._g2c)}")

        for r in range(1, self.rounds + 1):
            logging.info(f"=== Global Round {r} ===")
            train_clients_ids = train_schedule.get(str(r), [])
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
                if r > 1:
                    cid = client.client_id
                    gid = self._c2g[cid]

                    # Retrieve and send the leader model
                    if gid in inactive_clusters:
                        logging.info(f"Client {cid} in inactive cluster {gid}, skipping leader send.")
                    leader_state = self._leader_models_by_group.get(gid)
                    if leader_state is None:
                        logging.info(f"Not Sending leader model to client {cid} from group {gid}.")
                    if leader_state is not None:
                        payload = {
                            "state_dict": leader_state,
                            "ema_cluster_loss": ema_loss[gid],
                        }
                        client.receive_leader(payload)

                #----------------------------------------------------
                # Train the client with FedHiKoD local updates
                # receive trained model from client           
                # receive num of data points and class distribution 
                # ----------------------------------------------------
                client.train(r, 
                            max_local_round)
                
                num_data_points[client.client_id] = client.get_num_datapoints()
                self._client_distributions[client.client_id] = client.get_class_distribution()

                try:
                    #score = self._collect_self_reports(client)
                    score = client.get_loss_at_global()
                    reports[client.client_id] = score

                except Exception as e:
                    logging.error(f"Client {client.client_id} reporting failed: {e}")
                logging.info(f"\n")

            inactive_clusters = [
                gid for gid, client_ids in self._g2c.items()
                if all(cid not in train_clients for cid in client_ids)
            ]

            print(f"Reports: {reports}")

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
                #group_models = [train_clients[cid].get_model() for cid in valid_clients]
                group_clients = {
                    cid: train_clients[cid]
                    for cid in valid_clients
                }
                group_weights_dict = {
                    cid: num_data_points[cid] / sum(num_data_points[c] for c in valid_clients)
                    for cid in valid_clients
                }
                group_weights =[group_weights_dict[cid] for cid in valid_clients]

                # Build the group leader
                logging.info("Subglobal (group:%s) model aggregation weights: %s", group_id, group_weights)

                valid_clients = {cid: train_clients[cid] 
                 for cid in client_ids 
                 if cid in train_clients}
                
                group_leader = self._aggregate(group_clients, group_weights_dict)

                # Store as CPU state_dict (for later KD)
                self._leader_models_by_group[group_id] = {
                    k: v.detach().to('cpu', copy=False) for k, v in group_leader.state_dict().items()
                }

            cluster_loss = self._cluster_loss(reports, self._c2g)
            ema_loss = self._ema_cluster_loss(cluster_loss, beta=self.beta)
            cluster_q = self._cluster_quality_from_loss(ema_loss, alpha=1.0, q_exp=3, R=3.0)
            logging.info(f"Cluster Quality at round {r}: {cluster_q}")
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
            self.global_model = self._aggregate(train_clients, weights=weights)            

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{r}.pt")

        return self.global_model

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

class FedTiltServer(Server):

    """
    The federated learning server class for FedTilt.
    """

    def __init__(self, *args, fitness_cfg: FitnessCfg = FitnessCfg(), **kwargs):
        super().__init__(*args, **kwargs)
        self.fitness_cfg = fitness_cfg

        self._g_ema = None             
        self._g_beta = 0.9             # momentum factor for g_EMA
    

    def _collect_self_reports(self, client: Client) -> dict[str, dict]:
        cfg = self.fitness_cfg
        recipe_hash = hashlib.sha256(f"metric:loss|clip:{cfg.clip_range}".encode()).hexdigest()

        g_mom = None if self._g_ema is None else self._g_ema.detach().cpu()

        if not hasattr(client, "report_fitness"):
            raise RuntimeError(f"Client {client.client_id} lacks report_fitness()")
        
        # these are hyperparams
        rpt = client.report_fitness(
            recipe_hash=recipe_hash, 
        )

        return rpt
    

    def _fair_hybrid_weights(self, data_weights, client_reports, w_p=1.2, w_f=0.3, gamma=1, lam=None):
        cids = sorted(data_weights.keys())
        # 1) soften size prior: p_i ∝ n_i^gamma (gamma=1 → FedAvg; gamma<1 → more client-fair)
        p_unnorm = {c: float(data_weights[c]) ** gamma for c in cids}
        Zp = sum(p_unnorm.values()) or 1.0
        p = {c: p_unnorm[c] / Zp for c in cids}

        # 2) progress + fairness (robust z-score on post_loss)
        print("client_reports", client_reports)
        losses = [float(client_reports[c]["post_loss"]) for c in cids]
        med = statistics.median(losses)
        mad = statistics.median([abs(x - med) for x in losses]) or 1e-6
        z = {c: (float(client_reports[c]["post_loss"]) - med) / (mad + 1e-6) for c in cids}
        prog = {c: float(client_reports[c]["progress"]) for c in cids}
        #s = {c: w_p * prog[c] + w_f * z[c] for c in cids}
        s = {c: max(min(w_p*prog[c] + w_f*z[c], 4.0), -4.0) for c in cids}

        # 3) pick lambda robustly (std can be deceiving on non-IID)
        if lam is None:
            s_vals = list(s.values())
            s_med = statistics.median(s_vals)
            s_mad = statistics.median([abs(v - s_med) for v in s_vals]) or 1e-6
            lam = 1.4826 * s_mad   # robust std estimate; if tiny, this avoids crazy tilt
            if lam < 1e-3:
                lam = 1.0

        # 4) KL-tilt around softened FedAvg prior
        exps = {c: p[c] * math.exp(s[c] / lam) for c in cids}
        Z = sum(exps.values()) or 1.0
        w = {c: exps[c] / Z for c in cids}
        return cids, w
 
    def _aggregate(self, trained_clients, weights):
        """
        Aggregate the models of the clients using FedSMO aggregation strategy.
        """
        return weighted_avg(self.global_model, trained_clients, weights)

    
    def train(self, train_schedule: dict, max_local_round: int, threshold: float, patience: int,
              multithreading: bool = False) -> torch.nn.Module:
            
        for round in range(1, self.rounds + 1):
            logging.info(f"\n=== Global Round {round} ===")
            print(f"\n=== Global Round {round} ===")
            train_clients_ids = train_schedule.get(str(round), [])
            reports = {}

            train_clients = {cid: self.client_dict[cid] for cid in train_clients_ids}
            self._broadcast(self.global_model, train_clients)

            num_data_points = {}            
            for client in train_clients.values():
                client.train(round, 
                                max_local_round, 
                                lambda_l=0.02)
                num_data_points[client.client_id] = client.get_num_datapoints()
                
                try:
                    score = self._collect_self_reports(client)
                    reports[client.client_id] = score
                except Exception as e:
                    logging.error(f"Client {client.client_id} reporting failed: {e}")

                logging.info(f"\n")
                print(f"\n")

            total_points = sum(num_data_points.values())


            weights = {
                    cid: num_data_points[cid] / total_points
                    for cid in train_clients.keys()
                }
                        
            per_weights = {cid: rpt["weight_report"] for cid,rpt in reports.items()}

            _, hybrid_weights = self._fair_hybrid_weights(weights, per_weights)

            print("weights", weights)

            cid_order = list(train_clients.keys())
            model_list = [train_clients[cid].get_model() for cid in cid_order]
            w_list = [hybrid_weights[cid] for cid in cid_order]
            
            self.global_model = self._aggregate(model_list, w_list)

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{round}.pt")

        return self.global_model
    
class FedHiKoDServerOld(Server):

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
  

    def _global_aggregate(self, trained_clients: dict, q: float = 1.5):
        """
        Loss-powered fairness aggregation (qFedAvg-style).
        w_k ∝ p_k * (F_k)^q
        """

        device = self.device
        client_ids = list(trained_clients.keys())
        K = len(client_ids)

        # -----------------------------------------------------
        # 1. Collect sample counts and losses
        # -----------------------------------------------------
        n_k = torch.tensor(
            [trained_clients[cid].get_num_datapoints() for cid in client_ids],
            dtype=torch.float32, device=device
        )

        p_k = n_k / n_k.sum()

        F_k = torch.tensor(
            [trained_clients[cid].get_loss_at_global() for cid in client_ids],
            dtype=torch.float32, device=device
        )

        # -----------------------------------------------------
        # 2. Fairness-aware weights (stable, nonlinear)
        # -----------------------------------------------------
        # Detach losses so aggregation does not backprop through them
        w_k = p_k * (F_k.detach() + 1e-6).pow(q)

        # Normalize
        w_k = w_k / w_k.sum()

        # -----------------------------------------------------
        # 3. Aggregate client models
        # -----------------------------------------------------
        state_dicts = [
            trained_clients[cid].get_model().state_dict()
            for cid in client_ids
        ]

        global_state = self.global_model.state_dict()

        with torch.no_grad():
            for key in global_state.keys():
                stacked = torch.stack(
                    [state_dicts[i][key].float().to(device) * w_k[i]
                    for i in range(K)],
                    dim=0
                )
                global_state[key].copy_(stacked.sum(dim=0))

        self.global_model.load_state_dict(global_state)
        return self.global_model


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


    def hierarchical_clustering(self, client_ids, jsd_condensed, jsd_threshold):
        from scipy.cluster.hierarchy import linkage, fcluster
        from collections import defaultdict

        Z = linkage(jsd_condensed, method='average')
        clusters = fcluster(Z, t=jsd_threshold, criterion='distance')

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
        from scipy.cluster.hierarchy import linkage

        Z = linkage(jsd_condensed, method='average', optimal_ordering=True)

        best_k, sil_scores = self.evaluate_silhouette_scores(Z, jsd_condensed, k_min=2, k_max=20)

        return best_k, sil_scores


    def evaluate_silhouette_scores(self, Z, jsd_condensed, k_min=2, k_max=20):
        """
        """
        from scipy.cluster.hierarchy import fcluster
        from scipy.spatial.distance import squareform
        from sklearn.metrics import silhouette_score

        # Convert condensed distances to a full matrix for silhouette_score
        jsd_matrix = squareform(jsd_condensed)
        n_samples = jsd_matrix.shape[0]
        print(f"Number of samples for silhouette evaluation: {n_samples}")

        sil_scores = {}

        for k in range(k_min, min(k_max, n_samples - 1) + 1):
            print(f"Evaluating silhouette score for k={k}")
            clusters_k = fcluster(Z, k, criterion='maxclust')

            # Skip if clustering degenerates (e.g., all samples in one cluster)
            if len(np.unique(clusters_k)) < 2:
                continue

            score = silhouette_score(jsd_matrix, clusters_k, metric='precomputed')
            sil_scores[k] = score

        best_k = max(sil_scores, key=sil_scores.get)

        return best_k, sil_scores

    def set_client_regret(self, regret: float | None) -> None:
        """
        Server sets regret R_k = loss_ema_k / median_loss_ema.
        """
        self.client_regret = None if regret is None else float(regret)

    def compute_cluster_support(self,
        cluster_to_clients: dict,
        client_distributions: dict,
        num_classes: int,
        min_frac: float = 0.01,   # 1% of cluster data
         ):
        """
        Returns:
            cluster_support: dict
                cluster_id -> torch.Tensor [num_classes] of {0,1}
        """

        cluster_support = {}

        for cluster_id, client_ids in cluster_to_clients.items():
            class_counts = torch.zeros(num_classes, dtype=torch.float32)

            # Sum class counts over clients in this cluster
            for cid in client_ids:
                dist = client_distributions[cid]  # {class: count}
                for cls, cnt in dist.items():
                    class_counts[cls] += cnt

            total = class_counts.sum()
            if total == 0:
                # Degenerate cluster (should not happen)
                support = torch.zeros(num_classes)
            else:
                class_frac = class_counts / total
                support = (class_frac >= min_frac).float()

            cluster_support[cluster_id] = support

        return cluster_support

    
    def train(self, train_schedule: dict, max_local_round: int, threshold: float, patience: int,
              multithreading: bool = False) -> torch.nn.Module:
        
        logging.info("Starting FedHiKoD training.............")
        logging.info(f"Client dict: {self.client_dict}")

        self._client_distributions = {}
        for client in self.client_dict.values():
            self._client_distributions[client.client_id] = client.get_class_distribution()

        logging.info(f"Client distributions: {self._client_distributions}")

        client_ids, jsd_condensed = self._embed_for_clustering(self._client_distributions, total_classes=10, alpha=1)
        #logging.info(f"Determined number of clusters: {num_clusters}")
        #logging.info(f"Silhouette scores for different cluster counts: {sil_scores}")

        self._c2g, self._g2c = self.hierarchical_clustering(client_ids, jsd_condensed, 0.60)

        print(self._g2c)

        cluster_support = self.compute_cluster_support(
                self._g2c,
                self._client_distributions,
                num_classes=10,
                min_frac=0.02
            )
        
        print("cluster_support", cluster_support)

        for r in range(1, self.rounds + 1):
            logging.info(f"=== Global Round {r} ===")
            train_clients_ids = train_schedule.get(str(r), [])
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
                if r > 1:
                    cid = client.client_id
                    gid = self._c2g[cid]

                    # Retrieve and send the leader model
                    if gid in inactive_clusters:
                        logging.info(f"Client {cid} in inactive cluster {gid}, skipping leader send.")
                    leader_state = self._leader_models_by_group.get(gid)
                    if leader_state is None:
                        logging.info(f"Not Sending leader model to client {cid} from group {gid}.")
                    if leader_state is not None:
                        payload = {
                            "state_dict": leader_state,
                            "class_support": cluster_support[int(gid)]  # tensor [num_classes] of 0/1
                        }
                        client.receive_leader(payload)

                #----------------------------------------------------
                # Train the client with FedHiKoD local updates
                # receive trained model from client           
                # receive num of data points and class distribution 
                # ----------------------------------------------------
                client.train(r, 
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

                valid_clients = {cid: train_clients[cid] 
                 for cid in client_ids 
                 if cid in train_clients}
                
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
                weighted_size[cid] = n #* q

            total_weight = sum(weighted_size.values())
            weights = {cid: weighted_size[cid] / total_weight for cid in weighted_size}
        
            cid_order = list(train_clients.keys())
            model_list = [train_clients[cid].get_model() for cid in cid_order]
            w_list = [weights[cid] for cid in cid_order]

            logging.info(f"Global model aggregation weights: {w_list}")
            self.global_model = self._aggregate(model_list, weights=w_list)

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

            self.save_checkpt(self.global_model, f"{self.checkpoint_path}/checkpoints/ckpt_{r}.pt")

            self._prev_cluster_q = getattr(self, "_prev_cluster_q", cluster_q)
            cluster_q = {
                g: 0.7 * self._prev_cluster_q.get(g, q) + 0.3 * q
                for g, q in cluster_q.items()
            }
            self._prev_cluster_q = cluster_q

        return self.global_model