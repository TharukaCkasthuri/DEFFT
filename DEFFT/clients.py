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
import copy
import torch
import math
import logging
import copy
import numpy as np

from torch.utils.data import DataLoader
from utils import get_device
from datasets.mnist.preprocess import MNISTDataset
from datasets.femnist.preprocess import FEMNISTDataset
from datasets.cifar10.preprocess import CIFARDataset
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F

from typing import Dict

class Client:
    """
    Client class for federated learning.

    Parameters:
    ------------
    client_id: str; client id.
    train_dataset: torch.utils.data.Dataset object; training dataset.
    test_dataset: torch.utils.data.Dataset object; validation dataset.
    loss_fn: torch.nn.Module object; loss function.
    train_batch_size: int; train batch size.
    learning_rate: float; learning rate for clients.
    weight_decay: float; weight decay for optimizer.
    local_model: torch.nn.Module object; model.
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: object,
        test_dataset: object,
        loss_fn: torch.nn.Module,
        train_batch_size: int,
        test_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        local_model: object = None,
    ) -> None:

        self.client_id: str = client_id
        self.loss_fn = copy.deepcopy(loss_fn)
        self.batch_size = train_batch_size
        self.device = get_device()



        if local_model is None:
            raise ValueError("local_model must be provided")
        self.local_model = local_model.to(self.device)

        # A non-trainable clone to hold the last broadcasted global model
        self.broadcast_model = copy.deepcopy(local_model).to(self.device)

        for p in self.broadcast_model.parameters():                        
            p.requires_grad_(False)
        self.broadcast_model.eval()

        self.train_dataset = train_dataset
        self.datapoints = len(train_dataset)
        

        self.traindl = DataLoader(
            train_dataset, train_batch_size, shuffle=True, drop_last=False
        )
        self.valdl = DataLoader(test_dataset, test_batch_size, shuffle=False, drop_last=False)
          
        self.optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )

    def get_num_datapoints(self) -> int:
        """
        Get the number of samples in the training dataset.
        """
        return self.datapoints
    

    def receive_global(self, model_weights: dict) -> None:                  
        """
        Accept the *broadcasted/global* weights from the server.
        Stores them into `broadcast_model` (for evaluation) 
        and also syncs `local_model` to start local training from these weights.
        """
        try:
            self.broadcast_model.load_state_dict(model_weights, strict=True)
        except Exception as e:
            logging.error(f"Client {self.client_id} broadcast load error: {e}. Keeping old weights.")
            return
        # sync local from broadcast
        local_state = {k: v.clone().detach() for k, v in model_weights.items()}
        self.local_model.load_state_dict(local_state, strict=True)


    def set_model(self, model_weights:dict) -> None:
        """
        Set the model for the client.

        Parameters:
        ------------
        model_weights: dict; state dictionary of model weights
        """
        self.receive_global(model_weights)

    def get_model(self) -> object:
        """
        Get the model of the client.
        """
        return self.local_model
    
    def _log_msg(self, msg, log_queue=None):
        if log_queue:
            log_queue.put(msg)
        else:
            print(msg)

    def train(self, global_round: int, max_local_round: int, threshold: float, patience: int):
        """
        Training the model, using the fedaboost-optima strategy.

        Parameters:
        ------------
        global_round: int; global round number.
        max_local_round: int; maximum number of local rounds, in case loss reduction threshold is not met.
        threshold: float; threshold for loss reduction.
        patience: int; number of patience rounds to wait for loss reduction.

        Returns:
        ------------
        model: torch.nn.Module object; trained model.
        """

        previous_loss_avg = float('inf')  
        no_improvement_rounds = 0 

        self.local_model.train()
        for epoch in range(max_local_round):
            batch_loss = []

            for batch_idx, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.local_model(x)
                
                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                    if y.ndim > 1:
                        if y.size(-1) == 1:
                            y = y.squeeze(-1)          # (N,1) -> (N,)
                        else:
                            y = y.argmax(dim=-1)       # one-hot -> indices
                    y = y.long()
                else:
                    pass

                loss = self.loss_fn(outputs, y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_loss.append(loss.item())

            loss_avg = sum(batch_loss) / len(batch_loss)

            # Thread-safe logging here
            logging.info(
                f"Client: {self.client_id:<10} Epoch: {epoch + 1:<2} Average Training Loss: {loss_avg:<10.6f} Global Round: {global_round}"
            )
            #print(f"Client: {self.client_id:<10} Epoch: {epoch + 1:<2} Average Training Loss: {loss_avg:<10.6f} Global Round: {global_round}")

            previous_loss_avg = loss_avg

        return self.local_model


    def evaluate(self, broadcast_model: bool = False, data_split: str = "train") -> tuple:
        """
        Evaluate the client local model.

        Returns:
        --------
        loss_avg : float
        acc      : float
        """

        model = self.broadcast_model if broadcast_model else self.local_model
        model.eval()

        batch_loss = []
        all_preds = []
        all_labels = []

        correct = 0
        total = 0

        if data_split == "train":
            data = self.traindl
        elif data_split == "val":
            data = self.valdl
        else:
            raise ValueError(f"Unknown data_split: {data_split}")

        with torch.no_grad():
            for x, y in data:
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = model(x)

                # Ensure class indices for CrossEntropyLoss
                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                    if y.ndim > 1:
                        if y.size(-1) == 1:
                            y = y.squeeze(-1)
                        else:
                            y = y.argmax(dim=-1)
                    y = y.long()

                loss = self.loss_fn(outputs, y)
                batch_loss.append(loss.item())

                preds = torch.argmax(outputs, dim=1)

                correct += (preds == y).sum().item()
                total += y.numel()

                all_preds.append(preds.detach().cpu())
                all_labels.append(y.detach().cpu())

        # ---- Final aggregation (bulletproof) ----
        all_preds = torch.cat(all_preds).numpy().astype(np.int64)
        all_labels = torch.cat(all_labels).numpy().astype(np.int64)

        loss_avg = sum(batch_loss) / len(batch_loss)
        #acc = accuracy_score(all_labels, all_preds)
        acc = correct / total if total > 0 else 0.0

        return loss_avg, acc
    
class FedProxClient(Client):

    def _proximal_term(self, model: torch.nn.Module) -> torch.Tensor:
        proximal_term = torch.zeros((), device=self.device)
        ref = dict(model.named_parameters())  # assume model is broadcast_model (frozen)

        for name, p_local in self.local_model.named_parameters():
            p_ref = ref.get(name, None)
            if p_ref is None:
                continue
            p_ref = p_ref.detach()
            if p_ref.device != p_local.device:
                p_ref = p_ref.to(p_local.device)
            proximal_term += (p_local - p_ref).pow(2).sum()
        return proximal_term
    

    def train(
        self,
        global_round: int,
        max_local_round: int,
        mu: float = 0.0,
        threshold: float = 0.0, 
        patience: int = 1,               
    ) -> torch.nn.Module:
    
        self.local_model.train()

        same_ref = all(
        p1.data_ptr() == p2.data_ptr()
        for (_, p1), (_, p2) in zip(
            self.local_model.named_parameters(), 
            self.broadcast_model.named_parameters()
        )
        )
        print(f"[DEBUG] {self.client_id}: local_model and broadcast_model share memory? {same_ref}")
        # 
        for epoch in range(max_local_round):
            batch_loss = []

            for batch_idx, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.local_model(x)

                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                    if y.ndim > 1:
                        if y.size(-1) == 1:
                            y = y.squeeze(-1)          # (N,1) -> (N,)
                        else:
                            y = y.argmax(dim=-1)       # one-hot -> indices
                    y = y.long()
                else:
                    pass

                local_loss = self.loss_fn(outputs, y)
                prox = self._proximal_term(self.broadcast_model) if mu > 0.0 else 0.0
                loss = local_loss + 0.5 * mu * prox

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                batch_loss.append(loss.detach().item())

            loss_avg = float(sum(batch_loss) / max(len(batch_loss), 1))
            logging.info(
                f"Client: {self.client_id:<10} Epoch: {epoch + 1:<2} "
                f"FedProx Train Loss (data+prox): {loss_avg:<10.6f} Global Round: {global_round}"
            )

        return self.local_model

class QFFedAvgClient(Client):
    """
    QFFedAvg-specific client that extends the base Client class
    with privacy-preserving fitness reporting and peer-committee auditing.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._leader_state = None  # holds the current group's leader weights

    def get_loss_at_global(self):
        return self.pre_loss_at_global

    def train(self, global_round: int, max_local_round: int, threshold: float, patience: int):
        """
        Training the model, using the fedaboost-optima strategy.

        Parameters:
        ------------
        global_round: int; global round number.
        max_local_round: int; maximum number of local rounds, in case loss reduction threshold is not met.
        threshold: float; threshold for loss reduction.
        patience: int; number of patience rounds to wait for loss reduction.

        Returns:
        ------------
        model: torch.nn.Module object; trained model.
        """
        
        self.pre_loss_at_global, _  = self.evaluate(broadcast_model=True, data_split="val")
        self.local_model.train()

        for epoch in range(max_local_round):
            batch_loss = []

            for batch_idx, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.local_model(x)
                
                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                    if y.ndim > 1:
                        if y.size(-1) == 1:
                            y = y.squeeze(-1)          # (N,1) -> (N,)
                        else:
                            y = y.argmax(dim=-1)       # one-hot -> indices
                    y = y.long()

                loss = self.loss_fn(outputs, y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_loss.append(loss.item())

            loss_avg = sum(batch_loss) / len(batch_loss)

            # Thread-safe logging here
            logging.info(
                f"Client: {self.client_id:<10} Epoch: {epoch + 1:<2} Average Training Loss: {loss_avg:<10.6f} Global Round: {global_round}"
            )
            #print(f"Client: {self.client_id:<10} Epoch: {epoch + 1:<2} Average Training Loss: {loss_avg:<10.6f} Global Round: {global_round}")

            local_state = {k: v.clone().detach() for k, v in self.local_model.state_dict().items()}
            global_state = {k: v.clone().detach() for k, v in self.broadcast_model.state_dict().items()}

        return self.local_model

class FedHiKoDClient(Client):
    """
    FedHiKoD-specific client that extends the base Client class
    with privacy-preserving fitness reporting and peer-committee auditing.
    """

    def __init__(self, *args,
                 kd_alpha=0.5, kd_T=4.0, use_kd=True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kd_alpha = kd_alpha
        self.kd_T = kd_T
        self.use_kd = use_kd
        self._leader_state = None  # holds the current group's leader weights
        self.ema_cluster_loss = None  # holds the EMA of cluster loss

    def get_loss_at_global(self):
        return self.post_loss_at_global
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get the class distribution in the training dataset.
        Assumes classification with integer class labels.
        """
        labels = [labels for _, labels in self.train_dataset]

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        else:
            labels = np.array(labels)

        unique, counts = np.unique(labels, return_counts=True)
        class_counts = {int(k): int(v) for k, v in zip(unique, counts)}
        return class_counts


    def report_fitness(self, recipe_hash, clip_range=( -0.9, 0.9)) -> Dict[str, object]:
        """
        """
        pre_loss, _  = self.evaluate(broadcast_model=True)   # evaluate before local train in this round
        post_loss, _ = self.evaluate(broadcast_model=False)  # evaluate after local train
        delta_loss =  pre_loss - post_loss  # positive is good
        logging.info(f"Client {self.client_id} pre-loss: {pre_loss}, post-loss: {post_loss}, delta-loss: {delta_loss}")

        # 2) Clip + DP noise
        #a, b = clip_range
        #delta_loss = float(np.clip(delta_loss, a, b))

        # 4) Penalty for tiny validation sets
        n_eval = len(self.traindl.dataset)
        progress_term = delta_loss/(pre_loss + 1e-6)
        progress_term = np.clip(progress_term, 0.0, 0.5)

        logging.info(f"Progress term: {progress_term}")

        return {
            "client_id": self.client_id,
            "recipe_hash": recipe_hash,
            "n_eval": int(n_eval),
            "score": float(progress_term),
        }


    def receive_leader_old(self, leader_weights: dict | None) -> None:
        """
        Accept the current group's leader weights (parameter tensors only, CPU).
        """
        self._leader_state = None

        self._leader_state = {
            k: (v if torch.is_tensor(v) else torch.as_tensor(v)).detach().clone().to(self.device)
            for k, v in leader_weights.items()
        }


    def receive_leader(self, leader_payload: dict | None) -> None:
        """
        Accept the current group's leader payload.
        Supports:
        - state_dict only (backward compatible)
        - {"state_dict": ..., "class_support": ...}
        """
        self._leader_state = None

        if leader_payload is None:
            return

        # New format
        if "state_dict" in leader_payload:
            leader_weights = leader_payload["state_dict"]
        else:
            # Backward compatible: assume payload is a raw state_dict
            leader_weights = leader_payload

        # Store weights
        self._leader_state = {
            k: (v if torch.is_tensor(v) else torch.as_tensor(v)).detach().clone().to(self.device)
            for k, v in leader_weights.items()
        }

        self.ema_cluster_loss = leader_payload.get("ema_cluster_loss", None)


    def _build_teacher_from_leader(self) -> torch.nn.Module | None:
        """
        Build a frozen teacher model from self._leader_state.
        Returns None if no leader state is available.
        """
        leader_state = self._leader_state
        if leader_state is None:
            return None

        # Create a copy of the same architecture
        teacher = copy.deepcopy(self.local_model).to(self.device)

        # The leader weights you stored were only parameter tensors (no buffers).
        # strict=False lets us load what's available and ignore missing buffers (e.g., BN running stats).
        try:
            teacher.load_state_dict(leader_state, strict=False)
        except Exception as e:
            logging.error(f"[{self.client_id}] Failed to load leader state into teacher: {e}")
            return None

        # Freeze teacher
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher
        

    def adaptive_temperature(self, round_idx: int,
                         total_rounds: int,
                         T_min: float = 1.0,
                         T_max: float = 8.0,
                         mode: str = "linear") -> float:
        ratio = round_idx / max(total_rounds, 1)
        if mode == "linear":
            return T_max - (T_max - T_min) * ratio
        elif mode == "exp":
            return T_min + (T_max - T_min) * math.exp(-3 * ratio)
        else:
            return T_max
        

    @staticmethod
    def _kd_loss(student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                temperature: float) -> torch.Tensor:
        T = temperature

        # normalize logits to prevent scale collapse
        s_log = student_logits #/ (student_logits.std(dim=1, keepdim=True) + 1e-6)
        t_log = teacher_logits #/ (teacher_logits.std(dim=1, keepdim=True) + 1e-6)

        log_p_s = F.log_softmax(s_log / T, dim=1)
        p_t = F.softmax(t_log / T, dim=1)

        kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
        return kd
    

    def train(
        self,
        global_round: int,
        max_local_round: int,
        grad_clip: float = 1.0, 
        use_kd: bool = True,
    ) -> torch.nn.Module:

        # snapshot the global model for FedHiKoD's delta computation
        self.global_model = copy.deepcopy(self.local_model).to(self.device)
        self.local_model.train()

        # ----------------------------
        # Build teacher models
        # ----------------------------
        # compute pre-loss on global model
        self.preloss, _ = self.evaluate(broadcast_model=True, data_split="val")

        if self.use_kd and self.ema_cluster_loss is not None:
            delta = (self.preloss - self.ema_cluster_loss) / (self.ema_cluster_loss + 1e-8)

            progress = global_round / 150

            alpha_base = 0.5
            slope = 0.3
            alpha_min = 0.3
            alpha_max = 0.75 #- 0.25 * progress   # decay KD late

            if delta <= 0:
                # client is at or better than its cluster → do NOT regularize it
                self.kd_alpha = 0.3
            else:
                # struggling client → strong, bounded KD
                self.kd_alpha= float(
                    np.clip(alpha_base + slope * delta, alpha_min, alpha_max)
                )
        else:
            self.kd_alpha = self.kd_alpha

        if use_kd:
            teacher_leader = self._build_teacher_from_leader()
            if not teacher_leader:
                use_kd = False
                logging.info(f"[{self.client_id}] Failed to build teacher from leader state. KD disabled.")

        # ----------------------------
        # Local training
        # ----------------------------

        for epoch in range(max_local_round):
            batch_loss = []
            batch_total_loss = []

            for batch_idx, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)

                # forward pass (student)
                student_logits = self.local_model(x)

                # Supervised loss
                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                    if y.ndim > 1:
                        if y.size(-1) == 1:
                            y = y.squeeze(-1)
                        else:
                            y = y.argmax(dim=-1)
                    y = y.long()
                hard_loss = self.loss_fn(student_logits, y)
                # ----------------------------
                # Knowledge Distillation Loss
                # ----------------------------
                kd_term = torch.zeros((), device=self.device)
                if use_kd and teacher_leader is not None:
                    with torch.no_grad():
                        leader_logits = teacher_leader(x)
                    kd_term = self._kd_loss(student_logits, leader_logits, temperature=self.kd_T)
                    logging.debug(f"[{self.client_id}] KD term: {kd_term.item():.6f}")  
                else:
                    kd_term = torch.tensor(0.0, device=self.device)


                # ----------------------------
                # Total loss
                # ----------------------------
                if use_kd and kd_term is not None:
                    loss = (1 - self.kd_alpha) * hard_loss + self.kd_alpha * kd_term
                else:
                    loss = hard_loss 
                    
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), grad_clip)
                self.optimizer.step()

                batch_loss.append(hard_loss.item())
                batch_total_loss.append(loss.item())

            avg_loss = float(sum(batch_loss) / max(1, len(batch_loss)))
            avg_loss_total = float(sum(batch_total_loss) / max(1, len(batch_total_loss)))
            logging.info(
                f"Client: {self.client_id:<10} Epoch: {epoch + 1:<2} Average Training Loss: {avg_loss:<10.6f} "
                f"Average Total Loss (with KD): {avg_loss_total:<10.6f} "
                f"Global Round: {global_round}"
            )

        self._leader_state = None
        self.post_loss_at_global, _  = self.evaluate(broadcast_model=False, data_split="val")

        return self.local_model
    