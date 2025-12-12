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
from datasets.celeba.preprocess import CELEBADataset
from datasets.cifar10.preprocess import CIFARDataset
from sklearn.metrics import f1_score
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

        if isinstance(self.train_dataset, CIFARDataset):
            self.optimizer = torch.optim.SGD(
                    local_model.parameters(),
                    lr=learning_rate,
                    momentum=0.9,
                    weight_decay=weight_decay,
                )
          
        else:
            self.optimizer = torch.optim.SGD(
                self.local_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
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


    def evaluate(self, broadcast_model:bool = False) -> tuple:
        """
        Evaluate the client local model with validation dataset of the client.

        Returns:
        ------------
        loss_avg: float; average loss
        f1_avg: float; average F1 score
        """

        model = self.broadcast_model if broadcast_model else self.local_model
        model.eval()

        batch_loss = []
        all_preds, all_labels = [], []

        with torch.no_grad():
            for _, (x, y) in enumerate(self.valdl):
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                
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
                batch_loss.append(loss.item())
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
            
            loss_avg = sum(batch_loss) / len(batch_loss)
            f1_avg = f1_score(all_labels, all_preds, average='macro') 
        
        return loss_avg, f1_avg
    
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
        
        self.pre_loss_at_global, _  = self.evaluate(broadcast_model=True)
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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._leader_state = None  # holds the current group's leader weights


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
        delta_loss = post_loss - pre_loss  # negative is good

        # 2) Clip + DP noise
        a, b = clip_range
        delta_loss = float(np.clip(delta_loss, a, b))

        # 4) Penalty for tiny validation sets
        n_eval = len(self.valdl.dataset)
        progress_term = -delta_loss

        logging.info(f"Progress term: {progress_term}")

        return {
            "client_id": self.client_id,
            "recipe_hash": recipe_hash,
            "n_eval": int(n_eval),
            "score": float(progress_term),
        }


    def receive_leader(self, leader_weights: dict | None) -> None:
        """
        Accept the current group's leader weights (parameter tensors only, CPU).
        """
        self._leader_state = None

        self._leader_state = {
            k: (v if torch.is_tensor(v) else torch.as_tensor(v)).detach().clone().to(self.device)
            for k, v in leader_weights.items()
        }


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
        s_log = student_logits / (student_logits.std(dim=1, keepdim=True) + 1e-6)
        t_log = teacher_logits / (teacher_logits.std(dim=1, keepdim=True) + 1e-6)

        log_p_s = F.log_softmax(s_log / T, dim=1)
        p_t = F.softmax(t_log / T, dim=1)

        kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
        return kd
    

    def train(
        self,
        global_round: int,
        max_local_round: int,
        grad_clip: float = 1.0,
        kd_alpha: float = 0.4,      # KD weight
        kd_T: float = 2.0,          # fixed temperature
        use_kd: bool = True,
    ) -> torch.nn.Module:

        # snapshot the global model for FedHiKoD's delta computation
        self.global_model = copy.deepcopy(self.local_model).to(self.device)
        self.local_model.train()

        # ----------------------------
        # Build teacher models
        # ----------------------------

        if use_kd:
            teacher_leader = self._build_teacher_from_leader()
            if not teacher_leader:
                use_kd = False
                logging.info(f"[{self.client_id}] Failed to build teacher from leader state. KD disabled.")

        # ----------------------------
        # Local training
        # ----------------------------

        #kd_T = self.adaptive_temperature(
        #    round_idx=global_round,
        #    total_rounds=300,  
        #    T_min=1.0,
        #    T_max=8.0,
        #    mode="linear"
        #) if use_kd else kd_T

        for epoch in range(max_local_round):
            batch_loss = []

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
                    kd_term = self._kd_loss(student_logits, leader_logits, temperature=kd_T)
                    logging.debug(f"[{self.client_id}] KD term: {kd_term.item():.6f}")  
                else:
                    kd_term = torch.tensor(0.0, device=self.device)


                # ----------------------------
                # Total loss
                # ----------------------------
                if use_kd and kd_term is not None:
                    loss = (1 - kd_alpha) * hard_loss + kd_alpha * kd_term
                else:
                    loss = hard_loss 
               
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), grad_clip)
                self.optimizer.step()

                batch_loss.append(loss.item())

            avg_loss = float(sum(batch_loss) / max(1, len(batch_loss)))
            logging.info(
                f"Client: {self.client_id:<10} Epoch: {epoch + 1:<2} Average Training Loss: {avg_loss:<10.6f} Global Round: {global_round}"
            )

        self._leader_state = None
        return self.local_model
      
class BoostingClient(Client):
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
    num_classes: int; number of classes in the entire dataset.
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
        num_classes: int = 10,
        eta = 0.01,  # Boosting learning rate
        error_threshold: float = 0.5,  # Error threshold for boosting
    ) -> None:
        
        super().__init__(
            client_id, 
            train_dataset, 
            test_dataset, 
            loss_fn, 
            train_batch_size,
            test_batch_size, 
            learning_rate, 
            weight_decay, 
            local_model
        )

        """
        self.optimizer = torch.optim.AdamW(
                self.local_model.parameters(),
                lr=0.0002,
                weight_decay=1e-6,
            )
        """
        """
        self.optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,  # Momentum is often used in CIFAR-10 training
            )
        """
        self.optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,  # Momentum is used in CIFAR-10 training
            nesterov=True,
            )
        
        """
        self.optimizer = torch.optim.SGD([
                {'params': local_model.conv1.parameters(), 'lr': 0.005},
                {'params': local_model.conv2.parameters(), 'lr': 0.005},
                {'params': local_model.gn1.parameters(), 'lr': 0.005},
                {'params': local_model.gn2.parameters(), 'lr': 0.005},
                {'params': local_model.fc1.parameters(), 'lr': learning_rate},
                {'params': local_model.fc2.parameters(), 'lr': learning_rate},
            ], momentum=0.9, weight_decay=0.001, nesterov=True,)           
        """

        self.loss_fn = copy.deepcopy(loss_fn)
        self.eta = 0.01  # Boosting learning rate
        self.eta = eta
        self.error_threshold = error_threshold
        self.loss_fn.gamma = 0.0
        self.num_classes = train_dataset.num_classes() if num_classes is None else num_classes

    def train(self, global_round, max_local_round,k, threshold=0.01, patience=2,) -> tuple:
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
        alpha: float; alpha weight for the client aggregation in server.
        """

        previous_loss_avg = float('inf')  
        no_improvement_rounds = 0  

        error_rate, alpha = self.get_alpha()
        logging.info(f"Client {self.client_id} error rate Before start training: {error_rate}")
        print(f"Client {self.client_id} alpha for boosting weights: {alpha}")
        logging.info(f"Client {self.client_id} alpha for boosting weights: {alpha}")
        logging.info(f'Clients weights before update: {self.weight}')

        prev_weight = self.weight
        self.weight = self.update_weight(alpha, performance_indicator = (error_rate > self.error_threshold))
        logging.info(f'Clients weights after update: {self.weight}')
        logging.info(f'Clients weights change: {self.weight - prev_weight}')
        
        if (error_rate > self.error_threshold):
            print(f"The client training is boosted by: {self.weight}")
            logging.info(f"The client training is boosted by weight: {self.weight}")
            max_gamma = 3

            new_gamma = min(self.loss_fn.gamma + self.weight, max_gamma)
            self.loss_fn.update_gamma(new_gamma)
            logging.info(f"Client {self.client_id} gamma for training: {self.loss_fn.gamma}")
        else:
            logging.info(f"The client training is not boosted by weight: {self.weight}, because the error rate is less than the threshold.")
            logging.info(f"Client {self.client_id} gamma for training: {self.loss_fn.gamma}")


        print(f"Client: {self.client_id} \tTraining...")
        logging.info(f"Client: {self.client_id} \tTraining...")

        #val_loss, val_f1 = self.evaluate()
        #print(f"Client: {self.client_id} \tInitial Validation Loss: {val_loss:.4f} \tInitial Validation F1: {val_f1:.4f}")
        #logging.info(f"Client: {self.client_id} \tInitial Validation Loss before training: {val_loss:.4f} \tInitial Validation F1: {val_f1:.4f}")

        self.local_model.train()
        for epoch in range(max_local_round):
            batch_loss = []
            for batch_idx, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad() 
                outputs = self.local_model(x)

                if isinstance(self.train_dataset, FEMNISTDataset):
                    y = y.view(-1)
                elif isinstance(self.train_dataset, MNISTDataset):
                    y = torch.argmax(y, dim=1)
                elif isinstance(self.train_dataset, CIFARDataset):
                    y = torch.argmax(y, dim=1)
                else:
                    y = y.view(-1, 1)

                loss = self.loss_fn(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                batch_loss.append(loss.item())

    
            loss_avg = sum(batch_loss) / len(batch_loss)    
            print(f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg} \tGlobal Round: {global_round} \tGamma: {self.loss_fn.gamma}")
            logging.info(f"Client: {self.client_id} \tEpoch: {epoch + 1} \tAverage Training Loss: {loss_avg} \tGlobal Round: {global_round} \tGamma: {self.loss_fn.gamma}")

            # Dynamic loss reduction evaluation.

            
            loss_reduction = previous_loss_avg - loss_avg
            if loss_reduction < threshold:
                no_improvement_rounds += 1
                print(f"Loss reduction below threshold ({loss_reduction:.6f}). No improvement rounds: {no_improvement_rounds}")
                logging.info(f"Loss reduction below threshold ({loss_reduction:.6f}). No improvement rounds: {no_improvement_rounds}")
            else:
                no_improvement_rounds = 0

            if no_improvement_rounds >= patience:
                print(f"Stopping early at local epoch {epoch + 1} due to no significant improvement.")
                logging.info(f"Stopping early at local epoch {epoch + 1} due to no significant improvement.")
                break

            previous_loss_avg = loss_avg
            

        error_rate, alpha = self.get_alpha()
        alpha = np.clip(alpha, -3, 3)
        logging.info(f"Client {self.client_id} error rate After training: {error_rate}")
        #val_loss, val_f1 = self.evaluate()
        #print(f"Client: {self.client_id} \t Updated Validation Loss: {val_loss:.4f} \tUpdated Validation F1: {val_f1:.4f}")
        #logging.info(f"Client: {self.client_id} \tUpdated Loss: {val_loss:.4f} \tUpdated Validation F1: {val_f1:.4f}")

        return self.local_model, alpha, self.loss_fn.gamma

    def __get_error_rate(self, use_macro_f1: bool = False) -> float:
        """
        Evaluate the model on the validation dataset and return the error rate.

        Parameters:
        ----------------
        use_macro_f1: bool
            If True, use 1 - macro F1 score as the error.
            If False, use standard classification error rate.

        Returns:
        ----------------
        error_rate: float
            Error rate (proportion of incorrect predictions or 1 - F1)
        """
        self.local_model.eval()
        device = self.device

        incorrect_preds = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for _, (x, y) in enumerate(self.valdl):
                x, y = x.to(device), y.to(device)
                outputs = self.local_model(x)

                # Convert one-hot to class index if needed
                if y.dim() > 1:
                    y = torch.argmax(y, dim=1)
                else:
                    y = y.view(-1)

                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                incorrect_preds += (preds != y).sum().item()
                total_samples += y.size(0)

        if total_samples == 0:
            return 1.0  # Assume total error if no samples

        if use_macro_f1:
            f1 = f1_score(all_labels, all_preds, average='macro')
            error_rate = 1.0 - f1
        else:
            error_rate = incorrect_preds / total_samples

        return error_rate

    
    def get_alpha(self) -> float:
        """
        Calculate adjusted weight (alpha) for the client in the FL setting,
        giving higher weights to clients with lower errors with the global model.
        """
        error_rate = self.__get_error_rate(use_macro_f1=True)  # or False if you prefer
        eps = 1e-6
        error_rate = min(max(error_rate, eps), 1 - eps)
        alpha = np.log((1 - error_rate) / error_rate) + np.log(self.num_classes - 1)
        return error_rate, alpha
    
    def set_weight(self, weight:float) -> None:
        """
        Set the weight for the client.

        Parameters:
        ------------
        weight: float; weight
        """
        self.weight = weight
        return self.weight

    def update_weight(self, alpha, performance_indicator=1) -> None:
        """
        Update the weights for the client.

        Parameters:
        ------------
        weight: float; weight
        """
        alpha = np.clip(alpha, -5, 5)  #Ensure alpha is not too small
        self.weight = self.weight * math.exp(float(self.eta) *  -float(alpha) * int(performance_indicator))
        return self.weight

class FedTiltClient(Client):
    """
    FedTilt-specific client that extends the base Client class
    with privacy-preserving fitness reporting and peer-committee auditing.
    """

    
    def report_fitness(self, recipe_hash):

            pre_loss, _  = self.evaluate(broadcast_model=True)   # evaluate before local train in this round
            post_loss, _ = self.evaluate(broadcast_model=False)  # evaluate after local train

            progress_raw = (pre_loss - post_loss) / max(abs(pre_loss), 1e-6)
            progress = float(np.clip(progress_raw, -1.0, 1.0))

            return {
                "client_id": self.client_id,
                "recipe_hash": recipe_hash,
                "weight_report": {"client_id": self.client_id, 
                                  "pre_loss": float(pre_loss), 
                                  "post_loss": float(post_loss), 
                                  "progress": float(progress)}
            }

    def train(
        self,
        global_round: int,
        max_local_round: int,
        lambda_l: float = 0.0,                  
        grad_clip: float = 1.0,
    ) -> torch.nn.Module:

        self.global_model = copy.deepcopy(self.local_model).to(self.device)

        self.local_model.train()
        for epoch in range(max_local_round):
            batch_loss = []

            for batch_idx, (x, y) in enumerate(self.traindl):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.local_model(x)

                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                    if y.ndim > 1:
                        if y.size(-1) == 1:
                            y = y.squeeze(-1)          
                        else:
                            y = y.argmax(dim=-1)       
                    y = y.long()
                else:
                    pass

                local_loss = self.loss_fn(outputs, y)

                loss = local_loss 

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=grad_clip)
                self.optimizer.step()

                batch_loss.append(loss.detach().item())

            loss_avg = float(sum(batch_loss) / max(len(batch_loss), 1))
            logging.info(
                f"Client: {self.client_id:<10} Epoch: {epoch + 1:<2} "
                f"FedTilt Train Loss: {loss_avg:<10.6f} Global Round: {global_round}"
            )
            print(f"Client: {self.client_id:<10} Epoch: {epoch + 1:<2} "
                f"FedTilt Train Loss (data): {loss_avg:<10.6f} Global Round: {global_round}")

        return self.local_model
    
class DittoClient(Client):
    """
    DittoClient class extends the base Client for Ditto personalized FL.
    Maintains a separate `personal_model` and trains it with a
    regularization penalty towards the global (local_model) parameters.

    Parameters:
    ------------
    client_id: str; client id.
    train_dataset: torch.utils.data.Dataset object; training dataset.
    test_dataset: torch.utils.data.Dataset object; validation dataset.
    loss_fn: torch.nn.Module object; loss function.
    train_batch_size: int; train batch size.
    test_batch_size: int; test batch size.
    learning_rate: float; learning rate for clients.
    weight_decay: float; weight decay for optimizer.
    local_model: torch.nn.Module object; model.
    personal_learning_rate: float; personal model learning rate.
    ditto_lambda: float; regularization strength for Ditto.
    personalized: bool; whether to use personalized learning rates.
    checkpt_path: str; path to save personal model checkpoints.
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: object,
        test_dataset: object,
        loss_fn: torch.nn.Module,
        train_batch_size: int,
        test_batch_size: int,
        learning_rate: float,                   # Global model learning rate
        weight_decay: float,
        local_model: object = None,           
        personal_learning_rate: float = 0.01,   # Personal model learning rate
        ditto_lambda: float = 0.1,
        personalized: bool = True,
        checkpt_path: str = None,                
    ) -> None:
        
        super().__init__(
            client_id,
            train_dataset,
            test_dataset,
            loss_fn,
            train_batch_size,
            test_batch_size,
            learning_rate,
            weight_decay,
            local_model,
        )

        # Ditto-specific parameters
        self.ditto_lambda = ditto_lambda
        self.personalized = personalized
        self.checkpt_path = checkpt_path
        self.personal_lr = personal_learning_rate if self.personalized else learning_rate

        # Take Deep-copy the global/local model architecture as the personal model for Ditto
        self.personal_model = copy.deepcopy(self.local_model).to(self.device)

        self.personal_optimizer = torch.optim.SGD(
                self.personal_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        """
        # Personal optimizer (different LR is often used)
        conv_params = []
        fc_params = []

        for name, param in self.personal_model.named_parameters():
            if "conv" in name:
                conv_params.append(param)
            else:
                fc_params.append(param)

        self.personal_optimizer = torch.optim.SGD([
            {"params": conv_params, "lr": self.personal_lr * 2}, 
            {"params": fc_params, "lr": self.personal_lr}  
        ], weight_decay=0.0001, momentum=0.0)

        self.personal_optimizer = torch.optim.Adam(
            self.personal_model.parameters(),
            lr=self.personal_lr, 
            weight_decay=0.0001)
        """


    def train(
        self,
        global_round: int,
        max_local_round: int,
        threshold: float,
        patience: int
    ):
        """
        Overrides the parent train method to:
          1) Train the global model using fedAvg's standard procedure.
          2) Train the personal_model with an L2 penalty to keep it close to global model.
        
        Returns:
            global_model: The updated global model and personal model.
        """
        # Standard FL local training on local_model (global model copy).
        super().train(global_round, max_local_round, threshold, patience)

        # Personal model training.
        self._train_personal_model(global_round, max_local_round)

        personal_ckpt_path = f"{self.checkpt_path}/personal_ckpts/{self.client_id}/round_{global_round}.pt"
        self._save_checkpt(self.personal_model.eval(), personal_ckpt_path)

        return self.local_model

    def _train_personal_model(self, global_round: int, max_local_round: int):
        """
        Trains the personal_model for Ditto. Uses an L2 penalty (with strength ditto_lambda)
        against the current local_model’s parameters (which serve as the “anchor”).
        """
        print(f"Client: {self.client_id} \tTraining personal model for Ditto...")
        logging.info(f"Client: {self.client_id} \tTraining personal model for Ditto...")

        self.personal_model.train()

        #eval_loss, eval_f1 = self.evaluate_personal_model()
        #print(f"Client (personal): {self.client_id} \tInitial Validation Loss: {eval_loss:.6f} \tValidation F1: {eval_f1:.6f}")
        #logging.info(f"Client (personal): {self.client_id} \tInitial Validation Loss: {eval_loss:.6f} \tValidation F1: {eval_f1:.6f}")

        for epoch in range(max_local_round*1):
            batch_losses = []

            anchor_params = [p.detach() for p in self.local_model.parameters()]

            for x, y in self.traindl:
                x, y = x.to(self.device), y.to(self.device)
                predictions = self.personal_model(x)

                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                    if y.ndim > 1:
                        if y.size(-1) == 1:
                            y = y.squeeze(-1)          # (N,1) -> (N,)
                        else:
                            y = y.argmax(dim=-1)       # one-hot -> indices
                    y = y.long()

                loss = self.loss_fn(predictions, y)

                # L2 distance between personal_model & local_model
                ditto_penalty = 0.0
                for personal_param, anchor_param in zip(self.personal_model.parameters(), anchor_params):
                    ditto_penalty += torch.nn.functional.mse_loss(personal_param, anchor_param, reduction="sum")

                ditto_penalty = self.ditto_lambda * 0.5 * ditto_penalty  # 0.5 is optional scaling
                total_loss = loss + ditto_penalty
                self.personal_optimizer.zero_grad()
                total_loss.backward()

                self.personal_optimizer.step()
                batch_losses.append(total_loss.item())

            avg_loss = sum(batch_losses) / len(batch_losses)
            print(f"Client: {self.client_id} \tEpoch (personal): {epoch+1} \tAvg Loss: {avg_loss:.6f} \tGlobal Round: {global_round}")
            logging.info(f"Client: {self.client_id} \tEpoch (personal): {epoch+1} \tAvg Loss: {avg_loss:.6f} \tGlobal Round: {global_round}")
        
        #eval_loss, eval_f1 = self.evaluate_personal_model()
        #print(f"Client: {self.client_id} \tFinal Validation Loss: {eval_loss:.6f} \tValidation F1: {eval_f1:.6f}")
        #logging.info(f"Client: {self.client_id} \tFinal Validation Loss: {eval_loss:.6f} \tValidation F1: {eval_f1:.6f}")
    
    def evaluate_personal_model(self) -> tuple:
        """
        Evaluate the personal model with the validation dataset.
        """

        batch_loss = []
        all_preds = []
        all_labels = []

        for _, (x, y) in enumerate(self.valdl):
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.personal_model(x)

            if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                    if y.ndim > 1:
                        if y.size(-1) == 1:
                            y = y.squeeze(-1)          # (N,1) -> (N,)
                        else:
                            y = y.argmax(dim=-1)       # one-hot -> indices
                    y = y.long()

            loss = self.loss_fn(outputs, y)
            batch_loss.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        loss_avg = sum(batch_loss) / len(batch_loss)
        f1_avg = f1_score(all_labels, all_preds, average='macro') 
        
        return loss_avg, f1_avg
    
    def _save_checkpt(self, checkpoint: torch.nn.Module, ckptpath: str) -> None:
        """
        Saving the checkpoints.

        Parameters:
        ----------------
        checkpoint: Model at a specific checkpoint.
        ckptpath: str; Path to save the checkpoint. Default is None.
        """
        checkpoint.eval()
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