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

import torch
from typing import List, Dict, Tuple
import torch.nn.functional as F
import math

def fedAvg(global_model: torch.nn.Module, local_models: List[torch.nn.Module]) -> torch.nn.Module:
    """
    Federated averaging algorithm. Returns the updated global model.

    Parameters:
    ------------
    global_model: torch.nn.Module object; Global model.
    local_models: list; List of local models.

    Returns:
    ------------
    global_model: torch.nn.Module object; Updated global model.
    """
    # update global model parameters here
    state_dicts = [model.state_dict() for model in local_models]
    for key in global_model.track_layers.keys():
        global_model.track_layers[key].weight.data = torch.stack(
            [item[str(key) + ".weight"] for item in state_dicts]
        ).mean(dim=0)
        global_model.track_layers[key].bias.data = torch.stack(
            [item[str(key) + ".bias"] for item in state_dicts]
        ).mean(dim=0)
        # info here - https://discuss.pytorch.org/t/how-to-change-weights-and-bias-nn-module-layers/93065/2
    return global_model


def weighted_avg(global_model: torch.nn.Module, local_models: List[torch.nn.Module], weights: List[float]) -> torch.nn.Module:
    """
    Average model parameters using weighted averaging.
    
    Parameters:
    ------------
    global_model: torch.nn.Module object
        Global model.
    local_models: list
        List of local models.
    weights: list
        List of weights for each local model.
    
    Returns:
    ------------
    global_model: torch.nn.Module object
        Updated global model.
    """
    #w = torch.tensor(weights, dtype=torch.float32)
    #weights = F.softmax(w, dim=0).tolist()

    state_dicts = [model.state_dict() for model in local_models]

    with torch.no_grad():  
        for key in global_model.state_dict().keys():
            stacked_params = torch.stack(
                [state_dict[key] * weights[i] for i, state_dict in enumerate(state_dicts)], dim=0
            )
            global_model.state_dict()[key].copy_(stacked_params.sum(dim=0))  

    return global_model