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

"""
def weighted_avg(global_model: torch.nn.Module,
                 local_models: List[torch.nn.Module],
                 weights: List[float]) -> torch.nn.Module:

    # Normalize weights into a tensor
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()

    # Move work to CPU where memory is plentiful
    # Also stops MPS from hoarding model copies
    global_model_cpu = global_model.to("cpu")
    local_states = [m.state_dict() for m in local_models]

    with torch.no_grad():
        new_state = {}
        keys = global_model_cpu.state_dict().keys()

        for k in keys:
            agg = None
            for wi, state in zip(w, local_states):
                p = state[k].detach().to("cpu")   # important: avoid MPS
                if agg is None:
                    agg = wi * p
                else:
                    agg += wi * p
            new_state[k] = agg

        global_model_cpu.load_state_dict(new_state)

    device = next(global_model.parameters()).device
    return global_model_cpu.to(device)
"""

"""
def weighted_avg(global_model, local_models, weights):
    state_dicts = [m.state_dict() for m in local_models]
    new_state = {}

    with torch.no_grad():
        for key in global_model.state_dict().keys():
            # Start accumulation with first weighted tensor
            acc = state_dicts[0][key] * weights[0]

            # Add all remaining weighted tensors
            for i in range(1, len(state_dicts)):
                acc += state_dicts[i][key] * weights[i]

            new_state[key] = acc

    # Update the global model IN PLACE
    global_model.load_state_dict(new_state)
    return global_model
"""
