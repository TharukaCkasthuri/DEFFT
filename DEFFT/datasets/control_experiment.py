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
import json
import numpy as np
from set_training_clients import sample_clients, get_client_ids, parse_arguments, load_config

def main():
    args = parse_arguments()
    folder_path = args.dir
    config = load_config()

    client_ids = get_client_ids(f"{folder_path}/trainpt")
    training_sample = sample_clients(client_ids, 100, 100)

    if isinstance(training_sample, np.ndarray):
        training_sample = training_sample.tolist()
    training_samples = {i: training_sample for i in range(101)}

    output_file = f"{folder_path}/training_samples_ctrl_experiment_100.json"

    with open(output_file, 'w') as f:
        json.dump(training_samples, f, indent=4)

if __name__ == "__main__":
    main()