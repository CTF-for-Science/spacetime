import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
import os
import torch
import time
import numpy as np
from ctf4science.data_module import load_validation_dataset, get_validation_prediction_timesteps, parse_pair_ids
from ctf4science.eval_module import evaluate_custom

# Delete results directory - used for storing batch_results
file_dir = Path(__file__).parent

# Notes:
# K value larger than 10 results in invalid spatio-temporal loss
# Currently just overwriting config file and results file to save space
# Currently using init_data in hyperparameter optimization
# Currently not counting init_data in train_split amount

def main(config_path: str) -> None:
    """
    Main function to run the spacetime model on specified sub-datasets.

    Loads configuration, parses pair_ids, initializes the model, generates predictions,
    evaluates them, and saves results for each sub-dataset under a batch identifier.

    The evaluation function evaluates on validation data obtained from training data.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset name and get list of sub-dataset train/test pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    model_name = f"{config['model']['name']}"

    # batch_id is from optimize_parameters.py
    if 'batch_id' in config['model']:
        batch_id = config['model']['batch_id']
    else:
        batch_id = 0
 
    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': f"hyper_opt_{batch_id}",
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Process each sub-dataset
    for pair_id in pair_ids:
        # Prepare command
        cmd = \
        """\
        python\
        {spacetime_main_path}\
        --batch_id {batch_id}\
        --dataset {dataset}\
        --pair_id {pair_id}\
        --lag {lag}\
        --horizon {horizon}\
        --mlp_n_layers {mlp_n_layers}\
        --embedding_config {embedding_config}\
        --encoder_config {encoder_config}\
        --decoder_config {decoder_config}\
        --output_config {output_config}\
        --n_blocks {n_blocks}\
        --kernel_dim {kernel_dim}\
        --norm_order {norm_order}\
        --batch_size {batch_size}\
        --dropout {dropout}\
        --lr {lr}\
        --weight_decay {weight_decay}\
        --max_epochs {max_epochs}\
        --early_stopping_epochs {early_stopping_epochs}\
        --data_transform {data_transform}\
        --loss {loss}\
        --val_metric {val_metric}\
        --criterion_weights {criterion_weights}\
        --seed {seed}\
        --validation\
        {no_wandb}\
        """

        cmd_formatted = cmd.format(
            spacetime_main_path = file_dir / "main.py",
            batch_id = batch_id,
            dataset=config['dataset']['name'],
            pair_id = pair_id,
            lag=config['model']['lag'],
            horizon=config['model']['horizon'],
            mlp_n_layers=config['model']['mlp_n_layers'],
            embedding_config=config['model']['embedding_config'],
            encoder_config=config['model']['encoder_config'],
            decoder_config=config['model']['decoder_config'],
            output_config=config['model']['output_config'],
            n_blocks=config['model']['n_blocks'],
            kernel_dim=config['model']['kernel_dim'],
            norm_order=config['model']['norm_order'],
            batch_size=config['model']['batch_size'],
            dropout=config['model']['dropout'],
            lr=config['model']['lr'],
            weight_decay=config['model']['weight_decay'],
            max_epochs=config['model']['max_epochs'],
            early_stopping_epochs=config['model']['early_stopping_epochs'],
            data_transform=config['model']['data_transform'],
            loss=config['model']['loss'],
            val_metric=config['model']['val_metric'],
            criterion_weights=f"{config['model']['criterion_weights'][0]} {config['model']['criterion_weights'][1]} {config['model']['criterion_weights'][2]}",
            seed=config['model']['seed'],
            no_wandb="--no_wandb" if config['model']['no_wandb'] else "",
        )

        # Execute command
        print("---------------")
        print("Python running:")
        print(cmd_formatted)
        print("---------------")

        out = os.system(cmd_formatted)
        time.sleep(1) # to allow for ctrl+c

        print("---------------")
        print(f"Returned: {out}")
        print("---------------")

        if out != 0:
            raise Exception(f"Output code {out}")

        # Load predictions
        pred_data = torch.load(file_dir / 'tmp_pred' / f'output_mat_{batch_id}.torch', weights_only=False)
        pred_data = pred_data.T

        # Evaluate predictions using default metrics
        _, val_data, _ = load_validation_dataset(dataset_name, pair_id, 0.8)
        results = evaluate_custom(dataset_name, pair_id, val_data, pred_data)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

    # Save aggregated batch results
    results_file = file_dir / f"results_{batch_id}.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)