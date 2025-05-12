import os
import copy
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join
from omegaconf import OmegaConf

from dataloaders import initialize_data_functions, get_evaluation_loaders
from utils.logging import print_header, print_args, print_config
from optimizer import get_optimizer, get_scheduler
from loss import get_loss
from data_transforms import get_data_transforms
from train import train_model, evaluate_model, forecast_model, plot_forecasts

from setup import format_arg, seed_everything
from setup import initialize_args
from setup import load_model_config, load_main_config
from setup import initialize_experiment
from setup.configs.model import update_output_config_from_args  # For multivariate feature prediction

from model.network import SpaceTime

from ctf4science.data_module import load_dataset, load_validation_dataset, get_prediction_timesteps, get_validation_prediction_timesteps

file_dir = Path(__file__).parent

def main():
    print_header('*** EXPERIMENT ARGS ***')
    args = initialize_args()
    seed_everything(args.seed)
    experiment_configs = load_main_config(args, config_dir=file_dir / 'configs')
    
    load_data, visualize_data = initialize_data_functions(args)
    print_header('*** DATASET ***')
    print_config(experiment_configs['dataset'])
 
    print_header('*** LOADER ***')
    print_config(experiment_configs['loader'])
    
    print_header('*** OPTIMIZER ***')
    print_config(experiment_configs['optimizer'])
    
    print_header('*** SCHEDULER ***')
    print_config(experiment_configs['scheduler'])
    
    # Loading Data
    experiment_configs['dataset']['validation'] = args.validation
    dataloaders = load_data(experiment_configs['dataset'], 
                            experiment_configs['loader'])
    train_loader, val_loader, _ = dataloaders
    splits = ['train', 'val']
    dataloaders_by_split = {split: dataloaders[ix] 
                            for ix, split in enumerate(splits)}
    eval_loaders = get_evaluation_loaders(dataloaders, batch_size=args.batch_size)
    
    # Setup input_dim based on features
    x, y, *z = train_loader.dataset.__getitem__(0)
    args.input_dim = x.shape[1]  # L x D
    output_dim = y.shape[1]
    
    # Initialize Model
    model_configs = {'embedding_config': args.embedding_config,
                     'encoder_config':   args.encoder_config,
                     'decoder_config':   args.decoder_config,
                     'output_config':    args.output_config}
    model_configs = OmegaConf.create(model_configs)
    model_configs = load_model_config(model_configs, config_dir=file_dir / 'configs' / 'model',
                                      args=args)
    
    model_configs['inference_only'] = False
    model_configs['lag'] = args.lag
    model_configs['horizon'] = args.horizon
    
    if args.features == 'M':  # Update output
        update_output_config_from_args(model_configs['output_config'], args,
                                       update_output_dim=True, output_dim=output_dim)
        model_configs['output_config'].input_dim = model_configs['output_config'].kwargs.input_dim
        model_configs['output_config'].output_dim = model_configs['output_config'].kwargs.output_dim  
        print(model_configs['output_config'])
    
    model = SpaceTime(**model_configs)
    model.replicate = args.replicate  # Only used for testing specific things indicated by replicate
    model.set_lag(args.lag)
    model.set_horizon(args.horizon)
    
    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model, experiment_configs['optimizer'])
    scheduler = get_scheduler(model, optimizer, 
                              experiment_configs['scheduler'])
    
    # Save some model stats
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    args.model_parameter_count = params
    arg_dict = print_args(args, return_dict=True, verbose=args.verbose)
    
    # Setup logging
    wandb = initialize_experiment(args, experiment_name_id='',
                                  best_train_metric=1e10, 
                                  best_val_metric=1e10)
    try:
        pd.DataFrame.from_dict(arg_dict).to_csv(args.log_configs_path)
    except:
        pd.DataFrame.from_dict([arg_dict]).to_csv(args.log_configs_path)
        
    if args.verbose:
        print_header('*** MODEL ***')
        print(model)
        print_config(model_configs)
        
        from einops import rearrange
        _k = model.encoder.blocks[0].pre.get_kernel(rearrange(x, '(o l) d -> o d l', o=1))
        _k_diff = model.encoder.blocks[0].pre.diff_kernel
        _k_ma_r = model.encoder.blocks[0].pre.ma_r_kernel
        print_header(f'──> Preprocessing kernels (full: {_k.shape}, diff: {_k_diff.shape}, ma: {_k_ma_r.shape})')
        print(_k[:16, :_k_ma_r.shape[-1]])
                     
    
    print_header(f'*** TRAINING ***')
    print(f'├── Lag: {args.lag}')
    print(f'├── Horizon: {args.horizon}')
    print(f'├── Criterion: {args.loss}, weights: {args.criterion_weights}')
    print(f'├── Dims: input={args.input_dim}, model={args.model_dim}')
    print(f'├── Number trainable parameters: {params}')  # └── 
    print(f'├── Experiment name: {args.experiment_name}')
    print(f'├── Logging to: {args.log_results_path}')
    
    # Loss objectives
    criterions = {name: get_loss(name) for name in ['rmse', 'mse', 'mae', 'rse']}
    eval_criterions = criterions
    for name in ['rmse', 'mse', 'mae']:
        eval_criterions[f'informer_{name}'] = get_loss(f'informer_{name}')
    
    input_transform, output_transform = get_data_transforms(args.data_transform,
                                                            args.lag)
    
    model = train_model(model, optimizer, scheduler, dataloaders_by_split, 
                        criterions, max_epochs=args.max_epochs, config=args, 
                        input_transform=input_transform,
                        output_transform=output_transform,
                        val_metric=args.val_metric, wandb=wandb, 
                        return_best=True, early_stopping_epochs=args.early_stopping_epochs)    

    if experiment_configs['dataset']['_name_'] in ['ODE_Lorenz', 'PDE_KS', 'KS_Official', 'Lorenz_Official']:
        outputs = dict()
        outputs['reconstructions'] = dict()
        outputs['forecasts'] = dict()

        lag = experiment_configs['dataset']['size'][0]
        horizon = experiment_configs['dataset']['size'][2]

        # Generate reconstructions
        if args.pair_id in [2, 4]:
            # Get input data to initialize spacetime
            if args.validation:
                train_mats, _, init_data = load_validation_dataset(args.dataset, args.pair_id, transpose=True)
                output_timesteps = get_validation_prediction_timesteps(args.dataset, args.pair_id).shape[0]
            else:
                train_mats, init_data = load_dataset(args.dataset, args.pair_id, transpose=True)
                output_timesteps = get_prediction_timesteps(args.dataset, args.pair_id).shape[0]
            data_mat = train_mats[0]
            data_mat = torch.tensor((data_mat.T))
            output_mat_full = None
            start_idx = 0
            while output_mat_full is None or output_mat_full.shape[1] < output_timesteps:
                data_mat_tmp = data_mat[start_idx*horizon:start_idx*horizon + lag,:]
                # Generate the rest of the output
                output_mat = forecast_model(model, start_mat=data_mat_tmp, config=args,
                                                n_out=horizon, 
                                                input_transform=input_transform, 
                                                output_transform=output_transform)
                # Save output
                output_mat = np.asarray(output_mat.detach().cpu()).T
                data_mat_tmp = np.asarray(data_mat_tmp.squeeze().T)
                if start_idx == 0:
                    output_mat_full = np.concatenate([data_mat_tmp, output_mat], axis=1)
                else:
                    output_mat_full = np.concatenate([output_mat_full, output_mat], axis=1)

                # Update for next loop
                start_idx += 1
            output_mat_full = output_mat_full[:,0:output_timesteps]
            output_mat = output_mat_full

        # Generate forecasts
        else:
            # Get input data to initialize spacetime
            if args.validation:
                train_mats, _, init_mat = load_validation_dataset(args.dataset, args.pair_id, transpose=True)
                if args.pair_id in [8,9]:
                    data_mat = init_mat
                    output_timesteps = get_validation_prediction_timesteps(args.dataset, args.pair_id).shape[0] - lag
                else:
                    data_mat = train_mats[0]
                    output_timesteps = get_validation_prediction_timesteps(args.dataset, args.pair_id).shape[0]
            else:
                train_mats, init_mat = load_dataset(args.dataset, args.pair_id, transpose=True)
                if args.pair_id in [8,9]:
                    data_mat = init_mat
                    output_timesteps = get_prediction_timesteps(args.dataset, args.pair_id).shape[0] - lag
                else:
                    data_mat = train_mats[0]
                    output_timesteps = get_prediction_timesteps(args.dataset, args.pair_id).shape[0]

            data_mat = torch.tensor((data_mat.T))
            data_mat = data_mat[-lag:,:]
            # Generate the rest of the output
            output_mat = forecast_model(model, start_mat=data_mat, config=args,
                                            n_out=output_timesteps, 
                                            input_transform=input_transform, 
                                            output_transform=output_transform)
            # Save output
            output_mat = output_mat.detach().cpu()
            if args.pair_id in [8,9]:
                output_mat = torch.cat([data_mat,output_mat])
            output_mat = np.asarray(output_mat).T

        # Make tmp output dir
        (file_dir / 'tmp_pred').mkdir(exist_ok=True)

        # Save file
        torch.save(output_mat, file_dir / 'tmp_pred' / f'output_mat_{args.batch_id}.torch')
    
if __name__ == '__main__':
    main()
