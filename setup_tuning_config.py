from pathlib import Path
import yaml

config_path = Path(__file__).parent / 'tuning_config' / 'config_ocean_das_1.yaml'
with open(config_path, 'r') as f:
    hp_config = yaml.safe_load(f)
print(hp_config)

# Memory note: the first encoder block is 128 * (spatial dimension) channels wide,
# so activations scale with batch_size * lag * 128 * D. Large-D datasets (msfr: D=19400,
# mhd-2cyls: D=122400) therefore get small batches and short lags.
# Per-dataset settings: (batch_size, lag_upper, horizon_upper) for each pair group
dataset_settings = {
    # Lorenz-like low-dimensional ODE, 10000-step train matrices; pairs 6/7 have only
    # 100 train steps (80 in validation mode), pairs 8/9 a 100-step burn-in
    'pke-hpmr':  {'normal': (128, 512, 512), 'limited_train': (5, 45, 30), 'limited_burn': (128, 45, 45)},
    # D=123, 2500-step train matrices; 500-step limited train; 900-step burn-in
    'dynasty':   {'normal': (32, 256, 256),  'limited_train': (5, 45, 45), 'limited_burn': (128, 45, 45)},
    # D=19400
    'msfr':      {'normal': (4, 128, 256),   'limited_train': (5, 45, 45), 'limited_burn': (4, 45, 45)},
    # D=122400; batch 1 only. With the width-64 first block (width_configs below) lag <= 64 and
    # horizon <= 128 fit a 48 GB L40S
    'mhd-2cyls': {'normal': (1, 64, 128),    'limited_train': (1, 45, 45), 'limited_burn': (1, 45, 45)},
}
n_trials = {'mhd-2cyls': 5}
# mhd-2cyls: embedding width reduced from 128 to 64 to fit 48 GB GPU memory (L40S).
# The default first block (embedding/repeat + encoder/default_no_skip) is 128 * D wide, i.e.
# ~2.04B parameters for D=122400; the *_w64 configs halve it to ~1.03B. Measured peak at
# W=64: ~21 GB (lag 32) to ~29 GB (lag 64, horizon 128) of tensors, so the full tuning range fits.
# Fixed (not tuned) so no trial can run out of memory. Other datasets keep the base config.
width_configs = {
    'mhd-2cyls': {'embedding_config': 'embedding/repeat_w64', 'encoder_config': 'encoder/default_no_skip_w64'},
}
# Each mhd-2cyls epoch is slow (~2000 batch-1 steps on a 1B-parameter model, 12 GB checkpoints),
# so tune with the same epoch budget as the Official mhd-2cyls configs.
max_epochs = {'mhd-2cyls': 20}
base_model = dict(hp_config['model'])

for dataset in ["pke-hpmr", "dynasty", "msfr", "mhd-2cyls"]:
    for pair_id in range(1,9+1):

        # Fill data
        hp_config['dataset']['name'] = dataset
        hp_config['model']['name'] = 'spacetime'
        hp_config['model']['features'] = 'M'  # one output channel per state variable
        hp_config['model']['n_trials'] = n_trials.get(dataset, 20)
        hp_config['model']['max_epochs'] = max_epochs.get(dataset, base_model['max_epochs'])
        for key in ('embedding_config', 'encoder_config'):
            hp_config['model'][key] = width_configs.get(dataset, {}).get(key, base_model[key])
        hp_config['dataset']['pair_id'] = [pair_id]
        hp_config['hyperparameters']['lr']['lower_bound'] = 0.00001
        hp_config['hyperparameters']['lr']['upper_bound'] = 0.01

        if pair_id in list(range(6,7+1)):
            # limited training data 
            group = 'limited_train'
            hp_config['hyperparameters']['lag']['lower_bound'] = 10
            hp_config['hyperparameters']['horizon']['lower_bound'] = 10
        elif pair_id in list(range(8,10+1)):
            # limited burn-in data but not limited training data 
            group = 'limited_burn'
            hp_config['hyperparameters']['lag']['lower_bound'] = 10
            hp_config['hyperparameters']['horizon']['lower_bound'] = 10
        else:
            # normal data
            group = 'normal'
            hp_config['hyperparameters']['lag']['lower_bound'] = 32
            hp_config['hyperparameters']['horizon']['lower_bound'] = 32

        batch_size, lag_upper, horizon_upper = dataset_settings[dataset][group]
        hp_config['model']['batch_size'] = batch_size
        hp_config['hyperparameters']['lag']['upper_bound'] = lag_upper
        hp_config['hyperparameters']['horizon']['upper_bound'] = horizon_upper
 
        # save
        output_path = Path(__file__).parent / 'tuning_config' / f'config_{dataset}_{pair_id}.yaml'
        with open(output_path, 'w') as f:
            if dataset in width_configs:
                f.write(f'# {dataset}: embedding width reduced from 128 to 64 to fit 48 GB GPU memory.\n')
            yaml.dump(hp_config, f)
