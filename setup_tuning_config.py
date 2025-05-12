from pathlib import Path
import yaml

config_path = Path(__file__).parent / 'tuning_config' / 'config_KS_Official_1.yaml'
with open(config_path, 'r') as f:
    hp_config = yaml.safe_load(f)
print(hp_config)

for dataset in ['Lorenz_Official', "KS_Official"]:
    for pair_id in range(1,9+1):

        # Fill data
        hp_config['dataset']['name'] = dataset
        hp_config['dataset']['pair_id'] = [pair_id]
        hp_config['hyperparameters']['lr']['lower_bound'] = 0.00001
        hp_config['hyperparameters']['lr']['upper_bound'] = 0.01

        if pair_id in list(range(6,7+1)):
            # limited training data 
            hp_config['hyperparameters']['lag']['lower_bound'] = 10
            hp_config['hyperparameters']['lag']['upper_bound'] = 45
            hp_config['hyperparameters']['horizon']['lower_bound'] = 10
            hp_config['hyperparameters']['horizon']['upper_bound'] = 45
            hp_config['model']['batch_size'] = 5
        elif pair_id in list(range(8,10+1)):
            # limited burn-in data but not limited training data 
            hp_config['hyperparameters']['lag']['lower_bound'] = 10
            hp_config['hyperparameters']['lag']['upper_bound'] = 45
            hp_config['hyperparameters']['horizon']['lower_bound'] = 10
            hp_config['hyperparameters']['horizon']['upper_bound'] = 45
            hp_config['model']['batch_size'] = 128
        else:
            # normal data
            hp_config['hyperparameters']['lag']['lower_bound'] = 32
            hp_config['hyperparameters']['horizon']['lower_bound'] = 32

            if dataset in ['ODE_Lorenz', 'Lorenz_Official']:
                hp_config['model']['batch_size'] = 128
                hp_config['hyperparameters']['lag']['upper_bound'] = 512
                hp_config['hyperparameters']['horizon']['upper_bound'] = 512
            else:
                hp_config['model']['batch_size'] = 16
                hp_config['hyperparameters']['lag']['upper_bound'] = 256
                hp_config['hyperparameters']['horizon']['upper_bound'] = 256
 
        # save
        output_path = Path(__file__).parent / 'tuning_config' / f'config_{dataset}_{pair_id}.yaml'
        with open(output_path, 'w') as f:
            yaml.dump(hp_config, f)