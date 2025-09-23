from pathlib import Path

top_dir = Path(__file__).parent

# source ~/.virtualenvs/spacetime/bin/activate && cd ~/Git/CTF-for-Science/models/spacetime/ && export CUDA_VISIBLE_DEVICES=2 && python optimize_parameters.py --config-path tuning_config/config_seismo_6.yaml

bash_template_0 = \
"""\
repo="/home/alexey/Git/CTF-for-Science/models/spacetime"

# Create logs directory and set up logging
rm $repo/logs/*
mkdir -p $repo/logs
exec > >(tee -a $repo/logs/{log_filename}) 2>&1

# Set CUDA device
export CUDA_VISIBLE_DEVICES={cuda_device}

echo "Running Python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /home/alexey/.virtualenvs/spacetime/bin/activate

"""

bash_template_1 = \
"""\
python optimize_parameters.py --config-path tuning_config/config_{dataset}_{pair_id}.yaml --time-budget-hours 2.0

"""

bash_template_2 = \
"""\
echo "Finished running Python"

"""

# Parameters
n_parallel = 1
datasets = ["seismo"]
pair_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
validation = 0
recon_ctx = 50

# Create and clean up bash repo
bash_dir = top_dir / 'bash'
bash_dir.mkdir(exist_ok=True)
for file in bash_dir.glob('*.sh'):
    file.unlink()

device_counter = 0
devices = ["cuda:0"]
total_scripts = len(devices) * n_parallel

# Initialize bash scripts for each device and parallel index
bash_scripts = {}
for device in devices:
    for parallel_idx in range(n_parallel):
        script_key = f"{device}_{parallel_idx}"
        device_num = device.split(':')[1]
        log_filename = f"run_cuda_{device_num}_{parallel_idx}.log"
        bash_scripts[script_key] = bash_template_0.format(log_filename=log_filename, cuda_device=device_num)

for dataset in datasets:
    for pair_id in pair_ids:
        identifier = f"{dataset}_{pair_id}"

        # Determine which device and parallel script to use based on counter
        device_idx = device_counter % len(devices)
        parallel_idx = (device_counter // len(devices)) % n_parallel
        current_device = devices[device_idx]
        script_key = f"{current_device}_{parallel_idx}"
        
        cmd = bash_template_1.format(
            dataset=dataset,
            pair_id=pair_id,
        )

        # Add the command to the appropriate bash script
        bash_scripts[script_key] += cmd

        device_counter += 1

# Add the closing template to each script and write to files
for script_key, script_content in bash_scripts.items():
    script_content += bash_template_2
    
    # Parse device and parallel index from script_key
    device, parallel_idx = script_key.rsplit('_', 1)
    device_num = device.split(':')[1]  # Extract number from "cuda:X"
    filename = f"run_cuda_{device_num}_{parallel_idx}.sh"
    filepath = bash_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    filepath.chmod(0o755)
    
    print(f"Generated bash script: {filepath}")

print(f"Total jobs: {len(datasets) * len(pair_ids)}")
print(f"Total scripts generated: {total_scripts}")
print(f"Jobs per script: ~{len(datasets) * len(pair_ids) // total_scripts} (with remainder distributed)")
