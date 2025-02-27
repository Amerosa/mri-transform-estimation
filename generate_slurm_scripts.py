import yaml 
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

slurm_dir = "slurm_scripts"
log_dir = "logs"
os.makedirs(slurm_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


#Generate a SLURM script for each of the experiments
for experiment in config["experiments"]:
    experiment_name = experiment["experiment_name"]
    slurm_config = experiment["slurm"]
    os.makedirs(experiment["output_dir"], exist_ok=True)

    slurm_script = f"""#!/bin/bash
#SBATCH --account={slurm_config["account"]}
#SBATCH --job-name={experiment_name}
#SBATCH --output={log_dir}/{experiment_name}.out
#SBATCH --error={log_dir}/{experiment_name}.err
#SBATCH --time={slurm_config["time"]}
#SBATCH --gpus-per-node={slurm_config["gpus_per_node"]}
#SBATCH --mem-per-cpu={slurm_config["mem_per_cpu"]}

module purge
module load StdEnv/2023 python/3.11.5 scipy-stack/2024a cuda/12.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install sigpy pyyaml cupy

python main.py {slurm_dir}/{experiment_name}.yaml
"""
    
    #Saving the SLURM script
    slurm_file = os.path.join(slurm_dir, f"{experiment_name}.sh")
    with open(slurm_file, "w") as f:
        f.write(slurm_script)

    #Saving individual experiment config file
    experiment_config_file = os.path.join(slurm_dir, f"{experiment_name}.yaml")
    with open(experiment_config_file, "w") as f:
        yaml.dump(experiment, f)
    