#!/bin/bash
#SBATCH -n 1
#SBATCH -G 1
#SBATCH --gres=gpumem:20g
#SBATCH --time=01:20:00
#SBATCH --mem-per-cpu=20G
#SBATCH --output=sbatch_log/%j.out
#SBATCH --error=sbatch_err/%j.out
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=amitra@ethz.ch

source /cluster/apps/local/env2lmod.sh  # Switch to the new software stack
module load gcc/8.2.0 python/3.10.4 cuda/11.6.2 eth_proxy     # Load modules
cd /cluster/project/infk/cvg/students/amitra/thesis            # Change directory
export PYTHONPATH=.   
echo "Starting to activate virtual environment"
source ~/thesis/bin/activate               
echo "Activated virtual environment"
python scripts/new_pipeline.py general.verbose=True general.wandb=True general.debug=True "$@" # Execute the program