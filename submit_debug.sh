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
module load gcc/8.2.0 python/3.10.4 cuda/11.6.2 eth_proxy cudnn/8.0.5 pigz
cd /cluster/project/cvl/amitra/SlowFast_VRC            # Change directory
echo "Starting to activate virtual environment"
source /cluster/project/cvl/amitra/SlowFast_VRC/vrc/bin/activate # Activate virtual environment               
export PYTHONPATH=.   
echo "Activated virtual environment"
python tools/run_net.py --cfg "$@" #write the name of the config each time # Execute the program