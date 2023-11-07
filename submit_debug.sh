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
cd /cluster/project/cvl/amitra/SlowFast_VRC            # Change directory
echo "Starting to activate virtual environment"
source /cluster/project/cvl/amitra/SlowFast_VRC/vrc/bin/activate # Activate virtual environment               
export PYTHONPATH=.   
echo "Activated virtual environment"
python tools/run_net.py --cfg configs/VRC/X3D_L.yaml # Execute the program