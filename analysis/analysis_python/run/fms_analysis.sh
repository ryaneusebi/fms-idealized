#!/bin/bash -u
#SBATCH --ntasks=1
#SBATCH --time=4:00:00 
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --job-name=analyze_fms
#SBATCH --output=out_err/slurm_%j.out
#SBATCH --error=out_err/slurm_%j.err 
#SBATCH --export=ALL
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=$USER@caltech.edu


# 64 GB and 3 hours and 10 cpus for 200 days of T42
# 128 GB and 3 hours for 200 days of T85 and T127

# Activate conda environment manually (miniconda has broken hardcoded paths)
export PATH="/resnick/groups/esm/$USER/miniconda3/envs/fms_analysis/bin:$PATH"
export CONDA_PREFIX="/resnick/groups/esm/$USER/miniconda3/envs/fms_analysis"

python -u ../src/fms_analysis.py "$@"