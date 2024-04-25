#!/bin/bash
#SBATCH -p psych_scavenge
#SBATCH --job-name=geometric_ROI
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e
source env_setup.sh

echo python3 -u fig5/geometric_analysis/geometric.py "${SLURM_ARRAY_TASK_ID}"
python3 -u fig5/geometric_analysis/geometric.py "${SLURM_ARRAY_TASK_ID}"
echo "done"
