#!/bin/bash
#SBATCH -p psych_scavenge
#SBATCH --job-name=preprocess
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e

. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
cd /gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication
module load FSL
source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh
conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt

echo python3 -u data_preprocess/data_preprocess.py "${SLURM_ARRAY_TASK_ID}" "$1" "$2" "$3"
python3 -u data_preprocess/data_preprocess.py "${SLURM_ARRAY_TASK_ID}" "$1" "$2" "$3"
echo "done"
