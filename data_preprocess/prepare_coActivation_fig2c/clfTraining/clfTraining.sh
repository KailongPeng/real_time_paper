#!/bin/bash
#SBATCH -p psych_scavenge  #psych_day,psych_gpu,psych_scavenge,psych_week
#SBATCH --job-name=clfTraining
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
module load AFNI
module load FSL
source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh
module load dcm2niix
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/myrtSynth_rt

echo python3 -u data_preprocess/prepare_coActivation_fig2c/clfTraining/clfTraining.py "${SLURM_ARRAY_TASK_ID}"
python3 -u data_preprocess/prepare_coActivation_fig2c/clfTraining/clfTraining.py "${SLURM_ARRAY_TASK_ID}"
echo "done"
