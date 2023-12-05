#!/usr/bin/env bash
#SBATCH --output=logs/%A_%a.out
#SBATCH --job-name ROICurve
#SBATCH --partition=psych_scavenge  # psych_day,,psych_week,day,week
#SBATCH --time=2:00:00 #20:00:00
##SBATCH --mem=10000

set -e

. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh ;
conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt
echo SLURM_ARRAY_TASK_ID = "$SLURM_ARRAY_TASK_ID"
JobArrayStart=$1 #can be 10000
batch=$2
#plot_dir=$3
echo python3 -u data_preprocess/prepare_coActivation_fig2c/nonmonotonicCurve/ROI_nomonotonic_curve.py "$SLURM_ARRAY_TASK_ID" "$JobArrayStart" "$batch"
python3 -u data_preprocess/prepare_coActivation_fig2c/nonmonotonicCurve/ROI_nomonotonic_curve.py "$SLURM_ARRAY_TASK_ID" "$JobArrayStart" "$batch"

echo "done"