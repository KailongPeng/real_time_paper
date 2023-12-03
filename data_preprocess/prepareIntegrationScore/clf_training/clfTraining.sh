#!/bin/bash
#SBATCH -p psych_scavenge  #psych_day,psych_gpu,psych_scavenge,psych_week
#SBATCH --job-name=autoROI
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e
#module load FSL/5.0.10
#. /nexsan/apps/hpc/Apps/FSL/5.0.10/etc/fslconf/fsl.sh
#module load miniconda
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
#conda activate GLMsingle
cd /gpfs/milgram/project/turk-browne/projects/rt-cloud
module load AFNI
module load FSL
source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh
module load dcm2niix
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt


echo python3 -u /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/ROI/autoAlign_ses1ses5/clf_training/clfTraining.py "${SLURM_ARRAY_TASK_ID}"
python3 -u /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/ROI/autoAlign_ses1ses5/clf_training/clfTraining.py "${SLURM_ARRAY_TASK_ID}"
echo "done"
