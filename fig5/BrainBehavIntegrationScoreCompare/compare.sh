#!/bin/bash
#SBATCH -p psych_scavenge  # psych_day,psych_gpu,psych_scavenge,psych_week
#SBATCH --job-name=compare
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%J.out
#SBATCH --mem=10g
#SBATCH --requeue

set -e

. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
module load FSL
source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh
conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt

echo python3 -u fig5/BrainBehavIntegrationScoreCompare/compare.py
python3 -u fig5/BrainBehavIntegrationScoreCompare/compare.py
echo "done"


