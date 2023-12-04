#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=logs/%A_%a.out
#SBATCH --job-name ROI_real
#SBATCH --partition=psych_scavenge  # psych_day,,day
#SBATCH --time=1:00:00 #20:00:00
##SBATCH --mem=10000
##SBATCH -n 5
##SBATCH --mail-type=FAIL
##SBATCH --mail-user=kp578
set -e
cd /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt

module load AFNI ; module load FSL ;
source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh ;
module load dcm2niix ;
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh ;
conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt

jobArrayPath=$1
jobIDstart=$2
echo "$SLURM_ARRAY_TASK_ID" "${jobArrayPath}"
echo python -u /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/megaROI/withinSession/autoAlign/nonmonotonicCurve/simulatedFeedbackRun/rtSynth_rt_ABCD_ROIanalysis_unwarp.py "${SLURM_ARRAY_TASK_ID}" "${jobArrayPath}" "${jobIDstart}"
python -u /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/megaROI/withinSession/autoAlign/nonmonotonicCurve/simulatedFeedbackRun/rtSynth_rt_ABCD_ROIanalysis_unwarp.py "${SLURM_ARRAY_TASK_ID}" "${jobArrayPath}" "${jobIDstart}"

echo "done"
