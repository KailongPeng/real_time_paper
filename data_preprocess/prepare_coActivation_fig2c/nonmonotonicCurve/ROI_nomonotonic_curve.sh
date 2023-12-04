#!/usr/bin/env bash
#SBATCH --output=logs/%A_%a.out
#SBATCH --job-name ROICurve
#SBATCH --partition=psych_scavenge  # psych_day,,psych_week,day,week
#SBATCH --time=2:00:00 #20:00:00
##SBATCH --mem=10000
##SBATCH -n 5
##SBATCH --mail-type=FAIL
##SBATCH --mail-user=kp578
set -e
cd /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt
#module load AFNI ; module load FSL ;
#source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh ; module load dcm2niix ;
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh ;
conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt
echo SLURM_ARRAY_TASK_ID = "$SLURM_ARRAY_TASK_ID"
JobArrayStart=$1 #can be 10000
batch=$2
#plot_dir=$3
echo python3 -u /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/megaROI/withinSession/autoAlign/nonmonotonicCurve/ROI_nomonotonic_curve.py "$SLURM_ARRAY_TASK_ID" "$JobArrayStart" "$batch"
python3 -u /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/megaROI/withinSession/autoAlign/nonmonotonicCurve/ROI_nomonotonic_curve.py "$SLURM_ARRAY_TASK_ID" "$JobArrayStart" "$batch"

echo "done"

# cmd=f"sbatch --array=1-45 /gpfs/milgram/project/turk-browne/projects/rt-cloud//projects/rtSynth_rt/expScripts/recognition/ROI_analysis/ROI_nomonotonic_curve.sh 0"
