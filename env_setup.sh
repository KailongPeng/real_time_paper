#!/bin/bash

module purge
echo module load FSL/6.0.5.2-centos7_64
module load FSL/6.0.5.2-centos7_64
echo . /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh
. /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt
echo "$CONDA_DEFAULT_ENV"
echo "env_setup.sh done"
