#!/bin/bash
#SBATCH --partition=psych_day,psych_scavenge
#SBATCH --job-name=ASHS
#SBATCH --time=24:00:00
#SBATCH --output=logs/hippseg-%j.out
#SBATCH --mem=10000
set -e
cd /gpfs/milgram/project/turk-browne/projects/rt-cloud/
export ASHS_ROOT='./data/ASHS_tool/ashs-1.0.0'
export ASHS_ATLAS='./data/ASHS_tool/ashs_atlas_princeton'

# starting time
now=$(date +"%T")
echo "starting time : $now"

sub=$1
ANAT_dir="./data/subjects/${sub}/ses1/anat/"
ashs_dir="./data/subjects/${sub}/ses1/ashs/"
T1=${ANAT_dir}/T1.nii
T2=${ANAT_dir}/T2.nii
T1swap=${ANAT_dir}/T1_ashs.nii.gz
T2swap=${ANAT_dir}/T2_ashs.nii.gz

if [ ! -f "${T1swap}" ]; then
  fslswapdim "${T1}" -x z y "${T1swap}"
fi

if [ ! -f "${T2swap}" ]; then
  fslswapdim "${T2}" -x z y "${T2swap}"
fi

if [ ! -d "${ashs_dir}" ]; then
  mkdir "${ashs_dir}"
fi

$ASHS_ROOT/bin/ashs_main.sh -I "${sub}" -a ${ASHS_ATLAS} -g "${T1swap}" -f "${T2swap}" -w "${ashs_dir}"

echo "done"
# ending time
now=$(date +"%T")
echo "ending time : $now"

