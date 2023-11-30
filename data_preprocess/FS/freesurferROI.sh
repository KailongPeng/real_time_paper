#!/bin/bash
#SBATCH --partition=psych_day,psych_scavenge,psych_week,day,
#SBATCH --job-name=freesurfer
#SBATCH --time=40:00
#SBATCH --output=logs/%j.out
#SBATCH --mem=2g

set -e # 遇到问题的时候立即停止
module load FreeSurfer/6.0.0
module load FSL ; source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh 
SUBJ=${1}
SUBJECTS_DIR=/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/
cd ${SUBJECTS_DIR}/${SUBJ}/ses1/anat/


SUBJECT_SURF_DIR=${SUBJECTS_DIR}/${SUBJ}/ses1/anat/freesurfer/
ANAT=${SUBJECTS_DIR}/${SUBJ}/ses1/anat/T1_bet.nii
if [ ! -f "${ANAT}" ]; then
    gunzip ${ANAT}.gz ${ANAT}
fi
T2=${SUBJECTS_DIR}/${SUBJ}/ses1/anat/T2_bet.nii ; bet ${SUBJECTS_DIR}/${SUBJ}/ses1/anat/T2.nii ${T2}

SUBJECT=${SUBJECTS_DIR}/${SUBJ}/ses1/anat/freesurfer/
FILLTHRESH=0.0 # threshold for incorporating a voxel in the mask, default = 0
REGISTERDAT=${SUBJECT_SURF_DIR}/register.dat
echo "must run the following command from command line in interactive session"
echo "tkregister2 --s ${SUBJ}/ses1/anat/freesurfer/${SUBJ}/ --mov $ANAT --regheader --reg $REGISTERDAT"
bbregister --mov $ANAT --t1 --s ${SUBJ}/ses1/anat/freesurfer/${SUBJ}/ --init-fsl --reg $REGISTERDAT

# 通过开出使一个人与另一个人保持一致的转变，手动登记 manual registration by prescribing a transformation that brings one into alignment with the other
#--init-fsl : Initialize using FSL FLIRT.

# ROIS="V1 V2 perirhinal"
ROIS="V1 V2 perirhinal BA3b BA3a BA2 entorhinal BA45 BA1 BA44 BA6 MT BA4p BA4a"
## 如果想用功能体素大小，首先在解剖空间用applyisoxfm和flirt创建一个功能体，然后用它作为模板。 If wanting to go to functional voxel size, first create a functional in anatomical space using applyisoxfm and flirt, then use that as the template
## for mri_label2vol -temp and mri_convert -rl


#"convert label produced by freesurfer in T1 space to nifti files in T1 space. Convert T1 to T2 space"
OUT_DIR=rois_surf
mkdir -p $OUT_DIR
OUT_DIR=${OUT_DIR}/
mkdir -p ${OUT_DIR}
mkdir -p $OUT_DIR/t1space
mkdir -p $OUT_DIR/t2space

#convert labels to nifti files
for ROI in $ROIS ; do
    echo $ROI
    for HEMI in lh rh
    do
        LABEL=${SUBJECT_SURF_DIR}/${SUBJ}/label/${HEMI}.${ROI}_exvivo.label
        DEST=${OUT_DIR}/t1space/${HEMI}.${ROI}.nii.gz
        # DEST2=${OUT_DIR}/t2space/${HEMI}.${ROI}.nii.gz
        mri_label2vol --label $LABEL --temp $ANAT --o $DEST --fillthresh $FILLTHRESH --proj frac 0 1 0.1 --subject ${SUBJ}/ses1/anat/freesurfer/${SUBJ}/ --hemi $HEMI --surf white --reg $REGISTERDAT
        # flirt -ref $T2 -in $DEST -applyxfm -init $NIFTI_DIR/t1_to_t2.mat -out $DEST2 # > /dev/null
    done
done
fslinfo $ANAT #1*1*1*1  $DEST 1111 


# convert surface file into nifti file
freesurferSubDir=/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/${SUBJ}/ses1/anat/freesurfer/${SUBJ}/
# mri_convert -rl $AANT -rt nearest ${freesurferSubDir}/mri/aparc+aseg.mgz ${OUT_DIR}/aparc+aseg.nii.gz  > /dev/null 
mri_convert -rl ${freesurferSubDir}/mri/rawavg.mgz -rt nearest ${freesurferSubDir}/mri/aparc+aseg.mgz ${OUT_DIR}/aparc+aseg.nii.gz

#aparc.a2009s+aseg.mgz（Destrieux atlas）   aparc+aseg.mgz   aparc.DKTatlas+aseg.mgz
# https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI

fslinfo ${OUT_DIR}/aparc+aseg.nii.gz #1.000004*1*1*1

#ROIS=("IT" "HC" "EC" "PHC" "LOC" "Fus" "paraSub" "preSub" "Sub" "CA1" "CA3" "CA4" "DG" "HATA" "Fim" "Mol" "Fiss" "Tail") # names of labels can be looked up in $FREESURFER_HOME/FreeSurferColorLUT.txt


# 为每个ROI获得一个掩码 get a mask for each ROI
ROIS=("IT" "HC" "EC" "PHC" "LOC" "Fus")
LEFT=(1009 17 1006 1016 1011 1007) # 在$FREESURFER_HOME/FreeSurferColorLUT.txt 可以找到对应的ID
RIGHT=(2009 53 2006 2016 2011 2007)
lookUpTable=("ctx-lh-inferiortemporal" "Left-Hippocampus" "ctx-lh-entorhinal" "ctx-lh-parahippocampal" "ctx-lh-lateraloccipital" "ctx-lh-fusiform")
HEMIS="lh rh"
for hemi in $HEMIS; do
    fileList=""
    for ((i=0;i<${#ROIS[@]};++i)); do
        echo ${ROIS[i]}
        outfile2=$OUT_DIR/t1space/${hemi}.${ROIS[i]}.nii.gz
        # if [ $i -gt 5 ]; then basefile=${OUT_DIR}/${hemi}.hippoSfLabels.nii.gz; else basefile=${OUT_DIR}/aparc+aseg.nii.gz; fi
        basefile=${OUT_DIR}/aparc+aseg.nii.gz
            # aparc_aseg=nib.load(f"aparc+aseg.nii.gz").get_data()

        # 获取对应半球的mask
        if [ "$hemi" = "lh" ]; then 
            fslmaths $basefile -uthr ${LEFT[i]} -thr ${LEFT[i]} $outfile2
        else
            fslmaths $basefile -uthr ${RIGHT[i]} -thr ${RIGHT[i]} $outfile2
        fi

        fslmaths $outfile2 -bin $outfile2 # 二元化

        if [ "${ROIS[i]}" = "LOC" ]; then # 当目前的ROI是LOC的时候，用当前的LOC 减去V1 减去V2 ，就获得了LOC 
            fslmaths $outfile2 -sub $OUT_DIR/t1space/${hemi}.V1.nii.gz -sub $OUT_DIR/t1space/${hemi}.V2.nii.gz $outfile2
            fslmaths $outfile2 -bin $outfile2
        fi

        # flirt -ref $T2 -in $outfile2 -applyxfm -init $NIFTI_DIR/t1_to_t2.mat -out $outfile #转移到T2的空间中。
        fileList="$fileList $outfile2"
    done
done
# fslview_deprecated T1_bet.nii.gz rois_surf/t1space/lh.LOC.nii.gz
# fslinfo $basefile $outfile2 OUT_DIR/t1space/${hemi}.V1.nii.gz OUT_DIR/t1space/${hemi}.V2.nii.gz  #1.000004*1*1*1

# '合并半球并对ROI掩码文件进行降样处理 merge hemispheres and downsample the ROI mask files (originally 1*1*1*1.5 to newly 1.5*1.5*1.5*1)'

# ROIS=("V1" "V2" "perirhinal" "IT" "HC" "EC" "PHC" "LOC" "Fus" "paraSub" "preSub" "Sub" "CA1" "CA3" "CA4" "DG" "HATA" "Fim" "Mol" "Fiss" "Tail")
funcROI=rois_surf/func/
mkdir -p ${funcROI}
# ROIS=("V1" "V2" "perirhinal" "IT" "HC" "EC" "PHC" "LOC" "Fus")
ROIS=("V1" "V2" "perirhinal" "BA3b" "BA3a" "BA2" "entorhinal" "BA45" "BA1" "BA44" "BA6" "MT" "BA4p" "BA4a" "IT" "HC" "EC" "PHC" "LOC" "Fus") 
for ((i=0;i<${#ROIS[@]};++i)); do
    echo "merging hemispheres for ${ROIS[i]}" # 合并半球。
    #outfile=$OUT_DIR/t2space/${ROIS[i]}.nii.gz
    outfile2=$OUT_DIR/t1space/${ROIS[i]}.nii.gz
    #fslmaths $OUT_DIR/t2space/lh.${ROIS[i]}.nii.gz -add $OUT_DIR/t2space/rh.${ROIS[i]}.nii.gz $outfile
    fslmaths $OUT_DIR/t1space/lh.${ROIS[i]}.nii.gz -add $OUT_DIR/t1space/rh.${ROIS[i]}.nii.gz $outfile2 # 合并所选ROI的两个半球的掩码  merge the mask from two hemisphere for selected ROI 
    fslmaths $outfile2 -bin $outfile2 # 二元化
    #fslmaths $outfile -bin $outfile
    ROI_func=${funcROI}/${ROIS[i]}_FreeSurfer.nii.gz  # 曾经是 ROI_func=${funcROI}/${ROIS[i]}_func.nii.gz
    tempFunc=/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/${SUBJ}/ses1/recognition/functional_bet.nii.gz
    anat2func=/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/${SUBJ}/ses1/recognition/anat2func.mat
    #DOWNSAMP=$OUT_DIR/t1space/${ROIS[i]}_1.5.nii.gz
    #IDENT=$FSLDIR/etc/flirtsch/ident.mat
    #flirt -ref $ANAT -in $outfile2 -out $DOWNSAMP -applyisoxfm 1.5 -init $IDENT
    flirt -ref $tempFunc -in $outfile2 -out $ROI_func -init $anat2func -applyxfm
    fslmaths $ROI_func -thr 0.5 -bin $ROI_func # 对func空间中的mask进行二元化。
done
# 很完美

# fslview_deprecated T1_bet.nii rois_surf/t1space/V1.nii.gz 
# fslview_deprecated rois_surf/func/V1_func.nii.gz ../recognition/templateFunctionalVolume_bet.nii.gz 

echo done
