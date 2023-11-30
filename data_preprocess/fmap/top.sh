module load FSL
. /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.0-centos7_64/etc/fslconf/fsl.sh
source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.0-centos7_64/etc/fslconf/fsl.sh
cd $1
topup --imain=AP_PA_b0.nii.gz --datain=acqparams.txt --config=b02b0.cnf --out=topup_AP_PA_b0
