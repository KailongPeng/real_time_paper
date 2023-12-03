testMode = False
import os, re

if 'watts' in os.getcwd():
    projectDir = "/home/watts/Desktop/ntblab/kailong/rt-cloud/projects/rtSynth_rt/"
elif 'kailong' in os.getcwd():
    projectDir = "/Users/kailong/Desktop/rtEnv/rt-cloud/projects/rtSynth_rt/"
elif 'milgram' in os.getcwd():
    projectDir = "/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
else:
    raise Exception('path error')
import sys

sys.path.append(projectDir)
sys.path.append(projectDir + "../../")
from subprocess import call
import subprocess
import nibabel as nib
import pydicom as dicom
import numpy as np
import time
from glob import glob
import shutil
import pandas as pd
from rtCommon.imageHandling import convertDicomImgToNifti, readDicomFromFile, convertDicomFileToNifti
from cfg_loading import mkdir, cfg_loading
from scipy.stats import zscore
import pickle5 as pickle
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
from tqdm import tqdm
import itertools
from sklearn.linear_model import LogisticRegression
import joblib
from scipy import stats

import os
import sys
import numpy as np
import pandas as pd
import re
import subprocess
import time
import pickle5 as pickle
import IPython.display as display
from PIL import Image
# import matplotlib.pyplot as plt
import scipy.optimize as opt
from tqdm import tqdm
from glob import glob

import nibabel as nib


if 'watts' in os.getcwd():
    projectDir = "/home/watts/Desktop/ntblab/kailong/rt-cloud/projects/rtSynth_rt/"
elif 'kailong' in os.getcwd():
    projectDir = "/Users/kailong/Desktop/rtEnv/rt-cloud/projects/rtSynth_rt/"
elif 'milgram' in os.getcwd():
    projectDir = "/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
else:
    raise Exception('path error')
os.chdir(projectDir)

sys.path.append(projectDir)
sys.path.append(projectDir + "../../")
sys.path.append(projectDir + "/OrganizedScripts/")

# 在jupyterLab中如果失败了, 可能是因为使用的Python核心有问题, 使用下面的代码强制改变jupyter的核心以产生加载util.py中的函数.
## %%script /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt/bin/python3.7
# import sys
# print(sys.executable)
# print(sys.version)
# print(sys.version_info)
# from utils import *

from utils import save_obj, load_obj, mkdir, getjobID_num, kp_and, kp_or, kp_rename, kp_copy, kp_run, kp_remove
from utils import wait, check, checkEndwithDone, checkDone, check_jobIDs, check_jobArray, waitForEnd, \
    jobID_running_myjobs
from utils import readtxt, writetxt, deleteChineseCharactor, get_subjects, init
from utils import getMonDate, checkDate
from utils import get_ROIMethod, bar, get_ROIList

# batch = 12  # 29表示只对sub029 sub030 sub031 运行  # 在某些时候我只想在除了29 30 31 之外的被试身上运行, 此时就使用batch99
# subjects, scan_asTemplates = get_subjects(batch=batch)


def other(target):
    other_objs = [i for i in ['bed', 'bench', 'chair', 'table'] if i not in target]
    return other_objs


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def readtxt(file):
    f = open(file, "r")
    return f.read()


def kp_and(l):
    t = True
    for i in l:
        t = t * np.asarray(i)
    return t


def kp_or(l):
    t = np.asarray(l[0])
    for i in l:
        t = t + np.asarray(i)
    return t


def wait(tmpFile, waitFor=0.1):
    while not os.path.exists(tmpFile):
        time.sleep(waitFor)
    return 1


def normalize(X):
    _X = X.copy()
    _X = zscore(_X, axis=0)
    _X[np.isnan(_X)] = 0
    return _X


def check(sbatch_response):
    print(sbatch_response)
    if "Exception" in sbatch_response or "Error" in sbatch_response or "Failed" in sbatch_response or "not" in sbatch_response:
        raise Exception(sbatch_response)


os.chdir("/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/real_time_paper/")
assert os.getcwd().endswith('real_time_paper'), "working dir should be 'real_time_paper'"
workingDir = os.getcwd()


if testMode:
    [sub, ses] = ['sub024', 5]
else:
    jobarrayDict = np.load(f"data_preprocess/unwarp/unwarp_jobID.npy",
                           allow_pickle=True)
    jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
    jobarrayID = int(float(sys.argv[1]))
    [sub, ses] = jobarrayDict[jobarrayID]
print(f"sub={sub}, ses={ses}")
# cfg = cfg_loading(f"{sub}.ses{ses}.toml")
# batch = cfg.batch
# print(f"batch={batch}")

SesFolder = f"{workingDir}/data/subjects/{sub}/ses{ses}/"
mkdir(f"{SesFolder}/fmap/")
os.chdir(f"{SesFolder}/fmap/")
if not os.path.exists(f"{SesFolder}/fmap/topup_AP_PA_b0_fieldcoef.nii.gz"):
    cmd = f"bash {workingDir}/data_preprocess/fmap/top.sh {SesFolder}fmap/"
    kp_run(cmd)

# use topup_AP_PA_fmap to unwarp all functional data
runRecording = pd.read_csv(f"{SesFolder}/runRecording.csv")
recogRuns = list(
    runRecording['run'].iloc[list(np.where(1 == 1 * (runRecording['type'] == 'recognition'))[0])])
feedbackRuns = list(
    runRecording['run'].iloc[list(np.where(1 == 1 * (runRecording['type'] == 'feedback'))[0])])
for scan in recogRuns:
    scan_file = f"{SesFolder}/recognition/run_{scan}"
    if not os.path.exists(f"{scan_file}_unwarped.nii.gz"):
        cmd = (f"applytopup --imain={scan_file}.nii --topup=topup_AP_PA_b0 --datain=acqparams.txt --inindex=1 "
               f"--out={scan_file}_unwarped --method=jac")
        kp_run(cmd)

for scan in feedbackRuns:
    scan_file = f"{SesFolder}/feedback/run_{scan}"
    if not os.path.exists(f"{scan_file}_unwarped.nii.gz"):
        cmd = (f"applytopup --imain={scan_file}.nii --topup=topup_AP_PA_b0 --datain=acqparams.txt --inindex=1 "
               f"--out={scan_file}_unwarped --method=jac")
        kp_run(cmd)


def saveMiddleVolumeAsTemplate(SesFolder='',
                               scan_asTemplate=1):  # expScripts/recognition/recognitionDataAnalysis/GM_modelTrain.py
    nii = nib.load(f"{SesFolder}/recognition/run_{scan_asTemplate}_unwarped.nii.gz")
    frame = nii.get_fdata()
    TR_number = frame.shape[3]
    frame = frame[:, :, :, int(TR_number / 2)]
    frame = nib.Nifti1Image(frame, affine=nii.affine)
    unwarped_template = f"{SesFolder}/recognition/templateFunctionalVolume_unwarped.nii"
    nib.save(frame, unwarped_template)


# Take the middle volume of the first recognition run as a template
saveMiddleVolumeAsTemplate(SesFolder=SesFolder, scan_asTemplate=scan_asTemplates[sub][f"ses{ses}"])


def align_with_template(SesFolder='',
                        scan_asTemplate=1):  # expScripts/recognition/recognitionDataAnalysis/GM_modelTrain.py
    from utils import kp_run
    unwarped_template = f"{SesFolder}/recognition/templateFunctionalVolume_unwarped.nii"
    if not os.path.exists(f"{SesFolder}/recognition/functional_bet.nii.gz"):
        cmd = (f"bet {SesFolder}/recognition/templateFunctionalVolume_unwarped.nii "
               f"{SesFolder}/recognition/templateFunctionalVolume_unwarped_bet.nii.gz")

        kp_run(cmd)
        shutil.copyfile(f"{SesFolder}/recognition/templateFunctionalVolume_unwarped_bet.nii.gz",
                        f"{SesFolder}/recognition/functional_bet.nii.gz")

    # Align all recognition runs and feedback runs with the functional template of the current session
    for scan in tqdm(recogRuns):
        head = f"{SesFolder}/recognition/run_{scan}"
        if not os.path.exists(f"{head}_unwarped_mc.nii.gz"):
            cmd = f"mcflirt -in {head}_unwarped.nii.gz -out {head}_unwarped_mc.nii.gz"
            from utils import kp_run
            kp_run(cmd)
            wait(f"{head}_unwarped_mc.nii.gz")  # mcflirt for motion correction

        # Align the motion-corrected functional data with the unwarped_template of the current session, which is the funcTemplate corrected by topup.
        # Then, transfer the motion-corrected data to the func space corrected by topup.
        # The reason for two steps here is that flirt does not directly produce the -out result for multiple volumes.
        cmd = f"flirt -in {head}_unwarped_mc.nii.gz " \
              f"-out {head}_temp.nii.gz " \
              f"-ref {unwarped_template} " \
              f"-dof 6 " \
              f"-omat {SesFolder}/recognition/scan{scan}_to_unwarped.mat"

        def kp_run(cmd):
            print()
            print(cmd)
            import subprocess
            sbatch_response = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            check(sbatch_response.stdout)
            return sbatch_response.stdout

        kp_run(cmd)

        cmd = f"flirt " \
              f"-in {head}_unwarped_mc.nii.gz " \
              f"-out {head}_unwarped_mc.nii.gz " \
              f"-ref {unwarped_template} -applyxfm " \
              f"-init {SesFolder}/recognition/scan{scan}_to_unwarped.mat"
        from utils import kp_run
        kp_run(cmd)
        kp_remove(f"{head}_temp.nii.gz")

    for scan in tqdm(feedbackRuns):
        cmd = f"mcflirt -in {SesFolder}/feedback/run_{scan}_unwarped.nii.gz " \
              f"-out {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz"
        kp_run(cmd)
        wait(f"{SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz")

        cmd = f"flirt -in {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz " \
              f"-out {SesFolder}/feedback/run_{scan}_temp.nii.gz " \
              f"-ref {unwarped_template} -dof 6 -omat {SesFolder}/feedback/scan{scan}_to_unwarped.mat"
        kp_run(cmd)

        cmd = f"flirt -in {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz " \
              f"-out {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz " \
              f"-ref {unwarped_template} -applyxfm -init {SesFolder}/feedback/scan{scan}_to_unwarped.mat"
        kp_run(cmd)
        kp_remove(f"{SesFolder}/feedback/run_{scan}_temp.nii.gz")

        cmd = f"fslinfo {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz"
        kp_run(cmd)


batch = 12  # meaning both batch 1 and batch 2
subjects, scan_asTemplates = get_subjects(batch=batch)
align_with_template(SesFolder=SesFolder, scan_asTemplate=scan_asTemplates[sub][f"ses{ses}"])

print("done")
