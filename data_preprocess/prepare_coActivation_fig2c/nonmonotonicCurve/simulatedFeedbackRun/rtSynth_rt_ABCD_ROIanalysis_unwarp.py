# 测试的时候使用 jupLab
import os
import sys
import numpy as np
import pandas as pd
import re
import subprocess
import time
import pickle5 as pickle
import matplotlib.pyplot as plt
import scipy.optimize as opt
from tqdm import tqdm

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

# from recognition_dataAnalysisFunctions import \
#         interimAnalysis  # rtSynth_rt/expScripts/recognition/recognition_dataAnalysisFunctions.py
# interimAnalysis()
# def interimAnalysis():  # 这个代码的目的是 为了获得ses1和ses5的行为学结果，查看是否有显著的行为的整合。
sys.path.append(f"{projectDir}expScripts/catPer/")

# 找出实际上使用的chosenMask使用的是哪些ROI
from cfg_loading import mkdir, cfg_loading
import nibabel as nib
import numpy as np
from tqdm import tqdm
from glob import glob
from scipy.stats import zscore
import pandas as pd

import os
import numpy as np
import pandas as pd
import joblib
import itertools
from sklearn.linear_model import LogisticRegression

import random
# from rtCommon.imageHandling import convertDicomImgToNifti, readDicomFromFile, convertDicomFileToNifti
from subprocess import call
import shutil

sys.path.append(f"{projectDir}expScripts/recognition/")
from recognition_dataAnalysisFunctions import normalize, classifierProb


def convertDicomFileToNifti(dicomFilename, niftiFilename):
    # global binPath
    # binPath = '/gpfs/milgram/apps/hpc.rhel7/software/dcm2niix/3-Jan-2018/'
    binPath = '/gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt/bin/'
    if binPath is None:
        result = subprocess.run(['which', 'dcm2niix'], stdout=subprocess.PIPE)
        binPath = result.stdout.decode('utf-8')
        binPath = os.path.dirname(binPath)
    dcm2niiCmd = os.path.join(binPath, 'dcm2niix')
    outPath, outName = os.path.split(niftiFilename)
    if outName.endswith('.nii'):
        outName = os.path.splitext(outName)[0]  # remove extention
    __cmd = [dcm2niiCmd, '-s', 'y', '-b', 'n', '-o', outPath, '-f', outName, dicomFilename]
    cmd = ' '.join(__cmd)
    kp_run(cmd)


batch = 12  # 29表示只对sub029 sub030 sub031 运行  # 在某些时候我只想在除了29 30 31 之外的被试身上运行, 此时就使用batch99
subjects, scan_asTemplates = get_subjects(batch=batch)

# 首先获得绝对和最开始收集数据的时候一样的 brain_run 和 behav_run .
testMode = False
if testMode:
    [sub, feedbackSes, runNum, scanNum, useNewClf] = ['sub014', 2, 3, 5, True]
else:
    jobIDstart = int(sys.argv[3])
    print(f"jobIDstart={jobIDstart}")
    jobArrayPath = sys.argv[2]
    print(f"jobArrayPath={jobArrayPath}")
    SLURM_ARRAY_TASK_ID_dict = load_obj(jobArrayPath)
    SLURM_ARRAY_TASK_ID = int(sys.argv[1]) + jobIDstart
    print(f'SLURM_ARRAY_TASK_ID={SLURM_ARRAY_TASK_ID}')
    [sub, feedbackSes, runNum, scanNum, chosenMask, forceReRun, useNewClf] = SLURM_ARRAY_TASK_ID_dict[
        SLURM_ARRAY_TASK_ID]
    print(
        f"sub={sub}, ses={feedbackSes}, runNum={runNum}, scanNum={scanNum}, chosenMask={chosenMask}, forceReRun={forceReRun}")
    assert feedbackSes in [2, 3, 4]
    assert chosenMask == 'megaROI'
    print(f"useNewClf = {useNewClf}")


assert useNewClf == True

print(f"sub={sub}, feedbackSes={feedbackSes}, scanNum={scanNum}, runNum={runNum}")

cfg = cfg_loading(f"{sub}.ses{feedbackSes}.toml")


def doRuns(cfg, scanNum=None, runNum=None):
    def get_TrialNumber():
        currTrial = 1
        TrialNumber = 12
        TRduration = 2
        trial_list = pd.DataFrame(columns=['Trial', 'time', 'TR', 'state', 'newWobble'])
        curTime = 0
        curTR = 0
        state = ''
        trial_list = pd.concat([trial_list, pd.DataFrame({
            'Trial': None,
            'time': None,
            'TR': None,
            'state': None,
            'newWobble': None}, index=[0])], ignore_index=True)
        for currTrial in range(1, 1 + TrialNumber):

            # ITI
            ITINum = 4 if currTrial == 1 else 6
            for i in range(ITINum):  # should be 6TR=12s
                trial_list = pd.concat([trial_list, pd.DataFrame({
                    'Trial': currTrial,
                    'time': curTime,
                    'TR': curTR,
                    'state': 'ITI',
                    'newWobble': 0}, index=[0])], ignore_index=True)
                curTime = curTime + TRduration
                curTR = curTR + 1

            # waiting for metric calculation
            for i in range(3):  # should be 3TR=6s
                trial_list = pd.concat([trial_list, pd.DataFrame({
                    'Trial': currTrial,
                    'time': curTime,
                    'TR': curTR,
                    'state': 'waiting',
                    'newWobble': 1}, index=[0])], ignore_index=True)
                curTime = curTime + TRduration
                curTR = curTR + 1

            # feedback trial: try minimize the whobbling
            for i in range(5):  # 5TR=10s
                trial_list = pd.concat([trial_list, pd.DataFrame({
                    'Trial': currTrial,
                    'time': curTime,
                    'TR': curTR,
                    'state': 'feedback',
                    'newWobble': 1}, index=[0])], ignore_index=True)
                curTime = curTime + TRduration
                curTR = curTR + 1

        # ITI
        for i in range(6):  # should be 6TR=12s
            trial_list = pd.concat([trial_list, pd.DataFrame({
                'Trial': currTrial,
                'time': curTime,
                'TR': curTR,
                'state': 'ITI',
                'newWobble': 0}, index=[0])], ignore_index=True)
            curTime = curTime + TRduration
            curTime = curTime + TRduration
            curTR = curTR + 1
        return trial_list

    trial_list = get_TrialNumber()

    # variables we'll use throughout
    # scanNum = cfg.scanNum[0]
    # runNum = cfg.runNum[0]

    print(f"Doing run {runNum}, scan {scanNum}")
    print(f"cfg.dicomDir={cfg.dicomDir}")

    tmp_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/temp/{sub}/ses{feedbackSes}/run{runNum}/"

    # print(f"rmtree {tmp_dir}")
    # if os.path.exists(tmp_dir):
    #     shutil.rmtree(tmp_dir)
    mkdir(tmp_dir)

    mask = np.load(f"{cfg.chosenMask}")
    assert cfg.chosenMask == f"{cfg.subjects_dir}{cfg.subjectName}/ses1/recognition/chosenMask.npy"
    print(f"loading {cfg.chosenMask}")

    imcodeDict = {
        'A': 'bed',
        'B': 'chair',
        'C': 'table',
        'D': 'bench'}

    # next subject sub016 is batch2.
    if useNewClf:
        model_folder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/subjects/" \
                       f"{cfg.subjectName}/ses{cfg.session - 1}/megaROI/clf/"
        AB_clf = joblib.load(model_folder + 'bedbench_bedchair.joblib')
        CD_clf = joblib.load(model_folder + 'bedtable_tablebench.joblib')

        AC_clf = joblib.load(model_folder + 'bedbench_bedtable.joblib')
        AD_clf = joblib.load(model_folder + 'bedchair_bedbench.joblib')

        BC_clf = joblib.load(
            model_folder + 'benchchair_chairtable.joblib')  # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib
        BD_clf = joblib.load(
            model_folder + 'bedchair_chairbench.joblib')  # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib

        CA_clf = joblib.load(model_folder + 'benchtable_tablebed.joblib')
        CB_clf = joblib.load(model_folder + 'benchtable_tablechair.joblib')

        DA_clf = joblib.load(
            model_folder + 'benchtable_benchbed.joblib')  # benchtable_benchbed benchchair_benchbed bedtable_bedbench bedchair_bedbench
        DB_clf = joblib.load(
            model_folder + 'benchtable_benchchair.joblib')  # benchtable_benchchair bedbench_benchchair chairtable_chairbench bedchair_chairbench
    else:
        model_folder = cfg.usingModel_dir
        AB_clf = joblib.load(model_folder + 'bedbench_bedchair.joblib')
        CD_clf = joblib.load(model_folder + 'bedtable_tablebench.joblib')

        AC_clf = joblib.load(model_folder + 'bedbench_bedtable.joblib')
        AD_clf = joblib.load(model_folder + 'bedchair_bedbench.joblib')

        BC_clf = joblib.load(
            model_folder + 'benchchair_chairtable.joblib')  # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib
        BD_clf = joblib.load(
            model_folder + 'bedchair_chairbench.joblib')  # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib

        CA_clf = joblib.load(model_folder + 'benchtable_tablebed.joblib')
        CB_clf = joblib.load(model_folder + 'benchtable_tablechair.joblib')

        DA_clf = joblib.load(
            model_folder + 'benchtable_benchbed.joblib')  # benchtable_benchbed benchchair_benchbed bedtable_bedbench bedchair_bedbench
        DB_clf = joblib.load(
            model_folder + 'benchtable_benchchair.joblib')  # benchtable_benchchair bedbench_benchchair chairtable_chairbench bedchair_chairbench

    probs = []
    maskedData = 0
    megaROI_recognition_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/subjects/" \
                              f"{cfg.subjectName}/ses{cfg.session}/recognition/"
    mega_feedback_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/subjects/" \
                        f"{cfg.subjectName}/ses{cfg.session}/feedback/"
    try:
        mkdir(mega_feedback_dir)
    except:
        pass
    templateFunctionalVolume_converted = f"{megaROI_recognition_dir}/templateFunctionalVolume_converted.nii"

    dicomFilenames = glob(f"{cfg.dicomDir}/001_{str(scanNum).zfill(6)}_*.dcm")
    dicomFilenames.sort()
    print(f"scanNum={scanNum}, runNum={runNum}, len(dicomFilenames)={len(dicomFilenames)}")
    num_total_trials = 12
    num_total_TRs = min(int((num_total_trials * 28 + 12) / 2) + 8,
                        len(dicomFilenames) + 1)  # number of TRs to use for example 1
    history = pd.DataFrame()

    # dicomFilenames
    for this_TR in tqdm(np.arange(1, num_total_TRs)):
        print(f"milgramTR_ID={this_TR}")
        if this_TR > len(dicomFilenames) or this_TR > len(trial_list)-1:
            break
        else:
            dicomFilename = dicomFilenames[this_TR - 1]
        niiFileName = f"{tmp_dir}/{dicomFilename.split('/')[-1].replace('.dcm', '')}"
        convertDicomFileToNifti(dicomFilename, niiFileName)

        if os.path.exists(f"{niiFileName}_reorient.nii"):
            os.remove(f"{niiFileName}_reorient.nii")
        command = f"/gpfs/milgram/apps/hpc.rhel7/software/AFNI/2023.0.07/3dresample \
            -master {templateFunctionalVolume_converted} \
            -prefix {niiFileName}_reorient.nii \
            -input {niiFileName}.nii"
        if os.path.exists(f"{niiFileName}_reorient.nii"):
            os.remove(f"{niiFileName}_reorient.nii")
        kp_run(command)
        if not os.path.exists(f"{niiFileName}_reorient.nii"):
            raise Exception(f"3dresample failed for {niiFileName}")

        if os.path.exists(f"{niiFileName}_aligned.nii"):
            os.remove(f"{niiFileName}_aligned.nii")
        command = f"/gpfs/milgram/apps/hpc.rhel7/software/AFNI/2023.0.07/3dvolreg \
                -base {templateFunctionalVolume_converted} \
                -prefix  {niiFileName}_aligned.nii \
                {niiFileName}_reorient.nii"
        if os.path.exists(f"{niiFileName}_aligned.nii"):
            os.remove(f"{niiFileName}_aligned.nii")
        kp_run(command)
        if not os.path.exists(f"{niiFileName}_aligned.nii"):
            raise Exception(f"3dvolreg failed for {niiFileName}")

        niftiObject = nib.load(f"{niiFileName}_aligned.nii")
        nift_data = niftiObject.get_fdata()

        curr_volume = np.expand_dims(nift_data[mask == 1], axis=0)
        maskedData = curr_volume if this_TR == 1 else np.concatenate((maskedData, curr_volume), axis=0)
        _maskedData = normalize(maskedData)

        print(f"_maskedData.shape={_maskedData.shape}")

        X = np.expand_dims(_maskedData[-1], axis=0)

        def get_prob(showingImage="A", drivingTarget="B", otherAxis1="C", otherAxis2="D",
                     drivingClf1=None, drivingClf2=None,  # drivingClf1='BC_clf', drivingClf2='BD_clf', X="X"
                     _X=None):  # X is the current volume
            _Y = imcodeDict[drivingTarget]  # is chair when batch1
            print(f"classifierProb({drivingTarget}{otherAxis1}_clf,X,Y)={classifierProb(drivingClf1, _X, _Y)}")
            print(f"classifierProb({drivingTarget}{otherAxis2}_clf,X,Y)={classifierProb(drivingClf2, _X, _Y)}")
            prob1 = classifierProb(drivingClf1, _X, _Y)[0]
            prob2 = classifierProb(drivingClf2, _X, _Y)[0]
            print(f"{drivingTarget}{otherAxis1}_{drivingTarget}_prob={prob1}")
            print(f"{drivingTarget}{otherAxis1}_{drivingTarget}_prob={prob2}")
            _prob = float((prob1 + prob2) / 2)
            print(f"{drivingTarget}_prob={_prob}")
            print(f"| {drivingTarget}_prob for TR {this_TR} is {_prob}")
            return _prob

        # if cfg.batch == 'batch1':
        #     oldVersion = False
        #     if oldVersion:
        #         Y = 'chair'
        #         print(f"classifierProb(BC_clf,X,Y)={classifierProb(BC_clf, X, Y)}")
        #         print(f"classifierProb(BD_clf,X,Y)={classifierProb(BD_clf, X, Y)}")
        #         BC_B_prob = classifierProb(BC_clf, X, Y)[0]
        #         BD_B_prob = classifierProb(BD_clf, X, Y)[0]
        #         print(f"BC_B_prob={BC_B_prob}")
        #         print(f"BD_B_prob={BD_B_prob}")
        #         B_prob = float((BC_B_prob + BD_B_prob) / 2)
        #         print(f"B_prob={B_prob}")
        #         print("| B_prob for TR %d is %f" % (this_TR, B_prob))
        #         prob = B_prob
        #     else:
        #         prob = get_prob(showingImage="A", drivingTarget="B", otherAxis1="C", otherAxis2="D",
        #                         drivingClf1=BC_clf, drivingClf2=BD_clf, _X=X)  # X is the current volume
        #
        #     probs.append(prob)
        # elif cfg.batch == 'batch2':
        #     prob = get_prob(showingImage="C", drivingTarget="D", otherAxis1="A", otherAxis2="B",
        #                     drivingClf1=DA_clf, drivingClf2=DB_clf, _X=X)  # X is the current volume
        #     probs.append(prob)

        # A prob
        Aprob = get_prob(drivingTarget="A", showingImage="B", otherAxis1="C", otherAxis2="D", drivingClf1=AC_clf,
                         drivingClf2=AD_clf, _X=X)
        # B prob
        Bprob = get_prob(drivingTarget="B", showingImage="A", otherAxis1="C", otherAxis2="D", drivingClf1=BC_clf,
                         drivingClf2=BD_clf, _X=X)
        # C prob
        Cprob = get_prob(drivingTarget="C", showingImage="D", otherAxis1="A", otherAxis2="B", drivingClf1=CA_clf,
                         drivingClf2=CB_clf, _X=X)
        # D prob
        Dprob = get_prob(drivingTarget="D", showingImage="C", otherAxis1="A", otherAxis2="B", drivingClf1=DA_clf,
                         drivingClf2=DB_clf, _X=X)

        # AB_clf 的 A prob
        AB_clf_A = classifierProb(AB_clf, X, "bed")
        AB_clf_B = classifierProb(AB_clf, X, "chair")

        CD_clf_C = classifierProb(CD_clf, X, "table")
        CD_clf_D = classifierProb(CD_clf, X, "bench")

        if cfg.batch == 'batch1':
            prob = Bprob
            probs.append(prob)
        elif cfg.batch == 'batch2':
            prob = Dprob
            probs.append(prob)
        if cfg.batch == "batch1":
            history = history.append({
                'Sub': sub,
                # 'Run': run,
                # "TR_scanner":TR[0],
                "TR_milgram": this_TR,
                "Xprob": Aprob,
                "Yprob": Bprob,
                "Mprob": Cprob,
                "Nprob": Dprob,
                "XxY": Aprob * Bprob,
                "min(Xprob, Yprob)": min(Aprob, Bprob),
                "XY_clf_X": AB_clf_A,
                "XY_clf_Y": AB_clf_B,
                "MN_clf_M": CD_clf_C,
                "MN_clf_N": CD_clf_D,
                # "morphParam":morphParam,
                # "timestamp":timestamp,
                # "points":points,
                "states": trial_list.loc[this_TR - 1, 'state']
            },
                ignore_index=True)
        elif cfg.batch == "batch2":
            history = history.append({
                'Sub': sub,
                # 'Run': run,
                # "TR_scanner":TR[0],
                "TR_milgram": this_TR,
                "Xprob": Cprob,
                "Yprob": Dprob,
                "Mprob": Aprob,
                "Nprob": Bprob,
                "XxY": Cprob * Dprob,
                "min(Xprob, Yprob)": min(Cprob, Dprob),
                "XY_clf_X": CD_clf_C,
                "XY_clf_Y": CD_clf_D,
                "MN_clf_M": AB_clf_A,
                "MN_clf_N": AB_clf_B,
                # "morphParam":morphParam,
                # "timestamp":timestamp,
                # "points":points,
                "states": trial_list.loc[this_TR - 1, 'state']
            },
                ignore_index=True)

    # save probs
    if useNewClf:
        print(f'saving {mega_feedback_dir}/probs_{scanNum}_useNewClf')
        np.save(f'{mega_feedback_dir}/probs_{scanNum}_useNewClf', probs)  # save
        history.to_csv(f"{mega_feedback_dir}/history_runNum_{runNum}.csv", index=False)  # save
    else:
        print(f'saving {mega_feedback_dir}/probs_{scanNum}')
        np.save(f'{mega_feedback_dir}/probs_{scanNum}', probs)
    return


doRuns(cfg, scanNum=scanNum, runNum=runNum)
print('done')
