testMode = False
# normActivationFlag = False
# UsedTRflag = "feedback"

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
import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt
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

sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/')
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/')
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/')
from recognition_dataAnalysisFunctions import normalize, classifierProb
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
imcodeDict = {
    'A': 'bed',
    'B': 'chair',
    'C': 'table',
    'D': 'bench'}


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
    if not os.path.exists(tmpFile):
        print(f"waiting for {tmpFile}")
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


os.chdir("/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/")
if testMode:
    [sub, chosenMask, normActivationFlag, UsedTRflag] = ['sub024', 'V1_FreeSurfer', True, "feedback"]
else:
    jobarrayDict = np.load(f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
                           f"OrganizedScripts/ROI/autoAlign_ses1ses5/XYactivation/prepareData_jobID.npy",
                           allow_pickle=True)
    jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
    jobarrayID = int(float(sys.argv[1]))
    [sub, chosenMask, normActivationFlag, UsedTRflag] = jobarrayDict[jobarrayID]

tag = f"normActivationFlag_{normActivationFlag}_UsedTRflag_{UsedTRflag}"
print(f"sub={sub}, choseMask={chosenMask}")
autoAlignFlag = True  # 是否使用自动对齐的mat而不是手动对齐的mat.
print(f"tag={tag}")


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def prepareData(sub=None, ses=None, runNum=1, scanNum=3, chosenMask='', testRun_training=None):
    history_file = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis_ses1ses5/" \
                   f"XYcoactivation/{sub}/ses{ses}/{chosenMask}/" \
                   f"testRun_training{testRun_training}/simulate_history/{sub}_{runNum}_history_rtSynth_rt.csv"
    if testMode:
        print(f"waiting for {history_file}")
    wait(history_file, 10)
    history = pd.read_csv(history_file)

    def getMxN(_history):
        _history['MxN'] = _history['Mprob'] * _history['Nprob']
        return _history

    history = getMxN(history)

    def Get_modifiedStates(history):  # 希望遍历每一个TR的state, 将feedback后面的两个ITI改为trail.
        states = list(history['states'])
        modifiedStates = []
        for ii in range(len(states)):

            # 如果当前状态是ITI，且上一个状态是feedback，那么当前状态就是trail
            if [states[ii - 1], states[ii]] == ['feedback', 'ITI']:
                modifiedStates.append('trail')

            # 如果当前状态和上一个状态是ITI，且上上一个状态是feedback，那么当前状态就是trail
            elif [states[ii - 2], states[ii - 1], states[ii]] == ['feedback', 'ITI', 'ITI']:
                modifiedStates.append('trail')

            else:
                modifiedStates.append(states[ii])
        history['modifiedStates'] = modifiedStates

        # 定义当前的TR属于哪一个feedback trial.
        # 每一个trial包含了 6TR ITI + 3TR waiting + 5TR feedback + 6TR ITI
        # 每一个trial包含了 4TR ITI + 3TR waiting + 5TR feedback + 2TR trail 4TR ITI
        # 每一个run有 12 个trial也就是 5*12 = 60 feedback TR ; 2*12 = 24 trail TR
        trialID = []
        curr_trial = 0
        for ii in range(len(states)):
            if [states[ii - 1], states[ii]] == ['ITI', 'waiting']:
                curr_trial += 1
            trialID.append(curr_trial)
        history['trialID'] = trialID
        return history

    def normXY(history):
        # 之前的代码是没有进行每一个feedback run的归一化的，如果选择归一化, 那么就是用这个函数进行归一化.
        __Xprob = history['Xprob']
        __Yprob = history['Yprob']
        XYprob = history['XxY']
        MNprob = history['MxN']
        minXYprob = history['min(Xprob, Yprob)']
        __Xprob = (__Xprob - np.mean(__Xprob)) / np.std(__Xprob)
        __Yprob = (__Yprob - np.mean(__Yprob)) / np.std(__Yprob)
        XYprob = (XYprob - np.mean(XYprob)) / np.std(XYprob)
        MNprob = (MNprob - np.mean(MNprob)) / np.std(MNprob)
        minXYprob = (minXYprob - np.mean(minXYprob)) / np.std(minXYprob)
        history['Xprob'] = __Xprob
        history['Yprob'] = __Yprob
        history['XxY'] = XYprob
        history['MxN'] = MNprob
        history['min(Xprob, Yprob)'] = minXYprob

        return history

    history = Get_modifiedStates(history)

    if normActivationFlag:
        print(f"running history = normXY(history)")
        history = normXY(history)
        assert np.mean(history['MxN']) < 1e-10
        assert np.std(history['MxN']) - 1 < 1e-10
        assert np.mean(history['XxY']) < 1e-10
        assert np.std(history['XxY']) - 1 < 1e-10
    # history = float_history(history)
    # getTrialMean_feedabckTR
    # getTrialMean_ITI_TR

    if UsedTRflag == "feedback_trail":
        temp = history[
            kp_or([
                history['modifiedStates'] == 'feedback',
                history['modifiedStates'] == 'trail'  # 这里说明我使用的是 feedback 和 trail的TR
            ])
        ]
    elif UsedTRflag == "feedback":
        temp = history[
            history['modifiedStates'] == 'feedback'
            ]
    elif UsedTRflag == "all":
        temp = history
    else:
        raise Exception(f"UsedTRFlag={UsedTRflag} is not supported")

    _Xprobs = temp['Xprob'].mean()
    _Yprobs = temp['Yprob'].mean()
    _XxYs = temp['XxY'].mean()
    _MxNs = temp['MxN'].mean()
    _min_Xprobs_Yprobs = temp['min(Xprob, Yprob)'].mean()
    _Number_of_feedback_TR = len(temp)
    if testMode:
        print(f"Xprobs={_Xprobs}")
        print(f"Yprobs={_Yprobs}")
        print(f"XxYs={_XxYs}")
        print(f"min_Xprobs_Yprobs={_min_Xprobs_Yprobs}")
        print(f"Number_of_feedback_TR={_Number_of_feedback_TR}")
    print(f"Xprobs={_Xprobs}, "
          f"Yprobs={_Yprobs}, "
          f"XxYs={_XxYs}, "
          f"MxNs={_MxNs}, "
          f"min_Xprobs_Yprobs={_min_Xprobs_Yprobs}, "
          f"Number_of_feedback_TR={_Number_of_feedback_TR}")

    # np.save(f"{history_dir}/XYcoactivation_runNum{runNum}.npy",
    #         [Xprobs, Yprobs, XxYs, min_Xprobs_Yprobs, Number_of_feedback_TR])
    return _Xprobs, _Yprobs, _XxYs, _MxNs, _min_Xprobs_Yprobs, _Number_of_feedback_TR


allResults = pd.DataFrame()
for ses in [2, 3, 4]:
    runRecording = pd.read_csv(
        f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/"
        f"{sub}/ses{ses}/runRecording.csv")
    feedback_runRecording = runRecording[runRecording['type'] == 'feedback'].reset_index()
    print(f"ses{ses} len(feedback_runRecording)={len(feedback_runRecording)} ")
    for testRun_training in range(1, 9):
        for currFeedbackRun in range(len(feedback_runRecording)):
            runNum = currFeedbackRun + 1  # feedback run ID (not scan ID)
            scanNum = feedback_runRecording.loc[currFeedbackRun, 'run']  # scan ID according to runRecording.csv
            print(f"scanNum={scanNum}")
            [Xprobs, Yprobs, XxYs, MxNs, min_Xprobs_Yprobs, Number_of_feedback_TR] = prepareData(
                sub=sub, ses=ses,
                runNum=runNum,
                scanNum=scanNum,
                chosenMask=chosenMask,
                testRun_training=testRun_training)
            allResults = pd.concat([allResults, pd.DataFrame({
                'sub': [sub],
                'ses': [ses],
                'testRun_training': [testRun_training],
                'currFeedbackRun': [currFeedbackRun],
                'runNum': [runNum],
                'scanNum': [scanNum],
                'Xprobs': [Xprobs * Number_of_feedback_TR],
                'Yprobs': [Yprobs * Number_of_feedback_TR],
                'XxYs': [XxYs * Number_of_feedback_TR],
                'MxNs': [MxNs * Number_of_feedback_TR],
                'min_Xprobs_Yprobs': [min_Xprobs_Yprobs * Number_of_feedback_TR],
                'Number_of_feedback_TR': [Number_of_feedback_TR]
            }, index=[0])], ignore_index=True)

save_folder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis_ses1ses5/" \
              f"XYcoactivation/prepareData/{sub}/{chosenMask}/" \
              f"/{tag}/"
mkdir(save_folder)
allResults.to_csv(f"{save_folder}/{sub}_{chosenMask}_allResults.csv")

print("done")


#
# if testMode:
#     testRun_training = 1
#     currFeedbackRun = 2
#     runNum = currFeedbackRun + 1  # feedback run ID (not scan ID)
#     scanNum = feedback_runRecording.loc[currFeedbackRun, 'run']  # scan ID according to runRecording.csv
#     print(f"scanNum={scanNum}")
#     rtSynth_rt_ABCD_ROIanalysis_unwarp_ses1ses5(sub=sub, ses=ses,
#                                                 runNum=runNum, scanNum=scanNum,
#                                                 chosenMask=chosenMask,
#                                                 testRun_training=testRun_training)
#     prepareData(sub=sub, ses=ses,
#                 runNum=runNum, scanNum=scanNum,
#                 chosenMask=chosenMask,
#                 testRun_training=testRun_training)
# else:
#     for testRun_training in tqdm(range(1, 9)):
#         for currFeedbackRun in range(len(feedback_runRecording)):
#             runNum = currFeedbackRun + 1  # feedback run ID (not scan ID)
#             scanNum = feedback_runRecording.loc[currFeedbackRun, 'run']  # scan ID according to runRecording.csv
#             print(f"scanNum={scanNum}")
#             if normActivationFlag and UsedTRflag == 'feedback':
#                 rtSynth_rt_ABCD_ROIanalysis_unwarp_ses1ses5(sub=sub, ses=ses,
#                                                             runNum=runNum, scanNum=scanNum,
#                                                             chosenMask=chosenMask,
#                                                             testRun_training=testRun_training)
#     allResults = pd.DataFrame()
#     for testRun_training in tqdm(range(1, 9)):
#         for currFeedbackRun in range(len(feedback_runRecording)):
#             runNum = currFeedbackRun + 1  # feedback run ID (not scan ID)
#             scanNum = feedback_runRecording.loc[currFeedbackRun, 'run']  # scan ID according to runRecording.csv
#             print(f"scanNum={scanNum}")
#             [Xprobs, Yprobs, XxYs, min_Xprobs_Yprobs, Number_of_feedback_TR] = prepareData(
#                 sub=sub, ses=ses,
#                 runNum=runNum,
#                 scanNum=scanNum,
#                 chosenMask=chosenMask,
#                 testRun_training=testRun_training)
#             allResults = pd.concat([allResults, pd.DataFrame({
#                 'sub': [sub],
#                 'ses': [ses],
#                 'testRun_training': [testRun_training],
#                 'currFeedbackRun': [currFeedbackRun],
#                 'runNum': [runNum],
#                 'scanNum': [scanNum],
#                 'Xprobs': [Xprobs],
#                 'Yprobs': [Yprobs],
#                 'XxYs': [XxYs],
#                 'min_Xprobs_Yprobs': [min_Xprobs_Yprobs],
#                 'Number_of_feedback_TR': [Number_of_feedback_TR]
#             }, index=[0])], ignore_index=True)
#
#     save_folder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis_ses1ses5/" \
#                   f"XYcoactivation/{sub}/ses{ses}/{chosenMask}/" \
#                   f"/{tag}/"
#     mkdir(save_folder)
#     allResults.to_csv(f"{save_folder}/{sub}_{ses}_{chosenMask}_allResults.csv")
#

# def rtSynth_rt_ABCD_ROIanalysis_unwarp_ses1ses5(sub=None, ses=None, runNum=1, scanNum=3, chosenMask='',
#                                                 testRun_training=None):
#     cfg = cfg_loading(f"{sub}.ses{ses}.toml")
#     assert ses in [2, 3, 4]
#
#     def get_TrialNumber():
#         TrialNumber = 12
#         TRduration = 2
#         trial_list = pd.DataFrame(columns=['Trial', 'time', 'TR', 'state', 'newWobble'])
#         curTime = 0
#         curTR = 0
#         state = ''
#         # trial_list.append({'Trial': None,
#         #                    'time': None,
#         #                    'TR': None,
#         #                    'state': None,
#         #                    'newWobble': None},
#         #                   ignore_index=True)
#         trial_list = pd.concat([trial_list, pd.DataFrame({'Trial': None,
#                                                           'time': None,
#                                                           'TR': None,
#                                                           'state': None,
#                                                           'newWobble': None}, index=[0])],
#                                ignore_index=True)
#
#         for currTrial in range(1, 1 + TrialNumber):
#
#             # ITI
#             ITINum = 4 if currTrial == 1 else 6
#             for i in range(ITINum):  # should be 6TR=12s
#                 # trial_list = trial_list.append({'Trial': currTrial,
#                 #                                 'time': curTime,
#                 #                                 'TR': curTR,
#                 #                                 'state': 'ITI',
#                 #                                 'newWobble': 0},
#                 #                                ignore_index=True)
#                 trial_list = pd.concat([trial_list, pd.DataFrame({'Trial': currTrial,
#                                                                   'time': curTime,
#                                                                   'TR': curTR,
#                                                                   'state': 'ITI',
#                                                                   'newWobble': 0}, index=[0])],
#                                        ignore_index=True)
#                 curTime = curTime + TRduration
#                 curTR = curTR + 1
#
#             # waiting for metric calculation
#             for i in range(3):  # should be 3TR=6s
#                 # trial_list = trial_list.append({'Trial': currTrial,
#                 #                                 'time': curTime,
#                 #                                 'TR': curTR,
#                 #                                 'state': 'waiting',
#                 #                                 'newWobble': 1},
#                 #                                ignore_index=True)
#                 trial_list = pd.concat([trial_list, pd.DataFrame({'Trial': currTrial,
#                                                                   'time': curTime,
#                                                                   'TR': curTR,
#                                                                   'state': 'waiting',
#                                                                   'newWobble': 1}, index=[0])],
#                                        ignore_index=True)
#                 curTime = curTime + TRduration
#                 curTR = curTR + 1
#
#             # feedback trial: try minimize the whobbling
#             for i in range(5):  # 5TR=10s
#                 # trial_list = trial_list.append({'Trial': currTrial,
#                 #                                 'time': curTime,
#                 #                                 'TR': curTR,
#                 #                                 'state': 'feedback',
#                 #                                 'newWobble': 1},
#                 #                                ignore_index=True)
#                 trial_list = pd.concat([trial_list, pd.DataFrame({'Trial': currTrial,
#                                                                   'time': curTime,
#                                                                   'TR': curTR,
#                                                                   'state': 'feedback',
#                                                                   'newWobble': 1}, index=[0])],
#                                        ignore_index=True)
#                 curTime = curTime + TRduration
#                 curTR = curTR + 1
#
#         # ITI
#         for i in range(6):  # should be 6TR=12s
#             # trial_list = trial_list.append({'Trial': currTrial,
#             #                                 'time': curTime,
#             #                                 'TR': curTR,
#             #                                 'state': 'ITI',
#             #                                 'newWobble': 0},
#             #                                ignore_index=True)
#             trial_list = pd.concat([trial_list, pd.DataFrame({'Trial': currTrial,
#                                                               'time': curTime,
#                                                               'TR': curTR,
#                                                               'state': 'ITI',
#                                                               'newWobble': 0}, index=[0])],
#                                    ignore_index=True)
#
#             curTime = curTime + TRduration
#             curTR = curTR + 1
#         return trial_list
#
#     trial_list = get_TrialNumber()
#
#     def loadMask(sub=None, chosenMask=None):
#         subjectFolder = "/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/"
#         if autoAlignFlag:
#             maskFolder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/" \
#                          f"autoAlignMask/{sub}/func/"
#         else:
#             maskFolder = f"{subjectFolder}/{sub}/ses1/recognition/mask/"
#         print(f"maskFolder={maskFolder}")
#         try:
#             mask = nib.load(f"{maskFolder}/{chosenMask}.nii").get_data()
#             print(f"loading {maskFolder}/{chosenMask}.nii")
#         except:
#             mask = nib.load(f"{maskFolder}/{chosenMask}.nii.gz").get_data()
#             print(f"loading {maskFolder}/{chosenMask}.nii.gz")
#         return mask
#
#     mask = loadMask(sub=sub, chosenMask=chosenMask)
#
#     if autoAlignFlag:
#         # autoAlign_ROIFolder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/" \
#         #                       f"autoAlign_ROIanalysis_ses1ses5/" \
#         #                       f"subjects/{sub}/ses1/{chosenMask}/"
#         model_folder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/" \
#                        f"autoAlign_ROIanalysis_ses1ses5/subjects/{sub}/ses1/{chosenMask}/clf/"
#     else:
#         raise Exception("not supported")
#     clfDict = {
#         "AB": 'bedtable_bedchair',
#         "AC": 'bedbench_bedtable',
#         "AD": 'bedchair_bedbench',
#         "BC": 'benchchair_chairtable',
#         "BD": 'bedchair_chairbench',
#         "CD": 'bedtable_tablebench'
#     }
#     # model_folder = f"{cfg.usingModel_dir}/../ROI_analysis/{chosenMask}/clf/"
#     AB_clf = joblib.load(f"{model_folder}/{clfDict['AB']}_testRun{testRun_training}.joblib")
#     CD_clf = joblib.load(f"{model_folder}/{clfDict['CD']}_testRun{testRun_training}.joblib")
#     AC_clf = joblib.load(f"{model_folder}/{clfDict['AC']}_testRun{testRun_training}.joblib")
#     AD_clf = joblib.load(f"{model_folder}/{clfDict['AD']}_testRun{testRun_training}.joblib")
#     BC_clf = joblib.load(f"{model_folder}/{clfDict['BC']}_testRun{testRun_training}.joblib")
#     BD_clf = joblib.load(f"{model_folder}/{clfDict['BD']}_testRun{testRun_training}.joblib")
#
#     num_total_trials = 12
#     num_total_TRs = int((num_total_trials * 28 + 12) / 2) + 8  # number of TRs to use for example 1
#     print(f"num_total_TRs={num_total_TRs}")
#
#     _alignedData = nib.load(f"{cfg.feedback_dir}/run{scanNum}.nii.gz").get_data()
#
#     history = pd.DataFrame()
#     _alignedData = nib.load(f"{cfg.feedback_dir}/run{scanNum}.nii.gz").get_data()  # 我找不到这个是从哪里来的. 是否是经过了unwarp的处理?
#     _alignedData = np.transpose(_alignedData, [3, 0, 1, 2])
#     print(f"_alignedData.shape={_alignedData.shape}")  # (176, 64, 64, 36)
#     if len(_alignedData) == 145:  # 其实应该是145的长度，因为sub014 ses4 的 scan 16错误地使用了recognition run 的sequence
#         num_total_TRs = 1 + len(_alignedData)  # 177
#     elif len(_alignedData) >= 176:  # 这是因为我在batch1的时候
#         num_total_TRs = len(trial_list)  # 173
#     else:
#         raise Exception(f"len(_alignedData)={len(_alignedData)}")
#     # num_total_TRs=1+len(_alignedData) #177
#     # num_total_TRs=len(trial_list) #173
#     print(f"new num_total_TRs={num_total_TRs}")
#     history_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis_ses1ses5/" \
#                   f"XYcoactivation/{cfg.subjectName}/ses{cfg.session}/{chosenMask}/" \
#                   f"testRun_training{testRun_training}/simulate_history/"
#     mkdir(history_dir)
#     history_file = history_dir + f"{sub}_{runNum}_history_rtSynth_rt.csv"
#     print(history_file)
#     if os.path.exists(history_file):
#         os.remove(history_file)
#
#     for this_TR in tqdm(np.arange(1, 1 + num_total_TRs)):  # 1,177
#         # print(f"milgramTR_ID={this_TR}")
#
#         alignedData = _alignedData.copy()
#         maskedData = alignedData[0:this_TR, mask == 1]
#         # curr_volume = np.expand_dims(nift_data[mask==1], axis=0)
#         # maskedData=curr_volume if this_TR==1 else np.concatenate((maskedData,curr_volume),axis=0)
#         _maskedData = normalize(maskedData)
#         if testMode:
#             print(f"_maskedData.shape={_maskedData.shape}")
#
#         X = np.expand_dims(_maskedData[-1], axis=0)
#
#         def get_prob(drivingTarget="B", showingImage="A", otherAxis1="C", otherAxis2="D",
#                      drivingClf1='BC_clf', drivingClf2='BD_clf',
#                      X=None):  # X is the current volume
#             Y = imcodeDict[drivingTarget]  # is chair when batch1
#             if testMode:
#                 print(f"classifierProb({drivingTarget}{otherAxis1}_clf,X,Y)={classifierProb(drivingClf1, X, Y)}")
#                 print(f"classifierProb({drivingTarget}{otherAxis2}_clf,X,Y)={classifierProb(drivingClf2, X, Y)}")
#             prob1 = classifierProb(drivingClf1, X, Y)[0]
#             prob2 = classifierProb(drivingClf2, X, Y)[0]
#             if testMode:
#                 print(f"{drivingTarget}{otherAxis1}_{drivingTarget}_prob={prob1}")
#                 print(f"{drivingTarget}{otherAxis1}_{drivingTarget}_prob={prob2}")
#             prob = float((prob1 + prob2) / 2)
#             if testMode:
#                 print(f"{drivingTarget}_prob={prob}")
#                 print(f"| {drivingTarget}_prob for TR {this_TR} is {prob}")
#             return prob
#
#         # if cfg.batch=='batch1':
#         #     oldVersion=False
#         #     if oldVersion==True:
#         #         Y = 'chair'
#         #         print(f"classifierProb(BC_clf,X,Y)={classifierProb(BC_clf,X,Y)}")
#         #         print(f"classifierProb(BD_clf,X,Y)={classifierProb(BD_clf,X,Y)}")
#         #         BC_B_prob = classifierProb(BC_clf,X,Y)[0]
#         #         BD_B_prob = classifierProb(BD_clf,X,Y)[0]
#         #         print(f"BC_B_prob={BC_B_prob}")
#         #         print(f"BD_B_prob={BD_B_prob}")
#         #         B_prob = float((BC_B_prob+BD_B_prob)/2)
#         #         print(f"B_prob={B_prob}")
#         #         print("| B_prob for TR %d is %f" %(this_TR, B_prob))
#         #         prob = B_prob
#         #     else:
#         #         prob = get_prob(showingImage="A", drivingTarget="B", otherAxis1="C", otherAxis2="D", drivingClf1=BC_clf, drivingClf2=BD_clf, X=X) # X is the current volume
#
#         #     probs.append(prob)
#         # elif cfg.batch=='batch2':
#         #     prob = get_prob(showingImage="C", drivingTarget="D", otherAxis1="A", otherAxis2="B", drivingClf1=DA_clf, drivingClf2=DB_clf, X=X) # X is the current volume
#
#         # A prob
#         Aprob = get_prob(drivingTarget="A", showingImage="B", otherAxis1="C", otherAxis2="D", drivingClf1=AC_clf,
#                          drivingClf2=AD_clf, X=X)
#         # B prob
#         Bprob = get_prob(drivingTarget="B", showingImage="A", otherAxis1="C", otherAxis2="D", drivingClf1=BC_clf,
#                          drivingClf2=BD_clf, X=X)
#         # C prob
#         Cprob = get_prob(drivingTarget="C", showingImage="D", otherAxis1="A", otherAxis2="B", drivingClf1=AC_clf,
#                          drivingClf2=BC_clf, X=X)
#         # D prob
#         Dprob = get_prob(drivingTarget="D", showingImage="C", otherAxis1="A", otherAxis2="B", drivingClf1=AD_clf,
#                          drivingClf2=BD_clf, X=X)
#
#         # AB_clf 的 A prob
#
#         AB_clf_A = classifierProb(AB_clf, X, "bed")
#         AB_clf_B = classifierProb(AB_clf, X, "chair")
#
#         CD_clf_C = classifierProb(CD_clf, X, "table")
#         CD_clf_D = classifierProb(CD_clf, X, "bench")
#
#         # 在每一个TR来的时候都要保存history 中文
#         # history = history.append({
#         #         'Sub': sub,
#         #         # 'Run': run,
#         #         # "TR_scanner":TR[0],
#         #         "TR_milgram":this_TR,
#         #         "Aprob":Aprob,
#         #         "Bprob":Bprob,
#         #         "Cprob":Cprob,
#         #         "Dprob":Dprob,
#         #         "AB_clf_A":AB_clf_A,
#         #         "AB_clf_B":AB_clf_B,
#         #         "CD_clf_C":CD_clf_C,
#         #         "CD_clf_D":CD_clf_D,
#         #         # "morphParam":morphParam,
#         #         # "timestamp":timestamp,
#         #         # "points":points,
#         #         "states":trial_list.loc[this_TR-1,'state']
#         #     },
#         #     ignore_index=True)
#         if cfg.batch == "batch1":
#             # history = history.append({
#             #     'Sub': sub,
#             #     # 'Run': run,
#             #     # "TR_scanner":TR[0],
#             #     "TR_milgram": this_TR,
#             #     "Xprob": Aprob,
#             #     "Yprob": Bprob,
#             #     "Mprob": Cprob,
#             #     "Nprob": Dprob,
#             #     "XY_clf_X": AB_clf_A,
#             #     "XY_clf_Y": AB_clf_B,
#             #     "MN_clf_M": CD_clf_C,
#             #     "MN_clf_N": CD_clf_D,
#             #     # "morphParam":morphParam,
#             #     # "timestamp":timestamp,
#             #     # "points":points,
#             #     "states": trial_list.loc[this_TR - 1, 'state']
#             # },
#             #     ignore_index=True)
#             history = pd.concat([history, pd.DataFrame({
#                 'Sub': sub,
#                 # 'Run': run,
#                 # "TR_scanner":TR[0],
#                 "TR_milgram": this_TR,
#                 "Xprob": Aprob,
#                 "Yprob": Bprob,
#                 "Mprob": Cprob,
#                 "Nprob": Dprob,
#                 "XxY": Aprob * Bprob,
#                 "min(Xprob, Yprob)": min(Aprob, Bprob),
#                 "XY_clf_X": AB_clf_A,
#                 "XY_clf_Y": AB_clf_B,
#                 "MN_clf_M": CD_clf_C,
#                 "MN_clf_N": CD_clf_D,
#                 # "morphParam":morphParam,
#                 # "timestamp":timestamp,
#                 # "points":points,
#                 "states": trial_list.loc[this_TR - 1, 'state']
#             }, index=[0])], ignore_index=True)
#
#         elif cfg.batch == "batch2":
#             # history = history.append({
#             #     'Sub': sub,
#             #     # 'Run': run,
#             #     # "TR_scanner":TR[0],
#             #     "TR_milgram": this_TR,
#             #     "Xprob": Cprob,
#             #     "Yprob": Dprob,
#             #     "Mprob": Aprob,
#             #     "Nprob": Bprob,
#             #     "XY_clf_X": CD_clf_C,
#             #     "XY_clf_Y": CD_clf_D,
#             #     "MN_clf_M": AB_clf_A,
#             #     "MN_clf_N": AB_clf_B,
#             #     # "morphParam":morphParam,
#             #     # "timestamp":timestamp,
#             #     # "points":points,
#             #     "states": trial_list.loc[this_TR - 1, 'state']
#             # },
#             #     ignore_index=True)
#             history = pd.concat([history, pd.DataFrame({
#                 'Sub': sub,
#                 # 'Run': run,
#                 # "TR_scanner":TR[0],
#                 "TR_milgram": this_TR,
#                 "Xprob": Cprob,
#                 "Yprob": Dprob,
#                 "Mprob": Aprob,
#                 "Nprob": Bprob,
#                 "XxY": Cprob * Dprob,
#                 "min(Xprob, Yprob)": min(Cprob, Dprob),
#                 "XY_clf_X": CD_clf_C,
#                 "XY_clf_Y": CD_clf_D,
#                 "MN_clf_M": AB_clf_A,
#                 "MN_clf_N": AB_clf_B,
#                 # "morphParam":morphParam,
#                 # "timestamp":timestamp,
#                 # "points":points,
#                 "states": trial_list.loc[this_TR - 1, 'state']
#             }, index=[0])], ignore_index=True)
#
#         # print("saving")
#
#         # history_file = history_dir + f"{sub}_{runNum}_history_rtSynth_rt.csv"
#         history.to_csv(history_file)
#         print(f"history saved to {history_file}")
#

# if autoAlignFlag:
#     autoAlign_ROIFolder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/" \
#                           f"autoAlign_ROIanalysis_ses1ses5/subjects/{sub}/ses{ses}/{chosenMask}/"
#     mkdir(autoAlign_ROIFolder)
#     # np.save(f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis_ses1ses5/"
#     #         f"XYcoactivation/{cfg.subjectName}/ses{cfg.session}/{chosenMask}/"
#     #         f"testRun_training{testRun_training}/{tag}/XYcoactivation_runNum{runNum}.npy",
#     #         [Xprobs, Yprobs, XxYs, min_Xprobs_Yprobs, Number_of_feedback_TR])
