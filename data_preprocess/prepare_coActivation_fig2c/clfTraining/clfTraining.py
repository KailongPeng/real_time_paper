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

testMode = False

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


os.chdir("/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/")
# save_obj(SLURM_ARRAY_TASK_ID_dict,
#                  f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
#                  f"OrganizedScripts/ROI/autoAlign/clfTraining/clfTraining_ID_dict")
jobarrayDict = np.load(f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
                       f"OrganizedScripts/megaROI/withinSession/autoAlign/clfTraining/clfTraining_ID_dict.npy",
                       allow_pickle=True)
jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
jobarrayID = int(float(sys.argv[1]))
[sub, chosenMask, ses] = jobarrayDict[jobarrayID]
# [sub, ses, chosenMask] = ['sub024', 1, 'V1_FreeSurfer']
print(f"sub={sub}, ses={ses}, choseMask={chosenMask}")
assert chosenMask == "megaROI"
autoAlignFlag = True  # 是否使用自动对齐的mat而不是手动对齐的mat.


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if autoAlignFlag:
    megaROI_subSes_folder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/" \
                          f"subjects/{sub}/ses{ses}/{chosenMask}/"
    mkdir(megaROI_subSes_folder)


def minimalClass_ROI(sub='', ses=None,
                     chosenMask=''):  # 这个函数的目的是对于给定的被试, ses和ROI, 计算出 当前ses{ses} 加上紧邻前一个ses的8个run的模型, 同时还顺便计算了一下 留一run训练测试性能.
    # 对于 ses1/2/3, 可以使用保存的模型来计算 ses2/3/4 的测试性能 以及feedback的时候的Y的激活强度.
    # 对于 ses1 和ses5 的留一run训练测试性能结果, 可以用来计算  ROI ses1ses5 模式的分化整合程度.
    cfg = f"{sub}.ses{ses}.toml"
    cfg = cfg_loading(cfg)

    scanList = []
    files = glob(f"{cfg.dicom_dir}/*.dcm")
    for file in files:
        scanList.append(int(file.split(f"{cfg.subjectIDforxnat}/001_")[1].split('_')[0]))
    runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
    assert len(np.unique(np.asarray(scanList))) == len(runRecording)
    actualRuns = list(runRecording['run'].iloc[list(
        np.where(1 == 1 * (runRecording['type'] == 'recognition'))[0])])  # can be [1,2,3,4,5,6,7,8] or [1,2,4,5]

    if len(actualRuns) < 8:
        runRecording_preDay = pd.read_csv(
            f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session - 1}/recognition/../runRecording.csv")
        actualRuns_preDay = list(runRecording_preDay['run'].iloc[
                                     list(np.where(1 == 1 * (runRecording_preDay['type'] == 'recognition'))[0])])[
                            -(8 - len(actualRuns)):]  # might be [5,6,7,8]
    else:
        actualRuns_preDay = []
    assert len(actualRuns_preDay) + len(actualRuns) == 8

    objects = ['bed', 'bench', 'chair', 'table']

    new_run_indexs = []
    new_run_index = 1  # 使用新的run 的index，以便于后面的testRun selection的时候不会重复。正常的话 new_run_index 应该是1，2，3，4，5，6，7，8

    print(f"actualRuns={actualRuns}")
    assert (len(actualRuns) >= 4)
    print(f"actualRuns_preDay={actualRuns_preDay}")

    subjectFolder = "/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/"
    # maskFolder = cfg.chosenMask
    # assert maskFolder == f"{cfg.subjects_dir}{cfg.subjectName}/ses1/recognition/chosenMask.npy"
    # # if autoAlignFlag:
    # #     maskFolder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlignMask/{sub}/func/"
    # # else:
    # #     maskFolder = f"{subjectFolder}/{sub}/ses1/recognition/mask/"
    # print(f"maskFolder={maskFolder}")

    brain_data = None
    behav_data = None
    for ii, run in enumerate(actualRuns):  # load behavior and brain data for current session
        # t = np.load(f"{cfg.recognition_dir}brain_run{run}.npy")
        t = np.load(f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/"
                    f"megaROI_main/subjects/{sub}/ses{ses}/recognition/"
                    f"brain_run{run}.npy")
        # print(f"loading {maskFolder}/{chosenMask}")
        # mask = np.load(f"{maskFolder}/{chosenMask}")

        # maskPath = f"{maskFolder}/{chosenMask}.nii"
        maskPath = cfg.chosenMask  # cfg.chosenMask = f"{cfg.subjects_dir}{cfg.subjectName}/ses1/recognition/chosenMask.npy"
        assert maskPath == f"{cfg.subjects_dir}{cfg.subjectName}/ses1/recognition/chosenMask.npy"
        mask = np.load(maskPath)
        # try:
        #     mask = nib.load(maskPath).get_fdata()
        #     print(f"loading {maskFolder}/{chosenMask}.nii , mask size = {np.sum(mask)}")
        # except:
        #     mask = nib.load(f"{maskPath}.gz").get_fdata()
        #     print(f"loading {maskFolder}/{chosenMask}.nii.gz , mask size = {np.sum(mask)}")
        print(f"mask.shape={mask.shape}")

        t = t[:, mask == 1]
        t = normalize(t)
        brain_data = t if ii == 0 else np.concatenate((brain_data, t), axis=0)

        # t = pd.read_csv(f"{cfg.recognition_dir}behav_run{run}.csv")
        t = pd.read_csv(f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/"
                        f"megaROI_main/subjects/{sub}/ses{ses}/recognition/"
                        f"behav_run{run}.csv")
        t['run_num'] = new_run_index
        new_run_indexs.append(new_run_index)
        new_run_index += 1
        behav_data = t if ii == 0 else pd.concat([behav_data, t])

    for ii, run in enumerate(actualRuns_preDay):  # load behavior and brain data for previous session
        # t = np.load(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session - 1}/recognition/brain_run{run}.npy")
        t = np.load(f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/"
                    f"megaROI_main/subjects/{sub}/ses{ses-1}/recognition/"
                    f"brain_run{run}.npy")
        # print(f"loading {maskFolder}/{chosenMask}")
        # mask = np.load(f"{maskFolder}/{chosenMask}")
        # try:
        #     mask = nib.load(f"{maskFolder}/{chosenMask}.nii").get_fdata()
        #     print(f"loading {maskFolder}/{chosenMask}.nii , mask size = {np.sum(mask)}")
        # except:
        #     mask = nib.load(f"{maskFolder}/{chosenMask}.nii.gz").get_fdata()
        #     print(f"loading {maskFolder}/{chosenMask}.nii.gz , mask size = {np.sum(mask)}")

        maskPath = cfg.chosenMask  # cfg.chosenMask = f"{cfg.subjects_dir}{cfg.subjectName}/ses1/recognition/chosenMask.npy"
        assert maskPath == f"{cfg.subjects_dir}{cfg.subjectName}/ses1/recognition/chosenMask.npy"
        mask = np.load(maskPath)

        t = t[:, mask == 1]
        t = normalize(t)
        brain_data = np.concatenate((brain_data, t), axis=0)

        # t = pd.read_csv(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session - 1}/recognition/behav_run{run}.csv")
        t = pd.read_csv(f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/"
                        f"megaROI_main/subjects/{sub}/ses{ses-1}/recognition/"
                        f"behav_run{run}.csv")
        t['run_num'] = new_run_index
        new_run_indexs.append(new_run_index)
        new_run_index += 1
        behav_data = pd.concat([behav_data, t])

    print(f"brain_data.shape={brain_data.shape}")
    assert len(brain_data.shape) == 2

    # convert item colume to label colume
    imcodeDict = {
        'A': 'bed',
        'B': 'chair',
        'C': 'table',
        'D': 'bench'}
    label = []
    for curr_trial in range(behav_data.shape[0]):
        label.append(imcodeDict[behav_data['Item'].iloc[curr_trial]])
    behav_data['label'] = label  # merge the label column with the data dataframe

    accTable = pd.DataFrame()

    def train4wayClf(META, FEAT, accTable):
        runList = np.unique(list(META['run_num']))
        print(f"runList={runList}")
        accList = {}
        for testRun in runList:
            trainIX = META['run_num'] != int(testRun)
            testIX = META['run_num'] == int(testRun)

            # pull training and test data
            trainX = FEAT[trainIX]
            testX = FEAT[testIX]
            trainY = META.iloc[np.asarray(trainIX)].label
            testY = META.iloc[np.asarray(testIX)].label

            # Train your classifier
            clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', max_iter=1000,
                                     multi_class='multinomial').fit(trainX, trainY)

            # Monitor progress by printing accuracy (only useful if you're running a test set)
            acc = clf.score(testX, testY)
            print("acc=", acc)
            accList[testRun] = acc

            Fourway_acc = acc
            accTable = pd.concat([accTable, pd.DataFrame(
                {'testRun': [testRun], 'Fourway_acc': [Fourway_acc]})], ignore_index=True)
            #
            # accTable = accTable.append({
            #     'testRun': testRun,
            #     'Fourway_acc': Fourway_acc},
            #     ignore_index=True)

        print(f"new trained full rotation 4 way accuracy mean={np.mean(list(accList.values()))}")
        return accList, accTable

    accList, accTable = train4wayClf(behav_data, brain_data, accTable)

    if autoAlignFlag:

        accTable.to_csv(f"{megaROI_subSes_folder}/new_trained_full_rotation_4_way_accuracy.csv")
    else:
        mkdir(f"{cfg.recognition_dir}/ROI_analysis/")
        mkdir(f"{cfg.recognition_dir}/ROI_analysis/{chosenMask}/")
        accTable.to_csv(f"{cfg.recognition_dir}/ROI_analysis/{chosenMask}/new_trained_full_rotation_4_way_accuracy.csv")

    # 获得full rotation的2way clf的accuracy 平均值
    accs_rotation = []
    accs_rotation_df = pd.DataFrame()
    print(f"new_run_indexs={new_run_indexs}")
    for curr_testRun, testRun in enumerate(new_run_indexs):
        allpairs = itertools.combinations(objects, 2)
        accs = {}
        # Iterate over all the possible target pairs of objects
        for pair in allpairs:
            # Find the control (remaining) objects for this pair
            altpair = other(pair)
            for obj in pair:
                # foil = [i for i in pair if i != obj][0]
                for altobj in altpair:
                    # establish a naming convention where it is $TARGET_$CLASSIFICATION
                    # Target is the NF pair (e.g. bed/bench)
                    # Classificationis is btw one of the targets, and a control (e.g. bed/chair, or bed/table, NOT bed/bench)
                    naming = '{}{}_{}{}'.format(pair[0], pair[1], obj, altobj)

                    if testRun:  # behav_data, brain_data
                        trainIX = ((behav_data['label'] == obj) | (behav_data['label'] == altobj)) & (
                                behav_data['run_num'] != int(testRun))
                        testIX = ((behav_data['label'] == obj) | (behav_data['label'] == altobj)) & (
                                behav_data['run_num'] == int(testRun))
                    else:
                        trainIX = ((behav_data['label'] == obj) | (behav_data['label'] == altobj))
                        testIX = ((behav_data['label'] == obj) | (behav_data['label'] == altobj))

                    # pull training and test data
                    trainX = brain_data[trainIX]
                    testX = brain_data[testIX]
                    trainY = behav_data.iloc[np.asarray(trainIX)].label
                    testY = behav_data.iloc[np.asarray(testIX)].label

                    assert len(np.unique(trainY)) == 2

                    # Train your classifier
                    clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', max_iter=1000,
                                             multi_class='multinomial').fit(trainX, trainY)

                    # model_folder = cfg.trainingModel_dir

                    # Monitor progress by printing accuracy (only useful if you're running a test set)
                    acc = clf.score(testX, testY)
                    # print(naming, acc)
                    accs[naming] = acc

                    imcodeDict = {
                        'A': 'bed',
                        'B': 'chair',
                        'C': 'table',
                        'D': 'bench'}
                    codeImDict = {
                        'bed': 'A',
                        'chair': 'B',
                        'table': 'C',
                        'bench': 'D'}
                    accs_rotation_df = pd.concat([accs_rotation_df, pd.DataFrame(
                        {'subject': [cfg.subjectName],
                         'curr_testRun': [curr_testRun],
                         'testRun': [testRun],
                         'pair': [pair],
                         'obj': [obj],
                         'altobj': [altobj],
                         'axis': [codeImDict[obj] + codeImDict[altobj]],
                         'acc': [acc]})], ignore_index=True)

        for TwoWay_clf in ["AB", "CD", "AC", "AD", "BC", "BD"]:
            accTable.loc[curr_testRun, TwoWay_clf + '_acc'] = accs[cfg.twoWayClfDict[TwoWay_clf][0]]

        print(f"testRun = {testRun} : average 2 way clf accuracy={np.mean(list(accs.values()))}")
        accs_rotation.append(np.mean(list(accs.values())))
    print(f"accTable={accTable}")
    print(f"mean of 2 way clf acc full rotation = {np.mean(accs_rotation)}")
    # if autoAlignFlag:
    accs_rotation_df.to_csv(  # save
        f"{megaROI_subSes_folder}/2_way_clf_acc_full_rotation.csv")  # OrganizedScripts/ROI/autoAlign/clfTraining/clfTraining.py
    # else:
    #     accs_rotation_df.to_csv(f"{cfg.recognition_dir}/ROI_analysis/{chosenMask}/2_way_clf_acc_full_rotation.csv")

    # 用所有数据训练要保存并且使用的模型：
    allpairs = itertools.combinations(objects, 2)
    accs = {}
    # Iterate over all the possible target pairs of objects
    for pair in allpairs:
        # Find the control (remaining) objects for this pair
        altpair = other(pair)
        for obj in pair:
            # foil = [i for i in pair if i != obj][0]
            for altobj in altpair:  # behav_data, brain_data
                # establish a naming convention where it is $TARGET_$CLASSIFICATION
                # Target is the NF pair (e.g. bed/bench)
                # Classificationis is btw one of the targets, and a control (e.g. bed/chair, or bed/table, NOT bed/bench)
                naming = '{}{}_{}{}'.format(pair[0], pair[1], obj, altobj)

                trainIX = ((behav_data['label'] == obj) | (behav_data['label'] == altobj))
                testIX = ((behav_data['label'] == obj) | (behav_data['label'] == altobj))

                # pull training and test data
                trainX = brain_data[trainIX]
                testX = brain_data[testIX]
                trainY = behav_data.iloc[np.asarray(trainIX)].label
                testY = behav_data.iloc[np.asarray(testIX)].label

                assert len(np.unique(trainY)) == 2

                # Train your classifier
                clf = LogisticRegression(penalty='l2', C=1, solver='lbfgs', max_iter=1000,
                                         multi_class='multinomial').fit(trainX, trainY)

                # Save it for later use
                model_folder = f"{megaROI_subSes_folder}/clf/"
                mkdir(model_folder)
                # if autoAlignFlag:
                #     model_folder = f"{autoAlign_ROIFolder}/clf/"
                #     mkdir(model_folder)
                # else:
                #     model_folder = f"{cfg.recognition_dir}/ROI_analysis/{chosenMask}/clf/"
                joblib.dump(clf, f'{model_folder}/{naming}.joblib')

                # Monitor progress by printing accuracy (only useful if you're running a test set)
                acc = clf.score(testX, testY)
                # print(naming, acc)
                accs[naming] = acc
    print(f"average 2 way clf accuracy={np.mean(list(accs.values()))}")
    # if autoAlignFlag:
    accTable.to_csv(f"{megaROI_subSes_folder}/accTable.csv")  # save
    # else:
    #     accTable.to_csv(f"{cfg.recognition_dir}/ROI_analysis/{chosenMask}/accTable.csv")  # save
    return accTable


minimalClass_ROI(sub=sub, ses=ses, chosenMask=chosenMask)
print("done")
