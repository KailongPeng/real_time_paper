import itertools
from glob import glob

import joblib
from sklearn.linear_model import LogisticRegression

testMode = False
import os
import sys
os.chdir("/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/real_time_paper/")
assert os.getcwd().endswith('real_time_paper'), "working dir should be 'real_time_paper'"
workingDir = os.getcwd()
sys.path.append('.')
# print current dir
print(f"getcwd = {os.getcwd()}")


import shutil
from scipy.stats import zscore

import numpy as np
import pandas as pd
import time
import pickle5 as pickle
from tqdm import tqdm
import nibabel as nib

from utils import save_obj, load_obj, mkdir, getjobID_num, kp_and, kp_or, kp_rename, kp_copy, kp_run, kp_remove
from utils import wait, check, checkEndwithDone, checkDone, check_jobIDs, check_jobArray, waitForEnd, \
    jobID_running_myjobs
from utils import readtxt, writetxt, get_subjects


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


batch = 12  # meaning both batch 1 and batch 2
subjects, scan_asTemplates = get_subjects(batch=batch)


if testMode:
    [sub, chosenMask, ses, autoAlignFlag] = ['sub024', 'V1_FreeSurfer', 1, False]
else:
    jobarrayDict = np.load(f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
                           f"OrganizedScripts/ROI/autoAlign_ses1ses5/clf_training/clfTraining_jobID.npy",
                           allow_pickle=True)
    jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
    jobarrayID = int(float(sys.argv[1]))
    [sub, chosenMask, ses, autoAlignFlag] = jobarrayDict[jobarrayID]
    # [sub, ses, chosenMask] = ['sub024', 1, 'V1_FreeSurfer']
print(f"sub={sub}, ses={ses}, choseMask={chosenMask}")
assert ses == 1
# autoAlignFlag = True
print(f"autoAlignFlag={autoAlignFlag}")


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if autoAlignFlag:
    autoAlign_ROIFolder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis_ses1ses5/" \
                          f"subjects/{sub}/ses{ses}/{chosenMask}/"
    mkdir(autoAlign_ROIFolder)


def minimalClass_ROI(sub='', ses=None,
                     chosenMask=''):
    # scanList = []
    # files = glob(f"{cfg.dicom_dir}/*.dcm")
    # for file in files:
    #     scanList.append(int(file.split(f"{cfg.subjectIDforxnat}/001_")[1].split('_')[0]))
    runRecording = pd.read_csv(f"{workingDir}/data/subjects/{sub}/ses{ses}/runRecording.csv")
    # assert len(np.unique(np.asarray(scanList))) == len(runRecording)
    actualRuns = list(runRecording['run'].iloc[list(
        np.where(1 == 1 * (runRecording['type'] == 'recognition'))[0])])  # can be [1,2,3,4,5,6,7,8] or [1,2,4,5]

    if len(actualRuns) < 8:
        runRecording_preDay = pd.read_csv(
            f"{workingDir}/data/subjects/{sub}/ses{ses-1}/runRecording.csv")
        actualRuns_preDay = list(runRecording_preDay['run'].iloc[
                                     list(np.where(1 == 1 * (runRecording_preDay['type'] == 'recognition'))[0])])[
                            -(8 - len(actualRuns)):]  # might be [5,6,7,8]
    else:
        actualRuns_preDay = []
    assert len(actualRuns_preDay) + len(actualRuns) == 8

    objects = ['bed', 'bench', 'chair', 'table']

    new_run_indexs = []
    new_run_index = 1  # Use the new run index to avoid duplication during subsequent test run selections. Normally, the new run index should be 1, 2, 3, 4, 5, 6, 7, 8.

    print(f"actualRuns={actualRuns}")
    assert (len(actualRuns) >= 4)
    print(f"actualRuns_preDay={actualRuns_preDay}")

    subjectFolder = "/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/"
    if autoAlignFlag:
        maskFolder = f"{workingDir}/data/subjects/{sub}/ses1/recognition/mask/"
    else:
        maskFolder = f"{subjectFolder}/{sub}/ses1/recognition/mask/"
    print(f"maskFolder={maskFolder}")

    brain_data = None
    behav_data = None
    for ii, run in enumerate(actualRuns):  # load behavior and brain data for current session
        t = np.load(f"{cfg.recognition_dir}brain_run{run}.npy")
        # print(f"loading {maskFolder}/{chosenMask}")
        # mask = np.load(f"{maskFolder}/{chosenMask}")

        maskPath = f"{maskFolder}/{chosenMask}.nii"
        try:
            mask = nib.load(maskPath).get_fdata()
            print(f"loading {maskFolder}/{chosenMask}.nii , mask size = {np.sum(mask)}")
        except:
            mask = nib.load(f"{maskPath}.gz").get_fdata()
            print(f"loading {maskFolder}/{chosenMask}.nii.gz , mask size = {np.sum(mask)}")
        print(f"mask.shape={mask.shape}")

        t = t[:, mask == 1]
        t = normalize(t)
        brain_data = t if ii == 0 else np.concatenate((brain_data, t), axis=0)

        t = pd.read_csv(f"{cfg.recognition_dir}behav_run{run}.csv")
        t['run_num'] = new_run_index
        new_run_indexs.append(new_run_index)
        new_run_index += 1
        behav_data = t if ii == 0 else pd.concat([behav_data, t])

    for ii, run in enumerate(actualRuns_preDay):  # load behavior and brain data for previous session
        t = np.load(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session - 1}/recognition/brain_run{run}.npy")
        # print(f"loading {maskFolder}/{chosenMask}")
        # mask = np.load(f"{maskFolder}/{chosenMask}")
        try:
            mask = nib.load(f"{maskFolder}/{chosenMask}.nii").get_fdata()
            print(f"loading {maskFolder}/{chosenMask}.nii , mask size = {np.sum(mask)}")
        except:
            mask = nib.load(f"{maskFolder}/{chosenMask}.nii.gz").get_fdata()
            print(f"loading {maskFolder}/{chosenMask}.nii.gz , mask size = {np.sum(mask)}")

        t = t[:, mask == 1]
        t = normalize(t)
        brain_data = np.concatenate((brain_data, t), axis=0)

        t = pd.read_csv(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session - 1}/recognition/behav_run{run}.csv")
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

    # Obtain the average accuracy of the 2-way classification with full rotation
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
                        if testMode:
                            print(f"using testRun={testRun} as testRun")
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
                    if autoAlignFlag:
                        model_folder = f"{autoAlign_ROIFolder}/clf/"   # save
                        mkdir(model_folder)
                    else:
                        model_folder = f"{cfg.recognition_dir}/ROI_analysis/{chosenMask}/clf/"
                    # if not testMode:
                    joblib.dump(clf, f'{model_folder}/{naming}_testRun{testRun}.joblib')  # save

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
    # mkdir(f"{cfg.recognition_dir}/ROI_analysis/{chosenMask}/clf/")
    if autoAlignFlag:
        accs_rotation_df.to_csv(f"{autoAlign_ROIFolder}/2_way_clf_acc_full_rotation.csv")
    else:
        accs_rotation_df.to_csv(f"{cfg.recognition_dir}/ROI_analysis/{chosenMask}/2_way_clf_acc_full_rotation.csv")
    if autoAlignFlag:
        accTable.to_csv(f"{autoAlign_ROIFolder}/accTable.csv")
    else:
        accTable.to_csv(f"{cfg.recognition_dir}/ROI_analysis/{chosenMask}/accTable.csv")

    return accTable


minimalClass_ROI(sub=sub, ses=ses, chosenMask=chosenMask)
print("done")
