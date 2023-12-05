testMode = False
import joblib
import os
import sys
assert os.getcwd().endswith('real_time_paper'), "working dir should be 'real_time_paper'"
workingDir = os.getcwd()
sys.path.append('.')
# print current dir
print(f"getcwd = {os.getcwd()}")

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
    [sub, chosenMask, ses] = ['sub024', 'V1_FreeSurfer', 5]
else:
    jobarrayDict = np.load(f"data_preprocess/prepareIntegrationScore/integrationScore/integrationScore_jobID.npy",
                           allow_pickle=True)
    jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
    jobarrayID = int(float(sys.argv[1]))
    [sub, chosenMask, ses] = jobarrayDict[jobarrayID]
print(f"sub={sub}, ses={ses}, choseMask={chosenMask}")
assert ses == 5
autoAlignFlag = True

print(f"batch={batch}")


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


autoAlign_ROIFolder = (f"{workingDir}/data/"
                       f"result/subjects/{sub}/ses{ses}/{chosenMask}/")


def minimalClass_ROI(sub='', ses=None,
                     chosenMask=''):

    runRecording = pd.read_csv(f"{workingDir}/data/subjects/{sub}/ses{ses}/runRecording.csv")
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
    maskFolder = f"{workingDir}/data/subjects/{sub}/ses1/recognition/mask/"
    print(f"maskFolder={maskFolder}")

    brain_data = None
    behav_data = None
    recognition_dir = f"{workingDir}/data/subjects/{sub}/ses{ses}/recognition/"
    for ii, run in enumerate(actualRuns):  # load behavior and brain data for current session
        t = np.load(f"{recognition_dir}brain_run{run}.npy")

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

        t = pd.read_csv(f"{recognition_dir}behav_run{run}.csv")
        t['run_num'] = new_run_index
        new_run_indexs.append(new_run_index)
        new_run_index += 1
        behav_data = t if ii == 0 else pd.concat([behav_data, t])

    for ii, run in enumerate(actualRuns_preDay):  # load behavior and brain data for previous session
        recognition_prev_dir = f"{workingDir}/data/subjects/{sub}/ses{ses-1}/recognition/"
        t = np.load(f"{recognition_prev_dir}/brain_run{run}.npy")
        try:
            mask = nib.load(f"{maskFolder}/{chosenMask}.nii").get_fdata()
            print(f"loading {maskFolder}/{chosenMask}.nii , mask size = {np.sum(mask)}")
        except:
            mask = nib.load(f"{maskFolder}/{chosenMask}.nii.gz").get_fdata()
            print(f"loading {maskFolder}/{chosenMask}.nii.gz , mask size = {np.sum(mask)}")

        t = t[:, mask == 1]
        t = normalize(t)
        brain_data = np.concatenate((brain_data, t), axis=0)

        t = pd.read_csv(f"{recognition_prev_dir}/behav_run{run}.csv")
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
    codeImDict = {
        'bed': 'A',
        'chair': 'B',
        'table': 'C',
        'bench': 'D'}
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

    accs_rotation = []
    accs_rotation_df = pd.DataFrame()
    print(f"new_run_indexs={new_run_indexs}")
    # for curr_testRun, testRun in enumerate(new_run_indexs):
    # testRun = False

    accs = {}
    # Iterate over all the possible target pairs of objects
    for axis in ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']:
        objects = ['bed', 'bench', 'chair', 'table']
        clfDict = {
            "AB": 'bedtable_bedchair',
            "AC": 'bedbench_bedtable',
            "AD": 'bedchair_bedbench',
            "BC": 'benchchair_chairtable',
            "BD": 'bedchair_chairbench',
            "CD": 'bedtable_tablebench'
        }
        naming = clfDict[axis].split(".")[0]

        obj = imcodeDict[axis[0]]
        altobj = imcodeDict[axis[1]]

        if testMode:
            print(f"using all runs as testRun")
        testIX = ((behav_data['label'] == obj) | (behav_data['label'] == altobj))

        testX = brain_data[testIX]
        testY = behav_data.iloc[np.asarray(testIX)].label
        if testMode:
            print(f"testX.shape={testX.shape}")
            print(f"testX={testX}")
            print(f"testY={testY}")
        assert len(np.unique(testY)) == 2

        model_folder = f"{autoAlign_ROIFolder}/clf/"  # load
        for testRun_training in range(1, 9):
            clf = joblib.load(f'{model_folder}/{clfDict[axis]}_testRun{testRun_training}.joblib')

            # Monitor progress by printing accuracy (only useful if you're running a test set)
            acc = clf.score(testX, testY)
            # print(naming, acc)
            accs[f"{axis}_testRun{testRun_training}"] = acc

            accs_rotation_df = pd.concat([accs_rotation_df, pd.DataFrame(
                {'subject': [sub],
                 'testRun_training': [testRun_training],
                 'axis': axis,
                 'obj': [obj],
                 'altobj': [altobj],
                 'acc': [acc]})], ignore_index=True)

    for axis in ["AB", "CD", "AC", "AD", "BC", "BD"]:
        for testRun_training in range(1, 9):
            accTable.loc[testRun_training, axis + '_acc'] = accs[f"{axis}_testRun{testRun_training}"]

    accs_rotation.append(np.mean(list(accs.values())))
    print(f"accTable={accTable}")
    print(f"mean of 2 way clf acc full rotation = {np.mean(accs_rotation)}")
    # mkdir(f"{cfg.recognition_dir}/ROI_analysis/{chosenMask}/clf/")
    accs_rotation_df.to_csv(f"{autoAlign_ROIFolder}/2_way_clf_acc_full_rotation.csv")

    accTable.to_csv(f"{autoAlign_ROIFolder}/accTable.csv")

    return accTable


accTable = minimalClass_ROI(sub=sub, ses=ses, chosenMask=chosenMask)


def getIntegrationScore_ses1ses5(sub='', chosenMask=''):
    imcodeDict = {"A": "bed", "B": "Chair", "C": "table", "D": "bench"}
    clfDict = {
        "AB": 'bedtable_bedchair.joblib',
        "AC": 'bedbench_bedtable.joblib',
        "AD": 'bedchair_bedbench.joblib',
        "BC": 'benchchair_chairtable.joblib',
        "BD": 'bedchair_chairbench.joblib',
        "CD": 'bedtable_tablebench.joblib'
    }

    accTable_ses5 = pd.read_csv(f"/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/"
                                f"real_time_paper/data/result/subjects/{sub}/ses5/{chosenMask}/accTable.csv")
    accTable_ses1 = pd.read_csv(f"/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/"
                                f"real_time_paper/data/result/subjects/{sub}/ses1/{chosenMask}/accTable.csv")

    ses1ses5Acc = pd.DataFrame()
    for axis in ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']:
        ses1_acc = np.mean(accTable_ses1[f'{axis}_acc'])
        ses5_acc = np.mean(accTable_ses5[f'{axis}_acc'])

        ses1ses5Acc = pd.concat([ses1ses5Acc, pd.DataFrame({
            'subject': sub,
            'session': 999,  # this is not important
            'ses1_acc': ses1_acc,
            'ses5_acc': ses5_acc,
            'axis': axis,
            'chosenMask': chosenMask
        }, index=[0])], ignore_index=True)

    t = ses1ses5Acc[kp_and([
        ses1ses5Acc['subject'] == sub,
        ses1ses5Acc['session'] == 999,
        ses1ses5Acc['chosenMask'] == chosenMask
    ])]
    if batch == 'batch1':
        ses5_XY = float(t[t['axis'] == 'AB']['ses5_acc'])
        ses1_XY = float(t[t['axis'] == 'AB']['ses1_acc'])
        ses5_MN = float(t[t['axis'] == 'CD']['ses5_acc'])
        ses1_MN = float(t[t['axis'] == 'CD']['ses1_acc'])
    elif batch == 'batch2':
        ses5_XY = float(t[t['axis'] == 'CD']['ses5_acc'])
        ses1_XY = float(t[t['axis'] == 'CD']['ses1_acc'])
        ses5_MN = float(t[t['axis'] == 'AB']['ses5_acc'])
        ses1_MN = float(t[t['axis'] == 'AB']['ses1_acc'])
    else:
        raise Exception("batch is not defined")

    differentiation_ratio = (ses5_XY - ses1_XY) / (ses5_XY + ses1_XY) - (ses5_MN - ses1_MN) / (ses5_MN + ses1_MN)
    integration_ratio = - differentiation_ratio

    np.save(f"{workingDir}/data/"
            f"result/subjects/{sub}/ses{ses}/{chosenMask}/integration_ratio.npy",
            integration_ratio)
    np.save(f"{workingDir}/data/"
            f"result/subjects/{sub}/ses{ses}/{chosenMask}/integration_ratio_allData.npy",
            [ses1_XY, ses5_XY, ses1_MN, ses5_MN, differentiation_ratio, integration_ratio])
    return integration_ratio


_integration_ratio = getIntegrationScore_ses1ses5(sub=sub, chosenMask=chosenMask)


print("done")
