import os
import sys
assert os.getcwd().endswith('real_time_paper'), "working dir should be 'real_time_paper'"
workingDir = os.getcwd()
sys.path.append('.')
# print current dir
print(f"getcwd = {os.getcwd()}")

import sys
import numpy as np
import pandas as pd
from utils import save_obj, load_obj, mkdir, getjobID_num, kp_and, kp_or, kp_rename, kp_copy, kp_run, kp_remove
from utils import wait, check, checkEndwithDone, checkDone, check_jobIDs, check_jobArray, waitForEnd, \
    jobID_running_myjobs
from utils import readtxt, writetxt, get_subjects, init

import nibabel as nib
from tqdm import tqdm
from glob import glob
from scipy.stats import zscore
import joblib


def normalize(X, axis=0):
    _X = X.copy()
    _X = zscore(_X, axis=axis)
    _X[np.isnan(_X)] = 0
    return _X


def classifierProb(clf, X, Y):
    ID = np.where((clf.classes_ == Y) * 1 == 1)[0][0]
    p = clf.predict_proba(X)[:, ID]
    return p


batch = 12
subjects, scan_asTemplates = get_subjects(batch=batch)

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


sub_batch = f"batch{scan_asTemplates[sub]['batch']}"
print(f"sub_batch={sub_batch}")
assert useNewClf

print(f"sub={sub}, feedbackSes={feedbackSes}, scanNum={scanNum}, runNum={runNum}")

def doRuns(sub=None, ses=None, scanNum=None, runNum=None):
    megaROI_subSes_folder = (f"{workingDir}"
                             f"data/result/megaROI_main/subjects/{sub}/ses{ses}/{chosenMask}/")

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

    print(f"Doing run {runNum}, scan {scanNum}")
    # print(f"cfg.dicomDir={cfg.dicomDir}")

    tmp_dir = (f"/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/"
               f"real_time_paper/data/result/megaROI_main/temp/{sub}/ses{feedbackSes}/run{runNum}/")

    mkdir(tmp_dir)

    maskPath = f"{workingDir}/data/subjects/{sub}/ses1/recognition/mask/chosenMask.npy"
    mask = np.load(maskPath)

    imcodeDict = {
        'A': 'bed',
        'B': 'chair',
        'C': 'table',
        'D': 'bench'}

    model_folder = (f"{workingDir}"
                    f"/data/result/megaROI_main/subjects/{sub}/ses{ses-1}/{chosenMask}/clf/")
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

    brain_data_path = (f"{workingDir}/data/subjects/{sub}/ses{ses}/"
                       f"feedback/beforeUnwarp/run{scanNum}.nii.gz")  # from noUnwarpPreprocess
    brain_data = nib.load(brain_data_path).get_fdata()
    brain_data = np.transpose(brain_data, (3, 0, 1, 2))

    print(f"scanNum={scanNum}, runNum={runNum}")
    num_total_trials = 12
    num_total_TRs = min(int((num_total_trials * 28 + 12) / 2) + 8,
                        len(brain_data) + 1)  # number of TRs to use for example 1
    history = pd.DataFrame()

    # dicomFilenames
    for this_TR in tqdm(np.arange(1, num_total_TRs)):
        print(f"milgramTR_ID={this_TR}")
        nift_data = brain_data[this_TR - 1]

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

        AB_clf_A = classifierProb(AB_clf, X, "bed")
        AB_clf_B = classifierProb(AB_clf, X, "chair")

        CD_clf_C = classifierProb(CD_clf, X, "table")
        CD_clf_D = classifierProb(CD_clf, X, "bench")

        if sub_batch == 'batch1':
            prob = Bprob
            probs.append(prob)
        elif sub_batch == 'batch2':
            prob = Dprob
            probs.append(prob)
        if sub_batch == "batch1":
            history = history.append({
                'Sub': sub,
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
                "states": trial_list.loc[this_TR - 1, 'state']
            },
                ignore_index=True)
        elif sub_batch == "batch2":
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
                "states": trial_list.loc[this_TR - 1, 'state']
            },
                ignore_index=True)

    mkdir(f"{megaROI_subSes_folder}/feedback/")
    print(f'saving {megaROI_subSes_folder}/feedback/probs_{scanNum}_useNewClf')
    np.save(f'{megaROI_subSes_folder}/feedback/probs_{scanNum}_useNewClf',
            probs)  # save
    history.to_csv(f"{megaROI_subSes_folder}/feedback/"
                   f"history_runNum_{runNum}.csv", index=False)  # save
    return


doRuns(sub=sub, ses=feedbackSes, scanNum=scanNum, runNum=runNum)
print('done')
