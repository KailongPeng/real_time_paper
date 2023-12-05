testMode = False
import subprocess
import os
import sys

os.chdir("/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/real_time_paper/")
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
    [sub, ses] = ['sub024', 5]
else:
    jobarrayDict = np.load(f"data_preprocess/unwarp/recognition_preprocess_unwarped_jobID.npy",
                           allow_pickle=True)
    jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
    jobarrayID = int(float(sys.argv[1]))
    [sub, ses] = jobarrayDict[jobarrayID]
print(f"sub={sub}, ses={ses}")


def behaviorDataLoading(sub, ses, curr_run):
    """
        extract the labels which is selected by the subject and coresponding TR and time
        check if the subject's response is correct. When Item is A,bed, response should be 1, or it is wrong
    """
    recognition_dir = f"{workingDir}/data/subjects/{sub}/ses{ses}/recognition/"
    behav_data = pd.read_csv(f"{recognition_dir}{ses}_{curr_run}.csv")

    # the item(imcode) colume of the data represent each image in the following correspondence
    # When the imcode code is "A", the correct response should be '1', "B" should be '2'
    correctResponseDict = {
        'A': 1,
        'B': 2,
        'C': 1,
        'D': 2}

    SwitchCorrectResponseDict = {
        'A': 2,
        'B': 1,
        'C': 2,
        'D': 1}
    # extract the labels which is selected by the subject and coresponding TR and time
    try:
        behav_data = behav_data[
            ['TR', 'image_on', 'Resp', 'Item', 'switchButtonOrientation']]  # the TR, the real time it was presented,
        randomButtion = True
    except:
        behav_data = behav_data[['TR', 'image_on', 'Resp', 'Item']]  # the TR, the real time it was presented,
        randomButtion = False
    print(f"randomButtion={randomButtion}")
    for curr_trial in range(behav_data.shape[0]):
        if behav_data['Item'].iloc[curr_trial] in ["A", "B", "C", "D"]:
            if curr_trial + 1 < behav_data.shape[0]:
                if behav_data['Resp'].iloc[curr_trial + 1] in [1.0, 2.0]:
                    behav_data['Resp'].iloc[curr_trial] = behav_data['Resp'].iloc[curr_trial + 1]

    behav_data = behav_data.dropna(subset=['Item'])

    isCorrect = []

    if randomButtion:
        for curr_trial in range(behav_data.shape[0]):
            if behav_data['switchButtonOrientation'].iloc[curr_trial]:
                isCorrect.append(
                    SwitchCorrectResponseDict[behav_data['Item'].iloc[curr_trial]] == behav_data['Resp'].iloc[
                        curr_trial])
            else:
                isCorrect.append(
                    correctResponseDict[behav_data['Item'].iloc[curr_trial]] == behav_data['Resp'].iloc[curr_trial])
    else:
        for curr_trial in range(behav_data.shape[0]):
            isCorrect.append(
                correctResponseDict[behav_data['Item'].iloc[curr_trial]] == behav_data['Resp'].iloc[curr_trial])

    print(f"behavior pressing accuracy for run {curr_run} = {np.mean(isCorrect)}")
    assert np.mean(isCorrect) > 0.9
    print("assert np.mean(isCorrect)>0.9")

    behav_data['isCorrect'] = isCorrect  # merge the isCorrect clumne with the data dataframe
    behav_data['subj'] = [sub for i in range(len(behav_data))]
    behav_data['run_num'] = [int(curr_run) for i in range(len(behav_data))]
    behav_data = behav_data[behav_data['isCorrect']]  # discard the trials where the subject made wrong selection
    print(f"behav_data correct trial number = {len(behav_data)}")
    return behav_data


def recognition_preprocess_unwarped(sub, ses, scan_asTemplate, backupMode=False):
    from tqdm import tqdm
    runRecording = pd.read_csv(f"{workingDir}/data/subjects/{sub}/ses{ses}/runRecording.csv")
    actualRuns = list(runRecording['run'].iloc[list(np.where(1 == 1 * (runRecording['type'] == 'recognition'))[
                                                        0])])  # one example is  [1, 2, 13, 14] or [1, 2, 3, 4, 5, 6, 7, 8]
    feedbackActualRuns = list(runRecording['run'].iloc[list(np.where(1 == 1 * (runRecording['type'] == 'feedback'))[
                                                                0])])
    recognition_dir = f"{workingDir}/data/subjects/{sub}/ses{ses}/recognition/"
    # Transfer data from sessions 2, 3, 4, and 5 to the functional template of the first session
    if ses in [2, 3, 4, 5]:
        templateFunctionalVolume_converted = f"{recognition_dir}/templateFunctionalVolume_converted.nii.gz"
        cmd = (
            f"flirt "
            f"-ref {workingDir}/data/subjects/{sub}/ses1/recognition/templateFunctionalVolume_unwarped_bet.nii.gz \
            -in {recognition_dir}/templateFunctionalVolume_unwarped_bet.nii.gz \
            -out {templateFunctionalVolume_converted} \
            -dof 6 \
            -omat {recognition_dir}/convert_2_ses1FuncTemp.mat ")

        print(cmd)
        sbatch_response = subprocess.getoutput(cmd)
        print(sbatch_response)

        for curr_run in tqdm(actualRuns):
            kp_copy(f"{recognition_dir}/run_{curr_run}_unwarped_mc.nii.gz",
                    f"{recognition_dir}/run{curr_run}.nii.gz")

            # Transfer all recognition runs from the current session using the existing transformation matrix to the ses1funcTemplate space
            cmd = f"flirt -ref {templateFunctionalVolume_converted} \
                -in {recognition_dir}run{curr_run}.nii.gz \
                -out {recognition_dir}run{curr_run}.nii.gz -applyxfm \
                -init {recognition_dir}/convert_2_ses1FuncTemp.mat "
            print(cmd)
            sbatch_response = subprocess.getoutput(cmd)
            print(sbatch_response)
            cmd = f"fslinfo {recognition_dir}run{curr_run}.nii.gz"
            print(cmd)
            sbatch_response = subprocess.getoutput(cmd)
            print(sbatch_response)

        # Apply the same operation to the feedback runs, meaning to transfer all feedback runs from the current session using the existing transformation matrix to the ses1funcTemplate space
        for curr_run in tqdm(feedbackActualRuns):  # feedbackActualRuns one example is [3,4,5,6,7,8,9,10,11,12]
            feedback_dir = f"{workingDir}/data/subjects/{sub}/ses{ses}/feedback/"
            kp_copy(f"{feedback_dir}/feedback_scan{curr_run}_ses{ses}_unwarped_mc.nii.gz",
                    f"{feedback_dir}/run{curr_run}.nii.gz")
            print(f"renaming {feedback_dir}/feedback_scan{curr_run}_ses{ses}_unwarped_mc.nii.gz")
            # Transfer all feedback runs from the current session using the existing transformation matrix to the ses1funcTemp space
            cmd = f"flirt -ref {templateFunctionalVolume_converted} \
                -in {feedback_dir}run{curr_run}.nii.gz \
                -out {feedback_dir}run{curr_run}.nii.gz -applyxfm \
                -init {recognition_dir}/convert_2_ses1FuncTemp.mat "
            print(cmd)
            sbatch_response = subprocess.getoutput(cmd)
            print(sbatch_response)
            cmd = f"fslinfo {feedback_dir}run{curr_run}.nii.gz"
            print(cmd)
            sbatch_response = subprocess.getoutput(cmd)
            print(sbatch_response)

    # For the data of the first session, rename files to allow fmap-calibrated and manually calibrated data to replace the original data
    elif ses == 1:
        kp_copy(f"{recognition_dir}/templateFunctionalVolume_unwarped.nii",
                f"{recognition_dir}/templateFunctionalVolume.nii")

        # If it is the first session, then the templateFunctionalVolume_converted for this session is itself. If it is a subsequent session, then the templateFunctionalVolume_converted for that session is the funcTemp of that session transformed into the space of the funcTemp of the first session

        kp_copy(f"{recognition_dir}/templateFunctionalVolume.nii",
                f"{recognition_dir}/templateFunctionalVolume_converted.nii")

        # Rename the run_{curr_run}_unwarped_mc.nii.gz, which has undergone fmap correction and motion correction, to the commonly used run{curr_run}.nii.gz
        for curr_run in actualRuns:
            kp_copy(f"{recognition_dir}/run_{curr_run}_unwarped_mc.nii.gz",
                    f"{recognition_dir}/run{curr_run}.nii.gz")
            print(f"renaming {recognition_dir}/run_{curr_run}_unwarped_mc.nii.gz")

    '''
    for each run,
        load behavior data
        push the behavior data back for 2 TRs
        save the brain TRs with images
        save the behavior data
    '''

    for curr_run_behav, curr_run in enumerate(actualRuns):
        print(f"curr_run_behav={curr_run_behav},curr_run={curr_run}")
        # load behavior data
        behav_data = behaviorDataLoading(sub, ses, curr_run_behav + 1)

        # brain data is first aligned by pushed back 2TR(4s)
        print(f"loading {recognition_dir}run{curr_run}.nii.gz")
        brain_data = nib.load(f"{recognition_dir}run{curr_run}.nii.gz").get_data()
        brain_data = np.transpose(brain_data, (3, 0, 1, 2))
        # len = 144
        Brain_TR = np.arange(brain_data.shape[0])  # Assuming there are 144 entries in brain_data, then the Brain_TR after adding 2 would be 2, 3, ..., 145, totaling 144 TRs
        Brain_TR = Brain_TR + 2

        # select volumes of brain_data by counting which TR is left in behav_data
        try:
            Brain_TR = Brain_TR[list(behav_data['TR'])]  # original TR begin with 0
        except:
            Brain_TR = Brain_TR[
                list(behav_data['TR'])[:-1]]  # If the number of TRs in brain data is not greater than the number of TRs in behavioral data, the tail of the TRs in the behavioral data is unused and can be discarded

        if Brain_TR[-1] >= brain_data.shape[
            0]:  # when the brain data is not as long as the behavior data, delete the last row
            print("Warning: brain data is not long enough, don't cut the data collection too soon!!!!")
            Brain_TR = Brain_TR[:-1]
            # behav_data = behav_data.drop([behav_data.iloc[-1].TR])
            behav_data.drop(behav_data.tail(1).index, inplace=True)

        brain_data = brain_data[Brain_TR]
        if brain_data.shape[0] < behav_data.shape[0]:
            behav_data.drop(behav_data.tail(1).index, inplace=True)

        np.save(f"{recognition_dir}brain_run{curr_run}.npy", brain_data)
        # save the behavior data
        behav_data.to_csv(f"{recognition_dir}behav_run{curr_run}.csv")


scan_asTemplate = scan_asTemplates[sub][f"ses{ses}"]
recognition_preprocess_unwarped(sub, ses, scan_asTemplate)

print("done")
