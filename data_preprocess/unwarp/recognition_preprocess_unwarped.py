import subprocess

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
    [sub, ses] = ['sub024', 5]
else:
    jobarrayDict = np.load(f"data_preprocess/unwarp/recognition_preprocess_unwarped_jobID.npy",
                           allow_pickle=True)
    jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
    jobarrayID = int(float(sys.argv[1]))
    [sub, ses] = jobarrayDict[jobarrayID]
print(f"sub={sub}, ses={ses}")


def behaviorDataLoading(cfg, curr_run):
    '''
    extract the labels which is selected by the subject and coresponding TR and time
    check if the subject's response is correct. When Item is A,bed, response should be 1, or it is wrong
    '''
    behav_data = pd.read_csv(f"{cfg.recognition_dir}{cfg.subjectName}_{curr_run}.csv")

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

    # 为了处理 情况 A.被试的反应慢了一个TR，或者 B.两个按钮都被按了(这种情况下按照第二个按钮处理)
    # 现在的问题是”下一个TR“可能超过了behav_data的长度
    # this for loop is to deal with the situation where Resp is late for 1 TR, or two buttons are pressed.
    # when Resp is late for 1 TR, set the current Resp as the later Response.
    # when two buttons are pressed, set the current Resp as the later Response because the later one should be the real choice
    for curr_trial in range(behav_data.shape[0]):
        if behav_data['Item'].iloc[curr_trial] in ["A", "B", "C", "D"]:
            if curr_trial + 1 < behav_data.shape[0]:  # 为了防止”下一个TR“超过behav_data的长度  中文
                if behav_data['Resp'].iloc[curr_trial + 1] in [1.0, 2.0]:
                    behav_data['Resp'].iloc[curr_trial] = behav_data['Resp'].iloc[curr_trial + 1]

    behav_data = behav_data.dropna(subset=['Item'])

    # check if the subject's response is correct. When Item is A,bed, response should be 1, or it is wrong
    isCorrect = []

    # for curr_trial in range(behav_data.shape[0]):
    #     isCorrect.append(correctResponseDict[behav_data['Item'].iloc[curr_trial]]==behav_data['Resp'].iloc[curr_trial])
    # print(f"behavior pressing accuracy for run {curr_run} = {np.mean(isCorrect)}")
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
    behav_data['subj'] = [cfg.subjectName for i in range(len(behav_data))]
    behav_data['run_num'] = [int(curr_run) for i in range(len(behav_data))]
    behav_data = behav_data[behav_data['isCorrect']]  # discard the trials where the subject made wrong selection
    print(f"behav_data correct trial number = {len(behav_data)}")
    return behav_data


def recognition_preprocess_unwarped(cfg, scan_asTemplate, backupMode=False):
    from tqdm import tqdm
    import datetime
    ct = datetime.datetime.now()
    timeStamp = f"{ct}".replace(" ", "__").replace("-", "_").replace(":", ".")
    runRecording = pd.read_csv(f"{workingDir}/data/subjects/{sub}/ses{ses}/runRecording.csv")
    actualRuns = list(runRecording['run'].iloc[list(np.where(1 == 1 * (runRecording['type'] == 'recognition'))[
                                                        0])])  # 一个例子是 [1, 2, 13, 14] 或者 [1, 2, 3, 4, 5, 6, 7, 8]
    feedbackActualRuns = list(runRecording['run'].iloc[list(np.where(1 == 1 * (runRecording['type'] == 'feedback'))[
                                                                0])])  # 一个例子是 [1, 2, 13, 14] 或者 [1, 2, 3, 4, 5, 6, 7, 8]

    # 做好一系列的操作使得一系列的没有unwarp的文件被unwarp之后的新文件完美取代
    beforeUnwarpFolder = f"{cfg.recognition_dir}/beforeUnwarpFolder/"
    mkdir(beforeUnwarpFolder)
    beforeUnwarpFolder_feedback = f"{cfg.feedback_dir}/beforeUnwarpFolder_feedback/"
    mkdir(beforeUnwarpFolder_feedback)

    # 对于第2 3 4 5 个session的数据进行转移，转移到第一个session的functional template中
    if cfg.session in [2, 3, 4, 5]:
        # 首先删除之前存在的将当前session的functional数据转移到第一个ses的functional template中的矩阵 cfg.templateFunctionalVolume_converted
        # 重新计算得到一个 将当前session的functional数据转移到第一个ses的functional template中的矩阵 cfg.templateFunctionalVolume_converted 。  注意：ses1 的 templateFunctionalVolume_unwarped.nii的来源是 expScripts/recognition/recognitionDataAnalysis/GM_modelTrain.py ； ses2 3 4 5 的 templateFunctionalVolume_unwarped.nii的来源是
        cmd = f"flirt -ref {cfg.templateFunctionalVolume} \
            -in {cfg.recognition_dir}/templateFunctionalVolume_unwarped.nii \
            -out {cfg.templateFunctionalVolume_converted} -dof 6 \
            -omat {cfg.recognition_dir}/convert_2_ses1FuncTemp.mat "
        # cfg.templateFunctionalVolume_converted = f"{cfg.recognition_dir}/templateFunctionalVolume_converted.nii"  # templateFunctionalVolume_converted is the current day run1 middle volume converted in day1 template space

        print(cmd)
        sbatch_response = subprocess.getoutput(cmd)
        print(sbatch_response)

        for curr_run in tqdm(actualRuns):
            kp_copy(f"{cfg.recognition_dir}/recognition_scan{curr_run}_ses{cfg.session}_unwarped_mc.nii.gz",
                    f"{cfg.recognition_dir}run{curr_run}.nii.gz")

            # 把当前session的所有recognition run使用已有的转移矩阵转移到 ses1funcTemplate 空间中去。
            cmd = f"flirt -ref {cfg.templateFunctionalVolume_converted} \
                -in {cfg.recognition_dir}run{curr_run}.nii.gz \
                -out {cfg.recognition_dir}run{curr_run}.nii.gz -applyxfm \
                -init {cfg.recognition_dir}/convert_2_ses1FuncTemp.mat "
            print(cmd)
            sbatch_response = subprocess.getoutput(cmd)
            print(sbatch_response)
            cmd = f"fslinfo {cfg.recognition_dir}run{curr_run}.nii.gz"
            print(cmd)
            sbatch_response = subprocess.getoutput(cmd)
            print(sbatch_response)

        # 将 feedback run 也做同样的操作, 也就是将当前session的所有feedback run使用已有的转移矩阵转移到 ses1funcTemplate 空间中去。
        for curr_run in tqdm(feedbackActualRuns):  # feedbackActualRuns 的一个例子是 [3,4,5,6,7,8,9,10,11,12]
            kp_copy(f"{cfg.feedback_dir}/feedback_scan{curr_run}_ses{cfg.session}_unwarped_mc.nii.gz",
                    f"{cfg.feedback_dir}/run{curr_run}.nii.gz")
            print(f"renaming {cfg.feedback_dir}/feedback_scan{curr_run}_ses{cfg.session}_unwarped_mc.nii.gz")
            # 把当前session的所有 feedback run 使用已有的转移矩阵转移到ses1funcTemp空间中去。
            cmd = f"flirt -ref {cfg.templateFunctionalVolume_converted} \
                -in {cfg.feedback_dir}run{curr_run}.nii.gz \
                -out {cfg.feedback_dir}run{curr_run}.nii.gz -applyxfm \
                -init {cfg.recognition_dir}/convert_2_ses1FuncTemp.mat "
            print(cmd)
            sbatch_response = subprocess.getoutput(cmd)
            print(sbatch_response)
            cmd = f"fslinfo {cfg.feedback_dir}run{curr_run}.nii.gz"
            print(cmd)
            sbatch_response = subprocess.getoutput(cmd)
            print(sbatch_response)

    # 对于第1个session的数据，重命名文件以使得fmap校准过的，人工校准过的数据能够替代原来的数据。
    elif cfg.session == 1:

        kp_copy(f"{cfg.recognition_dir}/../anat/gm_func_unwarp.nii.gz", f"{cfg.recognition_dir}/../anat/gm_func.nii.gz")

        kp_copy(f"{cfg.recognition_dir}/anat2func_unwarp.mat", f"{cfg.recognition_dir}/anat2func.mat")
        kp_copy(f"{cfg.recognition_dir}/ANATinFUNC_unwarp.nii.gz", f"{cfg.recognition_dir}/ANATinFUNC.nii.gz")

        kp_copy(f"{cfg.recognition_dir}/templateFunctionalVolume_unwarped.nii",
                f"{cfg.recognition_dir}/templateFunctionalVolume.nii")

        # 如果是第一个session，那么这个session的templateFunctionalVolume_converted 就是本身。如果是后面的session，那么 那个session的templateFunctionalVolume_converted就是那个session的funcTemp转移到第一个session的funcTemp的空间当中。

        kp_copy(f"{cfg.recognition_dir}/templateFunctionalVolume.nii",
                f"{cfg.recognition_dir}/templateFunctionalVolume_converted.nii")

        # os.rename(f"{cfg.recognition_dir}/anat2func_beforeUnwarp.mat",f"{beforeUnwarpFolder}/anat2func_beforeUnwarp.mat")
        # os.rename(f"{cfg.recognition_dir}/ANATinFUNC_beforeUnwarp.nii.gz",f"{beforeUnwarpFolder}/ANATinFUNC_beforeUnwarp.nii.gz")

        # for curr_run in actualRuns:
        #     kp_copy(f"{cfg.recognition_dir}/run{curr_run}.nii.gz",f"{beforeUnwarpFolder}/run{curr_run}.nii.gz")

        # 把经过 fmap 校正和 motion correction 的 recognition_scan{curr_run}_ses{cfg.session}_unwarped_mc.nii.gz 重命名为 平常使用的 run{curr_run}.nii.gz
        for curr_run in actualRuns:
            kp_copy(f"{cfg.recognition_dir}/recognition_scan{curr_run}_ses{cfg.session}_unwarped_mc.nii.gz",
                    f"{cfg.recognition_dir}run{curr_run}.nii.gz")
            print(f"renaming {cfg.recognition_dir}/recognition_scan{curr_run}_ses{cfg.session}_unwarped_mc.nii.gz")

            # 做好一系列的操作使得一系列的没有unwarp的文件被unwarp之后的新文件完美取代

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
        behav_data = behaviorDataLoading(cfg, curr_run_behav + 1)  # behav_data 的数据的TR是从0开始的。brain_data 也是
        # len = 48 ，最后一个TR ID是 142

        # brain data is first aligned by pushed back 2TR(4s)
        print(f"loading {cfg.recognition_dir}run{curr_run}.nii.gz")
        brain_data = nib.load(f"{cfg.recognition_dir}run{curr_run}.nii.gz").get_data();
        brain_data = np.transpose(brain_data, (3, 0, 1, 2))
        # len = 144
        Brain_TR = np.arange(brain_data.shape[0])  # 假设brain_data 有144个，那么+2之后的Brain_TR就是2，3，。。。，145.一共144个TR。
        Brain_TR = Brain_TR + 2

        # select volumes of brain_data by counting which TR is left in behav_data
        try:
            Brain_TR = Brain_TR[list(behav_data['TR'])]  # original TR begin with 0 #筛选掉无用的TR，由于两个都是从0开始计数的，所以是可以的。
        except:
            Brain_TR = Brain_TR[
                list(behav_data['TR'])[:-1]]  # 如果大脑数据的TR数目没有行为的TR数目多的时候，此时行为的数据的TR的尾巴是没用的，可以丢掉一个（如果还不行的话，可以再丢）
        # 筛选掉之后的Brain_TR长度是 48 最后一个ID是144
        # Brain_TR[-1] 是想要的最后一个TR的ID，看看是否在brain_data里面？如果不在的话，那么删除最后一个Brain_TR，也删除behav里面的最后一行
        # 如果大脑数据的长度没有行为学数据长（比如大脑只收集到144个TR，然后我现在想要第145个TR的数据，这提醒我千万不要过早结束recognition run）
        if Brain_TR[-1] >= brain_data.shape[
            0]:  # when the brain data is not as long as the behavior data, delete the last row
            print("Warning: brain data is not long enough, don't cut the data collection too soon!!!!")
            Brain_TR = Brain_TR[:-1]
            # behav_data = behav_data.drop([behav_data.iloc[-1].TR])
            behav_data.drop(behav_data.tail(1).index, inplace=True)

        brain_data = brain_data[Brain_TR]
        # 如果在极端情况下，结束trial的同时就结束了fMRI数据收集，将会导致最后一个trial的大脑数据被遗漏，此时就需要丢掉最后一个行为学数据，即下面的if
        if brain_data.shape[0] < behav_data.shape[0]:
            print("如果在极端情况下，结束trial的同时就结束了fMRI数据收集，将会导致最后一个trial的大脑数据被遗漏，此时就需要丢掉最后一个行为学数据")
            behav_data.drop(behav_data.tail(1).index, inplace=True)

        np.save(f"{cfg.recognition_dir}brain_run{curr_run}.npy", brain_data)
        # save the behavior data
        behav_data.to_csv(f"{cfg.recognition_dir}behav_run{curr_run}.csv")


recognition_preprocess_unwarped(sub, ses)

print("done")
