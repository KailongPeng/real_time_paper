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
        if backupMode:
            kp_copy(cfg.templateFunctionalVolume_converted, f"{beforeUnwarpFolder}/templateFunctionalVolume_converted_{timeStamp}.nii")
            kp_copy(cfg.templateFunctionalVolume_converted + '.gz',
                    f"{beforeUnwarpFolder}/templateFunctionalVolume_converted_{timeStamp}.nii.gz")

        # 重新计算得到一个 将当前session的functional数据转移到第一个ses的functional template中的矩阵 cfg.templateFunctionalVolume_converted 。  注意：ses1 的 templateFunctionalVolume_unwarped.nii的来源是 expScripts/recognition/recognitionDataAnalysis/GM_modelTrain.py ； ses2 3 4 5 的 templateFunctionalVolume_unwarped.nii的来源是
        cmd = f"flirt -ref {cfg.templateFunctionalVolume} \
            -in {cfg.recognition_dir}/templateFunctionalVolume_unwarped.nii \
            -out {cfg.templateFunctionalVolume_converted} -dof 6 \
            -omat {cfg.recognition_dir}/convert_2_ses1FuncTemp.mat "
        # cfg.templateFunctionalVolume_converted = f"{cfg.recognition_dir}/templateFunctionalVolume_converted.nii"  # templateFunctionalVolume_converted is the current day run1 middle volume converted in day1 template space

        def test_nonBetFlirt():
            from utils import save_obj, load_obj, mkdir, getjobID_num, kp_and, kp_or, kp_rename, kp_copy, kp_run, \
                kp_remove
            from utils import wait, check, checkEndwithDone, checkDone, check_jobIDs, check_jobArray, waitForEnd, \
                jobID_running_myjobs
            from utils import readtxt, writetxt, deleteChineseCharactor, get_subjects, init
            from utils import getMonDate, checkDate, save_nib
            from utils import get_ROIMethod, bar, get_ROIList
            testDir = "/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/temp/test_nonBetFlirt/"
            os.chdir(testDir)
            """
                cp /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/sub030/ses1/recognition/templateFunctionalVolume.nii ./
            """
            cmd = f"flirt -ref /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/sub030/ses1/recognition/templateFunctionalVolume.nii \
                        -in /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/sub030/ses2/recognition/templateFunctionalVolume_unwarped.nii \
                        -out {testDir}/templateFunctionalVolume_converted.nii -dof 6 \
                        -omat {testDir}/convert_2_ses1FuncTemp.mat "
            kp_run(cmd)

            cmd = f"bet {testDir}/templateFunctionalVolume_converted.nii {testDir}/templateFunctionalVolume_converted.nii -f 0.3 -R"


            # 直接查看最终的结果, 通过肉眼观察的方法看看是否都是在ses1的functional template中的
            f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/{sub}/ses{ses}/recognition/run{recognitionScan}.nii.gz"
            # /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub005/ses1/recognition/run1.nii.gz /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub005/ses2/recognition/run1.nii.gz

            # 比较 cfg.templateFunctionalVolume_converted  cfg.templateFunctionalVolume
            # /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub005/ses1/recognition/templateFunctionalVolume.nii  /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub005/ses2/recognition/templateFunctionalVolume_converted.nii /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub005/ses2/recognition/templateFunctionalVolume_unwarped.nii
            # /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub009/ses1/recognition/templateFunctionalVolume.nii  /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub009/ses2/recognition/templateFunctionalVolume_converted.nii /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub009/ses2/recognition/templateFunctionalVolume_unwarped.nii
            # /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub012/ses1/recognition/templateFunctionalVolume.nii  /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub012/ses2/recognition/templateFunctionalVolume_converted.nii /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub012/ses2/recognition/templateFunctionalVolume_unwarped.nii
            # /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub012/ses1/recognition/templateFunctionalVolume.nii  /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub012/ses2/recognition/templateFunctionalVolume_converted.nii /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub012/ses2/recognition/templateFunctionalVolume_unwarped.nii /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub012/ses1/recognition/functional_bet.nii.gz

        print(cmd)
        sbatch_response = subprocess.getoutput(cmd)
        print(sbatch_response)
        if backupMode:
            kp_copy(f"{cfg.recognition_dir}/templateFunctionalVolume_unwarped.nii",
                    f"{beforeUnwarpFolder}/templateFunctionalVolume_unwarped_{timeStamp}.nii")

        # for curr_run in actualRuns:
        #     kp_copy(f"{cfg.recognition_dir}/run{curr_run}.nii.gz",f"{beforeUnwarpFolder}/run{curr_run}.nii.gz")

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

            # 做好一系列的操作使得一系列的没有unwarp的文件被unwarp之后的新文件完美取代
            if backupMode:
                kp_copy(f"{cfg.recognition_dir}/recognition_scan{curr_run}_ses{cfg.session}_unwarped.nii.gz",
                        f"{beforeUnwarpFolder}/recognition_scan{curr_run}_ses{cfg.session}_unwarped_{timeStamp}.nii.gz")
                kp_copy(f"{cfg.recognition_dir}/recognition_scan{curr_run}_ses{cfg.session}.nii",
                        f"{beforeUnwarpFolder}/recognition_scan{curr_run}_ses{cfg.session}_{timeStamp}.nii")

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

            # 做好一系列的操作使得一系列的没有unwarp的文件被unwarp之后的新文件完美取代
            if backupMode:
                kp_copy(f"{cfg.feedback_dir}/feedback_scan{curr_run}_ses{cfg.session}_unwarped.nii.gz",
                        f"{beforeUnwarpFolder_feedback}/feedback_scan{curr_run}_ses{cfg.session}_unwarped.nii.gz")
                print(f"renaming {cfg.feedback_dir}/feedback_scan{curr_run}_ses{cfg.session}_unwarped.nii.gz")
                kp_copy(f"{cfg.feedback_dir}/feedback_scan{curr_run}_ses{cfg.session}.nii",
                        f"{beforeUnwarpFolder_feedback}/feedback_scan{curr_run}_ses{cfg.session}.nii")
                print(f"renaming {cfg.feedback_dir}/feedback_scan{curr_run}_ses{cfg.session}.nii")

    # 对于第1个session的数据，重命名文件以使得fmap校准过的，人工校准过的数据能够替代原来的数据。
    elif cfg.session == 1:

        kp_copy(f"{cfg.recognition_dir}/../anat/gm_func_unwarp.nii.gz", f"{cfg.recognition_dir}/../anat/gm_func.nii.gz")

        kp_copy(f"{cfg.recognition_dir}/anat2func_unwarp.mat", f"{cfg.recognition_dir}/anat2func.mat")
        kp_copy(f"{cfg.recognition_dir}/ANATinFUNC_unwarp.nii.gz", f"{cfg.recognition_dir}/ANATinFUNC.nii.gz")

        if backupMode:
            kp_copy(f"{cfg.recognition_dir}/templateFunctionalVolume.nii",
                    f"{beforeUnwarpFolder}/templateFunctionalVolume_{timeStamp}.nii")
        kp_copy(f"{cfg.recognition_dir}/templateFunctionalVolume_unwarped.nii",
                f"{cfg.recognition_dir}/templateFunctionalVolume.nii")
        if backupMode:
            kp_copy(f"{cfg.recognition_dir}/templateFunctionalVolume_bet.nii.gz",
                    f"{beforeUnwarpFolder}/templateFunctionalVolume_bet_{timeStamp}.nii.gz")

        # 如果是第一个session，那么这个session的templateFunctionalVolume_converted 就是本身。如果是后面的session，那么 那个session的templateFunctionalVolume_converted就是那个session的funcTemp转移到第一个session的funcTemp的空间当中。
        if backupMode:
            kp_copy(f"{cfg.recognition_dir}/templateFunctionalVolume_converted.nii",
                    f"{beforeUnwarpFolder}/templateFunctionalVolume_converted_{timeStamp}.nii")
        kp_copy(f"{cfg.recognition_dir}/templateFunctionalVolume.nii",
                f"{cfg.recognition_dir}/templateFunctionalVolume_converted.nii")

        # os.rename(f"{cfg.recognition_dir}/anat2func_beforeUnwarp.mat",f"{beforeUnwarpFolder}/anat2func_beforeUnwarp.mat")
        # os.rename(f"{cfg.recognition_dir}/ANATinFUNC_beforeUnwarp.nii.gz",f"{beforeUnwarpFolder}/ANATinFUNC_beforeUnwarp.nii.gz")
        if backupMode:
            kp_copy(f"{cfg.recognition_dir}/templateFunctionalVolume_unwarped.nii",
                    f"{beforeUnwarpFolder}/templateFunctionalVolume_unwarped_{timeStamp}.nii")
            kp_copy(f"{cfg.recognition_dir}/templateFunctionalVolume_unwarped_bet.nii.gz",
                    f"{beforeUnwarpFolder}/templateFunctionalVolume_unwarped_bet_{timeStamp}.nii.gz")

        # for curr_run in actualRuns:
        #     kp_copy(f"{cfg.recognition_dir}/run{curr_run}.nii.gz",f"{beforeUnwarpFolder}/run{curr_run}.nii.gz")

        # 把经过 fmap 校正和 motion correction 的 recognition_scan{curr_run}_ses{cfg.session}_unwarped_mc.nii.gz 重命名为 平常使用的 run{curr_run}.nii.gz
        for curr_run in actualRuns:
            kp_copy(f"{cfg.recognition_dir}/recognition_scan{curr_run}_ses{cfg.session}_unwarped_mc.nii.gz",
                    f"{cfg.recognition_dir}run{curr_run}.nii.gz")
            print(f"renaming {cfg.recognition_dir}/recognition_scan{curr_run}_ses{cfg.session}_unwarped_mc.nii.gz")

            # 做好一系列的操作使得一系列的没有unwarp的文件被unwarp之后的新文件完美取代
            if backupMode:
                kp_copy(f"{cfg.recognition_dir}/recognition_scan{curr_run}_ses{cfg.session}_unwarped.nii.gz",
                        f"{beforeUnwarpFolder}/recognition_scan{curr_run}_ses{cfg.session}_unwarped_{timeStamp}.nii.gz")
                print(f"renaming {cfg.recognition_dir}/recognition_scan{curr_run}_ses{cfg.session}_unwarped.nii.gz")
                kp_copy(f"{cfg.recognition_dir}/recognition_scan{curr_run}_ses{cfg.session}.nii",
                        f"{beforeUnwarpFolder}/recognition_scan{curr_run}_ses{cfg.session}_{timeStamp}.nii")
                print(f"renaming {cfg.recognition_dir}/recognition_scan{curr_run}_ses{cfg.session}.nii")

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


#
# SesFolder = f"{workingDir}/data/subjects/{sub}/ses{ses}/"
# mkdir(f"{SesFolder}/fmap/")
# os.chdir(f"{SesFolder}/fmap/")
# if not os.path.exists(f"{SesFolder}/fmap/topup_AP_PA_b0_fieldcoef.nii.gz"):
#     cmd = f"bash {workingDir}/data_preprocess/fmap/top.sh {SesFolder}fmap/"
#     kp_run(cmd)
#
# # use topup_AP_PA_fmap to unwarp all functional data
# runRecording = pd.read_csv(f"{SesFolder}/runRecording.csv")
# recogRuns = list(
#     runRecording['run'].iloc[list(np.where(1 == 1 * (runRecording['type'] == 'recognition'))[0])])
# feedbackRuns = list(
#     runRecording['run'].iloc[list(np.where(1 == 1 * (runRecording['type'] == 'feedback'))[0])])
# for scan in recogRuns:
#     scan_file = f"{SesFolder}/recognition/run_{scan}"
#     if not os.path.exists(f"{scan_file}_unwarped.nii.gz"):
#         cmd = (f"applytopup --imain={scan_file}.nii --topup=topup_AP_PA_b0 --datain=acqparams.txt --inindex=1 "
#                f"--out={scan_file}_unwarped --method=jac")
#         kp_run(cmd)
#
# for scan in feedbackRuns:
#     scan_file = f"{SesFolder}/feedback/run_{scan}"
#     if not os.path.exists(f"{scan_file}_unwarped.nii.gz"):
#         cmd = (f"applytopup --imain={scan_file}.nii --topup=topup_AP_PA_b0 --datain=acqparams.txt --inindex=1 "
#                f"--out={scan_file}_unwarped --method=jac")
#         kp_run(cmd)
#
#
# def saveMiddleVolumeAsTemplate(SesFolder='',
#                                scan_asTemplate=1):  # expScripts/recognition/recognitionDataAnalysis/GM_modelTrain.py
#     nii = nib.load(f"{SesFolder}/recognition/run_{scan_asTemplate}_unwarped.nii.gz")
#     frame = nii.get_fdata()
#     TR_number = frame.shape[3]
#     frame = frame[:, :, :, int(TR_number / 2)]
#     frame = nib.Nifti1Image(frame, affine=nii.affine)
#     unwarped_template = f"{SesFolder}/recognition/templateFunctionalVolume_unwarped.nii"
#     nib.save(frame, unwarped_template)
#
#
# # Take the middle volume of the first recognition run as a template
# saveMiddleVolumeAsTemplate(SesFolder=SesFolder, scan_asTemplate=scan_asTemplates[sub][f"ses{ses}"])
#
#
# def align_with_template(SesFolder='',
#                         scan_asTemplate=1):  # expScripts/recognition/recognitionDataAnalysis/GM_modelTrain.py
#     from utils import kp_run
#     unwarped_template = f"{SesFolder}/recognition/templateFunctionalVolume_unwarped.nii"
#     if not os.path.exists(f"{SesFolder}/recognition/functional_bet.nii.gz"):
#         cmd = (f"bet {SesFolder}/recognition/templateFunctionalVolume_unwarped.nii "
#                f"{SesFolder}/recognition/templateFunctionalVolume_unwarped_bet.nii.gz")
#
#         kp_run(cmd)
#         shutil.copyfile(f"{SesFolder}/recognition/templateFunctionalVolume_unwarped_bet.nii.gz",
#                         f"{SesFolder}/recognition/functional_bet.nii.gz")
#
#     # Align all recognition runs and feedback runs with the functional template of the current session
#     for scan in tqdm(recogRuns):
#         head = f"{SesFolder}/recognition/run_{scan}"
#         if not os.path.exists(f"{head}_unwarped_mc.nii.gz"):
#             cmd = f"mcflirt -in {head}_unwarped.nii.gz -out {head}_unwarped_mc.nii.gz"
#             from utils import kp_run
#             kp_run(cmd)
#             wait(f"{head}_unwarped_mc.nii.gz")  # mcflirt for motion correction
#
#         # Align the motion-corrected functional data with the unwarped_template of the current session, which is the funcTemplate corrected by topup.
#         # Then, transfer the motion-corrected data to the func space corrected by topup.
#         # The reason for two steps here is that flirt does not directly produce the -out result for multiple volumes.
#         cmd = f"flirt -in {head}_unwarped_mc.nii.gz " \
#               f"-out {head}_temp.nii.gz " \
#               f"-ref {unwarped_template} " \
#               f"-dof 6 " \
#               f"-omat {SesFolder}/recognition/scan{scan}_to_unwarped.mat"
#
#         def kp_run(cmd):
#             print()
#             print(cmd)
#             import subprocess
#             sbatch_response = subprocess.run(cmd, shell=True, capture_output=True, text=True)
#             check(sbatch_response.stdout)
#             return sbatch_response.stdout
#
#         kp_run(cmd)
#
#         cmd = f"flirt " \
#               f"-in {head}_unwarped_mc.nii.gz " \
#               f"-out {head}_unwarped_mc.nii.gz " \
#               f"-ref {unwarped_template} -applyxfm " \
#               f"-init {SesFolder}/recognition/scan{scan}_to_unwarped.mat"
#         from utils import kp_run
#         kp_run(cmd)
#         kp_remove(f"{head}_temp.nii.gz")
#
#     for scan in tqdm(feedbackRuns):
#         cmd = f"mcflirt -in {SesFolder}/feedback/run_{scan}_unwarped.nii.gz " \
#               f"-out {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz"
#         kp_run(cmd)
#         wait(f"{SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz")
#
#         cmd = f"flirt -in {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz " \
#               f"-out {SesFolder}/feedback/run_{scan}_temp.nii.gz " \
#               f"-ref {unwarped_template} -dof 6 -omat {SesFolder}/feedback/scan{scan}_to_unwarped.mat"
#         kp_run(cmd)
#
#         cmd = f"flirt -in {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz " \
#               f"-out {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz " \
#               f"-ref {unwarped_template} -applyxfm -init {SesFolder}/feedback/scan{scan}_to_unwarped.mat"
#         kp_run(cmd)
#         kp_remove(f"{SesFolder}/feedback/run_{scan}_temp.nii.gz")
#
#         cmd = f"fslinfo {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz"
#         kp_run(cmd)
#
#
# align_with_template(SesFolder=SesFolder, scan_asTemplate=scan_asTemplates[sub][f"ses{ses}"])

print("done")
