testMode = False
autoAlignFlag = True

# 这段代码来自于/Users/kailong/Desktop/rtEnv/rt-cloud/projects/rtSynth_rt/expScripts/recognition/ROI_analysis/ROI_AB_coactivation_Integration_analysis.py
# 希望通过平行化来获得所有的ROI的曲线。

import os, re

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")

# sys.path.append("/home/kp578/.local/lib/python3.7/site-packages/")
# sys.path.append('/home/kp578/.local/lib/python3.7/site-packages/matplotlib/')
import matplotlib.pyplot as plt

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
sys.path.append(projectDir + "/expScripts/recognition/")
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

# import statsmodels.api as sm
# import statsmodels.formula.api as smf
from tqdm import tqdm
import itertools
from sklearn.linear_model import LogisticRegression
import joblib
from scipy import stats

# import seaborn as sns

print(f"numpy version = {np.__version__}")
print(f"pandas version = {pd.__version__}")
# which python version am I running?
print(sys.executable)
print(sys.version)
print(sys.version_info)
print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")


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


sys.path.append(
    '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/expScripts/recognition/ROI_analysis/')
# from ROI_AB_coactivation_Integration_analysis_functions import check_jobArray, check_jobIDs

# 尝试获得所有ROI的曲线。总体逻辑是：
# 一共运行了ROI的数量个代码，也就是44(除去16个无用的海马ROI只有28个ROI)个代码。每一个代码产生一条曲线。
# 对于当前代码被 argv 指定的ROI，设计一个特上个月的sub_ROI_ses_relevant_using并且保存。


if testMode:
    [interestedROI, plot_dir, batch, _normActivationFlag, _UsedTRflag, useNewClf] = [
        'megaROI',
        f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/"
        f"autoAlign_ROIanalysis/cubicFit/batch12/normActivationFlag_True_UsedTRflag_feedback/",
        12, True, "feedback", True]
else:
    # "$SLURM_ARRAY_TASK_ID" "$JobArrayStart" "$batch"
    _batch = int(float(sys.argv[3]))
    print(f"_batch={_batch}")
    jobarrayDict = np.load(
        f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
        f"OrganizedScripts/megaROI/withinSession/autoAlign/nonmonotonicCurve/ROI_nomonotonic_curve_batch_{_batch}_jobID.npy",
        allow_pickle=True)
    jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
    jobarrayID = int(float(sys.argv[1]))
    [interestedROI, plot_dir, batch, _normActivationFlag, _UsedTRflag, useNewClf] = jobarrayDict[jobarrayID]
    assert interestedROI == 'megaROI'
    assert batch == _batch


print(f"useNewClf={useNewClf}")

print(f"interestedROI={interestedROI} plot_dir={plot_dir} batch={batch} "
      f"_normActivationFlag={_normActivationFlag} _UsedTRflag={_UsedTRflag}")
print(f"autoAlignFlag={autoAlignFlag} "
      f"batch={batch} "
      f"testMode={testMode}")
if autoAlignFlag:
    Folder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/"
    [subjects, ROIList, sub_ROI_ses_relevant] = load_obj(
        f"{Folder}/ROI_nomonotonic_curve_batch_autoAlign_batch_{batch}")
    # sub_ROI_ses_relevant
    # 	sub	ses	chosenMask	Relevant
    # 0	sub003	1	megaROI	True
    # 1	sub003	2	megaROI	True
    # 2	sub003	3	megaROI	True
    # 3	sub004	1	megaROI	True
    # 4	sub004	2	megaROI	True
    # 5	sub004	3	megaROI	True
    # 6	sub005	1	megaROI	True
    # 7	sub005	2	megaROI	True
    # 8	sub005	3	megaROI	True
    # 9	sub006	1	megaROI	True
    # 10	sub006	2	megaROI	True
    # 11	sub006	3	megaROI	True
    # 12	sub008	1	megaROI	True
    # 13	sub008	2	megaROI	True
    # 14	sub008	3	megaROI	True
    # 15	sub009	1	megaROI	True
    # 16	sub009	2	megaROI	True
    # 17	sub009	3	megaROI	True
    # 18	sub012	1	megaROI	True
    # 19	sub012	2	megaROI	True
    # 20	sub012	3	megaROI	True
    # 21	sub013	1	megaROI	True
    # 22	sub013	2	megaROI	True
    # 23	sub013	3	megaROI	True
    # 24	sub014	1	megaROI	True
    # 25	sub014	2	megaROI	True
    # 26	sub014	3	megaROI	True
    # 27	sub015	1	megaROI	True
    # 28	sub015	2	megaROI	True
    # 29	sub015	3	megaROI	True
    # 30	sub018	1	megaROI	True
    # 31	sub018	2	megaROI	True
    # 32	sub018	3	megaROI	True
    # 33	sub021	1	megaROI	True
    # 34	sub021	2	megaROI	True
    # 35	sub021	3	megaROI	True
    # 36	sub022	1	megaROI	True
    # 37	sub022	2	megaROI	True
    # 38	sub022	3	megaROI	True
    # 39	sub023	1	megaROI	True
    # 40	sub023	2	megaROI	True
    # 41	sub023	3	megaROI	True
    # 42	sub024	1	megaROI	True
    # 43	sub024	2	megaROI	True
    # 44	sub024	3	megaROI	True
    # 45	sub026	1	megaROI	True
    # 46	sub026	2	megaROI	True
    # 47	sub026	3	megaROI	True
    # 48	sub027	1	megaROI	True
    # 49	sub027	2	megaROI	True
    # 50	sub027	3	megaROI	True
    # 51	sub029	1	megaROI	True
    # 52	sub029	2	megaROI	True
    # 53	sub029	3	megaROI	True
    # 54	sub030	1	megaROI	True
    # 55	sub030	2	megaROI	True
    # 56	sub030	3	megaROI	True
    # 57	sub031	1	megaROI	True
    # 58	sub031	2	megaROI	True
    # 59	sub031	3	megaROI	True
else:
    raise Exception("not implemented")
    # [subjects, ROIList, sub_ROI_ses_relevant] = load_obj(
    #     f"/gpfs/milgram/scratch60/turk-browne/kp578/ROI_nomonotonic_curve_batch_{batch}")

print(f"interestedROI={interestedROI}")
print(f"sub_ROI_ses_relevant={sub_ROI_ses_relevant}")
print(f"len(sub_ROI_ses_relevant)={len(sub_ROI_ses_relevant)}")

sub_ROI_ses_relevant_using = sub_ROI_ses_relevant[
    sub_ROI_ses_relevant['chosenMask'] == interestedROI].copy().reset_index(drop=True)

print(f"len(sub_ROI_ses_relevant_using)={len(sub_ROI_ses_relevant_using)}")
print(f"sub_ROI_ses_relevant_using={sub_ROI_ses_relevant_using}")


def getIntegrationScore(sub='', ses=1, chosenMask=''):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    # import seaborn as sns
    import pandas as pd
    from sklearn import datasets, linear_model
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    from scipy import stats
    import os
    cfg = f"{sub}.ses{ses}.toml"
    cfg = cfg_loading(cfg)
    if testMode:
        print(f"cfg.batch={cfg.batch}")
    if cfg.batch == "batch1":
        batchIm = {
            "A": "X",
            "B": "Y",
            "C": "M",
            "D": "N",
        }
        batchAxis = {
            "AB": "XY",
            "AC": "XM",
            "AD": "XN",
            "BC": "YM",
            "BD": "YN",
            "CD": "MN"
        }
    elif cfg.batch == "batch2":
        batchIm = {
            "C": "X",
            "D": "Y",
            "A": "M",
            "B": "N",
        }
        batchAxis = {
            "AB": "MN",
            "AC": "XM",
            "AD": "YM",
            "BC": "XN",
            "BD": "YN",
            "CD": "XY"
        }
    else:
        raise Exception("cfg.batch should be batch1 or batch2")

    # 获得纵坐标的每一个session的分化值大小
    # 对于每一个sub，对于 2 3 4 session，对于【AB AC AD BC BD CD】加载前一天训练好的clf，然后对于这个session的前两个run的数据进行score，并且保存在一个有如下项目的dataframe当中 subject	session	run12_acc	run34_acc	axis
    subjectFolder = "/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/"
    imcodeDict = {"A": "bed", "B": "Chair", "C": "table", "D": "bench"}
    clfDict = {
        "AB": 'bedtable_bedchair.joblib',
        "AC": 'bedbench_bedtable.joblib',
        "AD": 'bedchair_bedbench.joblib',
        "BC": 'benchchair_chairtable.joblib',
        "BD": 'bedchair_chairbench.joblib',
        "CD": 'bedtable_tablebench.joblib'
    }
    session234BeforeAfterAcc = pd.DataFrame()
    nextSession = int(ses + 1)  # 当ses的mask的relevant的时候，说明下一个session的 feedback 数据可以用。因此使用nextSession
    for axis in ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']:
        def loadFEAT_META(sub='', nextSession=2, axis='', chosenMask=''):
            # 加载这个session的4个recognition run
            runRecording = pd.read_csv(f"{subjectFolder}{sub}/ses{nextSession}/runRecording.csv")  # load
            actualRuns = list(runRecording['run'].iloc[list(np.where(1 == 1 * (runRecording['type'] == 'recognition'))[
                                                                0])])  # can be [1,2,3,4,5,6,7,8] or [1,2,4,5]
            if testMode:
                print(f"actualRuns={actualRuns}")
            assert (len(actualRuns) == 4)

            mask = np.load(f"{cfg.chosenMask}")  # load
            assert cfg.chosenMask == f"{cfg.subjects_dir}{cfg.subjectName}/ses1/recognition/chosenMask.npy"
            print(f"loading {cfg.chosenMask}")

            recognition_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/subjects/" \
                              f"{sub}/ses{nextSession}/recognition/"
            new_run_indexs = []
            new_run_index = 1
            brain_data = None
            behav_data = None
            for ii, run in enumerate(actualRuns):  # load behavior and brain data for current session
                t_brain = np.load(f"{recognition_dir}brain_run{run}.npy")  # load
                # mask = np.load(f"{chosenMask}")
                # print(f"loading {chosenMask}")
                # try:
                #     mask = nib.load(f"{maskFolder}/{chosenMask}.nii").get_fdata()
                # except:
                #     mask = nib.load(f"{maskFolder}/{chosenMask}.nii.gz").get_fdata()
                t_brain = t_brain[:, mask == 1]
                t_brain = normalize(t_brain)
                brain_data = t_brain if ii == 0 else np.concatenate((brain_data, t_brain), axis=0)

                t_behav = pd.read_csv(f"{recognition_dir}behav_run{run}.csv")  # load
                t_behav['run_num'] = new_run_index
                new_run_indexs.append(new_run_index)
                new_run_index += 1
                behav_data = t_behav if ii == 0 else pd.concat([behav_data, t_behav])
            _FEAT = brain_data
            if testMode:
                print(f"FEAT.shape={_FEAT.shape}")
            assert len(_FEAT.shape) == 2
            _META = behav_data
            return _FEAT, _META

        FEAT, META = loadFEAT_META(sub=sub, nextSession=nextSession, axis=axis, chosenMask=chosenMask)

        # convert item colume to label colume
        label = []
        for curr_trial in range(META.shape[0]):
            label.append(imcodeDict[META['Item'].iloc[curr_trial]])
        META['label'] = label  # merge the label column with the data dataframe
        if testMode:
            print(f"np.unique(list(META['run_num']))={np.unique(list(META['run_num']))}")
        # 加载前一个session训练好的clf
        if useNewClf:
            model_folder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/" \
                           f"subjects/{sub}/ses{ses}/{chosenMask}/clf/"  # load
        else:
            model_folder = f"{cfg.recognition_dir}/clf/"  # load
        # t_clf = joblib.load(f"{subjectFolder}{sub}/ses{ses}/recognition/ROI_analysis/{chosenMask}/clf/{clfDict[axis]}")
        t_clf = joblib.load(f"{model_folder}/{clfDict[axis]}")  # load

        # 筛选数据只有对应的axis的两种数据，比如A，B或者C，D
        testIX = kp_and([
            (META['label'] == imcodeDict[axis[0]]) | (META['label'] == imcodeDict[axis[1]]),
            META['run_num'] < 2.5
        ])
        testX = FEAT[testIX]
        testY = META.iloc[np.asarray(testIX)].label
        run12_acc = t_clf.score(testX, testY)

        testIX = kp_and([
            (META['label'] == imcodeDict[axis[0]]) | (META['label'] == imcodeDict[axis[1]]),
            META['run_num'] > 2.5
        ])
        testX = FEAT[testIX]
        testY = META.iloc[np.asarray(testIX)].label
        run34_acc = t_clf.score(testX, testY)

        if testMode:
            print(f"num of trials from 34 run {np.sum(META['run_num'] > 2.5)}")

        session234BeforeAfterAcc = pd.concat([
            session234BeforeAfterAcc,
            pd.DataFrame({
                'subject': sub,
                'session': nextSession,
                'run12_acc': run12_acc,
                'run34_acc': run34_acc,
                'axis': axis,
                'batchAxis': batchAxis[axis]}, index=[0])
        ], ignore_index=True)

        # session234BeforeAfterAcc = session234BeforeAfterAcc.append(
        #     {
        #         'subject': sub,
        #         'session': nextSession,
        #         'run12_acc': run12_acc,
        #         'run34_acc': run34_acc,
        #         'axis': axis,
        #         'batchAxis': batchAxis[axis]
        #     }, ignore_index=True
        # )

    # 获得这一个sub这一个session这一个chosenMask的整合的数值
    t = session234BeforeAfterAcc[kp_and([
        session234BeforeAfterAcc['subject'] == sub,
        session234BeforeAfterAcc['session'] == nextSession,
    ])]
    # run34_AB = float(t[t['axis']=='AB']['run34_acc'])
    # run12_AB = float(t[t['axis']=='AB']['run12_acc'])
    # run34_CD = float(t[t['axis']=='CD']['run34_acc'])
    # run12_CD = float(t[t['axis']=='CD']['run12_acc'])
    run34_XY = float(t[t['batchAxis'] == 'XY']['run34_acc'])
    run12_XY = float(t[t['batchAxis'] == 'XY']['run12_acc'])
    if testMode:
        print(f"cfg.batch={cfg.batch} ; using as experiment axis: {t[t['batchAxis'] == 'XY']['axis']}")

    run34_MN = float(t[t['batchAxis'] == 'MN']['run34_acc'])
    run12_MN = float(t[t['batchAxis'] == 'MN']['run12_acc'])
    if testMode:
        print(f"cfg.batch={cfg.batch} ; using as control axis:  {t[t['batchAxis'] == 'MN']['axis']}")

    # differentiation_ratio = (run34_AB - run12_AB)/(run34_AB + run12_AB) - (run34_CD - run12_CD)/(run34_CD + run12_CD)
    differentiation_ratio = (run34_XY - run12_XY) / (run34_XY + run12_XY) - \
                            (run34_MN - run12_MN) / (run34_MN + run12_MN)
    integration_ratio = - differentiation_ratio

    return integration_ratio, run34_XY, run12_XY, run34_MN, run12_MN


integration_ratios = []
run34_XYs = []
run12_XYs = []
run34_MNs = []
run12_MNs = []
for i in tqdm(range(len(sub_ROI_ses_relevant_using))):
    sub = sub_ROI_ses_relevant_using.loc[i, 'sub']
    ses = int(sub_ROI_ses_relevant_using.loc[i, 'ses'])
    chosenMask = sub_ROI_ses_relevant_using.loc[i, 'chosenMask']
    if testMode:
        print(f"sub={sub} ses={ses} chosenMask={chosenMask}")
    integration_ratio, run34_XY, run12_XY, run34_MN, run12_MN = getIntegrationScore(sub=sub, ses=ses, chosenMask=chosenMask)
    integration_ratios.append(integration_ratio)
    run34_XYs.append(run34_XY)
    run12_XYs.append(run12_XY)
    run34_MNs.append(run34_MN)
    run12_MNs.append(run12_MN)

sub_ROI_ses_relevant_using['integration_ratios'] = integration_ratios
sub_ROI_ses_relevant_using['run34_XYs'] = run34_XYs
sub_ROI_ses_relevant_using['run12_XYs'] = run12_XYs
sub_ROI_ses_relevant_using['run34_MNs'] = run34_MNs
sub_ROI_ses_relevant_using['run12_MNs'] = run12_MNs

# def getNumberOfRuns(subjects='', sub_ROI_ses_relevant_using=None, argvID=1, batch=0):  # 计算给定的被试都有多少个feedback run
#     jobArray = {}
#     count = 0
#     for curr_row in range(len(sub_ROI_ses_relevant_using)):
#         sub = sub_ROI_ses_relevant_using.loc[curr_row, 'sub']
#         ses_i = int(sub_ROI_ses_relevant_using.loc[curr_row, 'ses'])
#         nextSes_i = int(ses_i + 1)
#         assert ses_i in [1, 2, 3]
#         assert nextSes_i in [2, 3, 4]
#         chosenMask = sub_ROI_ses_relevant_using.loc[curr_row, 'chosenMask']
#         assert chosenMask == 'megaROI'
#
#         runRecording = pd.read_csv(
#             f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/"
#             f"{sub}/ses{nextSes_i}/runRecording.csv")
#         feedback_runRecording = runRecording[runRecording['type'] == 'feedback'].reset_index()
#
#         # print(f"sub={sub} ses={nextSes_i} chosenMask={chosenMask}")
#         for currFeedbackRun in range(len(feedback_runRecording)):
#             count += 1
#             forceReRun = True
#             if not forceReRun:
#                 if autoAlignFlag:
#                     history_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/subjects/" \
#                                   f"{sub}/ses{nextSes_i}/feedback/"
#                     # history_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis/" \
#                     #               f"subjects/{sub}/ses{nextSes_i}/{chosenMask}/rtSynth_rt_ABCD_ROIanalysis/"
#                 else:
#                     raise Exception("not implemented")
#                 history_file = f"{history_dir}/history_runNum_{currFeedbackRun + 1}.csv"
#                 if not os.path.exists(history_file):
#                     jobArray[count] = [
#                         sub,
#                         f"ses{nextSes_i}",  # assert nextSes_i in [2, 3, 4]
#                         currFeedbackRun + 1,  # runNum
#                         feedback_runRecording.loc[currFeedbackRun, 'run'],  # scanNum
#                         chosenMask,
#                         forceReRun
#                     ]
#             else:
#                 jobArray[count] = [
#                     sub,
#                     f"ses{nextSes_i}",  # assert nextSes_i in [2, 3, 4]
#                     currFeedbackRun + 1,
#                     feedback_runRecording.loc[currFeedbackRun, 'run'],
#                     chosenMask,
#                     forceReRun
#                 ]
#
#             if autoAlignFlag:
#                 history_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis/" \
#                               f"subjects/{sub}/ses{nextSes_i}/{chosenMask}/rtSynth_rt_ABCD_ROIanalysis/"
#                 # history_dir = f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session}/recognition//ROI_analysis/{chosenMask}/rtSynth_rt_ABCD_ROIanalysis/"
#             else:
#                 raise Exception("not implemented yet")
#             mkdir(history_dir)
#     jobArrayPath = f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/" \
#                    f"OrganizedScripts/ROI/autoAlign/nonmonotonicCurve/simulatedFeedbackRun/" \
#                    f"rtSynth_rt_ABCD_ROIanalysis_unwarp_batch{batch}_argvID_{argvID}"
#     save_obj(jobArray, jobArrayPath)
#     return len(jobArray), jobArrayPath
#
#
# # 已经在simulate运行过了, 不需要重复运行
# if batch in []:  # [1, 2, 12] # 当单独处理第一批或者第二批数据的时候，要模拟feedback的处理。   但是，当合并处理1 2 批数据的时候，直接使用之前的结果，不重复计算。
#     # 离线运行 feedback 实时处理。
#     jobNumber, jobArrayPath = getNumberOfRuns(
#         subjects=subjects,
#         sub_ROI_ses_relevant_using=sub_ROI_ses_relevant_using,
#         argvID=argvID,
#         batch=batch)
#     cmd = f"sbatch --requeue --array=1-{jobNumber} /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/" \
#           f"rtSynth_rt/OrganizedScripts/ROI/autoAlign/nonmonotonicCurve/simulatedFeedbackRun/" \
#           f"rtSynth_rt_ABCD_ROIanalysis_unwarp.sh {jobArrayPath} "  # 大约需要一分钟结束
#     print(cmd)
#     sbatch_response = subprocess.getoutput(cmd)
#     print(sbatch_response)
#     print("takes about 1 min to finish")
#     from recognition_dataAnalysisFunctions import jobID_running_myjobs, getjobID_num, waitForEnd
#
#     jobID = getjobID_num(sbatch_response)
#     waitForEnd(jobID)
#     completed = check_jobArray(jobID=jobID, jobarrayNumber=jobNumber)  # 检查所有运行的job都是成功完成的。

def float_history(history):  # 解决history一部分的数据不是float的问题
    _float_history = pd.DataFrame()
    for i in range(len(history)):
        # _float_history = _float_history.append({
        #     'XY_clf_X': float(history.loc[i, 'XY_clf_X'][1:-1]),
        #     'XY_clf_Y': float(history.loc[i, 'XY_clf_Y'][1:-1]),
        #     'Xprob': history.loc[i, 'Xprob'],
        #     'Yprob': history.loc[i, 'Yprob'],
        #     'MN_clf_M': float(history.loc[i, 'MN_clf_M'][1:-1]),
        #     'MN_clf_N': float(history.loc[i, 'MN_clf_N'][1:-1]),
        #     'Mprob': history.loc[i, 'Mprob'],
        #     'Nprob': history.loc[i, 'Nprob'],
        #     'Sub': history.loc[i, 'Sub'],
        #     'TR_milgram': history.loc[i, 'TR_milgram'],
        #     'states': history.loc[i, 'states'],
        #     'modifiedStates': history.loc[i, 'modifiedStates'],
        #     'trialID': history.loc[i, 'trialID'],
        # }, ignore_index=True)
        _float_history = pd.concat([_float_history, pd.DataFrame({
            'XY_clf_X': float(history.loc[i, 'XY_clf_X'][1:-1]),
            'XY_clf_Y': float(history.loc[i, 'XY_clf_Y'][1:-1]),
            'Xprob': history.loc[i, 'Xprob'],
            'Yprob': history.loc[i, 'Yprob'],
            'XxY': history.loc[i, 'XxY'],
            'MxN': history.loc[i, 'MxN'],
            'min(Xprob, Yprob)': history.loc[i, 'min(Xprob, Yprob)'],
            'MN_clf_M': float(history.loc[i, 'MN_clf_M'][1:-1]),
            'MN_clf_N': float(history.loc[i, 'MN_clf_N'][1:-1]),
            'Mprob': history.loc[i, 'Mprob'],
            'Nprob': history.loc[i, 'Nprob'],
            'Sub': history.loc[i, 'Sub'],
            'TR_milgram': history.loc[i, 'TR_milgram'],
            'states': history.loc[i, 'states'],
            'modifiedStates': history.loc[i, 'modifiedStates'],
            'trialID': history.loc[i, 'trialID'],
        }, index=[0])], ignore_index=True)

    return _float_history


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
    trialID = []
    curr_trial = 0
    for ii in range(len(states)):
        if [states[ii - 1], states[ii]] == ['ITI', 'waiting']:
            curr_trial += 1
        trialID.append(curr_trial)
    history['trialID'] = trialID

    return history


def getTrialMean_feedabckTR(history, target='Xprob'):
    target_value = []
    for currTrialID in range(1, int(1 + np.max(history['trialID']))):
        t = history[
            kp_and([history['trialID'] == currTrialID,
                    history['states'] == 'feedback'
                    ])
        ]
        # print(t)
        target_value.append(np.mean(t[target]))
    return target_value


def getTrialMean_ITI_TR(history, target='Xprob'):
    target_value = []
    for currTrialID in range(1, int(1 + np.max(history['trialID']))):
        t = history[
            kp_and([history['trialID'] == currTrialID,
                    history['modifiedStates'] == 'ITI'
                    ])
        ]
        target_value.append(np.mean(t[target]))
    return target_value


sys.path.append("/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/")
from utils import getMonDate, checkDate


def loadSimiulatedData(normActivationFlag=None, UsedTRflag=None):
    tag = f"normActivationFlag_{normActivationFlag}_UsedTRflag_{UsedTRflag}"
    print(f"tag={tag}")

    # plot_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/" \
    #            f"/cubicFit/batch{int(batch)}/{tag}/"
    # plot_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/" \
    #            f"cubicFit/batch{int(batch)}/{tag}/"

    print(f"plot_dir={plot_dir}")
    mkdir(plot_dir)

    # XY 共激活的程度？ 对于所有的被记住的 ROI sub session 的名字，计算 第 session+1 个session 的 feedback session 的所有的 TR 的 BC_clf_B BD_clf_B AC_clf_A AD_clf_A XY_clf_X
    allTRs_all = pd.DataFrame()
    for curr_row in tqdm(range(len(sub_ROI_ses_relevant_using))):
        sub = sub_ROI_ses_relevant_using.loc[curr_row, 'sub']
        ses = int(sub_ROI_ses_relevant_using.loc[curr_row, 'ses'])
        nextSession = int(ses + 1)
        assert ses in [1, 2, 3]
        assert nextSession in [2, 3, 4]
        chosenMask = sub_ROI_ses_relevant_using.loc[curr_row, 'chosenMask']
        print(f"sub={sub} nextSession={nextSession} chosenMask={chosenMask}")
        # currRun = 0

        runRecording = pd.read_csv(
            f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/"
            f"{sub}/ses{nextSession}/runRecording.csv")
        feedback_runRecording = runRecording[runRecording['type'] == 'feedback'].reset_index()

        # print(f"sub={sub} ses={nextSes_i} chosenMask={chosenMask}")
        for currRun in range(1, 1 + len(feedback_runRecording)):
            # while True:
            # currRun = 1history_runNum_{runNum}.csv
            if autoAlignFlag:
                # csvPath = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis/" \
                #           f"subjects/{sub}/ses{nextSession}/{chosenMask}/rtSynth_rt_ABCD_ROIanalysis/" \
                #           f"{sub}_{currRun}_history_rtSynth_RT_ABCD.csv"
                csvPath = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/subjects/" \
                          f"{sub}/ses{nextSession}/feedback/history_runNum_{currRun}.csv"
            else:
                raise Exception("not implemented")
                # 我可以选择只使用旧的chosenmask.npy, 但是训练新的clf, 也可以使用旧的chosenmask和旧的clf, 还可以使用旧的chosenmask旧的clf和旧的history.csv

                # csvPath = f'/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/' \
                #           f'{sub}/ses{nextSession}/recognition/ROI_analysis/{chosenMask}/rtSynth_rt_ABCD_ROIanalysis/' \
                #           f'{sub}_{currRun}_history_rtSynth_RT_ABCD.csv'

            history = pd.read_csv(csvPath)

            def getMxN(_history):
                _history['MxN'] = _history['Mprob'] * _history['Nprob']
                return _history

            history = getMxN(history)
            history = Get_modifiedStates(history)
            history = float_history(history)

            if normActivationFlag:
                # normalize within each run, to get zscored XY activation like What I did in sl.
                print(f"running history = normXY(history)")

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

                history = normXY(history)
                assert np.mean(history['MxN']) < 1e-10
                assert np.std(history['MxN']) - 1 < 1e-10
                assert np.mean(history['XxY']) < 1e-10
                assert np.std(history['XxY']) - 1 < 1e-10

            # 保留所有的TR 的数值
            feedback_X = list(history['Xprob'])
            feedback_Y = list(history['Yprob'])
            feedback_M = list(history['Mprob'])
            feedback_N = list(history['Nprob'])
            states = list(history['states'])
            modifiedStates = list(history['modifiedStates'])
            feedback_XxY = list(history['XxY'])
            feedback_MxN = list(history['MxN'])
            feedback_minXY = list(history['min(Xprob, Yprob)'])

            feedback_XY_clf_X = list(history['XY_clf_X'])
            feedback_XY_clf_Y = list(history['XY_clf_Y'])
            feedback_MN_clf_M = list(history['MN_clf_M'])
            feedback_MN_clf_N = list(history['MN_clf_N'])

            trial_ID = list(history['trialID'])
            assert len(feedback_X) == len(feedback_Y)
            assert len(feedback_X) == len(feedback_M)
            assert len(feedback_X) == len(feedback_N)
            for i in range(len(feedback_X)):
                # allTRs_all = allTRs_all.append({
                #     'sub': sub,
                #     'currSes': nextSession,
                #     'currRun': currRun,
                #     'trial_ID': trial_ID[i],
                #     'states': states[i],
                #     'modifiedStates': modifiedStates[i],
                #
                #     'X': feedback_X[i],
                #     'Y': feedback_Y[i],
                #     'M': feedback_M[i],
                #     'N': feedback_N[i],
                #     'XxY': feedback_XxY[i],
                #     'minXY': feedback_minXY[i],
                #
                #     'XY_clf_X': feedback_XY_clf_X[i],
                #     'XY_clf_Y': feedback_XY_clf_Y[i],
                #     'MN_clf_M': feedback_MN_clf_M[i],
                #     'MN_clf_N': feedback_MN_clf_N[i],
                # }, ignore_index=True)
                allTRs_all = pd.concat([allTRs_all, pd.DataFrame({
                    'sub': sub,
                    'currSes': nextSession,
                    'currRun': currRun,
                    'trial_ID': trial_ID[i],
                    'states': states[i],
                    'modifiedStates': modifiedStates[i],

                    'X': feedback_X[i],
                    'Y': feedback_Y[i],
                    'M': feedback_M[i],
                    'N': feedback_N[i],
                    'XxY': feedback_XxY[i],
                    'MxN': feedback_MxN[i],
                    'minXY': feedback_minXY[i],

                    'XY_clf_X': feedback_XY_clf_X[i],
                    'XY_clf_Y': feedback_XY_clf_Y[i],
                    'MN_clf_M': feedback_MN_clf_M[i],
                    'MN_clf_N': feedback_MN_clf_N[i],
                }, index=[0])], ignore_index=True)

            if currRun > 15:
                break

    print(f"allTRs_all.shape={allTRs_all.shape}")
    print(f"allTRs_all={allTRs_all}")
    XY_coactivation = []
    MN_coactivation = []
    XY_min = []
    X_probs = []
    Y_probs = []
    for i in range(len(sub_ROI_ses_relevant_using)):
        sub = sub_ROI_ses_relevant_using.loc[i, 'sub']
        ses = int(sub_ROI_ses_relevant_using.loc[i, 'ses'])
        nextSes = int(ses + 1)
        chosenMask = sub_ROI_ses_relevant_using.loc[i, 'chosenMask']

        if UsedTRflag == "feedback":
            t = allTRs_all[kp_and([
                allTRs_all['sub'] == sub,
                allTRs_all['currSes'] == nextSes,
                allTRs_all['states'] == 'feedback',  # 这里使用的是state而不是 modifiedStates, 说明我没有采用trail的TR.
            ])]
        elif UsedTRflag == "feedback_trail":
            t = allTRs_all[kp_and([
                allTRs_all['sub'] == sub,
                allTRs_all['currSes'] == nextSes,
                kp_or([allTRs_all['modifiedStates'] == 'feedback', allTRs_all['modifiedStates'] == 'trail'])
                # 这里使用的是state而不是 modifiedStates, 说明我没有采用trail的TR.
            ])]
        elif UsedTRflag == "all":
            t = allTRs_all[kp_and([
                allTRs_all['sub'] == sub,
                allTRs_all['currSes'] == nextSes
            ])]
        else:
            raise Exception(f"UsedTRflag={UsedTRflag} is not defined")
        X_prob = np.mean(t['X'])  # 这个被试这个session的所有feedback的X激活值, 不包括trail 的TR.
        Y_prob = np.mean(t['Y'])  # 这个被试这个session的所有feedback的Y激活值, 不包括trail 的TR.
        XxY_prob = np.mean(t['XxY'])  # 这个被试这个session的所有feedback的XxY激活值, 不包括trail 的TR.
        MxN_prob = np.mean(t['MxN'])  # 这个被试这个session的所有feedback的MxN激活值, 不包括trail 的TR.
        minXY_prob = np.mean(t['minXY'])  # 这个被试这个session的所有feedback的minXY激活值, 不包括trail 的TR.

        X_probs.append(X_prob)
        Y_probs.append(Y_prob)
        XY_coactivation.append(XxY_prob)
        MN_coactivation.append(MxN_prob)
        XY_min.append(minXY_prob)
        # XY_coactivation.append(np.mean(t['X'] * t['Y']))  # 不应该是均值后相乘，应该是均值前相乘。
        # t_XY_min = np.min(np.asarray(t[['X', 'Y']]),
        #                   axis=1)  # 这里对每一个feedback TR 的 X and Y （batch1 的时候X是bed Y是chair）进行均值。然后再每一个session求一个均值。换句话说，XY_min是一个list，其中的每一个元素都是一个session的 均值。
        # assert len(t_XY_min.shape) == 1
        # print(f"t_XY_min={t_XY_min}")
        # XY_min.append(np.mean(t_XY_min))

    sub_ROI_ses_relevant_using['X_probs'] = X_probs
    sub_ROI_ses_relevant_using['Y_probs'] = Y_probs
    sub_ROI_ses_relevant_using['XY_coactivation'] = XY_coactivation
    sub_ROI_ses_relevant_using['MN_coactivation'] = MN_coactivation
    sub_ROI_ses_relevant_using['XY_min'] = XY_min

    save_obj(sub_ROI_ses_relevant_using, f"{plot_dir}/sub_ROI_ses_relevant_using_{interestedROI}_batch{batch}")  # save
    print(f"saved {plot_dir}/sub_ROI_ses_relevant_using_{interestedROI}_batch{batch}")

    # 作图
    plotFigure = False
    if plotFigure:
        plt.figure()
        plt.plot(sub_ROI_ses_relevant_using['XY_coactivation'], sub_ROI_ses_relevant_using['integration_ratios'], '.')
        plt.title(f"X*Y versus integration {interestedROI}")
        plt.xlabel("X*Y")
        plt.ylabel("integration score")
        plt.savefig(f"{plot_dir}XY_versus_integration_{interestedROI}_batch{batch}.jpg")  # save
        plt.close()

        plt.figure()
        plt.plot(sub_ROI_ses_relevant_using['Y_probs'], sub_ROI_ses_relevant_using['integration_ratios'], '.')
        plt.title(f"Y versus integration {interestedROI}")
        plt.xlabel("Y")
        plt.ylabel("integration score")
        plt.savefig(f"{plot_dir}Y_versus_integration_{interestedROI}_batch{batch}.jpg")  # save
        plt.close()

        plt.figure()
        plt.plot(sub_ROI_ses_relevant_using['XY_min'], sub_ROI_ses_relevant_using['integration_ratios'], '.')
        plt.title(f"min(X,Y) versus integration {interestedROI}")
        plt.xlabel("min(X,Y)")
        plt.ylabel("integration score")
        plt.savefig(f"{plot_dir}min_XY_versus_integration_{interestedROI}_batch{batch}.jpg")  # save
        plt.close()


print(f"running loadSimiulatedData {_normActivationFlag}, {_UsedTRflag}")
loadSimiulatedData(normActivationFlag=_normActivationFlag, UsedTRflag=_UsedTRflag)
# if testMode:
#     _normActivationFlag = False
#     _UsedTRflag = "feedback"
#     loadSimiulatedData(normActivationFlag=_normActivationFlag, UsedTRflag=_UsedTRflag)
# else:
#     for [_normActivationFlag, _UsedTRflag] in [[True, 'feedback'],
#                                                [False, 'feedback'],
#                                                [True, 'feedback_trail'],
#                                                [False, 'feedback_trail'],
#                                                [True, 'all'],
#                                                [False, 'all']]:
#         loadSimiulatedData(normActivationFlag=_normActivationFlag, UsedTRflag=_UsedTRflag)

print("done")
