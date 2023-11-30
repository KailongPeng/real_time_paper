import os
import sys
import re
import pandas as pd
import subprocess
import numpy as np
import time
import pickle5 as pickle
from tqdm import tqdm

print(f"numpy version = {np.__version__}")
print(f"pandas version = {pd.__version__}")
# which python version am I running?
print(sys.executable)
print(sys.version)
print(sys.version_info)
print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        # pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    if name[-3:] == 'pkl':
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def getjobID_num(sbatch_response):  # 根据subprocess.Popen输出的proc，获得sbatch的jpobID
    import re
    jobID = re.findall(r'\d+', sbatch_response)[0]
    return jobID


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


def kp_rename(file1, file2):
    cmd = f"mv {file1} {file2}";
    print(cmd);
    sbatch_response = subprocess.getoutput(cmd);
    print(sbatch_response)
    return 1


def kp_copy(file1, file2):
    cmd = f"cp {file1} {file2}";
    print(cmd);
    sbatch_response = subprocess.getoutput(cmd);
    print(sbatch_response)
    return 1


def kp_run(cmd):
    print()
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd)
    check(sbatch_response)
    return sbatch_response


def kp_remove(fileName):
    cmd = f"rm {fileName}"
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd)
    print(sbatch_response)


def wait(tmpFile, waitFor=0.1):
    while not os.path.exists(tmpFile):
        time.sleep(waitFor)
    return 1


def check(sbatch_response):
    print(sbatch_response)
    if "Exception" in sbatch_response or "Error" in sbatch_response or "Failed" in sbatch_response or "not" in sbatch_response or 'Unrecognised' in sbatch_response:
        raise Exception(sbatch_response)


def checkEndwithDone(filename):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                last_line = f.readlines()[-1]
            EndwithDone = last_line == "done\n"
            # EndwithDone = (last_line == "done\n") or (last_line == "done")
        except UnicodeDecodeError:
            with open(filename, 'br') as f:
                last_line = f.readlines()[-1]
            EndwithDone = last_line == b"done\n"
            # EndwithDone = (last_line == b"done\n") or (last_line == b"done")
    else:
        EndwithDone = False
    return EndwithDone


def checkDone(jobIDs):
    from tqdm import tqdm
    completed = {}
    for jobID in tqdm(jobIDs):
        filename = f"./logs/{jobID}.out"
        completed[jobID] = checkEndwithDone(filename)
    if np.mean(list(completed.values())) == 1:
        status = True
    else:
        status = False
    return completed, status


def check_jobIDs(jobIDs):
    completed, status = checkDone(jobIDs)
    if status == True:
        pass
    else:
        print(completed)
        assert status == True
    return completed


def check_jobArray(jobID='', jobarrayNumber=10):
    arrayIDrange = np.arange(1, 1 + jobarrayNumber)
    jobIDs = []
    for arrayID in arrayIDrange:
        jobIDs.append(f"{jobID}_{arrayID}")
    completed = check_jobIDs(jobIDs)
    return completed


def waitForEnd(jobID):
    while jobID_running_myjobs(jobID):
        print(f"waiting for {jobID} to end")
        time.sleep(5)
    print(f"{jobID} finished")


def jobID_running_myjobs(jobID):
    jobID = str(jobID)
    cmd = "squeue -u kp578"
    sbatch_response = subprocess.getoutput(cmd)
    if jobID in sbatch_response:
        return True
    else:
        return False


def readtxt(file):
    # f = open(file, "r")
    f = open(file, "r", encoding="utf8")
    return f.read()


def writetxt(file, content):
    # with open(file, 'w') as f:
    with open(file, 'w', encoding="utf8") as f:
        f.write(content)


def get_subjects(batch=0):  # batch = 1 2 12
    # 对哪些被试进行分析？

    if batch == 1:
        subjects = []
        for sub_i in [3, 4, 5, 6, 8, 9, 12, 13, 14, 15]:
            subjects.append(f"sub{str(sub_i).zfill(3)}")
        scan_asTemplates = {
            'sub003': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub004': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub005': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 2, 'batch': 1},
            'sub006': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 2, 'ses5': 1, 'batch': 1},
            'sub008': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub009': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub012': {'ses1': 2, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub013': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub014': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub015': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1}
        }
    elif batch == 2:
        subjects = []
        for sub_i in [18, 21, 22, 23, 24, 26, 27, 29, 30, 31]:
            subjects.append(f"sub{str(sub_i).zfill(3)}")
        scan_asTemplates = {
            'sub018': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 2, 'ses5': 1, 'batch': 2},
            'sub021': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub022': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub023': {'ses1': 1, 'ses2': 3, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub024': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub026': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub027': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub029': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub030': {'ses1': 1, 'ses2': 3, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub031': {'ses1': 1, 'ses2': 1, 'ses3': 2, 'ses4': 1, 'ses5': 1, 'batch': 2}
        }
    elif batch == 12:
        subjects = []
        for sub_i in [3, 4, 5, 6, 8, 9, 12, 13, 14, 15, 18, 21, 22, 23, 24, 26, 27, 29, 30, 31]:
            subjects.append(f"sub{str(sub_i).zfill(3)}")
        scan_asTemplates = {
            'sub003': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub004': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub005': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 2, 'batch': 1},
            'sub006': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 2, 'ses5': 1, 'batch': 1},
            'sub008': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub009': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub012': {'ses1': 2, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub013': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub014': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},
            'sub015': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 1},

            'sub018': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 2, 'ses5': 1, 'batch': 2},
            'sub021': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub022': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub023': {'ses1': 1, 'ses2': 3, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub024': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub026': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub027': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub029': {'ses1': 1, 'ses2': 1, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub030': {'ses1': 1, 'ses2': 3, 'ses3': 1, 'ses4': 1, 'ses5': 1, 'batch': 2},
            'sub031': {'ses1': 1, 'ses2': 1, 'ses3': 2, 'ses4': 1, 'ses5': 1, 'batch': 2}
        }
    else:
        raise Exception
    return subjects, scan_asTemplates


def getMonDate():
    from datetime import date
    import datetime
    today = date.today()
    month_num = str(today.month)
    datetime_object = datetime.datetime.strptime(month_num, "%m")
    month_name = datetime_object.strftime("%b")
    if len(str(today.day)) == 1:
        return f"{month_name}  {today.day}"
    else:
        return f"{month_name} {today.day}"


def checkDate(csvPath, assertTrue=True):
    cmd = f"ls -al {csvPath}";
    print(cmd)
    sbatch_response = subprocess.getoutput(cmd);
    print(sbatch_response)
    print(f"getMonDate = {getMonDate()}")
    if assertTrue:
        # assert f"{getMonDate()}" in sbatch_response
        assert (("Oct 10" in sbatch_response) or
                ("Oct 11" in sbatch_response) or
                ("Oct 12" in sbatch_response) or
                ("Oct 13" in sbatch_response) or
                ("Jan 31" in sbatch_response) or
                ("Feb 15" in sbatch_response))
        # newFlag = f"{getMonDate()}" in sbatch_response
        newFlag = (("Oct 10" in sbatch_response) or
                   ("Oct 11" in sbatch_response) or
                   ("Oct 12" in sbatch_response) or
                   ("Oct 13" in sbatch_response) or
                   ("Jan 31" in sbatch_response) or
                   ("Feb 15" in sbatch_response))
    else:
        # newFlag = f"{getMonDate()}" in sbatch_response
        newFlag = (("Oct 10" in sbatch_response) or
                   ("Oct 11" in sbatch_response) or
                   ("Oct 12" in sbatch_response) or
                   ("Oct 13" in sbatch_response) or
                   ("Jan 31" in sbatch_response) or
                   ("Feb 15" in sbatch_response))
    return newFlag


def init():
    import os
    import sys

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
    sys.path.append(projectDir + "OrganizedScripts")

    # from utils import *


def cal_resample(data=None, times=5000, return_all=False):
    # 这个函数的目的是为了针对输入的数据，进行有重复的抽取5000次，然后记录每一次的均值，最后输出这5000次重采样的均值分布    的   均值和5%和95%的数值。
    if data is None:
        raise Exception
    if type(data) == list:
        data = np.asarray(data)
    iter_mean = []
    for _ in range(times):
        iter_distri = data[np.random.choice(len(data), len(data), replace=True)]
        iter_mean.append(np.nanmean(iter_distri))
    _mean = np.mean(iter_mean)
    _5 = np.percentile(iter_mean, 5)
    _95 = np.percentile(iter_mean, 95)
    if return_all:
        return _mean, _5, _95, iter_mean
    else:
        return _mean, _5, _95


def get_ROIMethod(ROI):
    methods = ['lfseg_corr_nogray_', 'lfseg_heur_', 'lfseg_corr_usegray_', '_FreeSurfer']
    hippoSubfieldID = {
        1: "CA1",
        2: "CA2+3",
        3: "DG",
        4: "ERC",
        5: "PHC",
        6: "PRC",
        7: "SUB"
    }
    newROIname = None
    method_using = None
    for method in methods:
        if method in ROI:
            newROIname = ROI.split(method)[0] if len(ROI.split(method)[0]) > 0 else ROI.split(method)[1]

            if method in ['lfseg_corr_nogray_', 'lfseg_heur_', 'lfseg_corr_usegray_']:
                method_using = method[:-1]
                try:
                    newROIname = hippoSubfieldID[int(newROIname)]
                    # print(newROIname)
                except:
                    pass
            else:
                method_using = method[1:]
    return newROIname, method_using


def bar(means=None, upper=None, lower=None, ROINames=None, title=None, xLabel="", yLabel="", fontsize=50,
        setBackgroundColor=False,
        savePath=None, showFigure=True):
    import matplotlib.pyplot as plt
    # plot barplot with percentage error bar
    if type(means) == list:
        means = np.asarray(means)
    if type(upper) == list:
        upper = np.asarray(upper)
    if type(means) == list:
        lower = np.asarray(lower)

    # plt.figure(figsize=(fontsize, fontsize/2), dpi=70)
    positions = list(np.arange(len(means)))

    fig, ax = plt.subplots(figsize=(fontsize/2, fontsize/2))
    ax.bar(positions, means, yerr=[means - lower, upper - means], align='center', alpha=0.5, ecolor='black',
           capsize=10)
    if setBackgroundColor:
        ax.set_facecolor((242 / 256, 242 / 256, 242 / 256))
    ax.set_ylabel(yLabel, fontsize=fontsize)
    ax.set_xlabel(xLabel, fontsize=fontsize)
    ax.set_xticks(positions)
    ax.set_facecolor((242 / 256, 242 / 256, 242 / 256))
    # Increase y-axis tick font size
    ax.tick_params(axis='y', labelsize=fontsize)

    if ROINames is not None:
        xtick = ROINames
        ax.set_xticklabels(xtick, fontsize=fontsize, rotation=45, ha='right')
    ax.set_title(title, fontsize=fontsize)
    ax.yaxis.grid(True)
    _ = plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath)
    if showFigure:
        _ = plt.show()
    else:
        _ = plt.close()


def get_ROIList():
    ROIList = []
    for ashs_method in ['lfseg_corr_usegray']:  # ['lfseg_corr_nogray', 'lfseg_heur', 'lfseg_corr_usegray']
        for subfield in [1, 2, 3, 4, 5, 6, 7, 'hippocampus']:  # [1,2,3,4,5,6,7,'hippocampus']
            ROIList.append(f"{ashs_method}_{subfield}")

    FS_ROIs = ['V1', 'V2', 'LOC', 'IT', 'Fus', 'PHC']
    for ROI in FS_ROIs:
        ROIList.append(f"{ROI}_FreeSurfer")
    print(f"len(ROIList)={len(ROIList)}")  # len(ROIList)=44
    return ROIList


def save_nib(toSave=None, fileName=None, affine=None, Print=True):
    import nibabel as nib
    toSave = toSave.astype('double')
    toSave[np.isnan(toSave)] = 0  # If there are nans we want this
    sl_nii = nib.Nifti1Image(toSave.astype('double'), affine)
    nib.save(sl_nii, fileName)
    if Print:
        print(f"{fileName} saved")


# setting up code testing environment:
# from rtCommon.cfg_loading import mkdir,cfg_loading ;cfg = cfg_loading('pilot_sub001.ses1.toml')


# def bar(means=None, upper=None, lower=None, ROINames=None, title=None):
#     # plot barplot with percentage error bar
#     plt.figure(figsize=(4, 3), dpi=70)
#     positions = list(np.arange(len(means)))
#
#     fig, ax = plt.subplots(figsize=(15, 15))
#     ax.bar(positions, means, yerr=[means - lower, upper - means], align='center', alpha=0.5, ecolor='black',
#            capsize=10)
#     ax.set_ylabel('correlation between y and y_pred', fontsize=25)
#     ax.set_xlabel('ROI', fontsize=25)
#     ax.set_xticks(positions)
#     xtick = ROINames
#     ax.set_xticklabels(xtick, fontsize=25, rotation=45, ha='right')
#     ax.set_title(title, fontsize=25)
#     ax.yaxis.grid(True)
#     plt.tight_layout()
#     # plt.savefig('bar_plot_with_error_bars.png')
#     plt.show()


hippoSubfieldID = {
    1: "CA1",
    2: "CA2+3",
    3: "DG",
    4: "ERC",
    5: "PHC",
    6: "PRC",
    7: "SUB"
}
BrodmannAreaMaps = {
    "BA1": "somatosensory area",
    "BA2": "somatosensory area",
    "BA3a": "somatosensory area",
    "BA3b": "somatosensory area",
    "BA4a": "primary motor area (anterior)",
    "BA4p": "primary motor area (posterier)",
    "BA6": "pre-motor area",
    "BA44": "Broca's area (pars opercularis)",
    "BA45": "Broca's area (pars triangularis)",
    "V1(BA17)": "primary visual area",
    "V2(BA18)": "secondary visual area",
    "V5/MT": "visual area, middle temporal",
}
imcodeDict = {
    'A': 'bed',
    'B': 'chair',
    'C': 'table',
    'D': 'bench'}



