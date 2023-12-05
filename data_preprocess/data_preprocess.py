import os
import numpy as np
import pandas as pd
from tqdm import tqdm

print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")
print(f"numpy version = {np.__version__}")
print(f"pandas version = {pd.__version__}")

from utils import save_obj, load_obj, mkdir, getjobID_num, kp_and, kp_or, kp_rename, kp_copy, kp_run, kp_remove
from utils import wait, check, checkEndwithDone, checkDone, check_jobIDs, check_jobArray, waitForEnd, \
    jobID_running_myjobs
from utils import readtxt, writetxt, get_subjects, init
from utils import getMonDate, checkDate, save_nib
from utils import get_ROIMethod, bar, get_ROIList
from utils import get_subjects
import nibabel as nib

os.chdir("/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/real_time_paper/")
assert os.getcwd().endswith('real_time_paper'), "working dir should be 'real_time_paper'"
workingDir = os.getcwd()
batch = 12  # meaning both batch 1 and batch 2
subjects, scan_asTemplates = get_subjects(batch=batch)
testMode = False


def unwarp_functionalData(scan_asTemplates):  # expScripts/recognition/recognitionDataAnalysis/GM_modelTrain.py
    jobarrayDict = {}
    jobarrayID = 1
    for sub in scan_asTemplates:
        for ses in range(1, 6):
            jobarrayDict[jobarrayID] = [sub, ses]
            jobarrayID += 1
    np.save(
        f"data_preprocess/unwarp/unwarp_jobID.npy",
        jobarrayDict)
    if testMode:
        cmd = f"sbatch --requeue --array=1-1 data_preprocess/unwarp/unwarp.sh"
    else:
        cmd = f"sbatch --requeue --array=1-{len(jobarrayDict)} data_preprocess/unwarp/unwarp.sh"

    def kp_run(cmd):
        print()
        print(cmd)
        import subprocess
        sbatch_response = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        check(sbatch_response.stdout)
        return sbatch_response.stdout

    sbatch_response = kp_run(cmd)

    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    if testMode:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
    else:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))


unwarp_functionalData(scan_asTemplates)


def preprocess(scan_asTemplates):
    jobarrayDict = {}
    jobarrayID = 1
    for sub in scan_asTemplates:
        for ses in range(1, 6):
            jobarrayDict[jobarrayID] = [sub, ses]
            jobarrayID += 1
    np.save(
        f"data_preprocess/unwarp/recognition_preprocess_unwarped_jobID.npy",
        jobarrayDict)
    if testMode:
        cmd = (f"sbatch --requeue --array=1-1 "
               f"data_preprocess/unwarp/recognition_preprocess_unwarped.sh")
    else:
        cmd = (f"sbatch --requeue --array=1-{len(jobarrayDict)} "
               f"data_preprocess/unwarp/recognition_preprocess_unwarped.sh")

    def kp_run(cmd):
        print()
        print(cmd)
        import subprocess
        sbatch_response = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        check(sbatch_response.stdout)
        return sbatch_response.stdout

    sbatch_response = kp_run(cmd)

    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    if testMode:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
    else:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))


preprocess(scan_asTemplates)


def prepareIntegrationScore_fig4b(scan_asTemplates=None):
    def clf_training(scan_asTemplates=None):
        ROIList = get_ROIList()
        autoAlignFlag = True
        jobarrayDict = {}
        jobarrayID = 1
        for sub in scan_asTemplates:
            for chosenMask in ROIList:
                for ses in [1]:
                    jobarrayDict[jobarrayID] = [sub, chosenMask, ses, autoAlignFlag]
                    jobarrayID += 1
        np.save(
            f"data_preprocess/prepareIntegrationScore/clf_training/clfTraining_jobID.npy",
            jobarrayDict)
        if testMode:
            cmd = (f"sbatch --requeue --array=1-1 "
                   f"data_preprocess/prepareIntegrationScore/clf_training/clfTraining.sh")
        else:
            cmd = (f"sbatch --requeue --array=1-{len(jobarrayDict)} "
                   f"data_preprocess/prepareIntegrationScore/clf_training/clfTraining.sh")
        sbatch_response = kp_run(cmd)

        jobID = getjobID_num(sbatch_response)
        waitForEnd(jobID)
        if testMode:
            completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
        else:
            completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))

    clf_training(scan_asTemplates=scan_asTemplates)

    def integrationScore(scan_asTemplates=None, testMode=None):
        ROIList = get_ROIList()
        jobarrayDict = {}
        jobarrayID = 1
        for sub in scan_asTemplates:
            for chosenMask in ROIList:
                for ses in [5]:
                    jobarrayDict[jobarrayID] = [sub, chosenMask, ses]
                    jobarrayID += 1
        np.save(
            f"data_preprocess/prepareIntegrationScore/integrationScore/integrationScore_jobID.npy",
            jobarrayDict)
        if testMode:
            cmd = (f"sbatch --requeue --array=1-1 "
                   f"data_preprocess/prepareIntegrationScore/integrationScore/integrationScore.sh")
        else:
            cmd = (f"sbatch --requeue --array=1-{len(jobarrayDict)} "
                   f"data_preprocess/prepareIntegrationScore/integrationScore/integrationScore.sh")
        sbatch_response = kp_run(cmd)
        jobID = getjobID_num(sbatch_response)
        waitForEnd(jobID)
        if testMode:
            completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
        else:
            completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))

    integrationScore(scan_asTemplates=scan_asTemplates, testMode=testMode)


prepareIntegrationScore_fig4b(scan_asTemplates=scan_asTemplates)


def prepare_coActivation_fig2c(scan_asTemplates=None,
                               testMode=None):
    ROIList = ['megaROI']
    autoAlignFlag = True

    def noUnwarpPreprocess(scan_asTemplates):
        jobarrayDict = {}
        jobarrayID = 1
        for sub in scan_asTemplates:
            for ses in range(1, 6):
                jobarrayDict[jobarrayID] = [sub, ses]
                jobarrayID += 1
        np.save(
            f"data_preprocess/prepare_coActivation_fig2c/"
            f"noUnwarpPreprocess/noUnwarpPreprocess_jobID.npy",
            jobarrayDict)
        if testMode:
            cmd = (f"sbatch --requeue --array=1-1 "
                   f"data_preprocess/prepare_coActivation_fig2c/noUnwarpPreprocess/noUnwarpPreprocess.sh")
        else:
            cmd = (f"sbatch --requeue --array=1-{len(jobarrayDict)} "
                   f"data_preprocess/prepare_coActivation_fig2c/noUnwarpPreprocess/noUnwarpPreprocess.sh")

        def kp_run(cmd):
            print()
            print(cmd)
            import subprocess
            sbatch_response = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            check(sbatch_response.stdout)
            return sbatch_response.stdout

        sbatch_response = kp_run(cmd)

        jobID = getjobID_num(sbatch_response)
        waitForEnd(jobID)
        if testMode:
            completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
        else:
            completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))

    noUnwarpPreprocess(scan_asTemplates)

    def clfTraining(scan_asTemplates=None, ROIList=None):

        def prepareForParalelROIclfTraining(scan_asTemplates=None, ROIList=None):
            SLURM_ARRAY_TASK_ID = 1
            SLURM_ARRAY_TASK_ID_dict = {}
            for sub in scan_asTemplates:
                for chosenMask in ROIList:
                    for ses in [1, 2, 3, 4, 5]:
                        SLURM_ARRAY_TASK_ID_dict[SLURM_ARRAY_TASK_ID] = [sub, chosenMask, ses]
                        SLURM_ARRAY_TASK_ID += 1  # ses=[1,2,3]:1321  ses=[1,2,3,4]:1761  ses=[1,2,3,4,5]:2201
            np.save(f"data_preprocess/prepare_coActivation_fig2c/clfTraining/clfTraining_ID_dict.npy",
                    SLURM_ARRAY_TASK_ID_dict)
            print(
                f"len(SLURM_ARRAY_TASK_ID_dict)={len(SLURM_ARRAY_TASK_ID_dict)}")
            return SLURM_ARRAY_TASK_ID_dict

        SLURM_ARRAY_TASK_ID_dict = prepareForParalelROIclfTraining(scan_asTemplates=scan_asTemplates, ROIList=ROIList)
        cmd = (f"sbatch --requeue --array=1-{len(SLURM_ARRAY_TASK_ID_dict)} "
               f"data_preprocess/prepare_coActivation_fig2c/clfTraining/clfTraining.sh")
        sbatch_response = kp_run(cmd)
        jobID = getjobID_num(sbatch_response)
        waitForEnd(jobID)
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(SLURM_ARRAY_TASK_ID_dict))  #

    clfTraining(scan_asTemplates=scan_asTemplates, ROIList=ROIList)

    def ROI_nomonotonic_curve(scan_asTemplates=None, ROIList=None, batch=None,
                              functionShape=None, useNewClf=True):  # ConstrainedCubic ConstrainedQuadratic Linear
        sub_ROI_ses_relevant = pd.DataFrame()
        for sub in tqdm(scan_asTemplates):
            for chosenMask in ROIList:
                for ses in [1, 2, 3]:
                    if autoAlignFlag:
                        autoAlign_ROIFolder = (
                            f"{workingDir}/"
                            f"data/result/megaROI_main/subjects/{sub}/ses{ses}/{chosenMask}/")
                        accTable_path = f"{autoAlign_ROIFolder}/accTable.csv"
                    else:
                        raise Exception("no such alignFlag")
                    if os.path.exists(accTable_path):
                        accTable = pd.read_csv(accTable_path)

                        Relevant = True  # checkWhetherRelevant(accTable, chosenMask)
                        sub_ROI_ses_relevant = pd.concat([sub_ROI_ses_relevant, pd.DataFrame({
                            'sub': [sub],
                            'ses': [ses],
                            'chosenMask': [chosenMask],
                            'Relevant': [Relevant]
                        })], ignore_index=True)
                    else:
                        print(accTable_path + ' missing')

        Folder = (f"{workingDir}data/result/megaROI_main/")
        save_obj([subjects, ROIList, sub_ROI_ses_relevant],
                 f"{Folder}/ROI_nomonotonic_curve_batch_autoAlign_batch_{batch}")

        def simulate():
            os.chdir(workingDir)

            def getNumberOfRuns(subjects=None,
                                batch=0):  # how many feedback runs there are for a specified subject
                jobArray = {}
                count = 0
                for chosenMask in ROIList:
                    for sub in subjects:
                        for ses_i in [1, 2, 3]:
                            nextSes_i = int(ses_i + 1)
                            assert ses_i in [1, 2, 3]
                            assert nextSes_i in [2, 3, 4]
                            runRecording = pd.read_csv(
                                f"{workingDir}/data/subjects/{sub}/ses{nextSes_i}/runRecording.csv")
                            feedback_runRecording = runRecording[runRecording['type'] == 'feedback'].reset_index()

                            for currFeedbackRun in range(len(feedback_runRecording)):
                                count += 1
                                forceReRun = True
                                jobArray[count] = [
                                    sub,
                                    nextSes_i,  # assert nextSes_i in [2, 3, 4]
                                    currFeedbackRun + 1,  # runNum
                                    feedback_runRecording.loc[currFeedbackRun, 'run'],  # scanNum
                                    chosenMask,
                                    forceReRun, useNewClf
                                ]  # [sub, ses, runNum, scanNum, chosenMask, forceReRun]

                                if autoAlignFlag:
                                    autoAlign_ROIFolder = (
                                        f"{workingDir}/"
                                        f"data/result/megaROI_main/subjects/{sub}/ses{nextSes_i}/{chosenMask}/")
                                    history_dir = f"{autoAlign_ROIFolder}/rtSynth_rt_ABCD_ROIanalysis/"
                                else:
                                    raise Exception("not implemented yet")
                                mkdir(history_dir)
                _jobArrayPath = (f"data_preprocess/prepare_coActivation_fig2c/nonmonotonicCurve/simulatedFeedbackRun/"
                                 f"rtSynth_rt_ABCD_ROIanalysis_unwarp_batch{batch}")
                save_obj(jobArray, _jobArrayPath)
                return len(jobArray), _jobArrayPath

            jobNumber, jobArrayPath = getNumberOfRuns(
                subjects=subjects,
                batch=batch)
            cmd = (f"sbatch --requeue --array=1-{jobNumber} "
                   f"data_preprocess/prepare_coActivation_fig2c/nonmonotonicCurve/simulatedFeedbackRun/"
                   f"rtSynth_rt_ABCD_ROIanalysis_unwarp.sh {jobArrayPath} 0")
            sbatch_response = kp_run(cmd)
            jobID = getjobID_num(sbatch_response)
            waitForEnd(jobID)
            completed = check_jobArray(jobID=jobID, jobarrayNumber=jobNumber)

        simulate()

    ROI_nomonotonic_curve(scan_asTemplates=scan_asTemplates, ROIList=ROIList, batch=batch,
                          functionShape="ConstrainedCubic")


prepare_coActivation_fig2c(scan_asTemplates=scan_asTemplates, testMode=testMode)
