import os
import shutil
import sys
import time

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


assert os.getcwd().endswith('real_time_paper'), "working dir should be 'real_time_paper'"

batch = 12  # meaning both batch 1 and batch 2
subjects, scan_asTemplates = get_subjects(batch=batch)
testMode = False

"""
run ashs and Freesurfer
unwarp the functional data
train clf
do real-time simulation

"""


def unwarp_functionalData(scan_asTemplates):  # expScripts/recognition/recognitionDataAnalysis/GM_modelTrain.py
    for sub in scan_asTemplates:
        for ses in range(1, 6):
            SesFolder = f"data/subjects/{sub}/ses{ses}/"
            mkdir(f"{SesFolder}/fmap/")
            os.chdir(f"{SesFolder}/fmap/")

            cmd = f"bash data_preprocess/fmap/top.sh {SesFolder}fmap/"
            kp_run(cmd)

            # use topup_AP_PA_fmap to unwarp all functional data
            runRecording = pd.read_csv(f"{SesFolder}/runRecording.csv")
            recogRuns = list(
                runRecording['run'].iloc[list(np.where(1 == 1 * (runRecording['type'] == 'recognition'))[0])])
            feedbackRuns = list(
                runRecording['run'].iloc[list(np.where(1 == 1 * (runRecording['type'] == 'feedback'))[0])])
            for scan in recogRuns:
                scan_file = f"{SesFolder}/recognition/run_{scan}"
                cmd = (f"applytopup --imain={scan_file}.nii --topup=topup_AP_PA_b0 --datain=acqparams.txt --inindex=1 "
                       f"--out={scan_file}_unwarped --method=jac")
                kp_run(cmd)

            for scan in feedbackRuns:
                scan_file = f"{SesFolder}/feedback/run_{scan}"
                cmd = (f"applytopup --imain={scan_file}.nii --topup=topup_AP_PA_b0 --datain=acqparams.txt --inindex=1 "
                       f"--out={scan_file}_unwarped --method=jac")
                kp_run(cmd)

            def saveMiddleVolumeAsTemplate(SesFolder='',
                                           scan_asTemplate=1):  # expScripts/recognition/recognitionDataAnalysis/GM_modelTrain.py
                nii = nib.load(f"{SesFolder}/recognition/run_{scan_asTemplate}_unwarped.nii.gz")
                frame = nii.get_data()
                TR_number = frame.shape[3]
                frame = frame[:, :, :, int(TR_number / 2)]
                frame = nib.Nifti1Image(frame, affine=nii.affine)
                unwarped_template = f"{SesFolder}/recognition/templateFunctionalVolume_unwarped.nii"
                nib.save(frame, unwarped_template)

            # Take the middle volume of the first recognition run as a template
            saveMiddleVolumeAsTemplate(SesFolder=SesFolder, scan_asTemplate=scan_asTemplates[sub][f"ses{ses}"])

            def align_with_template(SesFolder='',
                                    scan_asTemplate=1):  # expScripts/recognition/recognitionDataAnalysis/GM_modelTrain.py
                unwarped_template = f"{SesFolder}/recognition/templateFunctionalVolume_unwarped.nii"
                cmd = (f"bet {SesFolder}/recognition/templateFunctionalVolume_unwarped.nii "
                       f"{SesFolder}/recognition/templateFunctionalVolume_unwarped_bet.nii.gz")
                kp_run(cmd)
                shutil.copyfile(f"{SesFolder}/recognition/templateFunctionalVolume_unwarped_bet.nii.gz",
                                f"{SesFolder}/recognition/functional_bet.nii.gz")

                # Align all recognition runs and feedback runs with the functional template of the current session
                for scan in recogRuns:
                    head = f"{SesFolder}/recognition/run_{scan}"
                    cmd = f"mcflirt -in {head}_unwarped.nii.gz -out {head}_unwarped_mc.nii.gz"
                    kp_run(cmd)
                    wait(f"{head}_unwarped_mc.nii.gz")  # mcflirt for motion correction

                    # Align the motion-corrected functional data with the unwarped_template of the current session, which is the funcTemplate corrected by topup.
                    # Then, transfer the motion-corrected data to the func space corrected by topup.
                    # The reason for two steps here is that flirt does not directly produce the -out result for multiple volumes.
                    cmd = f"flirt -in {head}_unwarped_mc.nii.gz " \
                          f"-out {head}_temp.nii.gz " \
                          f"-ref {unwarped_template} " \
                          f"-dof 6 " \
                          f"-omat {SesFolder}/recognition/scan{scan}_to_unwarped.mat"
                    kp_run(cmd)

                    cmd = f"flirt " \
                          f"-in {head}_unwarped_mc.nii.gz " \
                          f"-out {head}_unwarped_mc.nii.gz " \
                          f"-ref {unwarped_template} -applyxfm " \
                          f"-init {SesFolder}/recognition/scan{scan}_to_unwarped.mat"
                    kp_run(cmd)
                    kp_remove(f"{head}_temp.nii.gz")

                for scan in feedbackRuns:
                    cmd = f"mcflirt -in {SesFolder}/feedback/run_{scan}_unwarped.nii.gz " \
                          f"-out {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz"
                    kp_run(cmd)
                    wait(f"{SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz")

                    cmd = f"flirt -in {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz " \
                          f"-out {SesFolder}/feedback/run_{scan}_temp.nii.gz " \
                          f"-ref {unwarped_template} -dof 6 -omat {SesFolder}/feedback/scan{scan}_to_unwarped.mat"
                    kp_run(cmd)

                    cmd = f"flirt -in {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz " \
                          f"-out {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz " \
                          f"-ref {unwarped_template} -applyxfm -init {SesFolder}/feedback/scan{scan}_to_unwarped.mat"
                    kp_run(cmd)
                    kp_remove(f"{SesFolder}/feedback/run_{scan}_temp.nii.gz")

                    cmd = f"fslinfo {SesFolder}/feedback/run_{scan}_unwarped_mc.nii.gz"
                    kp_run(cmd)

            align_with_template(SesFolder=SesFolder, scan_asTemplate=scan_asTemplates[sub][f"ses{ses}"])


unwarp_functionalData(scan_asTemplates)


def prepareIntegrationScore(scan_asTemplates=None):  # from OrganizedScripts/ROI/ROI_ses1ses5_autoAlign.py
    def clf_training(scan_asTemplates=None):  # OrganizedScripts/ROI/ROI_ses1ses5_autoAlign.py

        ROIList = get_ROIList()
        autoAlignFlag = True
        os.chdir("/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/")
        jobarrayDict = {}
        jobarrayID = 1
        for sub in scan_asTemplates:
            for chosenMask in ROIList:
                for ses in [1]:
                    jobarrayDict[jobarrayID] = [sub, chosenMask, ses, autoAlignFlag]
                    jobarrayID += 1
        np.save(
            f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
            f"OrganizedScripts/ROI/autoAlign_ses1ses5/clf_training/clfTraining_jobID.npy",
            jobarrayDict)
        if testMode:
            cmd = f"sbatch --requeue --array=1-1 /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/" \
                  f"OrganizedScripts/ROI/autoAlign_ses1ses5/clf_training/clfTraining.sh"
        else:
            cmd = f"sbatch --requeue --array=1-{len(jobarrayDict)} /gpfs/milgram/project/turk-browne/projects/rt-cloud/" \
                  f"projects/rtSynth_rt/" \
                  f"OrganizedScripts/ROI/autoAlign_ses1ses5/clf_training/clfTraining.sh"
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
        os.chdir("/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/")
        jobarrayDict = {}
        jobarrayID = 1
        for sub in scan_asTemplates:
            for chosenMask in ROIList:
                for ses in [5]:
                    jobarrayDict[jobarrayID] = [sub, chosenMask, ses]
                    jobarrayID += 1
        np.save(
            f"data_preprocess/integrationScore/integrationScore_jobID.npy",
            jobarrayDict)
        if testMode:
            cmd = f"sbatch --requeue --array=1-1 data_preprocess/integrationScore/integrationScore.sh"
        else:
            cmd = f"sbatch --requeue --array=1-{len(jobarrayDict)} data_preprocess/integrationScore/integrationScore.sh"
        sbatch_response = kp_run(cmd)
        jobID = getjobID_num(sbatch_response)
        waitForEnd(jobID)
        if testMode:
            completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
        else:
            completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))

    integrationScore(scan_asTemplates=scan_asTemplates, testMode=testMode)

    def prepareData(scan_asTemplates=None, testMode=None):
        # prepareData 这个函数的目的是为了把之前的函数的模拟的结果进行处理, 得到可以用来画图的数据.

        os.chdir("/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/")
        ROIList = get_ROIList()
        jobarrayDict = {}
        jobarrayID = 1
        for sub in scan_asTemplates:  # 20*29*6=3480
            for chosenMask in ROIList:
                for [normActivationFlag, UsedTRflag] in [[True, 'feedback'],
                                                         # [False, 'feedback'],
                                                         # [True, 'feedback_trail'],
                                                         # [False, 'feedback_trail'],
                                                         # [True, 'all'],
                                                         # [False, 'all']
                                                         ]:
                    # print(normActivationFlag, UsedTRflag)
                    # for normActivationFlag in [True, False]:
                    #     for UsedTRflag in ["feedback", "feedback_trail", "all"]:
                    # tag = f"normActivationFlag_{normActivationFlag}_UsedTRflag_{UsedTRflag}"
                    jobarrayDict[jobarrayID] = [sub, chosenMask, normActivationFlag, UsedTRflag]
                    jobarrayID += 1
        np.save(
            f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
            f"OrganizedScripts/ROI/autoAlign_ses1ses5/XYactivation/prepareData_jobID.npy",
            jobarrayDict)
        if testMode:
            cmd = f"sbatch --requeue --array=1-1 /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/" \
                  f"OrganizedScripts/ROI/autoAlign_ses1ses5/XYactivation/prepareData.sh"
        else:
            cmd = f"sbatch --requeue --array=1-{len(jobarrayDict)} /gpfs/milgram/project/turk-browne/projects/rt-cloud/" \
                  f"projects/rtSynth_rt/" \
                  f"OrganizedScripts/ROI/autoAlign_ses1ses5/XYactivation/prepareData.sh"
        sbatch_response = kp_run(cmd)

        jobID = getjobID_num(sbatch_response)
        waitForEnd(jobID)
        if testMode:
            completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
        else:
            completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))

    prepareData(scan_asTemplates=scan_asTemplates, testMode=testMode)


prepareIntegrationScore(scan_asTemplates=scan_asTemplates)


def prepare_coActivation_fig2c(scan_asTemplates=None,
                               testMode=None):  # OrganizedScripts/megaROI/withinSession/megaROI_withinSess.py
    ROIList = ['megaROI']
    autoAlignFlag = True

    def clfTraining(scan_asTemplates=None, ROIList=None):
        os.chdir("/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/")

        def prepareForParalelROIclfTraining(scan_asTemplates=None, ROIList=None):
            SLURM_ARRAY_TASK_ID = 1
            SLURM_ARRAY_TASK_ID_dict = {}
            for sub in scan_asTemplates:  # 对于被试 sub， ses，进行leave one run out 训练和测试。
                for chosenMask in ROIList:  # len(ROIList)=44
                    for ses in [1, 2, 3, 4, 5]:  # 这里可能可以使用1 2 3 4 5. 当前的1 2 3 的设计是为了使得 feedback 的离线模拟处理可以进行
                        SLURM_ARRAY_TASK_ID_dict[SLURM_ARRAY_TASK_ID] = [sub, chosenMask, ses]
                        SLURM_ARRAY_TASK_ID += 1  # ses=[1,2,3]:1321  ses=[1,2,3,4]:1761  ses=[1,2,3,4,5]:2201
            np.save(f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
                    f"OrganizedScripts/megaROI/withinSession/autoAlign/clfTraining/clfTraining_ID_dict.npy",
                    SLURM_ARRAY_TASK_ID_dict)
            print(
                f"len(SLURM_ARRAY_TASK_ID_dict)={len(SLURM_ARRAY_TASK_ID_dict)}")
            return SLURM_ARRAY_TASK_ID_dict

        SLURM_ARRAY_TASK_ID_dict = prepareForParalelROIclfTraining(scan_asTemplates=scan_asTemplates, ROIList=ROIList)
        cmd = f"sbatch --requeue --array=1-{len(SLURM_ARRAY_TASK_ID_dict)} /gpfs/milgram/project/turk-browne/" \
              f"projects/rt-cloud/projects/rtSynth_rt/" \
              f"OrganizedScripts/megaROI/withinSession/autoAlign/clfTraining/clfTraining.sh"
        sbatch_response = kp_run(cmd)
        jobID = getjobID_num(sbatch_response)
        waitForEnd(jobID)
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(SLURM_ARRAY_TASK_ID_dict))  # 检查所有运行的job都是成功完成的。

    clfTraining(scan_asTemplates=scan_asTemplates, ROIList=ROIList)

    def ROI_nomonotonic_curve(scan_asTemplates=None, ROIList=None, batch=None,
                              functionShape=None, useNewClf=True):  # ConstrainedCubic ConstrainedQuadratic Linear
        def checkWhetherRelevant(accTable, chosenMask):
            ROIisRelevant = []
            for clf in ['AB', 'CD', 'AC', 'AD', 'BC', 'BD']:
                t = list(accTable[f"{clf}_acc"])
                # tstat = stats.ttest_1samp(t, 0.5)
                # if tstat[0]>0 and tstat[1]<0.05:
                if np.mean(t) > 0.5:
                    ROIisRelevant.append(1)
                else:
                    ROIisRelevant.append(0)
            # print(f"{chosenMask}: good clf number out of 6 clf : {np.sum(ROIisRelevant)}")
            looseRelevant = "noThreshold"
            if looseRelevant == "noThreshold":
                threshold = 0
            elif looseRelevant == "looseThreshold":
                threshold = 6
            else:
                raise Exception("no such looseRelevant")
            if np.sum(ROIisRelevant) >= threshold:
                print(f"{chosenMask} is relevant")
                return True
            else:
                # print(f"{chosenMask} is not relevant")
                return False

        sub_ROI_ses_relevant = pd.DataFrame()
        for sub in tqdm(scan_asTemplates):
            for chosenMask in ROIList:
                for ses in [1, 2, 3]:
                    if autoAlignFlag:
                        # autoAlign_ROIFolder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis/" \
                        #                       f"subjects/{sub}/ses{ses}/{chosenMask}/"
                        autoAlign_ROIFolder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/subjects/" \
                                              f"{sub}/ses{ses}/{chosenMask}/"
                        accTable_path = f"{autoAlign_ROIFolder}/accTable.csv"
                    else:
                        raise Exception("no such alignFlag")
                        # accTable_path = f"{projectDir}/subjects/{sub}/ses{ses}/recognition/ROI_analysis/{chosenMask}/accTable.csv"
                    if os.path.exists(accTable_path):
                        accTable = pd.read_csv(accTable_path)

                        Relevant = checkWhetherRelevant(accTable, chosenMask)
                        sub_ROI_ses_relevant = pd.concat([sub_ROI_ses_relevant, pd.DataFrame({
                            'sub': [sub],
                            'ses': [ses],
                            'chosenMask': [chosenMask],
                            'Relevant': [Relevant]
                        })], ignore_index=True)
                    else:
                        print(accTable_path + ' missing')
        if autoAlignFlag:
            Folder = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/"
            save_obj([subjects, ROIList, sub_ROI_ses_relevant],
                     f"{Folder}/ROI_nomonotonic_curve_batch_autoAlign_batch_{batch}")
        else:
            raise Exception("no such alignFlag")
            # save_obj([subjects,ROIList,sub_ROI_ses_relevant], f"/gpfs/milgram/scratch60/turk-browne/kp578/ROI_nomonotonic_curve_193246289713648923694")
            # save_obj([subjects, ROIList, sub_ROI_ses_relevant],
            #          f"/gpfs/milgram/scratch60/turk-browne/kp578/ROI_nomonotonic_curve_batch_{batch}")  # 在 ROI_nomonotonic_curve.sh 中使用

        def simulate():
            os.chdir("/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/")

            def getNumberOfRuns(subjects=None,
                                batch=0):  # 计算给定的被试都有多少个feedback run
                jobArray = {}
                count = 0
                for chosenMask in ROIList:
                    for sub in subjects:
                        for ses_i in [1, 2, 3]:
                            nextSes_i = int(ses_i + 1)
                            assert ses_i in [1, 2, 3]
                            assert nextSes_i in [2, 3, 4]

                            runRecording = pd.read_csv(
                                f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/"
                                f"{sub}/ses{nextSes_i}/runRecording.csv")
                            feedback_runRecording = runRecording[runRecording['type'] == 'feedback'].reset_index()

                            # print(f"sub={sub} ses={nextSes_i} chosenMask={chosenMask}")
                            for currFeedbackRun in range(len(feedback_runRecording)):
                                count += 1
                                forceReRun = True
                                if not forceReRun:
                                    if autoAlignFlag:
                                        history_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/" \
                                                      f"megaROI_main/" \
                                                      f"subjects/{sub}/ses{nextSes_i}/{chosenMask}/rtSynth_rt_ABCD_ROIanalysis/"
                                    else:
                                        raise Exception("not implemented")
                                    history_file = f"{history_dir}/{sub}_{currFeedbackRun}_history_rtSynth_RT_ABCD.csv"
                                    if not os.path.exists(history_file):
                                        jobArray[count] = [
                                            sub,
                                            nextSes_i,  # assert nextSes_i in [2, 3, 4]
                                            currFeedbackRun + 1,  # runNum
                                            feedback_runRecording.loc[currFeedbackRun, 'run'],  # scanNum
                                            chosenMask,
                                            forceReRun, useNewClf
                                        ]
                                else:
                                    jobArray[count] = [
                                        sub,
                                        nextSes_i,  # assert nextSes_i in [2, 3, 4]
                                        currFeedbackRun + 1,  # runNum
                                        feedback_runRecording.loc[currFeedbackRun, 'run'],  # scanNum
                                        chosenMask,
                                        forceReRun, useNewClf
                                    ]  # [sub, ses, runNum, scanNum, chosenMask, forceReRun]

                                if autoAlignFlag:
                                    # history_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis/" \
                                    #               f"subjects/{sub}/ses{nextSes_i}/{chosenMask}/rtSynth_rt_ABCD_ROIanalysis/"
                                    history_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/" \
                                                  f"megaROI_main/" \
                                                  f"subjects/{sub}/ses{nextSes_i}/{chosenMask}/rtSynth_rt_ABCD_ROIanalysis/"
                                    # history_dir = f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session}/recognition//ROI_analysis/{chosenMask}/rtSynth_rt_ABCD_ROIanalysis/"
                                else:
                                    raise Exception("not implemented yet")
                                mkdir(history_dir)
                _jobArrayPath = f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/" \
                                f"OrganizedScripts/megaROI/withinSession/autoAlign/nonmonotonicCurve/simulatedFeedbackRun/" \
                                f"rtSynth_rt_ABCD_ROIanalysis_unwarp_batch{batch}"
                save_obj(jobArray, _jobArrayPath)
                return len(jobArray), _jobArrayPath

            jobNumber, jobArrayPath = getNumberOfRuns(
                subjects=subjects,
                batch=batch)
            cmd = f"sbatch --requeue --array=1-{jobNumber} /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/" \
                  f"rtSynth_rt/OrganizedScripts/megaROI/withinSession/autoAlign/nonmonotonicCurve/simulatedFeedbackRun/" \
                  f"rtSynth_rt_ABCD_ROIanalysis_unwarp.sh {jobArrayPath} 0"  # 大约需要一分钟结束
            sbatch_response = kp_run(cmd)
            jobID = getjobID_num(sbatch_response)
            waitForEnd(jobID)
            completed = check_jobArray(jobID=jobID, jobarrayNumber=jobNumber)  # 检查所有运行的job都是成功完成的。

        simulate()

        def prepareData():
            os.chdir("/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/")
            jobarrayDict = {}
            jobarrayID = 1
            # for sub in scan_asTemplates:
            for chosenMask in ROIList:
                for [_normActivationFlag, _UsedTRflag] in [[True, 'feedback'],
                                                           # [False, 'feedback'],
                                                           # [True, 'feedback_trail'],
                                                           # [False, 'feedback_trail'],
                                                           # [True, 'all'],
                                                           # [False, 'all']
                                                           ]:
                    tag = f"normActivationFlag_{_normActivationFlag}_UsedTRflag_{_UsedTRflag}"
                    plot_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/" \
                               f"cubicFit/batch{int(batch)}/{tag}/"
                    jobarrayDict[jobarrayID] = [chosenMask, plot_dir, batch, _normActivationFlag, _UsedTRflag,
                                                useNewClf]
                    jobarrayID += 1
            np.save(
                f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
                f"OrganizedScripts/megaROI/withinSession/autoAlign/nonmonotonicCurve/ROI_nomonotonic_curve_batch_{batch}_jobID.npy",
                jobarrayDict)

            cmd = f"sbatch --requeue --array=1-{len(jobarrayDict)} /gpfs/milgram/project/turk-browne/projects/rt-cloud/" \
                  f"projects/rtSynth_rt/" \
                  f"OrganizedScripts/megaROI/withinSession/autoAlign/nonmonotonicCurve/ROI_nomonotonic_curve.sh 0 {batch}"  # 每一个ROI单独运行一个代码。
            sbatch_response = kp_run(cmd)
            jobID = getjobID_num(sbatch_response)
            waitForEnd(jobID)
            completed = check_jobArray(jobID=jobID, jobarrayNumber=len(ROIList))

        prepareData()

    ROI_nomonotonic_curve(scan_asTemplates=scan_asTemplates, ROIList=ROIList, batch=batch, functionShape=functionShape)
