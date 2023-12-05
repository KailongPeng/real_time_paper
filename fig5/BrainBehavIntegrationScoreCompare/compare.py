import os
import sys
assert os.getcwd().endswith('real_time_paper'), "working dir should be 'real_time_paper'"
workingDir = os.getcwd()
sys.path.append('.')
# print current dir
print(f"getcwd = {os.getcwd()}")

import os
import sys
import numpy as np
import pandas as pd
import re
import subprocess
import time
import pickle5 as pickle
import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
import scipy.optimize as opt
from tqdm import tqdm
from glob import glob

testMode = False

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
sys.path.append(projectDir + "/OrganizedScripts/")

# 在jupyterLab中如果失败了, 可能是因为使用的Python核心有问题, 使用下面的代码强制改变jupyter的核心以产生加载util.py中的函数.
## %%script /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt/bin/python3.7
# import sys
# print(sys.executable)
# print(sys.version)
# print(sys.version_info)
# from utils import *

from utils import save_obj, load_obj, mkdir, getjobID_num, kp_and, kp_or, kp_rename, kp_copy, kp_run, kp_remove
from utils import wait, check, checkEndwithDone, checkDone, check_jobIDs, check_jobArray, waitForEnd, \
    jobID_running_myjobs
from utils import readtxt, writetxt, get_subjects, init
from utils import getMonDate, checkDate
from utils import get_ROIMethod, bar, get_ROIList


def get_file_created_time(file_path):
    import os
    import datetime

    # Get the creation time of the file in seconds since the epoch
    created_time = os.path.getctime(file_path)

    # Convert the epoch time to a human-readable format
    created_datetime = datetime.datetime.fromtimestamp(created_time)

    # Print the created date and time
    print(f"file_path={file_path}")
    print("File created date and time:", created_datetime)

    return created_datetime


autoAlignFlag = True
batch = 12  # 29表示只对sub029 sub030 sub031 运行  # 在某些时候我只想在除了29 30 31 之外的被试身上运行, 此时就使用batch99
subjects, scan_asTemplates = get_subjects(batch=batch)

dropCertainSub = []  # ['sub018']
if len(dropCertainSub) > 0:
    for sub in dropCertainSub:
        subjects.remove(sub)
ROIList = get_ROIList()  # ['lfseg_corr_usegray_hippocampus', 'lfseg_corr_usegray_MTL']  # get_ROIList()
ROIList = ROIList + ['megaROI']

"""    	
    code design:
        code name: BrainBehavIntegrationScoreCompare
        psuedo code
            load behavior catPer integrationScore
            if acrossSessionEffect:
                load ROI integrationScore for across session effect
            else:
                load ROI integrationScore for within session effect and average across session so that each subject has a single value.
            
            plot a scatter plot with behavior catPer integrationScore versus ROI integrationScore
            run a linear regression with behavior catPer integrationScore versus ROI integrationScore in a leave one sub out manner and obtain a averaged model performance.
            run a correlation between behavior catPer integrationScore versus ROI integrationScore.
"""

# load behavior catPer integrationScore
# Behav_slope = pd.read_csv(
#     '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/analysisResults/allSubs/catPer/Behav_slope.csv')  # load
# Behav_differentiations = pd.read_csv(
#     '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/analysisResults/'
#     'allSubs/catPer/Behav_differentiations.csv')  # load

Behav_differentiations = pd.read_csv(
        f'{workingDir}/data/result/analysisResults/allSubs/catPer/'
        f'Behav_differentiations_fixedCenter.csv')

if len(dropCertainSub) > 0:
    for sub in dropCertainSub:
        Behav_differentiations = Behav_differentiations[Behav_differentiations['sub'] != sub]

# Behav_differentiations
# sub	ses1_XY_acc	ses1_MN_acc	ses5_XY_acc	ses5_MN_acc	Behav_differentiation
# sub003	25.651651	25.494680	22.580585	27.620671	-0.103698
# sub004	43.045274	17.879099	47.857142	18.532048	0.035002
# sub005	40.229530	11.643448	40.479376	11.672306	0.001858


# load ROI integrationScore for within session effect and average across session so that each subject has a single value.
for interestedROI in tqdm(ROIList):
    Behav_differentiations[f"withinSes-{interestedROI}"] = None
    Behav_differentiations[f"acrossSes-{interestedROI}"] = None

for interestedROI in tqdm(ROIList):
    # Behav_differentiations[f"withinSes-{interestedROI}"] = None
    # Behav_differentiations[f"acrossSes-{interestedROI}"] = None

    # def load_withinSessionEffect(interestedROI=None, _Behav_differentiations=None):
    #     # withinSessionEffect
    #     if interestedROI != 'megaROI':
    #         # code and data from OrganizedScripts/ROI/autoAlign/constraint_cubit_analysis_ROI/constraint_cubit_analysis_ROI.py
    #         plot_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/" \
    #                    f"autoAlign_ROIanalysis/cubicFit/batch12/normActivationFlag_{True}_UsedTRflag_{'feedback'}/"
    #         sub_ROI_ses_relevant_using = load_obj(
    #             f"{plot_dir}/sub_ROI_ses_relevant_using_{interestedROI}_batch{int(batch)}")  # load
    #         get_file_created_time(f"{plot_dir}/sub_ROI_ses_relevant_using_{interestedROI}_batch{int(batch)}.pkl")
    #     else:
    #         # code and data from OrganizedScripts/megaROI/withinSession/autoAlign/constraint_cubit_analysis_ROI/constraint_cubit_analysis_ROI.py
    #         assert interestedROI == 'megaROI'
    #         plot_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/" \
    #                    f"cubicFit/batch{12}/normActivationFlag_{True}_UsedTRflag_{'feedback'}/"
    #         sub_ROI_ses_relevant_using = load_obj(
    #             f"{plot_dir}/sub_ROI_ses_relevant_using_{interestedROI}_batch{int(batch)}")
    #         get_file_created_time(f"{plot_dir}/sub_ROI_ses_relevant_using_{interestedROI}_batch{int(batch)}.pkl")
    #
    #     for sub in subjects:
    #         _Behav_differentiations.loc[_Behav_differentiations['sub'] == sub, f"withinSes-{interestedROI}"] = \
    #             np.mean(sub_ROI_ses_relevant_using[sub_ROI_ses_relevant_using['sub'] == sub]['integration_ratios'])
    #     return _Behav_differentiations
    #
    #
    # Behav_differentiations = load_withinSessionEffect(interestedROI=interestedROI,
    #                                                   _Behav_differentiations=Behav_differentiations)
    def load_acrossSessionEffect(interestedROI=None, _Behav_differentiations=None):
        # acrossSessionEffect
        for sub in subjects:
            # if interestedROI != 'megaROI':
            #     # code and data from OrganizedScripts/ROI/autoAlign_ses1ses5/cubicFit/constrainedCubicFit.py
            #     # integration_ratio = np.load(f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/"
            #     #                             f"autoAlign_ROIanalysis_ses1ses5/subjects/{sub}/ses5/{interestedROI}/"
            #     #                             f"integrationScore.npy")  # load
            [ses1_XY, ses5_XY, ses1_MN, ses5_MN, differentiation_ratio, integration_ratio] = np.load(
                f"{workingDir}/data/"
                f"result/subjects/{sub}/ses5/{interestedROI}/integration_ratio_allData.npy")  # load
            #
            #     # get_file_created_time(f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/"
            #     #                       f"autoAlign_ROIanalysis_ses1ses5/subjects/{sub}/ses5/{interestedROI}/"
            #     #                       f"integrationScore.npy")
            # else:
            #     # code and data from OrganizedScripts/megaROI/acrossSession/cubicFit/constrainedCubicFit.py
            #     assert interestedROI == 'megaROI'
            #     integration_ratio = np.load(
            #         f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/subjects/"
            #         f"{sub}/ses5/{interestedROI}/"
            #         f"integrationScore.npy")  # load integration score
            #     get_file_created_time(f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/subjects/"
            #                           f"{sub}/ses5/{interestedROI}/"
            #                           f"integrationScore.npy")
            # print(f"{sub} ROI={interestedROI} acrossSes integration_ratio={integration_ratio}")

            _Behav_differentiations.loc[_Behav_differentiations['sub'] == sub, f"acrossSes-{interestedROI}"] = \
                float(integration_ratio)
        return _Behav_differentiations


    Behav_differentiations = load_acrossSessionEffect(interestedROI=interestedROI,
                                                      _Behav_differentiations=Behav_differentiations)

# print(Behav_differentiations.columns)

num_rows, num_cols = 2, 3
corr_withinSes = {}
corr_acrossSes = {}
R2_withinSes = {}
R2_acrossSes = {}
# looPerformance_withinSes = {}
# looPerformance_acrossSes = {}
for ROIdict in [
    {
        'V1': 'V1_FreeSurfer',
        'V2': 'V2_FreeSurfer',
        'LO': 'LOC_FreeSurfer',
        'IT': 'IT_FreeSurfer',
        'FG': 'Fus_FreeSurfer',  # fusiform gyrus
        'PHC_FS': 'PHC_FreeSurfer',  # parahippocampal cortex
        # 'PHC_ASHS': 'lfseg_corr_usegray_5'
    },

    {
        # 'HC_FS': 'HC_FreeSurfer',
        'HC_ASHS': 'lfseg_corr_usegray_hippocampus',
        'CA1': 'lfseg_corr_usegray_1',
        'CA2/3': 'lfseg_corr_usegray_2',
        'DG': 'lfseg_corr_usegray_3',
        # 'ERC': 'lfseg_corr_usegray_4',
        # 'PRC': 'lfseg_corr_usegray_6',
        'SUB': 'lfseg_corr_usegray_7',
        'megaROI': 'megaROI'
    }]:

    resolution = 5
    # Create a new figure
    fig_within, axs_within = plt.subplots(num_rows, num_cols, figsize=(num_cols * resolution, num_rows * resolution))

    # Flatten the 2D array of axes to a 1D array for easier iteration
    axs_within = axs_within.ravel()

    # Create a new figure
    fig_across, axs_across = plt.subplots(num_rows, num_cols, figsize=(num_cols * resolution, num_rows * resolution))

    # Flatten the 2D array of axes to a 1D array for easier iteration
    axs_across = axs_across.ravel()

    for ii, interestedROI in enumerate(ROIdict):
        withinSess_integrationScore = np.asarray(Behav_differentiations[f"withinSes-{ROIdict[interestedROI]}"])
        acrossSess_integrationScore = np.asarray(Behav_differentiations[f"acrossSes-{ROIdict[interestedROI]}"])

        # convert differentiations to integrationScore
        behavior_catPer_integrationScore = - np.asarray(Behav_differentiations['Behav_differentiation'])

        # run a correlation between behavior catPer integrationScore versus ROI integrationScore.
        from scipy.stats import pearsonr

        print(f"{interestedROI}-pearsonr(behavior_catPer_integrationScore, withinSess_integrationScore)="
              f"{pearsonr(behavior_catPer_integrationScore, withinSess_integrationScore)}")
        print(f"{interestedROI}-pearsonr(behavior_catPer_integrationScore, acrossSess_integrationScore)="
              f"{pearsonr(behavior_catPer_integrationScore, acrossSess_integrationScore)}")

        corr_withinSes[interestedROI] = pearsonr(behavior_catPer_integrationScore, withinSess_integrationScore)[0]
        corr_acrossSes[interestedROI] = pearsonr(behavior_catPer_integrationScore, acrossSess_integrationScore)[0]


        def resample_leaveOO_linearRegression(X, y, resampleTimes=1000):
            """
            Resample the data n times .
            For each resamples, leave one dot out and fit a linear regression model.
            Predict the left out dot. Rotate the leave one dot out rotations and obtain len(X) predictions.
            correlate the predictions with the actual X and y values and save as a final correlation metric.
            after n times of resampling, return the final n metrics.
            For the final n metrics, save the 2.5% and 97.5% percentile as the confidence interval.
            """
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import LeaveOneOut
            # Create LeaveOneOut cross-validator
            loo = LeaveOneOut()

            # final_correlation_metrics = []
            correlations = []
            for curr_resample in tqdm(range(resampleTimes)):
                resampled_indices = np.random.choice(len(X), len(X), replace=True)
                resampled_X = X[resampled_indices]
                resampled_y = y[resampled_indices]

                y_preds = []
                y_reals = []
                # for currLeftOutDot in range(len(X)):
                for train_index, test_index in loo.split(resampled_X):
                    X_train, X_test = resampled_X[train_index], resampled_X[test_index]
                    y_train, y_test = resampled_y[train_index], resampled_y[test_index]

                    # Fit linear regression model
                    reg = LinearRegression()
                    reg.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

                    # Predict the left-out dot
                    y_pred = reg.predict(X_test.reshape(1, -1))
                    y_preds.append(float(y_pred))

                    y_reals.append(float(y_test))

                # Correlate the prediction with the actual X and y values
                # correlation = np.corrcoef(np.concatenate([y_preds, y_reals], axis=1).T)[0, 1]
                correlation = np.corrcoef(np.asarray(y_preds), np.asarray(y_reals))[0, 1]
                correlations.append(correlation)

                # # Calculate the mean correlation for this resample
                # mean_correlation = np.mean(correlations)
                # final_correlation_metrics.append(mean_correlation)

            # Calculate the confidence interval (2.5% and 97.5% percentiles)
            confidence_interval = np.percentile(correlations, [5, 95])

            return correlations, confidence_interval


        # define a function to plot a scatter plot with behavior catPer integrationScore versus ROI integrationScore and fit a linear regression line and report the R square value.
        def plotScatterAndLinearRegression(X, y, title, ax=None):
            from sklearn.linear_model import LinearRegression
            X = X.reshape(-1, 1)
            y = y.reshape(-1, 1)
            reg = LinearRegression()
            reg.fit(X, y)
            ax.scatter(X, y)
            ax.plot(X, reg.predict(X), color='red', linewidth=2)
            ax.set_ylabel('integrationScore')
            # Round reg.score(X, y) to 3 decimal places for title
            if testMode:
                resampleTimes = 10
            else:
                resampleTimes = 1000
            correlations, confidence_interval = resample_leaveOO_linearRegression(X, y, resampleTimes=resampleTimes)
            ax.set_title(f"{title}-R2={reg.score(X, y):.2f}-corr={pearsonr(X.reshape(-1), y.reshape(-1))[0]:.2f}")
            if np.prod(confidence_interval) > 0:
                significance = '*'
            else:
                significance = ''
            ax.set_xlabel(
                f'behavior_catPer-conf_interv={confidence_interval[0]:.2}~{confidence_interval[1]:.2}{significance}')

            return reg.score(X, y)


        # if interestedROI in plotROIs:
        R2_withinSes[interestedROI] = plotScatterAndLinearRegression(
            behavior_catPer_integrationScore, withinSess_integrationScore, f"within-{interestedROI}",
            ax=axs_within[ii])
        R2_acrossSes[interestedROI] = plotScatterAndLinearRegression(
            behavior_catPer_integrationScore, acrossSess_integrationScore, f"across-{interestedROI}",
            ax=axs_across[ii])
        # correlations, confidence_interval = resample_leaveOO_linearRegression(
        #     behavior_catPer_integrationScore, acrossSess_integrationScore)
    plt.show()

print("done")

# # plot a scatter plot with behavior catPer integrationScore versus ROI integrationScore
# plt.figure()
# plt.scatter(behavior_catPer_integrationScore, withinSess_integrationScore)
# plt.xlabel('behavior_catPer')
# plt.ylabel('withinSess_integrationScore')
# plt.title(f"withinSess_integrationScore {interestedROI}")
# plt.show()

# plt.figure()
# plt.scatter(behavior_catPer_integrationScore, acrossSess_integrationScore)
# plt.xlabel('behavior_catPer')
# plt.ylabel('acrossSess_integrationScore')
# plt.title(f"acrossSess_integrationScore {interestedROI}")
# plt.show()


# run a linear regression with behavior catPer integrationScore versus ROI integrationScore in a leave one sub out manner and obtain a averaged model performance.
# def linearRegression(X, y):
#     # leave 10% out
#     from sklearn.linear_model import LinearRegression
#     from sklearn.model_selection import LeaveOneOut
#     loo = LeaveOneOut()
#     loo.get_n_splits(X)
#     model_performance = []
#     for train_index, test_index in loo.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         X_train = X_train.reshape(-1, 1)
#         X_test = X_test.reshape(-1, 1)
#         y_train = y_train.reshape(-1, 1)
#         y_test = y_test.reshape(-1, 1)
#         reg = LinearRegression()
#         reg.fit(X_train, y_train)
#         model_performance.append(reg.score(X_test, y_test))
#     return np.mean(model_performance)
#
#     # from sklearn.linear_model import LinearRegression
#     # from sklearn.model_selection import LeaveOneOut
#     # loo = LeaveOneOut()
#     # loo.get_n_splits(X)
#     # model_performance = []
#     # for train_index, test_index in loo.split(X):
#     #     X_train, X_test = X[train_index], X[test_index]
#     #     y_train, y_test = y[train_index], y[test_index]
#     #     X_train = X_train.reshape(-1, 1)
#     #     X_test = X_test.reshape(-1, 1)
#     #     y_train = y_train.reshape(-1, 1)
#     #     y_test = y_test.reshape(-1, 1)
#     #     reg = LinearRegression()
#     #     reg.fit(X_train, y_train)
#     #     model_performance.append(reg.score(X_test, y_test))
#     #
#     # print(f"model_performance = {np.mean(model_performance)}")
#     # return np.mean(model_performance)
#
#
# withinSess_modelPerformance = linearRegression(behavior_catPer_integrationScore, withinSess_integrationScore)
# acrossSess_modelPerformance = linearRegression(behavior_catPer_integrationScore, acrossSess_integrationScore)


# ROI_integrationScore = pd.read_csv(
#     f"{projectDir}/subjects/{subjects[0]}/behavior/ROI_integrationScore.csv")
# ROI_integrationScore = ROI_integrationScore.drop(columns=['Unnamed: 0'])
# ROI_integrationScore = ROI_integrationScore.set_index('sub')
# ROI_integrationScore = ROI_integrationScore.loc[subjects]
# ROI_integrationScore = ROI_integrationScore.reset_index()
#
# # plot a scatter plot with behavior catPer integrationScore versus ROI integrationScore
# plt.scatter(behavior_catPer_integrationScore['catPer'], ROI_integrationScore['integrationScore'])
# plt.xlabel('behavior_catPer')
# plt.ylabel('ROI_integrationScore')
# plt.savefig(f"{projectDir}/results/behavior_catPer_vs_ROI_integrationScore.png")
# plt.close()
#
# # run a linear regression with behavior catPer integrationScore versus ROI integrationScore in a leave one sub out manner and obtain a averaged model performance.
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import LeaveOneOut
# loo = LeaveOneOut()
# loo.get_n_splits(behavior_catPer_integrationScore)
# X = behavior_catPer_integrationScore['catPer'].values.reshape(-1, 1)
# y = ROI_integrationScore['integrationScore'].values.reshape(-1, 1)
# model = LinearRegression()
# model.fit(X, y)
# print(model.coef_)
# print(model.intercept_)
# print(model.score(X, y))
#
# # run a correlation between behavior catPer integrationScore versus ROI integrationScore.
# from scipy.stats import pearsonr
# corr, _ = pearsonr(behavior_catPer_integrationScore['catPer'], ROI_integrationScore['integrationScore'])
# print(corr)
#
# # load ROI integrationScore for across session effect
# ROI_integrationScore_acrossSession = pd.read_csv(
#     f"{projectDir}/subjects/{subjects[0]}/behavior/ROI_integrationScore_acrossSession.csv")
# ROI_integrationScore_acrossSession = ROI_integrationScore_acrossSession.drop(columns=['Unnamed: 0'])
# ROI_integrationScore_acrossSession = ROI_integrationScore_acrossSession.set_index('sub')
# ROI_integrationScore_acrossSession = ROI_integrationScore_acrossSession.loc[subjects]
# ROI_integrationScore_acrossSession = ROI_integrationScore_acrossSession.reset_index()
#
# # plot a scatter plot with behavior catPer integrationScore versus ROI integrationScore
# plt.scatter(behavior_catPer_integrationScore['catPer'], ROI_integrationScore_acrossSession['integrationScore'])
# plt.xlabel('behavior_catPer')
# plt.ylabel('ROI_integrationScore_acrossSession')
# plt.savefig(f"{projectDir}/results/behavior_catPer_vs_ROI_integrationScore_acrossSession.png")
# plt.close()
#
# # run a linear regression with behavior catPer integrationScore versus ROI integrationScore in a leave one sub out manner and obtain a averaged model performance.
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import LeaveOneOut
# loo = LeaveOneOut()
# loo.get_n_splits(behavior_catPer_integrationScore)
# X = behavior_catPer_integrationScore['catPer'].values.reshape(-1, 1)
# y = ROI_integrationScore_acrossSession['integrationScore'].values.reshape(-1, 1)
# model = LinearRegression()
# model.fit(X, y)
# print(model.coef_)
# print(model.intercept_)
# print(model.score(X, y))
#
# # run a correlation between behavior catPer integrationScore versus ROI integrationScore.
# from scipy.stats import pearsonr
# corr, _ = pearsonr(behavior_catPer_integrationScore['catPer'], ROI_integrationScore_acrossSession['integrationScore'])
# print(corr)
#
# # load ROI integrationScore for within session effect
# ROI_integrationScore_withinSession = pd.read_csv(
#     f"{projectDir}/subjects/{subjects[0]}/behavior/ROI_integrationScore_withinSession.csv")
# ROI_integrationScore_withinSession = ROI_integrationScore_withinSession.drop(columns=['Unnamed: 0'])
# ROI_integrationScore_withinSession = ROI_integrationScore_withinSession.set_index('sub')
# ROI_integrationScore_withinSession = ROI_integrationScore_withinSession.loc[subjects]
# ROI_integrationScore_withinSession = ROI_integrationScore_withinSession.reset_index()
#
# # plot a scatter plot with behavior catPer integrationScore versus ROI integrationScore
# plt.scatter(behavior_catPer_integrationScore['catPer'], ROI_integrationScore_withinSession['integrationScore'])
# plt.xlabel('behavior_catPer')
# plt.ylabel('ROI_integrationScore_withinSession')
# plt.savefig(f"{projectDir}/results/behavior_catPer_vs_ROI_integrationScore_withinSession.png")
# plt.close()
#
# # run a linear regression with behavior catPer integrationScore versus ROI integrationScore in a leave one sub out manner and obtain a averaged model performance.
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import LeaveOneOut
# loo = LeaveOneOut()
# loo.get_n_splits(behavior_catPer_integrationScore)
# X = behavior_catPer_integrationScore['catPer'].values.reshape(-1, 1)
# y = ROI_integrationScore_withinSession['integrationScore'].values.reshape(-1, 1)
# model = LinearRegression()
# model.fit(X, y)
# print(model.coef_)
# print(model.intercept_)
# print(model.score(X, y))
#
# # run a correlation between behavior catPer integrationScore versus ROI integrationScore.
# from scipy.stats import pearsonr
# corr, _ = pearsonr(behavior_catPer_integrationScore['catPer'], ROI_integrationScore_withinSession['integrationScore'])
# print(corr)


# def get_autoAlign_ROI(scan_asTemplates):
#     # 这个函数的目的是使用自动对齐的mat准备func空间中的方便使用的ROI
#     jobarrayDict = {}
#     jobarrayID = 1
#     for sub in subjects:
#         # for ses in [2, 3, 4]:
#         jobarrayDict[jobarrayID] = [sub]
#         jobarrayID += 1
#     np.save(
#         f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
#         f"OrganizedScripts/ROI/autoAlign/get_autoAlign_ROI_jobID.npy",
#         jobarrayDict)
#     if testMode:
#         cmd = f"sbatch --requeue --array=1-1 /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/" \
#               f"OrganizedScripts/ROI/autoAlign/get_autoAlign_ROI.sh"
#     else:
#         cmd = f"sbatch --requeue --array=1-{len(jobarrayDict)} /gpfs/milgram/project/turk-browne/projects/rt-cloud/" \
#               f"projects/rtSynth_rt/" \
#               f"OrganizedScripts/ROI/autoAlign/get_autoAlign_ROI.sh"
#     sbatch_response = kp_run(cmd)
#     # 23653052_1-20  # -thr 0.1 -bin {subfield_func}  每一个需要2.5min
#     jobID = getjobID_num(sbatch_response)
#     waitForEnd(jobID)
#     if testMode:
#         completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
#     else:
#         completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))
#
#
# get_autoAlign_ROI(scan_asTemplates)
