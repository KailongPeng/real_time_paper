# % code:
#     % fig3c: OrganizedScripts/catPer/catPer_fixedCenter.py -> plot_allDots
#     % fig3c inset: OrganizedScripts/catPer/catPer_fixedCenter.py -> plot_slope_XYses1ses5_MNses1ses5

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from utils import get_subjects
from utils import cal_resample, bar, mkdir
import scipy.optimize as opt

def logit(subject, axis, _ax, which_subject, ses=1, plotFigure=False, color='red'):
    # /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub003/ses1/catPer/catPer_000000sub003_1.txt
    resp = pd.read_csv('./' + subject, sep='\t', lineterminator='\n',
                       header=None)  # /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub003/ses1/catPer/catPer_000000sub003_1.txt
    resp = resp.rename(columns={
        0: "workerid",
        1: "this_trial",
        2: "Image",
        3: "ImagePath",
        4: "category",
        5: "imageStart",
        6: "imageEnd",
        7: "response",
        8: "responseTime",
        9: "currentTime",
        10: "ButtonLeft",
        11: "ButtonRight"
    })

    singleAxisData = resp.loc[resp['category'] == axis]  # here resp['category'] is either bedChair or tableBench
    if len(singleAxisData) == 0:
        return None, None, None, None
    _x = np.asarray(singleAxisData['Image'])
    # X_dict={0:18, 1:26, 2:34, 3:42, 4:50, 5:58, 6:66, 7:74, 8:82, 9:18, 10:26, 11:34, 12:42, 13:50, 14:58, 15:66, 16:74, 17:82}
    # X_dict={0:18, 1:26, 2:34, 3:42, 4:50, 5:58, 6:66, 7:74, 8:82, 9:18, 10:26, 11:34, 12:42, 13:50, 14:58, 15:66, 16:74, 17:82}
    X_dict = {0: 18, 1: 26, 2: 34, 3: 38, 4: 42, 5: 46, 6: 50, 7: 54, 8: 58, 9: 62, 10: 66, 11: 74, 12: 82, 13: 18,
              14: 26, 15: 34, 16: 38, 17: 42, 18: 46, 19: 50, 20: 54, 21: 58, 22: 62, 23: 66, 24: 74, 25: 82}
    x = []
    for _ in _x:
        x.append(X_dict[_])
    y = np.asarray(singleAxisData['response'])
    # according to whether bottonLeft is Bed or Table, resave the response
    y_ = []
    if 'bed' in axis:
        button_good = list(singleAxisData['ButtonLeft'] == "Bed")
    else:
        button_good = list(singleAxisData['ButtonLeft'] == "Table")
    for i, j in enumerate(button_good):
        if j:
            y_.append(y[i])
        else:
            if y[i] == 1:
                y_.append(2)
            else:
                y_.append(1)
    y = np.asarray(y_)

    xy = np.concatenate((np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)), axis=0)
    xy = xy[:, xy[0, :].argsort()]
    # xy[1,:]=2-xy[1,:]
    methodFlag = "method1"
    if methodFlag == "method1":
        # method 1: use original datapoints for regression
        x = xy[0, :]
        y = xy[1, :] - 1
        x = x / 100
    else:
        # method 2: use frequency of choices for regression
        pass
        # prob=[]
        # for i in [18, 26, 34, 42, 50, 58, 66, 74, 82]:
        #     _prob=np.mean(xy[1,xy[0]==i])
        #     #print(xy[1,xy[0]==i])
        #     #print(_prob,end='\n\n')
        #     prob.append(_prob)
        # x=np.asarray([18, 26, 34, 42, 50, 58, 66, 74, 82], dtype=np.float128)/100
        # y=np.asarray(prob)-1

    morph18acc = round(np.mean(1 - y[x == 0.18]), 3)
    morph26acc = round(np.mean(1 - y[x == 0.26]), 3)
    morph74acc = round(np.mean(y[x == 0.74]), 3)
    morph82acc = round(np.mean(y[x == 0.82]), 3)
    # print("morph 1 acc=",morph1acc)
    # print("morph 21 acc=",morph21acc)
    # print("morph 80 acc=",morph80acc)
    # print("morph 100 acc=",morph100acc)

    if morph18acc > 0.8 and morph82acc > 0.8:
        title = '✓ '
        exclusion = "✓"
    else:
        title = 'X '
        exclusion = "X"

    def f(x, k, x0):
        return 1 / (1. + np.exp(-k * (x - x0)))

    # fit and plot the curve
    (k, x0), _ = opt.curve_fit(f, x, y)
    n_plot = 100
    x_plot = np.linspace(min(x), max(x), n_plot)
    y_fit = f(x_plot, k, x0)

    if plotFigure:
        resoluitionTimes = 3
        # fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        if methodFlag == "method1":
            # _ = ax.plot(rand_jitter(x), rand_jitter(y), '.', linewidth=8*resoluitionTimes)
            _ = _ax.scatter(rand_jitter(x), rand_jitter(y), s=4 * resoluitionTimes, c=color)
        else:
            _ = _ax.plot(x, y, 'o')
        _ = _ax.plot(x_plot, y_fit, '-', linewidth=1 * resoluitionTimes, c=color, label=f'ses{ses}')
        plt.setp(_ax.get_xticklabels(), color=(14 / 256, 53 / 256, 95 / 256))
        plt.setp(_ax.get_yticklabels(), color=(14 / 256, 53 / 256, 95 / 256))

        _ax.tick_params(axis="y", labelsize=40 * resoluitionTimes)
        _ax.tick_params(axis="x", labelsize=40 * resoluitionTimes)
        _ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
        # ax.set_facecolor((242 / 256, 242 / 256, 242 / 256))
        _ax.set_yticks([])
        _ax.set_xticks([])
        _ax.legend()

        # assert len(singleAxisData) == 156
        if len(singleAxisData) == 72:
            title = title + f"{axis} sub_{which_subject} ses{ses}\n k={np.round(k, 2)};x0={np.round(x0, 2)}; dataNum={len(singleAxisData)}"

            # title = title + "{} sub_{}\n k={};x0={}".format(axis,
            #                                                 which_subject,  # (subject.split("_")[1]).split(".")[0],
            #                                                 round(k, 2),
            #                                                 round(x0, 2),
            #                                                 )
        else:
            title = title + f"{axis} sub_{which_subject} ses{ses}\n k={np.round(k, 2)};x0={np.round(x0, 2)}; dataNum={len(singleAxisData)}"

            # title = "X {} sub_{}\n k={};x0={};dataNum={}".format(axis,
            #                                                      which_subject,
            #                                                      # (subject.split("_")[1]).split(".")[0],
            #                                                      round(k, 2),
            #                                                      round(x0, 2),
            #                                                      len(singleAxisData)
            #                                                      )
            exclusion = 'X'

        _ = _ax.set_title(title, fontdict={'fontsize': 10, 'fontweight': 'medium'})

    return morph18acc, morph26acc, morph74acc, morph82acc, k, x0, exclusion, x, y


def logit_fixedCenter(subject, axis, _ax, which_subject, ses=1, plotFigure=False, color='red', centerX0=None):
    print(f"centerX0={centerX0}")
    # /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub003/ses1/catPer/catPer_000000sub003_1.txt
    resp = pd.read_csv('./' + subject, sep='\t', lineterminator='\n',
                       header=None)  # /gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/sub003/ses1/catPer/catPer_000000sub003_1.txt
    resp = resp.rename(columns={
        0: "workerid",
        1: "this_trial",
        2: "Image",
        3: "ImagePath",
        4: "category",
        5: "imageStart",
        6: "imageEnd",
        7: "response",
        8: "responseTime",
        9: "currentTime",
        10: "ButtonLeft",
        11: "ButtonRight"
    })

    singleAxisData = resp.loc[resp['category'] == axis]  # here resp['category'] is either bedChair or tableBench
    if len(singleAxisData) == 0:
        return None, None, None, None
    _x = np.asarray(singleAxisData['Image'])
    # X_dict={0:18, 1:26, 2:34, 3:42, 4:50, 5:58, 6:66, 7:74, 8:82, 9:18, 10:26, 11:34, 12:42, 13:50, 14:58, 15:66, 16:74, 17:82}
    # X_dict={0:18, 1:26, 2:34, 3:42, 4:50, 5:58, 6:66, 7:74, 8:82, 9:18, 10:26, 11:34, 12:42, 13:50, 14:58, 15:66, 16:74, 17:82}
    X_dict = {0: 18, 1: 26, 2: 34, 3: 38, 4: 42, 5: 46, 6: 50, 7: 54, 8: 58, 9: 62, 10: 66, 11: 74, 12: 82, 13: 18,
              14: 26, 15: 34, 16: 38, 17: 42, 18: 46, 19: 50, 20: 54, 21: 58, 22: 62, 23: 66, 24: 74, 25: 82}
    x = []
    for _ in _x:
        x.append(X_dict[_])
    y = np.asarray(singleAxisData['response'])
    # according to whether bottonLeft is Bed or Table, resave the response
    y_ = []
    if 'bed' in axis:
        button_good = list(singleAxisData['ButtonLeft'] == "Bed")
    else:
        button_good = list(singleAxisData['ButtonLeft'] == "Table")
    for i, j in enumerate(button_good):
        if j:
            y_.append(y[i])
        else:
            if y[i] == 1:
                y_.append(2)
            else:
                y_.append(1)
    y = np.asarray(y_)

    xy = np.concatenate((np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)), axis=0)
    xy = xy[:, xy[0, :].argsort()]
    # xy[1,:]=2-xy[1,:]
    methodFlag = "method1"
    if methodFlag == "method1":
        # method 1: use original datapoints for regression
        x = xy[0, :]
        y = xy[1, :] - 1
        x = x / 100
    else:
        # method 2: use frequency of choices for regression
        pass
        # prob=[]
        # for i in [18, 26, 34, 42, 50, 58, 66, 74, 82]:
        #     _prob=np.mean(xy[1,xy[0]==i])
        #     #print(xy[1,xy[0]==i])
        #     #print(_prob,end='\n\n')
        #     prob.append(_prob)
        # x=np.asarray([18, 26, 34, 42, 50, 58, 66, 74, 82], dtype=np.float128)/100
        # y=np.asarray(prob)-1

    morph18acc = round(np.mean(1 - y[x == 0.18]), 3)
    morph26acc = round(np.mean(1 - y[x == 0.26]), 3)
    morph74acc = round(np.mean(y[x == 0.74]), 3)
    morph82acc = round(np.mean(y[x == 0.82]), 3)
    # print("morph 1 acc=",morph1acc)
    # print("morph 21 acc=",morph21acc)
    # print("morph 80 acc=",morph80acc)
    # print("morph 100 acc=",morph100acc)

    if morph18acc > 0.8 and morph82acc > 0.8:
        title = '✓ '
        exclusion = "✓"
    else:
        title = 'X '
        exclusion = "X"
    if ses == 1:

        def function_f(__x, __k, __x0):
            return 1 / (1. + np.exp(-__k * (__x - __x0)))

        # fit and plot the curve
        (k, x0), _ = opt.curve_fit(function_f, x, y)
        n_plot = 100
        x_plot = np.linspace(min(x), max(x), n_plot)
        y_fit = function_f(x_plot, k, x0)
    elif ses == 5:
        # centerX0
        def function_f(__x, __k):
            return 1 / (1. + np.exp(-__k * (__x - centerX0)))

        # fit and plot the curve
        (k), _ = opt.curve_fit(function_f, x, y)
        x0 = centerX0
        n_plot = 100
        x_plot = np.linspace(min(x), max(x), n_plot)
        y_fit = function_f(x_plot, k)
    else:
        raise Exception("error")

    if plotFigure:
        resoluitionTimes = 3
        # fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        if methodFlag == "method1":
            # _ = ax.plot(rand_jitter(x), rand_jitter(y), '.', linewidth=8*resoluitionTimes)
            _ = _ax.scatter(rand_jitter(x), rand_jitter(y), s=4 * resoluitionTimes, c=color)
        else:
            _ = _ax.plot(x, y, 'o')
        _ = _ax.plot(x_plot, y_fit, '-', linewidth=1 * resoluitionTimes, c=color, label=f'ses{ses}')
        plt.setp(_ax.get_xticklabels(), color=(14 / 256, 53 / 256, 95 / 256))
        plt.setp(_ax.get_yticklabels(), color=(14 / 256, 53 / 256, 95 / 256))

        _ax.tick_params(axis="y", labelsize=40 * resoluitionTimes)
        _ax.tick_params(axis="x", labelsize=40 * resoluitionTimes)
        _ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
        # ax.set_facecolor((242 / 256, 242 / 256, 242 / 256))
        _ax.set_yticks([])
        _ax.set_xticks([])
        _ax.legend()

        # assert len(singleAxisData) == 156
        if len(singleAxisData) == 72:
            title = title + f"{axis} sub_{which_subject} ses{ses}\n k={np.round(k, 2)};x0={np.round(x0, 2)}; dataNum={len(singleAxisData)}"

            # title = title + "{} sub_{}\n k={};x0={}".format(axis,
            #                                                 which_subject,  # (subject.split("_")[1]).split(".")[0],
            #                                                 round(k, 2),
            #                                                 round(x0, 2),
            #                                                 )
        else:
            title = title + f"{axis} sub_{which_subject} ses{ses}\n k={np.round(k, 2)};x0={np.round(x0, 2)}; dataNum={len(singleAxisData)}"

            # title = "X {} sub_{}\n k={};x0={};dataNum={}".format(axis,
            #                                                      which_subject,
            #                                                      # (subject.split("_")[1]).split(".")[0],
            #                                                      round(k, 2),
            #                                                      round(x0, 2),
            #                                                      len(singleAxisData)
            #                                                      )
            exclusion = 'X'

        _ = _ax.set_title(title, fontdict={'fontsize': 10, 'fontweight': 'medium'})

    return morph18acc, morph26acc, morph74acc, morph82acc, k, x0, exclusion, x, y


def checkVersion(subject):
    resp = pd.read_csv('./' + subject, sep='\t', lineterminator='\n', header=None)
    resp = resp.rename(columns={
        0: "workerid",
        1: "this_trial",
        2: "Image",
        3: "ImagePath",
        4: "category",
        5: "imageStart",
        6: "imageEnd",
        7: "response",
        8: "responseTime",
        9: "currentTime",
    })
    axes = {'bedChair': 'horizontal', 'benchBed': 'vertical', 'chairBench': 'diagonal'}
    for axis in ['bedChair', 'benchBed', 'chairBench']:
        data = resp.loc[resp['category'] == axis]
        if len(data) > 0:
            return axes[axis]


def rand_jitter(arr):
    stdev = .005 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def catPerDataAnalysis(sub='sub006', ses=1,
                       plotFigure=False, color='red', f=None,
                       ax=None,
                       x0Center=None):
    print(f"x0Center={x0Center}")

    if ses == 1:
        subFileName = f'000000{sub}_1'
    elif ses == 5:
        subFileName = f'000000{sub}_2'
    else:
        raise Exception("error")

    subject = f"catPer_{subFileName}.txt"

    if 'watts' in os.getcwd():
        projectDir = "/home/watts/Desktop/ntblab/kailong/rt-cloud/projects/rtSynth_rt/"
    elif 'kailong' in os.getcwd():
        projectDir = "/Users/kailong/Desktop/rtEnv/rt-cloud/projects/rtSynth_rt/"
    elif 'milgram' in os.getcwd():
        projectDir = "/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
    else:
        raise Exception('path error')
    if testMode:
        print(f"cd {projectDir}subjects/{sub}/ses{ses}/catPer")
    os.chdir(f"{projectDir}subjects/{sub}/ses{ses}/catPer")

    versionDict = {
        'horizontal': ['bedChair', 'tableBench'],  # AB, CD
        'vertical': ['benchBed', 'chairTable'],
        'diagonal': ['chairBench', 'bedTable']
    }
    version = checkVersion(subject)
    assert version == 'horizontal'

    slope = {}
    x0_container = {}
    if plotFigure:
        pass
    else:
        ax = [None, None]

    whichAxis = 0
    axis = versionDict[version][whichAxis]
    # if testMode:
    print(f"logit({subject}, {axis}, {ax[0]}, {sub}, plotFigure={plotFigure})")
    x_container = {}
    y_container = {}
    if ses == 1:
        (morph1acc, morph21acc, morph80acc, morph100acc, k, x0, exclusion, x, y) = logit(
            subject, axis, ax[whichAxis], sub, ses=ses,
            plotFigure=plotFigure, color=color)
    elif ses == 5:
        (morph1acc, morph21acc, morph80acc, morph100acc, k, x0, exclusion, x, y) = logit_fixedCenter(
            subject, axis, ax[whichAxis], sub,
            ses=ses,
            plotFigure=plotFigure,
            color=color,
            centerX0=x0Center['AB'])
    else:
        raise Exception("error")
    slope['AB'] = k
    x0_container['AB'] = x0
    x_container['AB'] = x
    y_container['AB'] = y

    whichAxis = 1
    axis = versionDict[version][whichAxis]
    if ses == 1:
        (morph1acc, morph21acc, morph80acc, morph100acc, k, x0, exclusion, x, y) = logit(
            subject, axis, ax[whichAxis], sub, ses=ses,
            plotFigure=plotFigure, color=color)
    elif ses == 5:
        (morph1acc, morph21acc, morph80acc, morph100acc, k, x0, exclusion, x, y) = logit_fixedCenter(
            subject, axis, ax[whichAxis], sub,
            ses=ses,
            plotFigure=plotFigure,
            color=color,
            centerX0=x0Center['CD'])
    else:
        raise Exception("error")
    slope['CD'] = k
    x0_container['CD'] = x0
    x_container['CD'] = x
    y_container['CD'] = y

    return slope, x0_container, x_container, y_container


Brain_differentiations = []

B_prob_slopes = []
batch = 12  # meaning both batch 1 and batch 2
subjects, scan_asTemplates = get_subjects(batch=batch)
testMode = False


def prepareData():
    Behav_differentiations = pd.DataFrame()
    Behav_slope = pd.DataFrame()
    x_ses1_XY = []
    x_ses5_XY = []
    x_ses1_MN = []
    x_ses5_MN = []

    y_ses1_XY = []
    y_ses5_XY = []
    y_ses1_MN = []
    y_ses5_MN = []

    for iii, sub in tqdm(enumerate(subjects)):
        f, ax = plt.subplots(1, 2, figsize=(20, 10))
        slope_ses1, x0_ses1, x_C_ses1, y_C_ses1 = catPerDataAnalysis(
            sub=sub, ses=1, plotFigure=True, color='red', f=f, ax=ax)
        slope_ses5, x0_ses5, x_C_ses5, y_C_ses5 = catPerDataAnalysis(
            sub=sub, ses=5, plotFigure=True, color='blue', f=f, ax=ax,
            x0Center=x0_ses1)
        plt.show()

        if scan_asTemplates[sub]['batch'] == 1:
            ses1_XY_slope, ses1_MN_slope, ses5_XY_slope, ses5_MN_slope = slope_ses1['AB'], slope_ses1['CD'], slope_ses5[
                'AB'], slope_ses5['CD']
            Behav_differentiation = (ses5_XY_slope - ses1_XY_slope) / (ses5_XY_slope + ses1_XY_slope) - (
                    ses5_MN_slope - ses1_MN_slope) / (ses5_MN_slope + ses1_MN_slope)

            x_ses1_XY.append(x_C_ses1['AB'])
            x_ses5_XY.append(x_C_ses5['AB'])
            x_ses1_MN.append(x_C_ses1['CD'])
            x_ses5_MN.append(x_C_ses5['CD'])

            y_ses1_XY.append(y_C_ses1['AB'])
            y_ses5_XY.append(y_C_ses5['AB'])
            y_ses1_MN.append(y_C_ses1['CD'])
            y_ses5_MN.append(y_C_ses5['CD'])
        elif scan_asTemplates[sub]['batch'] == 2:
            ses1_MN_slope, ses1_XY_slope, ses5_MN_slope, ses5_XY_slope = slope_ses1['AB'], slope_ses1['CD'], slope_ses5[
                'AB'], slope_ses5['CD']
            Behav_differentiation = (ses5_XY_slope - ses1_XY_slope) / (ses5_XY_slope + ses1_XY_slope) - (
                    ses5_MN_slope - ses1_MN_slope) / (ses5_MN_slope + ses1_MN_slope)

            x_ses1_XY.append(x_C_ses1['CD'])
            x_ses5_XY.append(x_C_ses5['CD'])
            x_ses1_MN.append(x_C_ses1['AB'])
            x_ses5_MN.append(x_C_ses5['AB'])

            y_ses1_XY.append(y_C_ses1['CD'])
            y_ses5_XY.append(y_C_ses5['CD'])
            y_ses1_MN.append(y_C_ses1['AB'])
            y_ses5_MN.append(y_C_ses5['AB'])
        else:
            raise Exception

        Behav_differentiations = pd.concat([Behav_differentiations, pd.DataFrame({
            'sub': sub,
            'ses1_XY_acc': ses1_XY_slope,
            # note here the key is named as acc because the other code was based on this, it actually means slope
            'ses1_MN_acc': ses1_MN_slope,
            'ses5_XY_acc': ses5_XY_slope,
            'ses5_MN_acc': ses5_MN_slope,
            'Behav_differentiation': Behav_differentiation,
            'Behav_integration': - Behav_differentiation
        }, index=[0])], ignore_index=True)

        Behav_slope = pd.concat([Behav_slope, pd.DataFrame({
            'sub': sub,
            'session': 1,
            'axis': 'XY',
            'acc': ses1_XY_slope
        }, index=[0])], ignore_index=True)
        Behav_slope = pd.concat([Behav_slope, pd.DataFrame({
            'sub': sub,
            'session': 1,
            'axis': 'MN',
            'acc': ses1_MN_slope
        }, index=[0])], ignore_index=True)
        Behav_slope = pd.concat([Behav_slope, pd.DataFrame({
            'sub': sub,
            'session': 5,
            'axis': 'XY',
            'acc': ses5_XY_slope
        }, index=[0])], ignore_index=True)
        Behav_slope = pd.concat([Behav_slope, pd.DataFrame({
            'sub': sub,
            'session': 5,
            'axis': 'MN',
            'acc': ses5_MN_slope
        }, index=[0])], ignore_index=True)

    mkdir(
        '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/analysisResults/allSubs/catPer/')
    # Behav_slope.to_csv(
    #     '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/analysisResults/allSubs/catPer/Behav_slope.csv')
    Behav_differentiations.to_csv(
        '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/'
        'analysisResults/allSubs/catPer/Behav_differentiations_fixedCenter.csv')

    # print(f"Brain_differentiations={Brain_differentiations}")
    print(f"Behav_differentiations={Behav_differentiations}")

    return (Behav_differentiations, Behav_slope,
            x_ses1_XY, x_ses5_XY, x_ses1_MN, x_ses5_MN, y_ses1_XY, y_ses5_XY, y_ses1_MN, y_ses5_MN)


(Behav_differentiations, Behav_slope,
 x_ses1_XY, x_ses5_XY, x_ses1_MN, x_ses5_MN, y_ses1_XY, y_ses5_XY, y_ses1_MN, y_ses5_MN) = prepareData()


def fig3c(x_ses1_XY, y_ses1_XY, x_ses5_XY, y_ses5_XY, x_ses1_MN, y_ses1_MN, x_ses5_MN, y_ses5_MN):

    def return_concatenate(list_of_lists):
        concatenated_list = []
        for sublist in list_of_lists:
            concatenated_list.extend(sublist)

        return concatenated_list

    _x_ses1_XY = return_concatenate(x_ses1_XY)
    _x_ses5_XY = return_concatenate(x_ses5_XY)
    _x_ses1_MN = return_concatenate(x_ses1_MN)
    _x_ses5_MN = return_concatenate(x_ses5_MN)

    _y_ses1_XY = return_concatenate(y_ses1_XY)
    _y_ses5_XY = return_concatenate(y_ses5_XY)
    _y_ses1_MN = return_concatenate(y_ses1_MN)
    _y_ses5_MN = return_concatenate(y_ses5_MN)

    def plot_ax(color='red',
                plotFigure=True,
                ax_=None,
                x=None,
                y=None,
                legend='ses',
                k_ses1=None,
                x0_ses1=None
                ):
        if legend == 'session1':
            def f__(_x, _k, _x0):
                return 1 / (1. + np.exp(-_k * (_x - _x0)))
        elif legend == 'session5':
            def f__(_x, _k):
                return 1 / (1. + np.exp(-_k * (_x - x0_ses1)))
        else:
            raise Exception("error")

        # fit and plot the curve
        if legend == 'session1':
            (k, x0), _ = opt.curve_fit(f__, x, y)
        elif legend == 'session5':
            (k), _ = opt.curve_fit(f__, x, y)
            x0 = x0_ses1
        else:
            raise Exception("error")

        n_plot = 100
        x_plot = np.linspace(min(x), max(x), n_plot)
        if legend == 'session1':
            y_fit = f__(x_plot, k, x0)
        elif legend == 'session5':
            y_fit = f__(x_plot, k)
        else:
            raise Exception("error")

        if plotFigure:
            resoluitionTimes = 1
            # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            methodFlag = "method1"
            if methodFlag == "method1":
                # _ = ax.plot(rand_jitter(x), rand_jitter(y), '.', linewidth=8*resoluitionTimes)
                _ = ax_.scatter(rand_jitter(x), rand_jitter(y), s=1 * resoluitionTimes, c=color)
            else:
                _ = ax_.plot(x, y, 'o')
            _ = ax_.plot(x_plot, y_fit, '-', linewidth=1 * resoluitionTimes, c=color, label=legend)
            ax_.legend(fontsize=50)  # Set the legend font size to 30

            plt.setp(ax_.get_xticklabels(), color=(14 / 256, 53 / 256, 95 / 256))
            plt.setp(ax_.get_yticklabels(), color=(14 / 256, 53 / 256, 95 / 256))

            ax_.tick_params(axis="y", labelsize=40 * resoluitionTimes)
            ax_.tick_params(axis="x", labelsize=40 * resoluitionTimes)
            ax_.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
            # ax.set_facecolor((242 / 256, 242 / 256, 242 / 256))
            ax_.set_yticks([])
            ax_.set_xticks([])
            ax_.legend()

            # _ = ax.set_title(title, fontdict={'fontsize': 10, 'fontweight': 'medium'})

        return k, x0

    figure, axs_ = plt.subplots(1, 2, figsize=(10, 5))

    k_ses1, _x0_ses1 = plot_ax(
        x=_x_ses1_XY, y=_y_ses1_XY,
        color='#5985D8',
        ax_=axs_[0], legend='session1')
    _, _ = plot_ax(
        x=_x_ses5_XY, y=np.asarray(_y_ses5_XY) + 0.02,
        color='#1DD3B0',
        ax_=axs_[0], legend='session5', k_ses1=k_ses1, x0_ses1=_x0_ses1)

    k_ses1, _x0_ses1 = plot_ax(
        x=_x_ses1_MN, y=_y_ses1_MN,
        color='#5985D8',
        ax_=axs_[1], legend='session1')
    _, _ = plot_ax(
        x=_x_ses5_MN, y=np.asarray(_y_ses5_MN) + 0.02,
        color='#1DD3B0',
        ax_=axs_[1], legend='session5', k_ses1=k_ses1, x0_ses1=_x0_ses1)
    savePath = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/temp/figures/allDots_logit_FC.pdf"
    figure.savefig(savePath, transparent=True)
    print(f"figure saved to {savePath}")
    figure.show()


fig3c(x_ses1_XY, y_ses1_XY, x_ses5_XY, y_ses5_XY, x_ses1_MN, y_ses1_MN, x_ses5_MN, y_ses5_MN)


def fig3c_inset(Behav_differentiations):
    ses5_XY_slope = np.asarray(list(Behav_differentiations['ses5_XY_acc']))
    ses1_XY_slope = np.asarray(list(Behav_differentiations['ses1_XY_acc']))
    ses5_MN_slope = np.asarray(list(Behav_differentiations['ses5_MN_acc']))
    ses1_MN_slope = np.asarray(list(Behav_differentiations['ses1_MN_acc']))

    # plot one bar plots with four bars, with error bars

    # Create a list of labels for the x-axis (e.g., session names)
    def cal_resample(data=None, times=5000, returnPvalue=False):

        if data is None:
            raise Exception
        iter_mean = []
        for _ in range(times):
            iter_distri = data[np.random.choice(len(data), len(data), replace=True)]
            iter_mean.append(np.nanmean(iter_distri))
        _mean = np.mean(iter_mean)
        _5 = np.percentile(iter_mean, 5)
        _95 = np.percentile(iter_mean, 95)
        pValue = 1 - np.sum(np.asarray(iter_mean) > 0) / len(iter_mean)
        # print(f"pValue={pValue}")
        _pValue = 1 - np.mean(np.asarray(iter_mean) > 0)
        # print(f"_pValue={_pValue}")
        assert pValue == _pValue
        if returnPvalue:
            return _mean, _5, _95, pValue
        else:
            return _mean, _5, _95

    def plotBar(title=None, data=None, errorBar_upper=None, errorBar_lower=None, sessions=None, fontSize=25):
        plt.figure(figsize=(3, 5))
        # Create a bar plot with error bars
        plt.bar(sessions, data, capsize=5, color=['#5985D8', '#1DD3B0'])

        # Add error bars to the plot
        plt.errorbar(sessions, data,
                     yerr=[np.array(data) - np.array(errorBar_lower), np.array(errorBar_upper) - np.array(data)],
                     fmt='none', capsize=5, color='black')
        plt.ylim([0, 30])
        # Set the y-ticks to only show 0 and 30
        plt.yticks([0, 15, 30], fontsize=fontSize)
        # Set the x-ticks font size to fontSize
        plt.xticks(fontsize=fontSize)
        # Add labels and title
        plt.xlabel(title, fontsize=fontSize)
        plt.ylabel('slope', fontsize=fontSize)
        plt.title(title, fontsize=fontSize)
        plt.savefig(f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/temp/figures/slope_{title}.pdf",
                    transparent=True)

    iterations = 5000
    sessions = ['ses1', 'ses5']

    data = []
    errorBar_upper = []
    errorBar_lower = []
    _mean, _5, _95, pValue = cal_resample(data=ses1_XY_slope, times=iterations, returnPvalue=True)
    data.append(_mean)
    errorBar_upper.append(_95)
    errorBar_lower.append(_5)
    _mean, _5, _95, pValue = cal_resample(data=ses5_XY_slope, times=iterations, returnPvalue=True)
    data.append(_mean)
    errorBar_upper.append(_95)
    errorBar_lower.append(_5)
    plotBar(title='XY', data=data, errorBar_upper=errorBar_upper, errorBar_lower=errorBar_lower, sessions=sessions)

    data = []
    errorBar_upper = []
    errorBar_lower = []
    _mean, _5, _95, pValue = cal_resample(data=ses1_MN_slope, times=iterations, returnPvalue=True)
    data.append(_mean)
    errorBar_upper.append(_95)
    errorBar_lower.append(_5)
    _mean, _5, _95, pValue = cal_resample(data=ses5_MN_slope, times=iterations, returnPvalue=True)
    data.append(_mean)
    errorBar_upper.append(_95)
    errorBar_lower.append(_5)
    plotBar(title='MN', data=data, errorBar_upper=errorBar_upper, errorBar_lower=errorBar_lower, sessions=sessions)


fig3c_inset(Behav_differentiations)

