import os
import sys
os.chdir("/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/real_time_paper/")
assert os.getcwd().endswith('real_time_paper'), "working dir should be 'real_time_paper'"
workingDir = os.getcwd()
sys.path.append('.')
# print current dir
print(f"getcwd = {os.getcwd()}")

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from utils import get_subjects, get_ROIList, kp_run, getjobID_num, waitForEnd, check_jobIDs

batch = 12  # meaning both batch 1 and batch 2
subjects, scan_asTemplates = get_subjects(batch=batch)


# def BrainBehavIntegrationScoreCompare():
#     cmd = "sbatch fig5/BrainBehavIntegrationScoreCompare/compare.sh"
#     sbatch_response = kp_run(cmd)
#     jobID = getjobID_num(sbatch_response)
#     waitForEnd(jobID)
#     completed = check_jobIDs([jobID])
#
#
# BrainBehavIntegrationScoreCompare()


def plot_integrationScore_components_brain_behav_compare(batch=None, testMode=None, fixedCenter=None, plotFig5=None):
    subjects, scan_asTemplates = get_subjects(batch=batch)
    # fixedCenter = True  # OrganizedScripts/catPer/catPer_fixedCenter.py
    if fixedCenter:
        fixedCenterFlag = ' FC'  # FixedCenter
    else:
        fixedCenterFlag = ''

    def prepare_data():
        dropCertainSub = []
        if len(dropCertainSub) > 0:
            for sub in dropCertainSub:
                subjects.remove(sub)
        ROIList = get_ROIList()
        ROIList = ROIList + ['megaROI']

        # load behavior catPer integrationScore

        if fixedCenter:
            __Behav_differentiations = pd.read_csv(
                '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/analysisResults/'
                'allSubs/catPer/Behav_differentiations_fixedCenter.csv')
        else:
            __Behav_differentiations = pd.read_csv(
                '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/analysisResults/'
                'allSubs/catPer/Behav_differentiations.csv')

        __Behav_differentiations.rename(columns=lambda x: x.replace('_acc', '_slope'), inplace=True)
        __Behav_differentiations = __Behav_differentiations.drop(columns=['Unnamed: 0'])

        if len(dropCertainSub) > 0:
            for sub in dropCertainSub:
                __Behav_differentiations = __Behav_differentiations[__Behav_differentiations['sub'] != sub]

        # Behav_differentiations
        # sub	ses1_XY_acc	ses1_MN_acc	ses5_XY_acc	ses5_MN_acc	Behav_differentiation
        # sub003	25.651651	25.494680	22.580585	27.620671	-0.103698
        # sub004	43.045274	17.879099	47.857142	18.532048	0.035002
        # sub005	40.229530	11.643448	40.479376	11.672306	0.001858

        # load ROI integrationScore for within session effect and average across session so that each subject has a single value.
        for interestedROI in tqdm(ROIList):
            # Behav_differentiations[f"withinSes-{interestedROI}"] = None
            __Behav_differentiations[f"acrossSes-{interestedROI}"] = None

        for interestedROI in tqdm(ROIList):
            def load_acrossSessionEffect(interestedROI=None, _Behav_differentiations=None):
                # acrossSessionEffect
                for sub in subjects:

                    if interestedROI == 'megaROI':
                        [ses1_XY, ses5_XY, ses1_MN, ses5_MN, differentiation_ratio, integration_ratio] = np.load(
                            f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/subjects/"
                            f"{sub}/ses5/{interestedROI}/integrationScore_allData.npy")
                    else:
                        [ses1_XY, ses5_XY, ses1_MN, ses5_MN, differentiation_ratio, integration_ratio] = np.load(
                            f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/"
                            f"autoAlign_ROIanalysis_ses1ses5/subjects/{sub}/ses5/{interestedROI}/"
                            f"integrationScore_allData.npy")

                    _Behav_differentiations.loc[_Behav_differentiations['sub'] == sub, f"acrossSes-{interestedROI}"] = \
                        float(integration_ratio)
                    _Behav_differentiations.loc[
                        _Behav_differentiations['sub'] == sub, f"acrossSes-{interestedROI}_XY_ses1"] = \
                        float(ses1_XY)
                    _Behav_differentiations.loc[
                        _Behav_differentiations['sub'] == sub, f"acrossSes-{interestedROI}_XY_ses5"] = \
                        float(ses5_XY)
                    _Behav_differentiations.loc[
                        _Behav_differentiations['sub'] == sub, f"acrossSes-{interestedROI}_MN_ses1"] = \
                        float(ses1_MN)
                    _Behav_differentiations.loc[
                        _Behav_differentiations['sub'] == sub, f"acrossSes-{interestedROI}_MN_ses5"] = \
                        float(ses5_MN)

                return _Behav_differentiations

            __Behav_differentiations = load_acrossSessionEffect(interestedROI=interestedROI,
                                                                _Behav_differentiations=__Behav_differentiations)
        return __Behav_differentiations

    Behav_differentiations = prepare_data()

    def compare_integration_conponent(ROIdict=None):
        num_rows, num_cols = len(ROIDict_), 2
        corr_t_acrossSes = {}
        corr_p_acrossSes = {}
        # R2_withinSes = {}
        R2_acrossSes = {}

        # for ROIdict in [
        #     {
        #         'V1': 'V1_FreeSurfer',
        #         'V2': 'V2_FreeSurfer',
        #         'LO': 'LOC_FreeSurfer',
        #         'IT': 'IT_FreeSurfer',
        #         'FG': 'Fus_FreeSurfer',  # fusiform gyrus
        #         'PHC_FS': 'PHC_FreeSurfer',  # parahippocampal cortex
        #         # 'PHC_ASHS': 'lfseg_corr_usegray_5'
        #     },
        #     {
        #         # 'HC_FS': 'HC_FreeSurfer',
        #         'HC_ASHS': 'lfseg_corr_usegray_hippocampus',
        #         'CA1': 'lfseg_corr_usegray_1',
        #         'CA2/3': 'lfseg_corr_usegray_2',
        #         'DG': 'lfseg_corr_usegray_3',
        #         # 'ERC': 'lfseg_corr_usegray_4',
        #         # 'PRC': 'lfseg_corr_usegray_6',
        #         'SUB': 'lfseg_corr_usegray_7',
        #         'megaROI': 'megaROI'
        #     }]:

        resolution = 5
        # # Create a new figure
        # fig_within, axs_within = plt.subplots(num_rows, num_cols,
        #                                       figsize=(num_cols * resolution, num_rows * resolution))
        #
        # # Flatten the 2D array of axes to a 1D array for easier iteration
        # axs_within = axs_within.ravel()

        # Create a new figure
        fig_, axs_ = plt.subplots(num_rows, num_cols,
                                  figsize=(num_cols * resolution, num_rows * resolution))

        # Flatten the 2D array of axes to a 1D array for easier iteration
        axs_ = axs_.ravel()
        curr_ax = 0
        for interestedROI in ROIdict:
            XY_ses1_neuro = np.asarray(Behav_differentiations[f"acrossSes-{ROIdict[interestedROI]}_XY_ses1"])
            XY_ses5_neuro = np.asarray(Behav_differentiations[f"acrossSes-{ROIdict[interestedROI]}_XY_ses5"])
            MN_ses1_neuro = np.asarray(Behav_differentiations[f"acrossSes-{ROIdict[interestedROI]}_MN_ses1"])
            MN_ses5_neuro = np.asarray(Behav_differentiations[f"acrossSes-{ROIdict[interestedROI]}_MN_ses5"])
            # behav_whichComponent = None
            for behav_whichComponent in ['XY_ses1-XY_ses5', 'MN_ses1-MN_ses5']:
                if behav_whichComponent == 'XY_ses1-XY_ses5':
                    # withinSess_integrationScore = np.asarray(Behav_differentiations[f"withinSes-{ROIdict[interestedROI]}"])
                    neuro_integrationScore_component = XY_ses1_neuro - XY_ses5_neuro
                    plotColor = '#AA4AC8'
                elif behav_whichComponent == 'MN_ses1-MN_ses5':
                    neuro_integrationScore_component = MN_ses1_neuro - MN_ses5_neuro
                    plotColor = '#BBBDBF'
                elif behav_whichComponent == '(XY_ses1-XY_ses5)/+':
                    neuro_integrationScore_component = (XY_ses1_neuro - XY_ses5_neuro) / (
                            XY_ses1_neuro + XY_ses5_neuro)
                    plotColor = '#BBBDBF'
                elif behav_whichComponent == '(MN_ses1-MN_ses5)/+':
                    neuro_integrationScore_component = (MN_ses1_neuro - MN_ses5_neuro) / (
                            MN_ses1_neuro + MN_ses5_neuro)
                    plotColor = '#BBBDBF'
                elif behav_whichComponent == 'integration score':
                    neuro_integrationScore_component = ((XY_ses1_neuro - XY_ses5_neuro) / (
                            XY_ses1_neuro + XY_ses5_neuro)) - \
                                                       ((MN_ses1_neuro - MN_ses5_neuro) / (
                                                               MN_ses1_neuro + MN_ses5_neuro))
                    plotColor = '#BBBDBF'
                else:
                    raise Exception(f"whichComponent={behav_whichComponent} is not defined")

                XY_ses1_behav = np.asarray(Behav_differentiations[f"ses1_XY_slope"])
                XY_ses5_behav = np.asarray(Behav_differentiations[f"ses5_XY_slope"])
                MN_ses1_behav = np.asarray(Behav_differentiations[f"ses1_MN_slope"])
                MN_ses5_behav = np.asarray(Behav_differentiations[f"ses5_MN_slope"])
                if behav_whichComponent == 'XY_ses1-XY_ses5':
                    # convert differentiations to integrationScore
                    # behavior_catPer_integrationScore = - np.asarray(Behav_differentiations['Behav_differentiation'])
                    behavior_catPer_integrationScore_component = XY_ses1_behav - XY_ses5_behav
                elif behav_whichComponent == 'MN_ses1-MN_ses5':
                    # acrossSess_integrationScore = np.asarray(Behav_differentiations[f"acrossSes-{ROIdict[interestedROI]}_MN_ses1"]) - \
                    #                                 np.asarray(Behav_differentiations[f"acrossSes-{ROIdict[interestedROI]}_MN_ses5"])
                    behavior_catPer_integrationScore_component = MN_ses1_behav - MN_ses5_behav
                elif behav_whichComponent == '(XY_ses1-XY_ses5)/+':
                    behavior_catPer_integrationScore_component = (XY_ses1_behav - XY_ses5_behav) / (
                            XY_ses1_behav + XY_ses5_behav)
                elif behav_whichComponent == '(MN_ses1-MN_ses5)/+':
                    behavior_catPer_integrationScore_component = (MN_ses1_behav - MN_ses5_behav) / (
                            MN_ses1_behav + MN_ses5_behav)
                elif behav_whichComponent == 'integration score':
                    behavior_catPer_integrationScore_component = ((XY_ses1_behav - XY_ses5_behav) / (
                            XY_ses1_behav + XY_ses5_behav)) - \
                                                                 ((MN_ses1_behav - MN_ses5_behav) / (
                                                                         MN_ses1_behav + MN_ses5_behav))
                else:
                    raise Exception(f"whichComponent={behav_whichComponent} is not defined")
                # print(f"acrossSess_integrationScore={acrossSess_integrationScore_component}")
                # print(f"behavior_catPer_integrationScore={behavior_catPer_integrationScore_component}")

                # run a correlation between behavior catPer integrationScore versus ROI integrationScore.
                from scipy.stats import pearsonr

                # print(f"{interestedROI}-pearsonr(behavior_catPer_integrationScore, withinSess_integrationScore)="
                #       f"{pearsonr(behavior_catPer_integrationScore, withinSess_integrationScore)}")
                print(f"{interestedROI}-pearsonr(behavior_catPer_integrationScore, acrossSess_integrationScore)="
                      f"{pearsonr(behavior_catPer_integrationScore_component, neuro_integrationScore_component)}")

                # corr_withinSes[interestedROI] = pearsonr(behavior_catPer_integrationScore, withinSess_integrationScore)[0]
                corr_t_acrossSes[f"{interestedROI}-{behav_whichComponent}"] = \
                    pearsonr(neuro_integrationScore_component, behavior_catPer_integrationScore_component)[0]
                corr_p_acrossSes[f"{interestedROI}-{behav_whichComponent}"] = \
                    pearsonr(neuro_integrationScore_component, behavior_catPer_integrationScore_component)[1]

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
                def plotScatterAndLinearRegression(X=None, y=None, title="", ax=None, notes='', _plotColor=None,
                                                   _behav_whichComponent=None):

                    xAxisType = 'neuro'
                    corr_t, corr_p = pearsonr(X.reshape(-1), y.reshape(-1))
                    title = f"{title} ct:{corr_t:.4f} cp:{corr_p:.4f}"  # ct:correlation t value; cp:correlation p value
                    from sklearn.linear_model import LinearRegression
                    X = X.reshape(-1, 1)
                    y = y.reshape(-1, 1)
                    reg = LinearRegression()
                    reg.fit(X, y)
                    ax.scatter(X, y, color=_plotColor)
                    ax.plot(X, reg.predict(X), color=plotColor, linewidth=2)

                    # Round reg.score(X, y) to 3 decimal places for title
                    if testMode:
                        resampleTimes = 10
                    else:
                        resampleTimes = 1000
                    # resampleTimes = 10
                    correlations, confidence_interval = resample_leaveOO_linearRegression(
                        X, y, resampleTimes=resampleTimes)
                    if plotFig5:
                        ax.set_title(
                            f"{title} {kp_linear_(X, y)}")
                    else:
                        ax.set_title(
                            f"{title} R2={reg.score(X, y):.2f} "
                            f"corr={pearsonr(X.reshape(-1), y.reshape(-1))[0]:.2f} {kp_linear_(X, y)}")
                    if np.prod(confidence_interval) > 0:
                        significance = '*'
                    else:
                        significance = ''
                    if xAxisType == 'neuro':
                        ax.set_xlabel(f"neuro {behav_whichComponent}")
                        if plotFig5:
                            ax.set_ylabel(
                                f'{notes}')
                        else:
                            ax.set_ylabel(
                                f'{notes} {confidence_interval[0]:.2}~{confidence_interval[1]:.2}{significance}')
                        if _behav_whichComponent == 'XY_ses1-XY_ses5':
                            ax.set_xlim([-0.10, 0.10])
                            ax.set_ylim([-10, 17])
                        else:
                            ax.set_xlim([-0.13, 0.14])
                            ax.set_ylim([-20, 43])
                        from matplotlib.ticker import MultipleLocator
                        x_tick_interval = 0.05
                        ax.xaxis.set_major_locator(MultipleLocator(base=x_tick_interval))
                    else:
                        ax.set_ylabel(f"neuro {behav_whichComponent}")
                        if plotFig5:
                            ax.set_xlabel(
                                f'{notes}')
                        else:
                            ax.set_xlabel(
                                f'{notes} {confidence_interval[0]:.2}~{confidence_interval[1]:.2}{significance}')
                        if _behav_whichComponent == 'XY_ses1-XY_ses5':
                            ax.set_ylim([-0.10, 0.10])
                            ax.set_xlim([-10, 17])
                        else:
                            ax.set_ylim([-0.13, 0.14])
                            ax.set_xlim([-20, 43])
                        from matplotlib.ticker import MultipleLocator
                        y_tick_interval = 0.05
                        ax.yaxis.set_major_locator(MultipleLocator(base=y_tick_interval))
                    return reg.score(X, y)

                R2_acrossSes[interestedROI] = plotScatterAndLinearRegression(
                    X=neuro_integrationScore_component,
                    y=behavior_catPer_integrationScore_component,
                    title=f"{interestedROI}",
                    notes=f"behav {behav_whichComponent}{fixedCenterFlag}",
                    _behav_whichComponent=behav_whichComponent,
                    ax=axs_[curr_ax],
                    _plotColor=plotColor
                )
                curr_ax += 1
                # correlations, confidence_interval = resample_leaveOO_linearRegression(
                #     behavior_catPer_integrationScore, acrossSess_integrationScore)

        fig_.savefig(f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/"
                     f"plot_integrationScore_components_brain_behav_compare/fig6_{fixedCenterFlag}{batch}.pdf",
                     transparent=True)  # bbox_inches='tight', format='pdf',
        fig_.show()

    ROIDict_ = {
        # 'V1': 'V1_FreeSurfer',
        # 'V2': 'V2_FreeSurfer',
        # 'LO': 'LOC_FreeSurfer',
        # 'IT': 'IT_FreeSurfer',
        # 'FG': 'Fus_FreeSurfer',  # fusiform gyrus

        'HC_ASHS': 'lfseg_corr_usegray_hippocampus',
        'CA1': 'lfseg_corr_usegray_1',
        'PHC_FS': 'PHC_FreeSurfer',  # parahippocampal cortex

        # 'HC_FS': 'HC_FreeSurfer',
        # 'CA2/3': 'lfseg_corr_usegray_2',
        # 'DG': 'lfseg_corr_usegray_3',
        # 'ERC_ASHS': 'lfseg_corr_usegray_4',
        # 'PRC_ASHS': 'lfseg_corr_usegray_6',
        # 'PHC_ASHS': 'lfseg_corr_usegray_5',
        # 'SUB': 'lfseg_corr_usegray_7',
        # 'megaROI': 'megaROI'
    }
    # 'HC_FS': 'HC_FreeSurfer',
    # 'HC_ASHS': 'lfseg_corr_usegray_hippocampus',
    # 'CA1': 'lfseg_corr_usegray_1',
    # 'CA2/3': 'lfseg_corr_usegray_2',
    # 'DG': 'lfseg_corr_usegray_3',
    # 'ERC': 'lfseg_corr_usegray_4',
    # 'PRC': 'lfseg_corr_usegray_6',
    # 'SUB': 'lfseg_corr_usegray_7',
    compare_integration_conponent(ROIdict=ROIDict_)


plot_integrationScore_components_brain_behav_compare(batch=12, testMode=False, fixedCenter=True, plotFig5=True)

