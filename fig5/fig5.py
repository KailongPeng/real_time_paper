import os
import sys
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
from utils import get_subjects, get_ROIList, kp_run, getjobID_num, waitForEnd, check_jobIDs, mkdir

batch = 12  # meaning both batch 1 and batch 2
subjects, scan_asTemplates = get_subjects(batch=batch)


def plot_integrationScore_components_brain_behav_compare(batch=None, testMode=None, fixedCenter=None, plotFig5=None):
    subjects, scan_asTemplates = get_subjects(batch=batch)
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
                f'{workingDir}/data/result/analysisResults/allSubs/catPer/Behav_differentiations_fixedCenter.csv')
        else:
            raise Exception("not defined")
        __Behav_differentiations.rename(columns=lambda x: x.replace('_acc', '_slope'), inplace=True)
        __Behav_differentiations = __Behav_differentiations.drop(columns=['Unnamed: 0'])

        if len(dropCertainSub) > 0:
            for sub in dropCertainSub:
                __Behav_differentiations = __Behav_differentiations[__Behav_differentiations['sub'] != sub]

        for interestedROI in tqdm(ROIList):
            __Behav_differentiations[f"acrossSes-{interestedROI}"] = None

        for interestedROI in tqdm(ROIList):
            def load_acrossSessionEffect(interestedROI=None, _Behav_differentiations=None):
                # acrossSessionEffect
                for sub in subjects:
                    [ses1_XY, ses5_XY, ses1_MN, ses5_MN, differentiation_ratio, integration_ratio] = np.load(
                        f"{workingDir}/data/"
                        f"result/subjects/{sub}/ses5/{interestedROI}/integration_ratio_allData.npy")

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
        R2_acrossSes = {}

        resolution = 5
        fig_, axs_ = plt.subplots(num_rows, num_cols,
                                  figsize=(num_cols * resolution, num_rows * resolution))
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
                    behavior_catPer_integrationScore_component = XY_ses1_behav - XY_ses5_behav
                elif behav_whichComponent == 'MN_ses1-MN_ses5':
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
                from scipy.stats import pearsonr

                print(f"{interestedROI}-pearsonr(behavior_catPer_integrationScore, acrossSess_integrationScore)="
                      f"{pearsonr(behavior_catPer_integrationScore_component, neuro_integrationScore_component)}")

                corr_t_acrossSes[f"{interestedROI}-{behav_whichComponent}"] = \
                    pearsonr(neuro_integrationScore_component, behavior_catPer_integrationScore_component)[0]
                corr_p_acrossSes[f"{interestedROI}-{behav_whichComponent}"] = \
                    pearsonr(neuro_integrationScore_component, behavior_catPer_integrationScore_component)[1]

                def resample_leaveOO_linearRegression(X, y, resampleTimes=1000):
                    from sklearn.linear_model import LinearRegression
                    from sklearn.model_selection import LeaveOneOut
                    loo = LeaveOneOut()

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

                        correlation = np.corrcoef(np.asarray(y_preds), np.asarray(y_reals))[0, 1]
                        correlations.append(correlation)
                    confidence_interval = np.percentile(correlations, [5, 95])

                    return correlations, confidence_interval

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
        scratchFolder = f"{workingDir}/data/result/analysisResults/plot_integrationScore_components_brain_behav_compare/"
        mkdir(scratchFolder)
        fig_.savefig(f"{scratchFolder}/fig6_{fixedCenterFlag}{batch}.pdf",
                     transparent=True)  # bbox_inches='tight', format='pdf',
        fig_.show()

    ROIDict_ = {
        'HC_ASHS': 'lfseg_corr_usegray_hippocampus',
        'CA1': 'lfseg_corr_usegray_1',
        'PHC_FS': 'PHC_FreeSurfer',  # parahippocampal cortex
    }

    compare_integration_conponent(ROIdict=ROIDict_)


plot_integrationScore_components_brain_behav_compare(batch=12, testMode=False, fixedCenter=True, plotFig5=True)

