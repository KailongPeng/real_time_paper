import os
import sys
os.chdir("/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/real_time_paper/")
assert os.getcwd().endswith('real_time_paper'), "working dir should be 'real_time_paper'"
workingDir = os.getcwd()
sys.path.append('.')
# print current dir
print(f"getcwd = {os.getcwd()}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from utils import get_subjects

batch = 12  # meaning both batch 1 and batch 2
subjects, scan_asTemplates = get_subjects(batch=batch)
testMode = False


def cal_resample(data=None, times=5000, returnPvalue=False):
    if data is None:
        raise Exception
    iter_mean = []
    np.random.seed(123)
    for _ in range(times):
        iter_distri = data[np.random.choice(len(data), len(data), replace=True)]
        iter_mean.append(np.nanmean(iter_distri))
    _mean = np.mean(iter_mean)
    _5 = np.percentile(iter_mean, 5)
    _95 = np.percentile(iter_mean, 95)
    pValue = 1 - np.sum(np.asarray(iter_mean) > 0) / len(iter_mean)
    print(f"pValue={pValue}")
    if returnPvalue:
        return _mean, _5, _95, pValue
    else:
        return _mean, _5, _95


def fig2c():
    [ROI, ROI_name, _clfType, _normActivationFlag, _UsedTRflag, xAxisMethod] = [
        'megaROI', 'megaROI', 'updatingClf', True, 'feedback', 'X_times_Y']

    def feedback_analysis(sub='sub006', plotFigure=False,
                          practiceEffectLevel="withinSub",
                          testMode=False,
                          dotSize=0.5):  # practiceEffectLevel = withinSub withinSess withinRun
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import pandas as pd
        import statsmodels.api as sm
        from scipy import stats
        import os

        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        def gaussian_fit(y=None, x=1):
            if y is None:
                y = [1, 2, 3, 4]
            y_values = np.linspace(min(y), max(y), 120)
            mu = np.mean(y)
            sig = np.std(y)
            # plt.text(x+0.04, mu, 'mu={0:.2f}\nsig={1:.2f}'.format(mu,sig), fontsize=12)
            plt.plot(x + 0.04 + 0.5 * gaussian(y_values, mu, sig), y_values)

        def assignTrialID(history):
            TrialsIDs = []
            currTrial = 0
            oldState = 'ITI'
            for state in history['states']:
                if oldState == 'ITI' and state == 'waiting':
                    currTrial += 1
                oldState = state
                TrialsIDs.append(currTrial)

            history['TrialID'] = TrialsIDs
            return history

        def loadDrivingTargetProb(sub=None, testMode=False, testRun_training=None):
            _feedbackTR_Yprobs = []
            _allTR_Yprobs = []
            _Yprobs = []
            _XYprobs = []
            _minXYprobs = []
            _feedbackTR_trialIDs = []
            _successful_trials = []
            _perfect_trials = []
            _whichSession = []

            _Yprob_df_ = pd.DataFrame()
            for curr_ses in range(2, 5):

                runRecording = pd.read_csv(f"{workingDir}/data/"
                                           f"subjects/{sub}/ses{curr_ses}/runRecording.csv")
                feedbackScanNum = list(
                    runRecording['run'].iloc[list(np.where(1 == 1 * (runRecording['type'] == 'feedback'))[0])])
                for _currRun in range(1, 1 + len(feedbackScanNum)):
                    # if ROI == 'megaROI':
                    #     if _clfType == 'updatingClf':
                    # mega_feedback_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/subjects/" \
                    #                     f"{sub}/ses{curr_ses}/feedback/"
                    megaROI_subSes_folder = (f"{workingDir}"
                                             f"data/result/megaROI_main/subjects/{sub}/ses{curr_ses}/{ROI}/")
                    history_fileName = f"{megaROI_subSes_folder}/feedback/history_runNum_{_currRun}.csv"
                    #     elif _clfType == 'ses1Clf':
                    #         history_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/XYcoactivation_acrossSes/" \
                    #                       f"{sub}/ses{curr_ses}/{ROI_name}/" \
                    #                       f"testRun_training{testRun_training}/simulate_history/"
                    #         history_fileName = history_dir + f"{sub}_{_currRun}_history_rtSynth_rt.csv"
                    #     else:
                    #         raise Exception("clfType must be updatingClf or ses1Clf")
                    # else:
                    #     if _clfType == 'updatingClf':
                    #         history_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis/" \
                    #                       f"subjects/{sub}/ses{curr_ses}/{ROI_name}/rtSynth_rt_ABCD_ROIanalysis/"
                    #         history_fileName = f"{history_dir}/{sub}_{_currRun}_history_rtSynth_RT_ABCD.csv"
                    #     elif _clfType == 'ses1Clf':
                    #         history_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis_ses1ses5/" \
                    #                       f"XYcoactivation/{sub}/ses{curr_ses}/{ROI_name}/" \
                    #                       f"testRun_training{testRun_training}/simulate_history/"
                    #
                    #         history_fileName = history_dir + f"{sub}_{_currRun}_history_rtSynth_rt.csv"
                    #     else:
                    #         raise Exception("clfType must be updatingClf or ses1Clf")
                    print(f"history_fileName={history_fileName}")
                    history = pd.read_csv(history_fileName)
                    history = assignTrialID(history)

                    def normXY(history):
                        __Xprob = history['Xprob']
                        __Yprob = history['Yprob']
                        XYprob = history['XxY']
                        minXYprob = history['min(Xprob, Yprob)']
                        __Xprob = (__Xprob - np.mean(__Xprob)) / np.std(__Xprob)
                        __Yprob = (__Yprob - np.mean(__Yprob)) / np.std(__Yprob)
                        XYprob = (XYprob - np.mean(XYprob)) / np.std(XYprob)
                        minXYprob = (minXYprob - np.mean(minXYprob)) / np.std(minXYprob)
                        history['Xprob'] = __Xprob
                        history['Yprob'] = __Yprob
                        history['XxY'] = XYprob
                        history['min(Xprob, Yprob)'] = minXYprob

                        return history

                    if _normActivationFlag:
                        print(f"normActivationFlag={_normActivationFlag}, doing normalization")
                        history = normXY(history)
                        assert np.mean(history['XxY']) < 1e-5

                    allTR_Yprob = list(history['Yprob'])

                    feedbackTR_Yprob = list(history[history['states'] == "feedback"]['Yprob'])
                    feedbackTR_XYprob = list(history[history['states'] == "feedback"]['XxY'])
                    feedbackTR_minXYprob = list(history[history['states'] == "feedback"]['min(Xprob, Yprob)'])
                    if testMode:
                        print(f"number of feedbackTR in this run = {len(feedbackTR_Yprob)}")
                    feedbackTR_trialID = list(history[history['states'] == "feedback"]['TrialID'])
                    _Yprob_df_ = pd.concat([_Yprob_df_, pd.DataFrame({
                        'subject': sub,
                        'session': curr_ses,
                        'currRun': _currRun,
                        'feedbackTR_YprobMean': np.mean(feedbackTR_Yprob),
                        'allTR_YprobMean': np.mean(allTR_Yprob)
                    }, index=[0])], ignore_index=True)
                    if len(feedbackTR_Yprob) < 60:
                        print(f"-------------------------\n error run sub {sub} ses {curr_ses} {_currRun}\n\n")
                        if sub == 'sub014' and curr_ses == 4 and _currRun == 7:  # this run used the recognition sequence which is 1min shorter
                            pass
                        else:
                            continue
                        # feedbackTR_Yprob = feedbackTR_Yprob + [np.nan] * (60 - len(feedbackTR_Yprob))

                    if sub == 'sub014' and curr_ses == 4 and _currRun == 7:  # this run used the recognition sequence which is 1min shorter
                        print(
                            f"still using data:  {sub} ses {curr_ses} {_currRun} : this run used the recognition sequence which is 1min shorter\n\n")

                    _allTR_Yprobs.append(allTR_Yprob)

                    if len(_Yprobs) == 0:
                        _Yprobs = np.expand_dims(feedbackTR_Yprob, 0)
                        _XYprobs = np.expand_dims(feedbackTR_XYprob, 0)
                        _minXYprobs = np.expand_dims(feedbackTR_minXYprob, 0)
                        _feedbackTR_trialIDs = np.expand_dims(feedbackTR_trialID, 0)
                    else:
                        _Yprobs = np.concatenate([_Yprobs, np.expand_dims(feedbackTR_Yprob, 0)], axis=0)
                        _XYprobs = np.concatenate([_XYprobs, np.expand_dims(feedbackTR_XYprob, 0)], axis=0)
                        _minXYprobs = np.concatenate([_minXYprobs, np.expand_dims(feedbackTR_minXYprob, 0)], axis=0)
                        _feedbackTR_trialIDs = np.concatenate([_feedbackTR_trialIDs,
                                                               np.expand_dims(feedbackTR_trialID, 0)], axis=0)
                    _whichSession.append(curr_ses)
                    if testMode:
                        print(f"ses {curr_ses} {_currRun}")
            return _feedbackTR_Yprobs, _allTR_Yprobs, [_Yprobs, _XYprobs,
                                                       _minXYprobs], _feedbackTR_trialIDs, _successful_trials, _perfect_trials, _Yprob_df_, _whichSession

        if _clfType == 'updatingClf':
            _, _, [Yprobs, XYprobs, minXYprobs], feedbackTR_trialIDs, _, _, _, whichSession = loadDrivingTargetProb(
                sub=sub, testMode=testMode)
        else:
            raise Exception

        if xAxisMethod == 'Y':  # , 'X_times_Y', 'min_X_Y']:
            yAxis_data = Yprobs
        elif xAxisMethod == 'X_times_Y':
            yAxis_data = XYprobs
        elif xAxisMethod == 'min_X_Y':
            yAxis_data = minXYprobs
        else:
            raise Exception

        slopes = []
        subData = pd.DataFrame()
        if practiceEffectLevel == "withinSub":
            y = yAxis_data.reshape(np.prod(yAxis_data.shape))
            X = []
            for currSess in whichSession:
                X += [currSess] * 60

            def get_y_mean(_X, _y):
                _X = np.asarray(_X)
                _y = np.asarray(_y)
                y_mean = [
                    np.mean(_y[_X == 2]),
                    np.mean(_y[_X == 3]),
                    np.mean(_y[_X == 4])]
                X_mean = [2, 3, 4]
                print(f"y_mean={y_mean}")
                print(f"X_mean={X_mean}")
                return X_mean, y_mean

            X_mean, y_mean = get_y_mean(X, y)

            X2 = sm.add_constant(X)
            est = sm.OLS(y, X2)
            est2 = est.fit()
            if testMode:
                print(est2.summary())
                print("Ordinary least squares")

            # slope, intercept, r_value, p_value, std_err = stats.linregress(X2,y)
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.asarray(X), y)

            if plotFigure:
                resoluitionTimes = 2
                fig, ax = plt.subplots(figsize=(15 * resoluitionTimes, 15 * resoluitionTimes))
                ax.set_facecolor((242 / 256, 242 / 256, 242 / 256))
                plt.scatter(X, y, s=dotSize * resoluitionTimes)
                fittedLineX = np.arange(np.min(X), np.max(X), 0.01)
                plt.plot(fittedLineX, slope * fittedLineX + intercept, 'r', linewidth=4 * resoluitionTimes)
                ax.spines[['right', 'top']].set_visible(False)
                print(f"slope={slope}")
                ax.set_yticks([])
                ax.set_xticks([])

            if testMode:
                print(f"slope={slope}, intercept={intercept}")
                print(f"r-squared: {r_value ** 2}")  # To get coefficient of determination (r_squared)
            slopes.append(slope)
            subData_temp = pd.DataFrame({
                "sub": sub,
                "slope": slope,
                # "intercept":intercept,
                'session': X_mean,
                'XxY': y_mean
            })
            print(f"subData_temp={subData_temp}")
            subData = pd.concat([subData, subData_temp], ignore_index=True)
        else:
            raise Exception
        return slopes, subData

    def praticeEffect_practiceEffectLevel(practiceEffectLevel="practiceEffectLevel", plotFigure=False):
        slopes_container = []
        subData_container = pd.DataFrame()
        for _sub_ in tqdm(subjects):
            slopes, subData = feedback_analysis(sub=_sub_, plotFigure=False, practiceEffectLevel=practiceEffectLevel,
                                       testMode=testMode)  # withinSub withinSess withinRun
            slopes_container = slopes_container + slopes
            subData_container = pd.concat([subData_container, subData], ignore_index=True)

        def cal_resample(data=None, times=5000, returnPvalue=False):
            if data is None:
                raise Exception
            iter_mean = []
            np.random.seed(123)
            for _ in range(times):
                iter_distri = data[np.random.choice(len(data), len(data), replace=True)]
                iter_mean.append(np.nanmean(iter_distri))
            _mean = np.mean(iter_mean)
            _5 = np.percentile(iter_mean, 5)
            _95 = np.percentile(iter_mean, 95)
            pValue = 1 - np.sum(np.asarray(iter_mean) > 0) / len(iter_mean)
            print(f"pValue={pValue}")
            _pValue = 1 - np.mean(np.asarray(iter_mean) > 0)
            print(f"_pValue={_pValue}")
            assert pValue == _pValue
            if returnPvalue:
                return _mean, _5, _95, pValue
            else:
                return _mean, _5, _95
        _mean, _5, _95, p_value = cal_resample(data=np.asarray(slopes_container), times=5000, returnPvalue=True)
        print(f"practiceEffectLevel={practiceEffectLevel}, resample mean={_mean} 5%={_5} 95%={_95} p_value={p_value}")
        # practiceEffectLevel=withinSub, resample mean=0.04208769650586326 5%=0.007142798332258992 95%=0.07947392568214129 p_value=0.022800000000000042
        if plotFigure:
            _ = plt.hist(slopes_container, bins=100)
            _ = plt.title(f"{practiceEffectLevel}, resample mean={_mean} 5%={_5} 95%={_95}")
        return _mean, _5, _95, subData_container

    _mean, _5, _95, subData_container = praticeEffect_practiceEffectLevel(practiceEffectLevel="withinSub", plotFigure=False)

    def plot_fig(sub_data, fig=None, ax=None):
        """
        Plot the threshold as y-axis vs session as x-axis as a grey half transparent line.
        Each dot represents one session. For the mean of all subs, plot the threshold as y-axis
        vs session as x-axis as a red line. Each dot represents one session.
        """

        # Plot individual sub data
        for sub in sub_data["sub"].unique():
            sub_data_sub = sub_data[sub_data["sub"] == sub]
            ax.plot(sub_data_sub["session"], sub_data_sub["XxY"], alpha=0.1, label=sub, linestyle='-',
                    # marker=".",
                    color="black")

        ax.set_title("XxY vs Session", fontsize=16)

        plt.xticks([2, 3, 4], fontsize=12)  # Set the x-axis ticks to show only 2, 3, and 4
        plt.yticks(fontsize=12)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_fig(subData_container, fig=fig, ax=ax)

    import seaborn as sns

    ax = sns.regplot(x='session', y='XxY', data=subData_container, ax=ax, color="#AA4AC8", marker='.', scatter=False)

    ax.set_xlabel("Session", fontsize=14)
    ax.set_ylabel("XxY", fontsize=14)
    ax.set_xlim(1.5, 4.5)
    from utils import mkdir
    mkdir(f"{workingDir}/data/result/temp/figures/")
    fig.savefig(
        f"{workingDir}/data/result/temp/figures/XxY_vs_session.pdf",
        transparent=True)
    plt.show()


fig2c()
