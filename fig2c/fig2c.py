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
                          dotSize=0.5):  # practiceEffectLevel = withinSub withinSess withinRun # 获取 session 234 的 B probability，并且作图计算出斜率
        # if testMode:
        #     sub = 'sub003'
        #     plotFigure = True
        #     practiceEffectLevel = "withinSub"
        #     dotSize = 0.5
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

        def loadDrivingTargetProb(sub='sub003', testMode=False, testRun_training=None):
            subPath = f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/subjects/"

            # 加载保存的 session 2/3/4 的 B probability. 加载方法主要是对于每一个session的每一个run, 使用保存的分类器, 对于这个run的数据计算 B的概率并且保存.
            _feedbackTR_Yprobs = []  # len = 32 ; 是一个32长度的list, 每一个元素都是一个数字, 表示这个run中的所有的feedbackTR的Yprob的均值.
            _allTR_Yprobs = []  # len = 32; 是一个32长度的list, 每一个元素都是一个173长度的list. 因为有32个run, 每一个run有173个TR
            _Yprobs = []  # (32, 60) 形状的一个array,  因为有32个feedback run, 每一个run有 5/14*173≈61个feedbackTR, 实际上考虑到一些冗余的头尾TR, 只有60个feedbackTR.
            _XYprobs = []
            _minXYprobs = []
            _feedbackTR_trialIDs = []  # feedbackTR_trialIDs 是 (32, 60) 形状的一个array,  因为有32个feedback run, 每一个run有 5/14*173≈61个feedbackTR, 实际上考虑到一些冗余的头尾TR, 只有60个feedbackTR. 每一个元素表示当前的TR是当前run的第几个trial.
            _successful_trials = []  # 32 长度的一个list, 记录的是这32个run中, 每一个run的成功的trial的数量
            _perfect_trials = []  # 32 长度的一个list, 记录的是这32个run中, 每一个run的完美的trial的数量
            _whichSession = []

            _Yprob_df_ = pd.DataFrame()
            for curr_ses in range(2, 5):  # 对于指定被试的 2 3 4 session
                # 对于当前session的所有的 feedback run
                runRecording = pd.read_csv(f"/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/"
                                           f"subjects/{sub}/ses{curr_ses}/runRecording.csv")
                feedbackScanNum = list(
                    runRecording['run'].iloc[list(np.where(1 == 1 * (runRecording['type'] == 'feedback'))[0])])
                for _currRun in range(1, 1 + len(feedbackScanNum)):
                    if ROI == 'megaROI':
                        if _clfType == 'updatingClf':
                            mega_feedback_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/subjects/" \
                                                f"{sub}/ses{curr_ses}/feedback/"
                            history_fileName = f"{mega_feedback_dir}/history_runNum_{_currRun}.csv"
                        elif _clfType == 'ses1Clf':
                            history_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/XYcoactivation_acrossSes/" \
                                          f"{sub}/ses{curr_ses}/{ROI_name}/" \
                                          f"testRun_training{testRun_training}/simulate_history/"
                            history_fileName = history_dir + f"{sub}_{_currRun}_history_rtSynth_rt.csv"
                        else:
                            raise Exception("clfType must be updatingClf or ses1Clf")
                    else:
                        if _clfType == 'updatingClf':
                            history_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis/" \
                                          f"subjects/{sub}/ses{curr_ses}/{ROI_name}/rtSynth_rt_ABCD_ROIanalysis/"
                            history_fileName = f"{history_dir}/{sub}_{_currRun}_history_rtSynth_RT_ABCD.csv"
                        elif _clfType == 'ses1Clf':
                            history_dir = f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/autoAlign_ROIanalysis_ses1ses5/" \
                                          f"XYcoactivation/{sub}/ses{curr_ses}/{ROI_name}/" \
                                          f"testRun_training{testRun_training}/simulate_history/"

                            history_fileName = history_dir + f"{sub}_{_currRun}_history_rtSynth_rt.csv"
                        else:
                            raise Exception("clfType must be updatingClf or ses1Clf")
                    print(f"history_fileName={history_fileName}")
                    history = pd.read_csv(history_fileName)
                    history = assignTrialID(history)

                    def normXY(history):
                        # 之前的代码是没有进行每一个feedback run的归一化的，如果选择归一化, 那么就是用这个函数进行归一化.
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
                        # print("assert np.mean(history['XxY']) < 1e-5")
                        assert np.mean(history['XxY']) < 1e-5

                    allTR_Yprob = list(history['Yprob'])  # 考虑到两个batch的history文件中有B_prob的差别.

                    # 找到当前feedback run中 是 feedback TR的TR的Y概率.
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
                    if len(feedbackTR_Yprob) < 60:  # 如果当前 feedback run 的有效feedback TR的个数不足60 (正常情况应该是多少呢??), 就不保存在probs中
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
            # 在withinSub的时候是否需要每一个sess做一下标准化? 因为每一个session的clf实际上是不一样的?
            # 可能需要做的就是用两种做法, 第一种就是不标准化, 看看是否有练习效应
            # 第二种做法就是标准化, 看看是否不同的session间有练习效应, 我推测应该没有, 因为一旦标准化所有的sess都具有0的均值, 就人为去掉了总体的趋势.

            # 每一个被试进行一次线性回归, 每一个点是一个feedbackTR的Yprob, 横坐标是这个点所在的session. 最终的结果是判断是否有 withSub 层面的练习效应.
            y = yAxis_data.reshape(np.prod(yAxis_data.shape))
            X = []
            for currSess in whichSession:
                X += [currSess] * 60
            # print(f"y={y}")
            # print(f"X={X}")

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

            # 线性回归A, 经过测试发现 线性回归A和B都是一样的结果, 区别就是A可以提供更多的统计细节, 比如斜率是否显著不同于零.
            X2 = sm.add_constant(X)
            est = sm.OLS(y, X2)
            est2 = est.fit()
            if testMode:
                print(est2.summary())
                print("Ordinary least squares")

            # 线性回归B
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
            # 这个函数的目的是为了针对输入的数据，进行有重复的抽取5000次，然后记录每一次的均值，最后输出这5000次重采样的均值分布    的   均值和5%和95%的数值。
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

    fig.savefig(
        f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/temp/figures/XxY_vs_session.pdf",
        transparent=True)
    plt.show()


fig2c()
