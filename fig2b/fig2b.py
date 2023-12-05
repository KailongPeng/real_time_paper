import os
import sys
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


def fig2b():
    slopes = []
    subData = pd.DataFrame()
    for sub in tqdm(subjects):
        data = pd.read_csv(
            f"{workingDir}/data/"
            f"subjects/{sub}/adaptiveThreshold_{sub}.csv")
        y = data['threshold']
        X = data['session']

        X_mean = [2, 3, 4]
        y_mean = [
            data[data['session'] == 2]['threshold'].mean(),
            data[data['session'] == 3]['threshold'].mean(),
            data[data['session'] == 4]['threshold'].mean()
        ]

        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()

        print(est2.summary())
        print("Ordinary least squares")

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.asarray(X), y)
        slopes.append(slope)
        subData_temp = pd.DataFrame({
            "sub": sub,
            "slope": slope,
            'session': X_mean,
            'threshold': y_mean
        })
        print(f"subData_temp={subData_temp}")
        subData = pd.concat([subData, subData_temp], ignore_index=True)
    _mean, _5, _95, p_value = cal_resample(data=np.asarray(slopes), times=5000, returnPvalue=True)
    print(
        f"resample mean={_mean} 5%={_5} 95%={_95} p_value={p_value}")

    def plot_fig(sub_data, fig=None, ax=None):
        """
        Plot the threshold as y-axis vs session as x-axis as a grey half transparent line.
        Each dot represents one session. For the mean of all subs, plot the threshold as y-axis
        vs session as x-axis as a red line. Each dot represents one session.
        """

        # Plot individual sub data
        for sub in sub_data["sub"].unique():
            sub_data_sub = sub_data[sub_data["sub"] == sub]
            ax.plot(sub_data_sub["session"], sub_data_sub["threshold"], alpha=0.1, label=sub, linestyle='-',
                    color="black")

        ax.set_title("Threshold vs Session", fontsize=16)

        plt.xticks([2, 3, 4], fontsize=12)  # Set the x-axis ticks to show only 2, 3, and 4
        plt.yticks(fontsize=12)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_fig(subData, fig=fig, ax=ax)

    ax = sns.regplot(x='session', y='threshold', data=subData, ax=ax, color="#AA4AC8", marker='.', scatter=False)

    ax.set_xlabel("Session", fontsize=14)
    ax.set_ylabel("Threshold", fontsize=14)
    ax.set_xlim(1.5, 4.5)

    from utils import mkdir
    mkdir(f"{workingDir}/data/result/temp/figures/")
    fig.savefig(
        f"{workingDir}/data/result/temp/figures/morph_param_threshold_vs_session.pdf",
        transparent=True)
    plt.show()


fig2b()
