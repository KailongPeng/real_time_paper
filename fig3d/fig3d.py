# fig3d: plot_integrationScore_components -> behav (updated)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_subjects


def plot_integrationScore_components(batch=None, fixedCenter=None, plot5Percentile=None):
    # fixedCenter : OrganizedScripts/catPer/catPer_fixedCenter.py
    scratchFolder = "/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/plot_integrationScore_components/"
    subjects, scan_asTemplates = get_subjects(batch=batch)

    def plot_components(XY_ses5, XY_ses1, MN_ses5, MN_ses1, title="", saveFigPath=None, length=3.33, height=3.33 * 3):
        XY_ses5 = np.asarray(XY_ses5)
        XY_ses1 = np.asarray(XY_ses1)
        MN_ses5 = np.asarray(MN_ses5)
        MN_ses1 = np.asarray(MN_ses1)

        def plot_ttest_bar(allYdata, ax=None, ylabel=''):
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy import stats

            p_values = {}
            means = {}
            stds = {}
            _5s = {}
            _95s = {}
            _2_5s = {}
            _97_5s = {}
            resample_P_values = {}

            def cal_resample(data=None, times=5000, return_all=False):
                if data is None:
                    raise Exception
                if type(data) == list:
                    data = np.asarray(data)
                iter_mean = []

                # Set the random seed
                np.random.seed(123)  # You can use any integer value as the seed

                for _ in range(times):
                    iter_distri = data[np.random.choice(len(data), len(data), replace=True)]
                    iter_mean.append(np.nanmean(iter_distri))
                _mean = np.mean(iter_mean)
                _5 = np.percentile(iter_mean, 5)
                _95 = np.percentile(iter_mean, 95)

                _2_5 = np.percentile(iter_mean, 2.5)
                _97_5 = np.percentile(iter_mean, 97.5)

                P_value = np.mean(np.asarray(iter_mean) <= 0)

                if return_all:
                    return _mean, _5, _95, _2_5, _97_5, P_value, iter_mean
                else:
                    return _mean, _5, _95, _2_5, _97_5, P_value

            # Perform t-test and calculate mean and standard deviation for each key
            for key, values in allYdata.items():
                t_statistic, p_value = stats.ttest_1samp(values, 0)
                mean = np.mean(values)
                std = np.std(values)
                p_values[key] = p_value
                # means[key] = mean
                stds[key] = std

                _mean, _5, _95, _2_5, _97_5, resample_P_value = cal_resample(data=values,
                                                                             times=5000,
                                                                             return_all=False)
                means[key] = _mean
                _5s[key] = _5
                _95s[key] = _95
                _2_5s[key] = _2_5
                _97_5s[key] = _97_5
                resample_P_values[key] = resample_P_value

            # Extract keys, corresponding means, stds, and p-values for plotting
            keys = list(means.keys())
            mean_values = list(means.values())
            std_values = list(stds.values())
            _5_values = list(_5s.values())
            _95_values = list(_95s.values())
            _2_5_values = list(_2_5s.values())
            _97_5_values = list(_97_5s.values())
            p_values_list = list(p_values.values())
            resample_P_values_list = list(resample_P_values.values())

            # Define the colors for the bars
            bar_colors = ['#AA4AC8', '#BBBDBF', '#BBBDBF', '#BBBDBF'][:len(allYdata)]

            # Plot the bars with error bars
            if plot5Percentile:
                bars = ax.bar(keys, mean_values,
                              yerr=[np.array(mean_values) - np.array(_5_values),
                                    np.array(_95_values) - np.array(mean_values)],
                              align='center', alpha=0.7, capsize=5, color=bar_colors)
            else:
                bars = ax.bar(keys, mean_values,
                              yerr=[np.array(mean_values) - np.array(_2_5_values),
                                    np.array(_97_5_values) - np.array(mean_values)],
                              align='center', alpha=0.7, capsize=5, color=bar_colors)

            # Scatter plot the data on top of each bar
            for i, values in enumerate(allYdata.values()):
                x_coords = np.random.normal(bars[i].get_x() + bars[i].get_width() / 2,
                                            0.05,
                                            len(values))
                ax.scatter(x_coords, values, s=50, alpha=0.5, label=f'{keys[i]} data', color=bar_colors[i],
                           edgecolor='black', linewidth=length / 3.33)
                # # Plot black outline circles
                # ax.scatter(x_coords, values, s=60, alpha=1, edgecolor='black', linewidth=1, label=f'{keys[i]} data',
                #            facecolors='none')
                # # Plot actual data points
                # ax.scatter(x_coords, values, s=50, alpha=0.5, label=f'{keys[i]} data', color=bar_colors[i])

            # Add "*" marker if p-value is significant
            for i, p_value in enumerate(p_values_list):
                ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                        _95_values[i] * 1.1,
                        f'P={resample_P_values_list[i]:.3f}',
                        # t p={p_value:.3f}\nr P={resample_P_values_list[i]:.3f}
                        ha='center',
                        va='bottom')

            ax.set_ylabel(ylabel)
            if plot5Percentile:
                ax.set_title(f"{title}")
            else:
                ax.set_title(f"{title} 97.5% CI")

            # Set x-axis tick labels rotation to 45 degrees
            ax.set_xticklabels(keys, rotation=45)

            # Set the y-axis tick interval to 20
            from matplotlib.ticker import MultipleLocator
            y_tick_interval = 20
            ax.yaxis.set_major_locator(MultipleLocator(base=y_tick_interval))
            #
            # # Increase font size for labels and text
            # plt.rcParams.update({'font.size': 14})
            #
            # plt.tight_layout()
            # plt.show()

        num_rows = 1
        num_cols = 5
        # length = 3.33
        # height = 3.33 * 3
        fig, axs = plt.subplots(num_rows, num_cols,
                                figsize=(num_cols * length, num_rows * height))
        axs = axs.ravel()
        data = {
            'XY_ses1': XY_ses1,
            'XY_ses5': XY_ses5,
            'MN_ses1': MN_ses1,
            'MN_ses5': MN_ses5,
        }
        plot_ttest_bar(data, ax=axs[0], ylabel='accuracy or slope')
        data = {
            'XY_ses1-XY_ses5': XY_ses1 - XY_ses5,
            'MN_ses1-MN_ses5': MN_ses1 - MN_ses5,
        }
        plot_ttest_bar(data, ax=axs[1], ylabel='')

        data = {
            '(XY_ses1-5)-(MN_ses1-5)': (XY_ses1 - XY_ses5) - (MN_ses1 - MN_ses5)
        }
        plot_ttest_bar(data, ax=axs[2], ylabel='')

        data = {
            '(XY_ses1-XY_ses5)/+': (XY_ses1 - XY_ses5) / (XY_ses1 + XY_ses5),
            '(MN_ses1-MN_ses5)/+': (MN_ses1 - MN_ses5) / (MN_ses1 + MN_ses5),
        }
        plot_ttest_bar(data, ax=axs[3], ylabel='normalized integration')

        data = {
            '(XY_ses1-5)/(XY_ses1+5)-control': ((XY_ses1 - XY_ses5) / (XY_ses1 + XY_ses5)) - (
                    (MN_ses1 - MN_ses5) / (MN_ses1 + MN_ses5))
        }
        plot_ttest_bar(data, ax=axs[4],
                       ylabel='((XY_ses1-XY_ses5)/(XY_ses1+XY_ses5))-((MN_ses1-MN_ses5)/(MN_ses1+MN_ses5))')

        fig.savefig(saveFigPath, bbox_inches='tight', format='png')
        print(f"figure saved at {saveFigPath}")
        fig.savefig(saveFigPath.replace('.png', '.pdf'), format='pdf', transparent=True)
        print(f"figure saved at {saveFigPath.replace('.png', '.pdf')}")
        plt.show()

    def behav():
        """
            data from OrganizedScripts/catPer/catPer.py
        """
        if fixedCenter:
            allResults = pd.read_csv(
                '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/analysisResults/'
                'allSubs/catPer/Behav_differentiations_fixedCenter.csv')
            plotTitle = 'behav fixedCenter'
        else:
            allResults = pd.read_csv(
                '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/OrganizedScripts/analysisResults/'
                'allSubs/catPer/Behav_differentiations.csv')
            plotTitle = 'behav'

        plot_components(XY_ses5=allResults['ses5_XY_acc'],
                        XY_ses1=allResults['ses1_XY_acc'],
                        MN_ses5=allResults['ses5_MN_acc'],
                        MN_ses1=allResults['ses1_MN_acc'],
                        title=plotTitle,
                        saveFigPath=f"{scratchFolder}/behav.png",
                        length=3.33 * 3 / 2, height=3.33 * 3 / 2
                        )

    behav()


plot_integrationScore_components(batch=12, fixedCenter=True, plot5Percentile=True)
