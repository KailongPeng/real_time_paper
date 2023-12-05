# fig4b: plot_integrationScore_components -> neuro (updated)
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import get_subjects


def plot_integrationScore_components(batch=None, fixedCenter=None, plot5Percentile=None):
    # fixedCenter : OrganizedScripts/catPer/catPer_fixedCenter.py
    scratchFolder = ("/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/"
                     "real_time_paper/data/result/plot_integrationScore_components/")
    if not os.path.exists(scratchFolder):
        os.mkdir(scratchFolder)
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

    def neuro(length=3.33, height=3.33 * 3):
        """
            data prepare code:
                OrganizedScripts/ROI/ROI_ses1ses5_autoAlign.py ->
                    OrganizedScripts/ROI/autoAlign_ses1ses5/integrationScore/integrationScore.py
                    (sbatch --array=1-1 ./OrganizedScripts/ROI/ROI_ses1ses5_autoAlign.sh 12 Linear) 25324548
                OrganizedScripts/megaROI/acrossSession/megaROI_acrossSess.py ->
                    OrganizedScripts/megaROI/acrossSession/integrationScore/integrationScore.py
                    (sbatch --array=1-1 ./OrganizedScripts/megaROI/acrossSession/megaROI_acrossSess.sh 12 Linear) 25325129
        """

        def dataPreparation(interestedROI=''):
            allResults = pd.DataFrame()
            for sub in tqdm(subjects):
                # if interestedROI == 'megaROI':
                #     [ses1_XY, ses5_XY, ses1_MN, ses5_MN, differentiation_ratio, integration_ratio] = np.load(
                #         f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/megaROI_main/subjects/"
                #         f"{sub}/ses5/{interestedROI}/integrationScore_allData.npy")
                # else:
                # np.save(
                #     f"/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/real_time_paper/data/"
                #     f"result/subjects/{sub}/ses{ses}/{chosenMask}/integration_ratio_allData.npy",
                #     [ses1_XY, ses5_XY, ses1_MN, ses5_MN, differentiation_ratio, integration_ratio])

                [ses1_XY, ses5_XY, ses1_MN, ses5_MN, differentiation_ratio, integration_ratio] = np.load(
                    f"/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/real_time_paper/data/"
                    f"result/subjects/{sub}/ses5/{interestedROI}/integration_ratio_allData.npy")

                # f"/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/"
                #                         f"autoAlign_ROIanalysis_ses1ses5/subjects/{sub}/ses5/{interestedROI}/"
                #                         f"integrationScore_allData.npy"
                allResults = pd.concat([allResults, pd.DataFrame({
                    'sub': sub,

                    'integration_ratios': integration_ratio,
                    'differentiation_ratios': differentiation_ratio,

                    'ses1_XY': ses1_XY,
                    'ses5_XY': ses5_XY,
                    'ses1_MN': ses1_MN,
                    'ses5_MN': ses5_MN,
                }, index=[0])], ignore_index=True)
            return allResults

        for curr_ROIdict, ROIdict in enumerate([
            {
                'V1': 'V1_FreeSurfer',
                'V2': 'V2_FreeSurfer',
                'LO': 'LOC_FreeSurfer',
                'IT': 'IT_FreeSurfer',
                'FG': 'Fus_FreeSurfer',  # fusiform gyrus
                'PHC_FS': 'PHC_FreeSurfer',  # parahippocampal cortex
            },
            {
                'HC_ASHS': 'lfseg_corr_usegray_hippocampus',
                'CA1': 'lfseg_corr_usegray_1',
                'CA2/3': 'lfseg_corr_usegray_2',
                'DG': 'lfseg_corr_usegray_3',
                'SUB': 'lfseg_corr_usegray_7',
                'ERC_ASHS': 'lfseg_corr_usegray_4',
                'PHC_ASHS': 'lfseg_corr_usegray_5',
                'PRC_ASHS': 'lfseg_corr_usegray_6',
                # 'megaROI': 'megaROI'
            }]):
            # hippoSubfieldID = {
            #         1: "CA1",
            #         2: "CA2+3",
            #         3: "DG",  # dentate gyrus
            #         4: "ERC",  # Entorhinal cortex
            #         5: "PHC",  # parahippocampal cortex
            #         6: "PRC",  # Perirhinal Cortex
            #         7: "SUB"  # subiculum
            #     }  # hippocampus: CA1+CA2+CA3+DG+SUB
            #     # hippocampus: 1+2+3+7   CA1+CA2+CA3+DG+SUB
            #     # MTL (Medial Temporal Lobe): 4+5+6

            _ylim = [[-0.14, 0.17], [-0.13, 0.11]][curr_ROIdict]
            num_rows = 1
            num_cols = len(ROIdict)
            fig, axs = plt.subplots(num_rows, num_cols,
                                    figsize=(num_cols * length, num_rows * height))
            axs = axs.ravel()
            for currROI, ROI in enumerate(ROIdict):
                allResults = dataPreparation(interestedROI=ROIdict[ROI])

                def plot_ROI_components(XY_ses5, XY_ses1, MN_ses5, MN_ses1, title="", saveFigPath=None, length=3.33,
                                        height=3.33 * 3, ax=None):
                    XY_ses5 = np.asarray(XY_ses5)
                    XY_ses1 = np.asarray(XY_ses1)
                    MN_ses5 = np.asarray(MN_ses5)
                    MN_ses1 = np.asarray(MN_ses1)

                    def plot_ttest_bar(allYdata, __ax=None, ylabel=''):
                        # 希望对于每一个ROI的integration score进行 t test.
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
                            # 这个函数的目的是为了针对输入的数据，进行有重复的抽取5000次，然后记录每一次的均值，最后输出这5000次重采样的均值分布    的   均值和5%和95%的数值。
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
                            bars = __ax.bar(keys, mean_values,
                                            yerr=[np.array(mean_values) - np.array(_5_values),
                                                  np.array(_95_values) - np.array(mean_values)],
                                            align='center', alpha=0.7, capsize=5, color=bar_colors)
                        else:
                            bars = __ax.bar(keys, mean_values,
                                            yerr=[np.array(mean_values) - np.array(_2_5_values),
                                                  np.array(_97_5_values) - np.array(mean_values)],
                                            align='center', alpha=0.7, capsize=5, color=bar_colors)

                        # Scatter plot the data on top of each bar
                        for i, values in enumerate(allYdata.values()):
                            x_coords = np.random.normal(bars[i].get_x() + bars[i].get_width() / 2,
                                                        0.05,
                                                        len(values))
                            __ax.scatter(x_coords, values, s=50, alpha=0.5, label=f'{keys[i]} data',
                                         color=bar_colors[i],
                                         edgecolor='black', linewidth=length / 3.33)
                            # # Plot black outline circles
                            # ax.scatter(x_coords, values, s=60, alpha=1, edgecolor='black', linewidth=1, label=f'{keys[i]} data',
                            #            facecolors='none')
                            # # Plot actual data points
                            # ax.scatter(x_coords, values, s=50, alpha=0.5, label=f'{keys[i]} data', color=bar_colors[i])

                        # Add "*" marker if p-value is significant
                        for i, p_value in enumerate(p_values_list):
                            __ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                                      _95_values[i] * 1.1,
                                      f'P={resample_P_values_list[i]:.3f}',
                                      # t p={p_value:.3f}\nr P={resample_P_values_list[i]:.3f}
                                      ha='center',
                                      va='bottom')

                        __ax.set_ylabel(ylabel)
                        if plot5Percentile:
                            __ax.set_title(f"{title}")
                        else:
                            __ax.set_title(f"{title} 97.5% CI")

                        # Set x-axis tick labels rotation to 45 degrees
                        __ax.set_xticklabels(keys, rotation=45)

                        setYtick_interval = False
                        if setYtick_interval:
                            # Set the y-axis tick interval to 20
                            from matplotlib.ticker import MultipleLocator
                            y_tick_interval = 20
                            __ax.yaxis.set_major_locator(MultipleLocator(base=y_tick_interval))
                        # if setYlim:
                        __ax.set_ylim(_ylim)
                        #
                        # # Increase font size for labels and text
                        # plt.rcParams.update({'font.size': 14})
                        #
                        # plt.tight_layout()
                        # plt.show()

                    #
                    # num_rows = 1
                    # num_cols = 5
                    # # length = 3.33
                    # # height = 3.33 * 3
                    # fig, axs = plt.subplots(num_rows, num_cols,
                    #                         figsize=(num_cols * length, num_rows * height))
                    # axs = axs.ravel()
                    # data = {
                    #     'XY_ses1': XY_ses1,
                    #     'XY_ses5': XY_ses5,
                    #     'MN_ses1': MN_ses1,
                    #     'MN_ses5': MN_ses5,
                    # }
                    # plot_ttest_bar(data, ax=axs[0], ylabel='accuracy or slope')
                    data = {
                        'XY_ses1-XY_ses5': XY_ses1 - XY_ses5,
                        'MN_ses1-MN_ses5': MN_ses1 - MN_ses5,
                    }
                    plot_ttest_bar(data, __ax=ax, ylabel='')

                    # data = {
                    #     '(XY_ses1-5)-(MN_ses1-5)': (XY_ses1 - XY_ses5) - (MN_ses1 - MN_ses5)
                    # }
                    # plot_ttest_bar(data, ax=axs[2], ylabel='')
                    #
                    # data = {
                    #     '(XY_ses1-XY_ses5)/+': (XY_ses1 - XY_ses5) / (XY_ses1 + XY_ses5),
                    #     '(MN_ses1-MN_ses5)/+': (MN_ses1 - MN_ses5) / (MN_ses1 + MN_ses5),
                    # }
                    # plot_ttest_bar(data, ax=axs[3], ylabel='normalized integration')
                    #
                    # data = {
                    #     '(XY_ses1-5)/(XY_ses1+5)-control': ((XY_ses1 - XY_ses5) / (XY_ses1 + XY_ses5)) - (
                    #             (MN_ses1 - MN_ses5) / (MN_ses1 + MN_ses5))
                    # }
                    # plot_ttest_bar(data, ax=axs[4],
                    #                ylabel='((XY_ses1-XY_ses5)/(XY_ses1+XY_ses5))-((MN_ses1-MN_ses5)/(MN_ses1+MN_ses5))')

                    # fig.savefig(saveFigPath, bbox_inches='tight', format='png')
                    # print(f"figure saved at {saveFigPath}")
                    # fig.savefig(saveFigPath.replace('.png', '.pdf'), format='pdf', transparent=True)
                    # print(f"figure saved at {saveFigPath.replace('.png', '.pdf')}")
                    # plt.show()

                plot_ROI_components(
                    XY_ses5=allResults['ses5_XY'],
                    XY_ses1=allResults['ses1_XY'],
                    MN_ses5=allResults['ses5_MN'],
                    MN_ses1=allResults['ses1_MN'],
                    title=f"{ROI}",
                    ax=axs[currROI],
                )
            saveFigPath = f"{scratchFolder}/plotfig_4_5/{curr_ROIdict}.pdf"
            fig.savefig(saveFigPath, transparent=True)

    neuro()


plot_integrationScore_components(batch=12, fixedCenter=True, plot5Percentile=True)
