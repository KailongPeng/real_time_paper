import os
import sys
assert os.getcwd().endswith('real_time_paper'), "working dir should be 'real_time_paper'"
workingDir = os.getcwd()
sys.path.append('.')
print(f"getcwd = {os.getcwd()}")
import numpy as np
import matplotlib.pyplot as plt
from utils import get_subjects, get_ROIList, kp_run, getjobID_num, waitForEnd, check_jobIDs, mkdir, load_obj, save_obj
from utils import cal_resample, check_jobArray

batch = 12  # meaning both batch 1 and batch 2
subjects, scan_asTemplates = get_subjects(batch=batch)

ROIList = get_ROIList()
PCA_n_components = None
use_norm = False
testMode = False
print(f"PCA_n_components={PCA_n_components}, use_norm={use_norm}")

# random_seed
random_seed = 42
np.random.seed(random_seed)


def run_geometric_ROI():
    jobarrayDict = {}
    jobarrayID = 1
    for ROI in ROIList:
        # for ses in [2, 3, 4]:
        jobarrayDict[jobarrayID] = [ROI, PCA_n_components, use_norm]
        jobarrayID += 1
    np.save(
        f"fig5/geometric_analysis/geometric_jobID.npy",
        jobarrayDict)
    if testMode:
        cmd = f"sbatch --requeue --array=1-1 fig5/geometric_analysis/geometric.sh"
    else:
        cmd = f"sbatch --requeue --array=1-{len(jobarrayDict)} fig5/geometric_analysis/geometric.sh"
    sbatch_response = kp_run(cmd)

    jobID = getjobID_num(sbatch_response)
    waitForEnd(jobID)
    if testMode:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=1)
    else:
        completed = check_jobArray(jobID=jobID, jobarrayNumber=len(jobarrayDict))


# run_geometric_ROI()  # this code runs the job array for geometric analysis, we provided a intermediate file below for better reproducibility


def fig5_geometric():
    # scratchFolder = "/gpfs/milgram/scratch60/turk-browne/kp578/rtSynth_rt/result/geometric/"
    scratchFolder = "path/to/scratch/folder/for/this/analysis/"  # for result storage
    chosenMask = "lfseg_corr_usegray_hippocampus"
    ROIname = "HC_ASHS"
    results = load_obj(
        f"{scratchFolder}/results_{chosenMask}_PCA_n_components{PCA_n_components}_use_norm{use_norm}")

    # Here an intermediate file is provided for better reproducibility
    results = load_obj(
        f"fig5/geometric_analysis/results_lfseg_corr_usegray_hippocampus_PCA_n_componentsNone_use_normFalse.pkl")

    def combined_analysis_updated_subplot(results, ax, title):
        analysis_names = [
            'Integration',
            'Shared Dimension Increasing', 'Unique Dimension Decreasing',
            # 'XY Shared Δ', 'MN Shared Δ', 'XY Unique Δ', 'MN Unique Δ',
            # 'Context Effect'
        ]
        all_results = []
        for analysis_name in analysis_names:
            current_results = []
            for sub, metrics in results.items():
                if analysis_name == 'Integration':
                    value1 = ((metrics['XY_metrics']['X_post_shared'] - metrics['XY_metrics']['X_pre_shared']) +
                              (metrics['XY_metrics']['Y_post_shared'] - metrics['XY_metrics']['Y_pre_shared']) -
                              (metrics['MN_metrics']['M_post_shared'] - metrics['MN_metrics']['M_pre_shared']) -
                              (metrics['MN_metrics']['N_post_shared'] - metrics['MN_metrics']['N_pre_shared']))
                    value2 = ((metrics['XY_metrics']['X_post_unique'] - metrics['XY_metrics']['X_pre_unique']) +
                              (metrics['XY_metrics']['Y_post_unique'] - metrics['XY_metrics']['Y_pre_unique']) -
                              (metrics['MN_metrics']['M_post_unique'] - metrics['MN_metrics']['M_pre_unique']) -
                              (metrics['MN_metrics']['N_post_unique'] - metrics['MN_metrics']['N_pre_unique']))
                    value = value1 - value2
                elif analysis_name == 'Shared Dimension Increasing':
                    # ((X_post_shared - X_pre_shared) + (Y_post_shared - Y_pre_shared))
                    # -
                    # ((M_post_shared - M_pre_shared) + (N_post_shared - N_pre_shared))
                    value = ((metrics['XY_metrics']['X_post_shared'] - metrics['XY_metrics']['X_pre_shared']) +
                             (metrics['XY_metrics']['Y_post_shared'] - metrics['XY_metrics']['Y_pre_shared']) -
                             (metrics['MN_metrics']['M_post_shared'] - metrics['MN_metrics']['M_pre_shared']) -
                             (metrics['MN_metrics']['N_post_shared'] - metrics['MN_metrics']['N_pre_shared']))
                elif analysis_name == 'Unique Dimension Decreasing':
                    # ((X_post_unique - X_pre_unique) + (Y_post_unique - Y_pre_unique))
                    # -
                    # ((M_post_unique - M_pre_unique) + (N_post_unique - N_pre_unique))
                    value = ((metrics['XY_metrics']['X_post_unique'] - metrics['XY_metrics']['X_pre_unique']) +
                             (metrics['XY_metrics']['Y_post_unique'] - metrics['XY_metrics']['Y_pre_unique']) -
                             (metrics['MN_metrics']['M_post_unique'] - metrics['MN_metrics']['M_pre_unique']) -
                             (metrics['MN_metrics']['N_post_unique'] - metrics['MN_metrics']['N_pre_unique']))
                elif analysis_name == 'XY Shared Δ':
                    # ((X_post_shared - X_pre_shared) + (Y_post_shared - Y_pre_shared))
                    value = \
                        ((results[sub]['XY_metrics']['X_post_shared'] - results[sub]['XY_metrics']['X_pre_shared']) +
                         (results[sub]['XY_metrics']['Y_post_shared'] - results[sub]['XY_metrics']['Y_pre_shared']))
                elif analysis_name == 'MN Shared Δ':
                    # ((M_post_shared - M_pre_shared) + (N_post_shared - N_pre_shared))
                    value = ((results[sub]['MN_metrics']['M_post_shared'] - results[sub]['MN_metrics'][
                        'M_pre_shared']) +
                             (results[sub]['MN_metrics']['N_post_shared'] - results[sub]['MN_metrics']['N_pre_shared']))
                elif analysis_name == 'XY Unique Δ':
                    # ((X_post_unique - X_pre_unique) + (Y_post_unique - Y_pre_unique))
                    value = \
                        ((results[sub]['XY_metrics']['X_post_unique'] - results[sub]['XY_metrics']['X_pre_unique']) +
                         (results[sub]['XY_metrics']['Y_post_unique'] - results[sub]['XY_metrics']['Y_pre_unique']))
                elif analysis_name == 'MN Unique Δ':
                    # ((M_post_unique - M_pre_unique) + (N_post_unique - N_pre_unique))
                    value = \
                        ((results[sub]['MN_metrics']['M_post_unique'] - results[sub]['MN_metrics']['M_pre_unique']) +
                         (results[sub]['MN_metrics']['N_post_unique'] - results[sub]['MN_metrics']['N_pre_unique']))
                elif analysis_name == 'Context Effect':
                    # (X_post_normal+Y_post_normal) - (M_post_normal+N_post_normal)
                    value = ((metrics['XY_metrics']['X_post_normal'] + metrics['XY_metrics']['Y_post_normal']) -
                             (metrics['MN_metrics']['M_post_normal'] + metrics['MN_metrics']['N_post_normal']))
                else:
                    raise Exception
                current_results.append(value)

            all_results.append(current_results)

        # Define specific colors for each analysis
        analysis_colors = {
            'Integration': '#FFF3F0',  # Light pink
            'Shared Dimension Increasing': '#FCFF6C',  # Yellow
            'Unique Dimension Decreasing': '#12130F'  # Dark grey/black
        }
        positions = np.arange(len(analysis_names))
        bar_width = 0.8  # Narrower bar width for closer spacing

        # Error bar style
        error_kw = {
            'capsize': 5,
            'capthick': 2,
            'elinewidth': 2,
            'ecolor': 'grey',  # Edge color
            'markeredgecolor': 'grey',
            'markeredgewidth': 2
        }
        for currAnalysis, (analysis, results) in enumerate(zip(analysis_names, all_results)):
            _mean, _5, _95, iter_mean = cal_resample(data=results, times=5000, return_all=True, random_seed=random_seed)
            yerr = np.array([[_mean - _5], [_95 - _mean]])
            color = analysis_colors.get(analysis, 'grey')
            ax.bar(positions[currAnalysis], _mean, yerr=yerr, label=analysis, color=color, capsize=5,
                   error_kw=error_kw,
                   width=bar_width)
            scatter_x = np.random.normal(positions[currAnalysis], 0.04, size=len(results))
            ax.scatter(scatter_x, results, edgecolors='grey', facecolors=color, alpha=0.7, s=30, linewidths=1)

            if _mean >= 0:
                pvalue = np.mean(np.array(iter_mean) < 0)
                ax.text(positions[currAnalysis], _95 + (2.4 * (_95 - _mean)), f'p={pvalue:.6f}', ha='center',
                        va='bottom')
            elif _mean < 0:
                pvalue = np.mean(np.array(iter_mean) > 0)
                ax.text(positions[currAnalysis], _5 - (2.4 * (_mean - _5)), f'p={pvalue:.6f}', ha='center', va='top')
        ax.set_xticks(positions)
        ax.set_xticklabels(analysis_names, rotation=45, ha="right")
        ax.set_ylabel('Mean Difference')
        ax.set_title(title)
        ax.axhline(0, color='grey', linestyle='--')

    fig, axes = plt.subplots(figsize=(4, 6))
    combined_analysis_updated_subplot(
        results,
        axes,
        title=f'{ROIname} pca_n={PCA_n_components} use_norm={use_norm}'
    )
    plt.tight_layout()
    plt.savefig('fig5/geometric_analysis/geometric_FSL_6.0.5.2-centos7_64__FSL_6.0.3-centos7_64_fsl_sh.pdf')
    plt.show()


fig5_geometric()  # note this function may need to be run in juptyer notebook to display the plot

