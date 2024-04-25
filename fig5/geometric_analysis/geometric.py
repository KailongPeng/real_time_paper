testMode = False
verbose = False
import os
assert os.getcwd().endswith('real_time_paper'), "working dir should be 'real_time_paper'"
workingDir = os.getcwd()

from scipy.stats import zscore
import joblib

import os
import sys
import numpy as np
import pandas as pd
import time
import pickle5 as pickle
from tqdm import tqdm

import nibabel as nib
from sklearn.decomposition import PCA

projectDir = "./"
os.chdir(projectDir)

sys.path.append(projectDir)


from utils import mkdir
from utils import get_subjects
from utils import get_ROIList


def other(target):
    other_objs = [i for i in ['bed', 'bench', 'chair', 'table'] if i not in target]
    return other_objs


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def readtxt(file):
    f = open(file, "r")
    return f.read()


def kp_and(l):
    t = True
    for i in l:
        t = t * np.asarray(i)
    return t


def kp_or(l):
    t = np.asarray(l[0])
    for i in l:
        t = t + np.asarray(i)
    return t


def wait(tmpFile, waitFor=0.1):
    while not os.path.exists(tmpFile):
        time.sleep(waitFor)
    return 1


def normalize(X):
    _X = X.copy()
    _X = zscore(_X, axis=0)
    _X[np.isnan(_X)] = 0
    return _X


def check(sbatch_response):
    print(sbatch_response)
    if "Exception" in sbatch_response or "Error" in sbatch_response or "Failed" in sbatch_response or "not" in sbatch_response:
        raise Exception(sbatch_response)


os.chdir("/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/rtSynth_rt/")


def pca_metrics_alex(A_pre, A_post, B_pre, B_post, is_normalize_all_vectors=False):
    """
    :params A_pre: (n_samples, n_units)
    :params A_post: (n_samples, n_units)
    :params B_pre: (n_samples, n_units)
    :params B_post: (n_samples, n_units)
    """

    def proj_u_on_v(u, v):
        normalized_u_dot_v = np.sum(u * v, axis=1) / np.linalg.norm(v, axis=1)  # shape: (n_samples,)
        return normalized_u_dot_v

    def vector_proj_u_on_v(u, v):
        """u, v have shape (n_samples, n_units)
        """
        normalized_u_dot_v = proj_u_on_v(u, v)  # shape: (n_samples,)

        return normalized_u_dot_v.reshape(-1, 1) * v  # shape: (n_samples, n_units)

    if is_normalize_all_vectors == True:
        A_pre /= np.linalg.norm(A_pre, axis=1, keepdims=True)
        A_post /= np.linalg.norm(A_post, axis=1, keepdims=True)
        B_pre /= np.linalg.norm(B_pre, axis=1, keepdims=True)
        B_post /= np.linalg.norm(B_post, axis=1, keepdims=True)

    # shared direction is bisector of A_pre and B_pre
    shared_direction = (A_pre + B_pre)
    shared_direction /= np.linalg.norm(A_pre + B_pre, axis=1, keepdims=True)  # normalize, shape: (n_samples, n_units)

    # unique direction is direction in plane perpendicular to shared direction
    unique_direction = A_pre - vector_proj_u_on_v(A_pre, shared_direction)
    unique_direction /= np.linalg.norm(unique_direction, axis=1,
                                       keepdims=True)  # normalize, shape: (n_samples, n_units)

    assert np.allclose(np.sum(shared_direction * unique_direction, axis=1),
                       0), "shared and unique direction should be orthogonal"

    # normal direction is direction in plane perpendicular to shared direction and unique direction,
    # for now we pick the direction closest to A_post
    normal_direction = A_post - vector_proj_u_on_v(A_post, shared_direction) - vector_proj_u_on_v(A_post,
                                                                                                  unique_direction)
    normal_direction /= np.linalg.norm(normal_direction, axis=1,
                                       keepdims=True)  # normalize, shape: (n_samples, n_units)

    assert np.allclose(np.sum(normal_direction * shared_direction, axis=1),
                       0), "normal and shared direction should be orthogonal"
    assert np.allclose(np.sum(normal_direction * unique_direction, axis=1),
                       0), "normal and unique direction should be orthogonal"

    # compute projection of A_post, B_post on shared, unique, normal directions
    A_post_shared = proj_u_on_v(A_post, shared_direction).mean()
    A_post_unique = proj_u_on_v(A_post, unique_direction).mean()
    A_post_normal = proj_u_on_v(A_post, normal_direction).mean()

    B_post_shared = proj_u_on_v(B_post, shared_direction).mean()
    B_post_unique = proj_u_on_v(B_post, unique_direction).mean()
    B_post_normal = proj_u_on_v(B_post, normal_direction).mean()

    return A_post_shared, A_post_unique, A_post_normal, B_post_shared, B_post_unique, B_post_normal


# this code first load the data from 8 runs from ses1 and 8 runs from ses5, from a single subject and single ROI.
subjects, scan_asTemplates = get_subjects(batch=12)
ROIList = get_ROIList()

if testMode:
    chosenMask = ROIList[8]
    PCA_n_components = 0.95
    use_norm = True
else:
    jobarrayDict = np.load(f"fig5/geometric_analysis/geometric_jobID.npy",
                           allow_pickle=True)
    jobarrayDict = dict(enumerate(jobarrayDict.flatten(), 1))[1]
    jobarrayID = int(float(sys.argv[1]))
    [chosenMask, PCA_n_components, use_norm] = jobarrayDict[jobarrayID]

print(f"choseMask={chosenMask}, PCA_n_components={PCA_n_components}, use_norm={use_norm}")


autoAlignFlag = True
print(f"chosenMask={chosenMask}, autoAlignFlag={autoAlignFlag}")
scratchFolder = "path/to/scratch/folder/for/this/analysis/"  # for result storage
mkdir(scratchFolder)

# random seed
random_seed = 42
np.random.seed(random_seed)


def project_on_directions(sub=None, chosenMask=None, autoAlignFlag=True):
    # load the data
    def load_data(sub, chosenMask, ses=1):
        """
        :param subject: subject name
        :param chosenMask: mask name
        :return:
        """
        # load the data
        runRecording = pd.read_csv(f"{workingDir}/data/subjects/{sub}/ses{ses}/runRecording.csv")
        # assert len(np.unique(np.asarray(scanList))) == len(runRecording)
        actualRuns = list(runRecording['run'].iloc[list(
            np.where(1 == 1 * (runRecording['type'] == 'recognition'))[0])])  # can be [1,2,3,4,5,6,7,8] or [1,2,4,5]
        assert len(actualRuns) == 8

        objects = ['bed', 'bench', 'chair', 'table']

        new_run_indexs = []
        new_run_index = 1

        maskFolder = f"{workingDir}/data/subjects/{sub}/ses1/recognition/mask/"
        print(f"maskFolder={maskFolder}")

        brain_data = None
        behav_data = None
        recognition_dir = f"{workingDir}/data/subjects/{sub}/ses{ses}/recognition/"
        for ii, run in enumerate(actualRuns):  # load behavior and brain data for current session
            t = np.load(f"{recognition_dir}/beforeUnwarp/brain_run{run}.npy")

            maskPath = f"{maskFolder}/{chosenMask}.nii"
            try:
                mask = nib.load(maskPath).get_fdata()
                if verbose:
                    print(f"loading {maskFolder}/{chosenMask}.nii , mask size = {np.sum(mask)}")
            except:
                mask = nib.load(f"{maskPath}.gz").get_fdata()
                if verbose:
                    print(f"loading {maskFolder}/{chosenMask}.nii.gz , mask size = {np.sum(mask)}")
            if verbose:
                print(f"mask.shape={mask.shape}")

            t = t[:, mask == 1]
            t = normalize(t)  # normalize along the time dimension
            brain_data = t if ii == 0 else np.concatenate((brain_data, t), axis=0)

            t = pd.read_csv(f"{recognition_dir}behav_run{run}.csv")
            t['run_num'] = new_run_index
            new_run_indexs.append(new_run_index)
            new_run_index += 1
            behav_data = t if ii == 0 else pd.concat([behav_data, t])

        print(f"brain_data.shape={brain_data.shape}")
        assert len(brain_data.shape) == 2

        # convert item colume to label colume
        imcodeDict = {
            'A': 'bed',
            'B': 'chair',
            'C': 'table',
            'D': 'bench'}
        label = []
        for curr_trial in range(behav_data.shape[0]):
            label.append(imcodeDict[behav_data['Item'].iloc[curr_trial]])
        behav_data['label'] = label  # merge the label column with the data dataframe

        return brain_data, behav_data

    brain_data_ses1, behav_data_ses1 = load_data(sub, chosenMask, ses=1)
    brain_data_ses5, behav_data_ses5 = load_data(sub, chosenMask, ses=5)

    def apply_and_save_pca(brain_data, sub, ses, scratchFolder, chosenMask=None):
        sub_ses_folder = os.path.join(scratchFolder, f"sub{sub}/ses{ses}")
        pca_filename = os.path.join(sub_ses_folder, f'pca_model_{chosenMask}_{PCA_n_components}.pkl')

        pca = PCA(n_components=PCA_n_components, random_state=random_seed)  # n_components=0.95
        transformed_data = pca.fit_transform(brain_data)
        print(f"brain_data.shape={brain_data.shape}")
        print(f"transformed_data.shape={transformed_data.shape}")

        mkdir(sub_ses_folder)
        joblib.dump(pca, pca_filename)  # save the model

        return transformed_data

    def mean_centering(data):
        mean = np.mean(data, axis=0)
        # assert mean < 1e-6, f"mean={mean}"
        print(f"original np.sum(mean)={np.sum(mean)}")
        data_centered = data - mean
        return data_centered

    brain_data_ses1 = mean_centering(brain_data_ses1)
    brain_data_ses5 = mean_centering(brain_data_ses5)

    brain_data_ses1_transformed = apply_and_save_pca(brain_data_ses1, sub, 1, scratchFolder,
                                                     chosenMask=chosenMask)

    sub_ses_folder = os.path.join(scratchFolder, f"sub{sub}/ses1")
    pca_model_path = os.path.join(sub_ses_folder, f'pca_model_{chosenMask}_{PCA_n_components}.pkl')
    pca_loaded = joblib.load(pca_model_path)
    brain_data_ses5_transformed = pca_loaded.transform(brain_data_ses5)

    def prepare_X_Y_M_N(brain_data_ses1, behav_data_ses1, brain_data_ses5, behav_data_ses5):
        """
        :param brain_data_ses1: brain data from ses1
        :param behav_data_ses1: behavior data from ses1
        :param brain_data_ses5: brain data from ses5
        :param behav_data_ses5: behavior data from ses5
        :return:
        """

        # Mean centering
        brain_data_ses1 = mean_centering(brain_data_ses1)
        brain_data_ses5 = mean_centering(brain_data_ses5)

        objects = ['bed', 'bench', 'chair', 'table']
        curr_batch = scan_asTemplates[sub]['batch']
        if curr_batch == 1:
            X = 'bed'
            Y = 'chair'
            M = 'table'
            N = 'bench'
        elif curr_batch == 2:
            X = 'table'
            Y = 'bench'
            M = 'chair'
            N = 'bed'
        else:
            raise Exception(f"curr_batch={curr_batch} is not supported.")

        X_pre = np.mean(brain_data_ses1[behav_data_ses1['label'] == X], axis=0)
        Y_pre = np.mean(brain_data_ses1[behav_data_ses1['label'] == Y], axis=0)
        M_pre = np.mean(brain_data_ses1[behav_data_ses1['label'] == M], axis=0)
        N_pre = np.mean(brain_data_ses1[behav_data_ses1['label'] == N], axis=0)

        X_post = np.mean(brain_data_ses5[behav_data_ses5['label'] == X], axis=0)
        Y_post = np.mean(brain_data_ses5[behav_data_ses5['label'] == Y], axis=0)
        M_post = np.mean(brain_data_ses5[behav_data_ses5['label'] == M], axis=0)
        N_post = np.mean(brain_data_ses5[behav_data_ses5['label'] == N], axis=0)

        return X_pre, Y_pre, M_pre, N_pre, X_post, Y_post, M_post, N_post

    X_pre, Y_pre, M_pre, N_pre, X_post, Y_post, M_post, N_post = prepare_X_Y_M_N(
        brain_data_ses1_transformed, behav_data_ses1, brain_data_ses5_transformed, behav_data_ses5)

    def normalize_vector(v):
        """
        Normalize a vector v
        """
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def pca_metrics(X_pre, X_post, Y_pre, Y_post, label_prefix=''):
        """
        Calculate and project on shared, unique, and normal dimensions.
        """

        def proj_u_on_v(u, v):
            """
            Project vector u on vector v.
            """
            return (np.dot(u, v) / np.dot(v, v)) * v

        # Normalizing the vectors
        X_pre_norm = normalize_vector(X_pre)
        Y_pre_norm = normalize_vector(Y_pre)
        X_post_norm = normalize_vector(X_post)
        Y_post_norm = normalize_vector(Y_post)

        # Calculating shared and unique dimensions
        shared_dimension = normalize_vector(X_pre_norm + Y_pre_norm)
        unique_dimension = normalize_vector(X_pre_norm - Y_pre_norm)

        # Calculating normal dimension
        temp = normalize_vector(X_post_norm + Y_post_norm)
        temp_proj_shared = proj_u_on_v(temp, shared_dimension)
        temp_proj_unique = proj_u_on_v(temp, unique_dimension)
        normal_dimension = normalize_vector(temp - temp_proj_shared - temp_proj_unique)

        dot_product_shared_unique = np.dot(shared_dimension, unique_dimension)
        assert np.allclose(dot_product_shared_unique, 0), "Shared and unique dimensions should be orthogonal."

        dot_product_shared_normal = np.dot(shared_dimension, normal_dimension)
        dot_product_unique_normal = np.dot(unique_dimension, normal_dimension)
        assert np.allclose(dot_product_shared_normal, 0), "Normal and shared dimensions should be orthogonal."
        assert np.allclose(dot_product_unique_normal, 0), "Normal and unique dimensions should be orthogonal."

        if use_norm:
            # Projecting vectors on dimensions
            X_pre_shared = np.dot(X_pre_norm, shared_dimension)
            X_pre_unique = np.dot(X_pre_norm, unique_dimension)
            X_post_shared = np.dot(X_post_norm, shared_dimension)
            X_post_unique = np.dot(X_post_norm, unique_dimension)

            Y_pre_shared = np.dot(Y_pre_norm, shared_dimension)
            Y_pre_unique = np.dot(Y_pre_norm, unique_dimension)
            Y_post_shared = np.dot(Y_post_norm, shared_dimension)
            Y_post_unique = np.dot(Y_post_norm, unique_dimension)

            # For normal dimension, since it's orthogonal to the shared/unique, pre projections are 0
            X_pre_normal = np.dot(X_pre_norm, normal_dimension)
            Y_pre_normal = np.dot(Y_pre_norm, normal_dimension)
            assert np.allclose(X_pre_normal, 0), "X_pre_normal should be 0"
            assert np.allclose(Y_pre_normal, 0), "Y_pre_normal should be 0"

            X_post_normal = np.dot(X_post_norm, normal_dimension)
            Y_post_normal = np.dot(Y_post_norm, normal_dimension)
        else:
            # Projecting vectors on dimensions
            X_pre_shared = np.dot(X_pre, shared_dimension)
            X_pre_unique = np.dot(X_pre, unique_dimension)
            X_post_shared = np.dot(X_post, shared_dimension)
            X_post_unique = np.dot(X_post, unique_dimension)

            Y_pre_shared = np.dot(Y_pre, shared_dimension)
            Y_pre_unique = np.dot(Y_pre, unique_dimension)
            Y_post_shared = np.dot(Y_post, shared_dimension)
            Y_post_unique = np.dot(Y_post, unique_dimension)

            # For normal dimension, since it's orthogonal to the shared/unique, pre projections are 0
            X_pre_normal = np.dot(X_pre, normal_dimension)
            Y_pre_normal = np.dot(Y_pre, normal_dimension)
            assert np.allclose(X_pre_normal, 0), "X_pre_normal should be 0"
            assert np.allclose(Y_pre_normal, 0), "Y_pre_normal should be 0"

            X_post_normal = np.dot(X_post, normal_dimension)
            Y_post_normal = np.dot(Y_post, normal_dimension)

        return {
            f'{label_prefix[0]}_pre_shared': X_pre_shared, f'{label_prefix[0]}_pre_unique': X_pre_unique, f'{label_prefix[0]}_pre_normal': X_pre_normal,
            f'{label_prefix[0]}_post_shared': X_post_shared, f'{label_prefix[0]}_post_unique': X_post_unique, f'{label_prefix[0]}_post_normal': X_post_normal,
            f'{label_prefix[1]}_pre_shared': Y_pre_shared, f'{label_prefix[1]}_pre_unique': Y_pre_unique, f'{label_prefix[1]}_pre_normal': Y_pre_normal,
            f'{label_prefix[1]}_post_shared': Y_post_shared, f'{label_prefix[1]}_post_unique': Y_post_unique, f'{label_prefix[1]}_post_normal': Y_post_normal
        }

    # run pca_metrics(X_pre, X_post, Y_pre, Y_post)
    XY_metrics = pca_metrics(X_pre, X_post, Y_pre, Y_post, label_prefix='XY')
    print(f"XY_metrics={XY_metrics}")  # should print the metrics as a dictionary

    # run pca_metrics(M_pre, M_post, N_pre, N_post)
    MN_metrics = pca_metrics(M_pre, M_post, N_pre, N_post, label_prefix='MN')
    print(f"MN_metrics={MN_metrics}")  # should print the metrics as a dictionary

    return XY_metrics, MN_metrics


results = {}
for sub in tqdm(subjects):
    XY_metrics, MN_metrics = project_on_directions(sub, chosenMask, autoAlignFlag)
    results[sub] = {'XY_metrics': XY_metrics, 'MN_metrics': MN_metrics}


save_obj(results, f"{scratchFolder}/results_{chosenMask}_PCA_n_components{PCA_n_components}_use_norm{use_norm}")
print(f"saved {scratchFolder}/results_{chosenMask}_PCA_n_components{PCA_n_components}_use_norm{use_norm}")

print("done")

