# %% [markdown]
# # MVPA
# based on https://peerherholz.github.io/workshop_weizmann/advanced/machine_learning_nilearn.html

# %%
from nilearn import plotting
import numpy as np
import nibabel as nb
import nibabel.processing as nbp
from nilearn.image import resample_to_img
import os

# %%
# define paths
root_dir = '/users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2'
fmriprep_dir = os.path.join(root_dir, 'derivatives/fmriprep23')
#dataset_dir = os.path.join(root_dir, 'derivatives/mvpa_extracted_data')
dataset_dir = os.path.join(root_dir, 'derivatives/mvpa_ls_a_data')
mask_dir = os.path.join(root_dir, 'derivatives','mni_icbm152_gm_tal_nlin_asym_09c.nii')

# %%
# list datasets and concatenate

# find all *_dataset.nii.gz files in dataset_dir
dataset_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('_musicnoise_confounds_dataset.nii.gz')]
dataset_files.sort()

# find all *_trial_types.txt files in dataset_dir
label_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('_musicnoise_confounds_trial_types.txt')]
label_files.sort()

# %%
# concatenate all datasets
D = nb.concat_images(dataset_files, axis=3)

# %%
# concatenate all labels into a single string array
labels = np.concatenate([np.loadtxt(l, dtype=str) for l in label_files])

# trim each label to remove the 2 digit number in the end
labels = np.array([l[:-2] for l in labels])

# %%
mask = nb.load(mask_dir)
mask_downsampled = nbp.resample_to_output(mask, [2,2,2])

mask_resampled = resample_to_img(mask_downsampled, D)

# Binarize ROI template
data_binary = np.array(mask_resampled.get_fdata()>=1, dtype=np.int8)

# Dilate binary mask once
from scipy.ndimage import binary_dilation
data_dilated = binary_dilation(data_binary, iterations=2).astype(np.int8)

# Save binary mask in NIfTI image
mask_resampled2 = nb.Nifti1Image(data_dilated, mask_resampled.affine, mask_resampled.header)
mask_resampled2.set_data_dtype('i1')

from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_resampled2, standardize=False, detrend=False)
samples = masker.fit_transform(D)

# %%
masked_epi = masker.inverse_transform(samples)

# %%
# generate an array of chunk labels
# 13 subjects, 10 noise and 9 x 2 music for each of the 4 runs
chunks = np.repeat(np.arange(1,14), 10*4 + 9*2*4)

# %%
# Let's specify the classifier
from sklearn.svm import LinearSVC
clf = LinearSVC(penalty='l2', loss='squared_hinge', max_iter=1000)

# %%
# Perform the cross validation (takes time to compute)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
cv_scores = cross_val_score(estimator=clf,
                            X=samples,
                            y=labels,
                            groups=chunks,
                            cv=LeaveOneGroupOut(),
                            n_jobs=30,
                            verbose=1)

# %%
print('Average accuracy = %.02f percent\n' % (cv_scores.mean() * 100))
print('Accuracy per fold:', cv_scores, sep='\n')

# %% export results
import pandas as pd
results1 = pd.DataFrame({'accuracy': cv_scores})
results1.to_csv(os.path.join(dataset_dir, 'mvpa-lsa-musicnoise-confounds-gm-results-1.csv'), index=False)
print('saved accuracy results to mvpa-musicnoise-confounds-gm-results-1.csv')

# # %%
# # Import the permuation function
# from sklearn.model_selection import permutation_test_score

# # %%
# # Run the permuation cross-validation
# null_cv_scores = permutation_test_score(estimator=clf,
#                                         X=samples,
#                                         y=labels,
#                                         groups=chunks,
#                                         cv=LeaveOneGroupOut(),
#                                         n_permutations=1000,
#                                         n_jobs=30,
#                                         verbose=1)

# # %%
# print('Prediction accuracy: %.02f' % (null_cv_scores[0] * 100),
#       'p-value: %.04f' % (null_cv_scores[2]),
#       sep='\n')

# # %%
# results2 = pd.DataFrame({'accuracy': null_cv_scores})
# results2.to_csv(os.path.join(dataset_dir, 'mvpa-lsa-results-2.csv'), index=False)
# print('saved accuracy results to mvpa-lsa-results-2.csv')
print('python script finished running.')