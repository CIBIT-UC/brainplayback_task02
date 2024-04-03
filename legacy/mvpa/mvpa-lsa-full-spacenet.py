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
mask_dir = os.path.join(root_dir, 'derivatives','mni_icbm152_t1_tal_nlin_asym_09c_mask.nii')

# %%
# list datasets and concatenate

# find all *_dataset.nii.gz files in dataset_dir
dataset_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('_full_dataset.nii.gz')]
dataset_files.sort()

# find all *_trial_types.txt files in dataset_dir
label_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('_full_trial_types.txt')]
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
# Create two masks that specify the training and the test set 
mask_test = chunks == 5
mask_train = np.invert(mask_test)

# Apply this sample mask to X (fMRI data) and y (behavioral labels)
from nilearn.image import index_img
X_train = index_img(D, mask_train)
y_train = labels[mask_train]

X_test = index_img(D, mask_test)
y_test = labels[mask_test]

# %%
# Let's specify the classifier
from nilearn.decoding import SpaceNetClassifier

# Fit model on train data and predict on test data
decoder = SpaceNetClassifier(penalty='tv-l1',
                             mask=mask_resampled2,
                             max_iter=1000,
                             cv=12,
                             standardize=True,
                             n_jobs=30,
                             verbose=1)

# starting decoder fit...
print('starting decoder fit...')
decoder.fit(X_train, y_train)

# %%
# Predict the labels of the test data
y_pred = decoder.predict(X_test)
mse = np.mean(np.abs(y_test - y_pred))
print(f"Mean square error (MSE) on the predicted class: {mse:.2f}")


# Re run average accuracy
accuracy = (y_pred == y_test).mean() * 100.
print("\nTV-l1  classification accuracy : %g%%" % accuracy)

import pandas as pd
results1 = pd.DataFrame({'testaccuracy': accuracy}, index=[0])
results1.to_csv(os.path.join(dataset_dir, 'mvpa-lsa-full-spacenet-results-1.csv'), index=False)
print('saved accuracy results to mvpa-lsa-full-spacenet-results-1.csv')

# %%
coef_img = decoder.coef_img_

# save coef_img
coef_img.to_filename(os.path.join(dataset_dir, 'mvpa-lsa-full-spacenet_coefimg.nii.gz'))
print('saved coef_img to mvpa-lsa-full-spacenet_coefimg.nii.gz')

# export decoder
import pickle
with open(os.path.join(dataset_dir, 'mvpa-lsa-full-spacenet-decoder.pkl'), 'wb') as f:
    pickle.dump(decoder, f)
print('saved decoder to mvpa-lsa-full-spacenet-decoder.pkl')

print('python script finished running.')