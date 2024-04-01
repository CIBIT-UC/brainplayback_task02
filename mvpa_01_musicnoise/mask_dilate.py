import nibabel as nb
import nibabel.processing as nbp
from nilearn.image import resample_to_img
import numpy as np
import os

root_dir = '/Volumes/T7/BIDS-BRAINPLAYBACK-TASK2'
mask_gm_file    = os.path.join(root_dir, 'derivatives', 'mni_icbm152_gm_tal_nlin_asym_09c.nii')

mask = nb.load(mask_gm_file)
mask_downsampled = nbp.resample_to_output(mask, [2,2,2])

D = os.path.join(root_dir, 'derivatives', 'mvpa_01_musicnoise', 'sub-01_ses-01_task-02a_run-1_musicnoise_confounds_dataset.nii.gz')
mask_resampled = resample_to_img(mask_downsampled, D)

# Binarize ROI template
data_binary = np.array(mask_resampled.get_fdata()>=1, dtype=np.int8)

# Dilate binary mask once
from scipy.ndimage import binary_dilation
data_dilated = binary_dilation(data_binary, iterations=2).astype(np.int8)

# Save binary mask in NIfTI image
mask_resampled2 = nb.Nifti1Image(data_dilated, mask_resampled.affine, mask_resampled.header)
mask_resampled2.set_data_dtype('i1')
mask_resampled2.to_filename(os.path.join(root_dir, 'derivatives', 'mni_icbm152_gm_tal_nlin_asym_09c_res-2_dilated.nii'))