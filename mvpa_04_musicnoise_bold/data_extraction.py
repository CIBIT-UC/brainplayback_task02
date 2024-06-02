#%% Import libraries
import os
from data_extraction_functions import create_brain_mask, get_mask, clean_func_image, extract_samples, extract_samples_with_atlas, convert_samples_to_features

#%% Define paths
data_root = '/Volumes/T7/BIDS-BRAINPLAYBACK-TASK2'
fmriprep_dir = os.path.join(data_root, 'derivatives','fmriprep23')
output_func_dir = os.path.join(data_root, 'derivatives', 'mvpa_04_musicnoise_bold', 'func_clean')
output_feat_dir = os.path.join(data_root, 'derivatives', 'mvpa_04_musicnoise_bold', 'features')

#%% Create brain mask with example functional image
#img_func = os.path.join(fmriprep_dir, 'sub-01', 'ses-01', 'func', 'sub-01_ses-01_task-02a_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')
#create_brain_mask(data_root, img_func)

#%% Iterate on the subjects and runs
img_mask = get_mask(data_root)
for subject in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17']:
    for run in ['1', '2', '3', '4']:
        img_crop = clean_func_image(fmriprep_dir, output_func_dir, img_mask, subject, run, False)
        #samples = extract_samples(img_crop, img_mask, subject, run)
        samples = extract_samples_with_atlas(img_crop, 'koelsch_spheres', subject, run)
        convert_samples_to_features(samples, data_root, output_feat_dir, subject, run)
