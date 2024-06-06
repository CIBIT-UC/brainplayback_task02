#%% Import libraries
import os
from data_extraction_functions import edit_events, get_mask, clean_func_image, extract_samples_with_atlas, convert_samples_to_features

#%% Define paths
data_root = '/Volumes/T7/BIDS-BRAINPLAYBACK-TASK2'
fmriprep_dir = os.path.join(data_root, 'derivatives','fmriprep23')
output_func_dir = os.path.join(data_root, 'derivatives', 'mvpa_04_musicnoise_bold', 'func_clean')
output_feat_dir = os.path.join(data_root, 'derivatives', 'mvpa_06_full_bold', 'features')

#%% Iterate on the subjects and runs
img_mask = get_mask(data_root)
for subject in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17']:
    for run in ['1', '2', '3', '4']:
        img_crop = clean_func_image(fmriprep_dir, output_func_dir, img_mask, subject, run, overwrite=False)
        samples = extract_samples_with_atlas(img_crop, 'koelsch_spheres_per_voxel', subject, run)
        events_split = edit_events(data_root, subject, run)
        convert_samples_to_features(samples, data_root, output_feat_dir, events_split, subject, run)
