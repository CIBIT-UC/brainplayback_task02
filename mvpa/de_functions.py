import os
import nibabel as nb
import pandas as pd
import numpy as np
from scipy.signal import detrend
from scipy.stats import zscore
from nilearn.image import mean_img, clean_img, concat_imgs, clean_img

from nilearn import masking
from nilearn.glm.first_level import FirstLevelModel

# define function to extract data from a single subject
def extract_data(root_dir, subj, task, run, brain_resampled, mask_resampled):

    print('Extracting data for subject ' + subj + ' and run ' + run)

    # define paths
    fmriprep_dir = os.path.join(root_dir, 'derivatives/fmriprep23')
    output_dir = os.path.join(root_dir, 'derivatives/mvpa_extracted_data')

    # load data
    func_dir = os.path.join(fmriprep_dir, subj, 'ses-01', 'func')
    func_file = os.path.join(func_dir, subj + '_ses-01_task-' + task + '_run-' + run + '_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')
    #func = nb.load(func_file)

    # Detrend and zscore data and save it under a new NIfTI file
    func_clean = clean_img(func_file, detrend=True, standardize=True, confounds=None, low_pass=None, high_pass=None, t_r=1, ensure_finite=True)

    # data = func.get_fdata()
    # data = detrend(data)
    # data = np.nan_to_num(zscore(data, axis=0)) # nilearn clean img
    # func_standardized = nb.Nifti1Image(data, func.affine, func.header)

    # apply brain mask to functional image
    #func_cleaned = math_img('img1 * img2',
    #                        img1=func_standardized, img2=mask_resampled.slicer[..., None])

    #func_crop = crop_img(func_cleaned)
    #func_crop = func_cleaned
    #func_crop = func_standardized

    # clear func from memory
    # func.uncache()

    # Load events file
    events_file = os.path.join(root_dir,subj,'ses-01','func',
                               subj + '_ses-01_task-' + task + '_run-' + run + '_events.tsv')
    events = pd.read_csv(events_file, sep='\t')

    # delete rows with trial_type 'Noise'
    events = events[events.trial_type != 'Noise']

    # round onset and duration to integer
    events['onset'] = events['onset'].round(0).astype(int)
    events['duration'] = events['duration'].round(0).astype(int)

    # add compensation for hemodynamic lag (4s)
    events['onset'] = events['onset'] + 4

    # Create new column named 'offset' with onset+duration
    events['offset'] = events['onset'] + events['duration']

    # fetch unique trial_types
    trial_types = events['trial_type'].unique()

    # Create empty dataframe with columns 'trial_type' and 'boldstd'
    df = pd.DataFrame(columns=['trial_type', 'boldstd'])

    # iterate on events dataframe
    for index, row in events.iterrows():

        # extract onset and offset
        onset = row['onset']
        offset = row['offset']
        # extract trial_type
        trial_type = row['trial_type']

        # extract mean data from functional image
        data = nb.Nifti1Image(func_clean.get_fdata()[...,onset:offset].mean(axis=-1), func_clean.affine, func_clean.header)

        # concat to dataframe
        df = pd.concat([df, pd.DataFrame([{'trial_type': trial_type, 'boldstd': data}])], ignore_index=True)
    
    # concatenate all images in the dataframe
    func_concat = nb.concat_images(df['boldstd'].tolist())

    # extract trial_type to list
    trial_types = np.ravel(df['trial_type'].tolist())

    # save concatenated images
    func_concat.to_filename(os.path.join(output_dir, 
                                         subj + '_ses-01_task-' + task + '_run-' + run + '_dataset.nii.gz'))

    # save trial_types
    np.savetxt(os.path.join(output_dir,
                            subj + '_ses-01_task-' + task + '_run-' + run + '_trial_types.txt'),
                        trial_types, fmt='%s')
    
    print('Done for subject ' + subj + ' and run ' + run + '\n')

def ls_a(root_dir, subj, task, run):

    print('Extracting data for subject ' + subj + ' and run ' + run)

    # define paths
    fmriprep_dir = os.path.join(root_dir, 'derivatives/fmriprep23')
    output_dir = os.path.join(root_dir, 'derivatives/mvpa_ls_a_data')

    # load data
    func_dir = os.path.join(fmriprep_dir, subj, 'ses-01', 'func')
    func_file = os.path.join(func_dir, subj + '_ses-01_task-' + task + '_run-' + run + '_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')

    # Load events file
    events_file = os.path.join(root_dir,subj,'ses-01','func',
                                subj + '_ses-01_task-' + task + '_run-' + run + '_events.tsv')
    events = pd.read_csv(events_file, sep='\t')

    # delete rows with trial_type 'Noise'
    events = events[events.trial_type != 'Noise'].reset_index(drop=True)

    # round onset and duration to integer
    events['onset'] = events['onset'].round(0).astype(int)
    events['duration'] = events['duration'].round(0).astype(int)

    # extract trial_type to list (before messing with the names)
    trial_types = np.ravel(events['trial_type'].tolist())

    # mess with the condition names
    # iterate on the even rows of the events dataframe and rename trial_type to the corresponding trial_type + '___1
    for i in range(0, len(events), 2):
        events.loc[i, 'trial_type'] = events.loc[i, 'trial_type'] + '___1'

    # iterate on the odd rows of the events dataframe and rename trial_type to the corresponding trial_type + '___2
    for i in range(1, len(events), 2):
        events.loc[i, 'trial_type'] = events.loc[i, 'trial_type'] + '___2'

    # fetch confounds to clean image
    confounds_file = os.path.join(fmriprep_dir,subj,'ses-01','func',
                                subj + '_ses-01_task-' + task + '_run-' + run + '_desc-confounds_timeseries.tsv')

    confounds = pd.read_csv(confounds_file, sep='\t')

    # only consider confounds that start with 'trans', 'rot', and 'cosine'
    confounds = confounds.filter(regex='trans|rot|cosine')

    confounds = confounds.fillna(0)

    # Create a EPI mask
    fmri_img_mean = mean_img(func_file)
    epi_mask = masking.compute_epi_mask(fmri_img_mean)

    # GLM
    lsa_glm = FirstLevelModel(t_r=1, 
                              hrf_model='spm', 
                              mask_img=epi_mask, 
                              drift_model=None, 
                              n_jobs=2, # beware that this is going to be paralelized outside of the function
                              standardize=False)

    lsa_glm.fit(func_file, events, confounds)

    contrast_mat_lsa = np.eye(len(events), len(events))  # +1 to account for the constant term

    # Estimate statistical maps for each condition of interest and trial
    z_maps_lsa = []

    for contrast in contrast_mat_lsa:
        z_map = lsa_glm.compute_contrast(contrast, output_type='z_score')

        # Drop the trial number from the condition name to get the original name
        z_maps_lsa.append(z_map)

    # 4D image with all the beta maps
    img_lsa = concat_imgs(z_maps_lsa)   

    # save concatenated images
    img_lsa.to_filename(os.path.join(output_dir, 
                                         subj + '_ses-01_task-' + task + '_run-' + run + '_dataset.nii.gz'))

    # save trial_types
    np.savetxt(os.path.join(output_dir,
                            subj + '_ses-01_task-' + task + '_run-' + run + '_trial_types.txt'),
                        trial_types, fmt='%s')

    print('Done for subject ' + subj + ' and run ' + run + '\n')


def ls_a_musicnoise(root_dir, subj, task, run):

    print('Extracting data for subject ' + subj + ' and run ' + run)

    # define paths
    fmriprep_dir = os.path.join(root_dir, 'derivatives/fmriprep23')
    output_dir = os.path.join(root_dir, 'derivatives/mvpa_ls_a_data')

    # load data
    func_dir = os.path.join(fmriprep_dir, subj, 'ses-01', 'func')
    func_file = os.path.join(func_dir, subj + '_ses-01_task-' + task + '_run-' + run + '_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')

    # Load events file
    events_file = os.path.join(root_dir,subj,'ses-01','func',
                                subj + '_ses-01_task-' + task + '_run-' + run + '_events.tsv')
    events = pd.read_csv(events_file, sep='\t')

    # rename all trial_types except 'Noise' to 'Music'
    events['trial_type'] = events['trial_type'].replace(events['trial_type'].unique()[1:], 'Music')

    # extract trial_type to list (before messing with the names)
    trial_types = np.ravel(events['trial_type'].tolist())

    # Add counter to each trial_type in the format '___01'
    events['trial_type'] = events['trial_type'] + events.groupby('trial_type').cumcount().add(1).astype(str).str.zfill(2)

    # round onset and duration to integer
    events['onset'] = events['onset'].round(0).astype(int)
    events['duration'] = events['duration'].round(0).astype(int)

    # fetch confounds to clean image
    confounds_file = os.path.join(fmriprep_dir,subj,'ses-01','func',
                                subj + '_ses-01_task-' + task + '_run-' + run + '_desc-confounds_timeseries.tsv')

    confounds = pd.read_csv(confounds_file, sep='\t')

    # only consider confounds that start with 'trans', 'rot', and 'cosine'
    confounds = confounds.filter(regex='trans|rot|cosine')

    confounds = confounds.fillna(0)

    # Create a EPI mask
    fmri_img_mean = mean_img(func_file)
    epi_mask = masking.compute_epi_mask(fmri_img_mean)

    # GLM
    lsa_glm = FirstLevelModel(t_r=1, 
                              hrf_model='spm', 
                              mask_img=epi_mask, 
                              drift_model=None, 
                              n_jobs=2, # beware that this is going to be paralelized outside of the function
                              standardize=False)

    lsa_glm.fit(func_file, events, confounds)

    contrast_mat_lsa = np.eye(len(events), len(events))  # +1 to account for the constant term

    # Estimate statistical maps for each condition of interest and trial
    z_maps_lsa = []

    for contrast in contrast_mat_lsa:
        z_map = lsa_glm.compute_contrast(contrast, output_type='z_score')

        # Drop the trial number from the condition name to get the original name
        z_maps_lsa.append(z_map)

    # 4D image with all the beta maps
    img_lsa = concat_imgs(z_maps_lsa)   

    # save concatenated images
    img_lsa.to_filename(os.path.join(output_dir, 
                                         subj + '_ses-01_task-' + task + '_run-' + run + '_musicnoise_dataset.nii.gz'))

    # save trial_types
    np.savetxt(os.path.join(output_dir,
                            subj + '_ses-01_task-' + task + '_run-' + run + '_musicnoise_trial_types.txt'),
                        trial_types, fmt='%s')

    print('Done for subject ' + subj + ' and run ' + run + '\n')