import os
import numpy as np
import pandas as pd
import nibabel as nb
from scipy.signal import detrend
from scipy.stats import zscore
from scipy.ndimage import binary_dilation
from nilearn.image import math_img, resample_to_img, crop_img, clean_img, binarize_img
from nilearn.input_data import NiftiMasker, NiftiSpheresMasker, NiftiLabelsMasker
from nilearn import datasets

def create_brain_mask(data_root, img_func):
    """ Create MNI-152 template brain mask """
    print('Creating brain mask...')
    brain = os.path.join(data_root, 'derivatives', 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii')
    img_roi = math_img("img1", img1=brain)
    img_resampled = resample_to_img(img_roi, img_func)
    data_binary = np.array(img_resampled.get_fdata()>=0.25, dtype=np.int8)
    data_dilated = binary_dilation(data_binary, iterations=2).astype(np.int8)
    img_mask = nb.Nifti1Image(data_dilated, img_resampled.affine, img_resampled.header)
    img_mask.set_data_dtype('i1')
    img_mask.to_filename(os.path.join(data_root, 'derivatives', 'mni_icbm152_t1_tal_nlin_asym_09c_mask_dilate_resample_crop.nii.gz'))
    print('Brain mask created.')

def get_mask(data_root):
    """ Load MNI-152 template brain mask """
    print('Loading brain mask...')
    mask_file = os.path.join(data_root, 'derivatives', 'mni_icbm152_t1_tal_nlin_asym_09c_mask_dilate_resample_crop.nii.gz')
    img_mask = nb.load(mask_file)
    print('Brain mask loaded.')
    return img_mask

def clean_func_image(fmriprep_dir, output_func_dir, img_mask, subject, run, overwrite=False):

    func_clean_path = os.path.join(output_func_dir, 
                    f'sub-{subject}_ses-01_task-02a_run-{run}_cleaned.nii.gz')
    
    if os.path.exists(func_clean_path) and overwrite is False:
        print(f'Functional image already cleaned for subject {subject}, run {run}.')
        return nb.load(func_clean_path)

    print(f'Cleaning functional image for subject {subject}, run {run}...')

    in_file = os.path.join(fmriprep_dir, f'sub-{subject}', 'ses-01',
                        'func', f'sub-{subject}_ses-01_task-02a_run-{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')

    # # Load functional image
    # img_func = nb.load(in_file)

    # # Detrend and zscore data and save it under a new NIfTI file
    # data = img_func.get_fdata()
    # data = detrend(data)
    # data = np.nan_to_num(zscore(data, axis=0))
    # #Todo: inspect this data in the auditory cortex for all subs
    # img_standardized = nb.Nifti1Image(data, img_func.affine, img_func.header)

    # # Multiply functional image with mask and crop image
    # img_cleaned = math_img('img1 * img2',
    #                     img1=img_standardized, img2=img_mask.slicer[..., None])
    # img_crop = crop_img(img_cleaned)

    confounds_file = os.path.join(fmriprep_dir, f'sub-{subject}', 'ses-01',
                        'func', f'sub-{subject}_ses-01_task-02a_run-{run}_desc-confounds_timeseries.tsv')
    
    confounds = pd.read_csv(confounds_file, sep='\t')

    confounds = confounds.filter(regex='csf|trans|rot').copy()
    confounds.drop('csf_wm', axis=1, inplace=True)
    confounds.fillna(0, inplace=True)

    img_clean = clean_img(in_file,
                          detrend=True,
                          standardize=True,
                          confounds=confounds,
                          high_pass=0.007,
                          t_r=1,
                          mask_img=img_mask)
    
    # Save cleaned image
    print(f'Saving cleaned image for subject {subject}, run {run}...')
    img_clean.to_filename(func_clean_path)

    print(f'Functional image cleaned for subject {subject}, run {run}.')

    return img_clean

def extract_samples(img_crop, img_mask, subject, run):

    print(f'Extracting samples for subject {subject}, run {run}...')

    masker = NiftiMasker(mask_img=img_mask, standardize=False, detrend=False)
    samples = masker.fit_transform(img_crop)

    # transform img_crop to 2D array (volumes x voxels)
    # samples = brain_data_masked.reshape(img_crop.shape[-1], -1)

    # save samples
    # np.save(os.path.join(output_func_dir, f'sub-{subject}_ses-01_task-02a_run-{run}_samples.npy'), samples)
    print(f'Samples extracted for subject {subject}, run {run}.')
    return samples

def extract_samples_with_atlas(img_crop, atlas_name, subject, run):

    print(f'Extracting samples from atlas {atlas_name} for subject {subject}, run {run}...')

    # Load atlas
    if atlas_name == 'pauli':
        atlas = datasets.fetch_atlas_pauli_2017(version='det')
        masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=False, detrend=False)

    elif atlas_name == 'power':
        atlas = datasets.fetch_coords_power_2011(legacy_format=False)
        coords = np.vstack((atlas.rois["x"], atlas.rois["y"], atlas.rois["z"])).T
        masker = NiftiSpheresMasker(seeds=coords, radius=5.0, 
                                    standardize=False, detrend=False)
    elif atlas_name == 'koelsch':
        atlas_path = os.path.join(os.getcwd(),'data','koelsch','Meta_analysis_C05_1k_clust_MNI.nii.gz')
        masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=False, detrend=False)
    
    elif atlas_name == 'koelsch_spheres':
        atlas_path = os.path.join(os.getcwd(),'data','koelsch','spheres','koelsch_spheres_atlas.nii.gz')
        masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=False, detrend=False)

    elif atlas_name == 'koelsch_spheres_per_voxel':
        #atlas_path = os.path.join(os.getcwd(),'data','koelsch','spheres','koelsch_spheres_atlas.nii.gz')
        atlas_path = '/Users/alexandresayal/GitHub/brainplayback_task02/data/koelsch/spheres/koelsch_spheres_atlas.nii.gz'
        atlas_resample = resample_to_img(atlas_path, img_crop, interpolation='nearest')
        atlas_resample_bin = binarize_img(atlas_resample)

        masker = NiftiMasker(mask_img=atlas_resample_bin, standardize=False, detrend=False)
    
    samples = masker.fit_transform(img_crop)

    print(f'Samples extracted for subject {subject}, run {run}.')
    return samples

def edit_events(root_dir, subject, run):
    """Edit events file to remove intersong trials and rename trial types."""

    print(f'Editing events for subject {subject}, run {run}...')
    events = pd.read_csv(
                    os.path.join(root_dir,f'sub-{subject}','ses-01','func',
                        f'sub-{subject}' + '_ses-01_task-02a_run-' + run + '_events.tsv'),
                    sep='\t')

    # round onset and duration to integer
    events['onset'] = events['onset'].round(0).astype(int)
    events['duration'] = events['duration'].round(0).astype(int)

    # remove first and last row - first and last noise trials
    events = events.iloc[1:-1]

    # remove all 'Noise_Intersong' trials
    events = events[events.trial_type != 'Noise_Intersong']

    # rename all trial_types except 'Noise' to 'Music'
    events.loc[events['trial_type'] != 'Noise', 'trial_type'] = 'Music'

    # add the hemodynamic delay of 4 volumes to all onsets
    events.loc[:, 'onset'] = events.loc[:, 'onset'] + 4

    # let's split the music trials into 4 segments of 6 seconds each
    # and the noise trials into 3 segments of 6 seconds each
    # we will use the onsets and durations from the events file to do this

    # create a new dataframe to store the new events
    new_events = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])

    # loop through the events and split the trials
    for _, row in events.iterrows():
        num_segments = 4 if row['trial_type'] == 'Music' else 3
        for i in range(num_segments):
            new_events = pd.concat([new_events, pd.DataFrame({'onset': row['onset'] + i*6,
                                                              'duration': 6,
                                                              'trial_type': row['trial_type']}, index=[0])], ignore_index=True)

    print(f'Events edited for subject {subject}, run {run}.')
    return new_events

def convert_samples_to_features(samples, data_root, output_func_dir, subject, run):

    print(f'Converting samples to features for subject {subject}, run {run}...')

    # Load events file
    events_split = edit_events(data_root, subject, run)

    # Initialize features numpy array to store the mean of the samples in each segment
    features = np.zeros((len(events_split), samples.shape[1]))

    # Calculate the mean of the sample in each segment from events_split
    for i, row in events_split.iterrows():
        features[i,:] = np.mean(samples[row['onset']:row['onset']+row['duration'], :], axis=0)
    
    # save features
    np.save(os.path.join(output_func_dir, f'sub-{subject}_ses-01_task-02a_run-{run}_features.npy'), features)

    # save the labels ('trial_type' from events_split)
    labels = events_split['trial_type'].values
    np.save(os.path.join(output_func_dir, f'sub-{subject}_ses-01_task-02a_run-{run}_labels.npy'), labels)

    print(f'Features extracted and labels saved for subject {subject}, run {run}.')
    #return features
