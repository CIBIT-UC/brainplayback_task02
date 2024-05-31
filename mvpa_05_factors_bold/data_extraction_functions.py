import os
import numpy as np
import pandas as pd
import nibabel as nb
from scipy.signal import detrend
from scipy.stats import zscore
from scipy.ndimage import binary_dilation
from nilearn.image import math_img, resample_to_img, crop_img
from nilearn.input_data import NiftiMasker, NiftiSpheresMasker, NiftiLabelsMasker
from nilearn import datasets

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

    # Load functional image
    img_func = nb.load(in_file)

    # Detrend and zscore data and save it under a new NIfTI file
    data = img_func.get_fdata()
    data = detrend(data)
    data = np.nan_to_num(zscore(data, axis=0))
    img_standardized = nb.Nifti1Image(data, img_func.affine, img_func.header)

    # Multiply functional image with mask and crop image
    img_cleaned = math_img('img1 * img2',
                        img1=img_standardized, img2=img_mask.slicer[..., None])
    img_crop = crop_img(img_cleaned)
    
    # Save cleaned image
    print(f'Saving cleaned image for subject {subject}, run {run}...')
    img_crop.to_filename(func_clean_path)

    print(f'Functional image cleaned for subject {subject}, run {run}.')

    return img_crop

def extract_samples_with_atlas(img_crop, output_samples_dir, atlas_name, subject, run):
    """Extract samples from the functional image using a specific atlas."""
    sample_file_path = os.path.join(output_samples_dir, 
                                    f'sub-{subject}_ses-01_task-02a_run-{run}_atlas-{atlas_name}_samples.npy')

    if os.path.exists(sample_file_path):
        print(f'Samples already extracted for subject {subject}, run {run}, atlas {atlas_name}.')
        return np.load(sample_file_path)
    
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
        
    samples = masker.fit_transform(img_crop)

    # save samples
    np.save(sample_file_path, samples)

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

    # remove all 'Noise' trials
    events = events[events.trial_type != 'Noise']

    # rename 'Wonder', 'Transcendence', 'Tenderness', 'Nostalgia' and 'Peacefulness' to 'Sublimity'
    events['trial_type'] = np.where(events['trial_type'].isin(['Wonder', 'Transcendence', 'Tenderness', 'Nostalgia', 'Peacefulness']), 'Sublimity', events['trial_type'])

    # rename 'Power' and 'JoyfulActivation' to 'Vitality'
    events['trial_type'] = np.where(events['trial_type'].isin(['Power', 'JoyfulActivation']), 'Vitality', events['trial_type'])

    # rename 'Tension' and 'Sadness' to 'Unease'
    events['trial_type'] = np.where(events['trial_type'].isin(['Tension', 'Sadness']), 'Unease', events['trial_type'])

    # let's split the music trials into 4 segments of 6 seconds each
    # and the noise trials into 3 segments of 6 seconds each
    # we will use the onsets and durations from the events file to do this

    # create a new dataframe to store the new events
    new_events = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])

    # loop through the events and split the trials
    num_segments = 4
    for _, row in events.iterrows():
        for i in range(1, num_segments): # excluding first segment
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
