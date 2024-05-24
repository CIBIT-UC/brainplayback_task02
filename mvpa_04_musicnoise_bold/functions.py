"""Functions for data extraction in music vs. noise with BOLD signal."""

#%% Imports
import os
import pandas as pd
import numpy as np
from nilearn.image import clean_img, mean_img, concat_imgs

def boldmeansegments_musicnoise(combinations):
    """Extract BOLD signal for music and noise segments."""
    root_dir, output_dir, subj, task, run = combinations[0], combinations[1], combinations[2], combinations[3], combinations[4]
    
    print('Starting ' + subj + ' run ' + run)

    #%% Settings

    # define paths
    func_dir = os.path.join(root_dir, 'derivatives','fmriprep23', subj, 'ses-01', 'func')

    func_file = os.path.join(func_dir, subj + '_ses-01_task-' + task + '_run-' + run +
                             '_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')

    # Load events file
    events = edit_events(root_dir, subj, task, run)

    # %% Regress out the confounds from the data using nilearn
    cleaned_img = clean_img_after_check(root_dir, func_file, func_dir, subj, task, run)

    # %% let's split the music trials into 4 segments of 6 seconds each
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

    # %% Calculate the mean of the images in each segment
    bold_values = [mean_img(cleaned_img.slicer[..., row['onset']:row['onset']+row['duration']]) for _, row in new_events.iterrows()]

    # %% Concatenate into a single 4D image
    img_bold = concat_imgs(bold_values)

    # %% Labels for the segments
    labels = new_events['trial_type'].values

    # %% Save the 4D image and labels
    newimage = os.path.join(output_dir, subj + '_ses-01_task-' + task +
                            '_run-' + run + '_musicnoise_bold_dataset.nii.gz')

    check_file_delete(newimage)

    img_bold.to_filename(newimage)

    # save labels
    labels_file = os.path.join(output_dir, subj + '_ses-01_task-' + task +
                               '_run-' + run + '_musicnoise_bold_labels.txt')

    check_file_delete(labels_file)

    np.savetxt(labels_file, labels, fmt='%s')

    print('Done with ' + subj + ' run ' + run)

def check_file_delete(file):
    """Check if file exists and delete it."""
    if os.path.exists(file):
        os.remove(file)
        print('Deleted existing file ' + file)

def edit_events(root_dir, subj, task, run):
    """Edit events file to remove intersong trials and rename trial types."""
    events = pd.read_csv(
                    os.path.join(root_dir,subj,'ses-01','func',
                        subj + '_ses-01_task-' + task + '_run-' + run + '_events.tsv'),
                    sep='\t')

    # round onset and duration to integer
    events['onset'] = events['onset'].round(0).astype(int)
    events['duration'] = events['duration'].round(0).astype(int)

    # remove first and last row - first and last noise trials
    events = events.iloc[1:-1]

    # Identify all Noise trials which duration is 6 seconds
    intersong_trials = events.query("trial_type == 'Noise' and duration > 5.5 and duration < 6.5")

    # rename noise_trials to 'intersong'
    events.loc[intersong_trials.index, "trial_type"] = "Intersong"

    # remove all 'intersong' trials
    events = events[events.trial_type != 'Intersong']

    # rename all trial_types except 'Noise' to 'Music'
    events.loc[events['trial_type'] != 'Noise', 'trial_type'] = 'Music'
    return events

def clean_img_after_check(root_dir, func_file, func_dir, subj, task, run):
    """Clean image and save it if it does not exist."""

    cleaned_file = os.path.join(root_dir, 'derivatives', 'func_clean',
                            subj + '_ses-01_task-' + task +
                            '_run-' + run + '_musicnoise_bold_cleaned.nii.gz')

    if os.path.exists(cleaned_file):
        print('Found existing cleaned image file')
        cleaned_img = cleaned_file
    else:
        print('Cleaning image')

        # fetch confounds to clean image
        confounds = pd.read_csv(
                        os.path.join(func_dir, subj + '_ses-01_task-' + task + '_run-' + run +
                            '_desc-confounds_timeseries.tsv')
                        , sep='\t')

        # only consider confounds that start with 'trans', 'rot', 'csf', 'cosine'
        confounds = confounds.filter(regex='trans|rot|csf|cosine')

        confounds = confounds.fillna(0)
        
        # brain mask
        mask_brain_file = os.path.join(root_dir, 'derivatives', 'mni_icbm152_t1_tal_nlin_asym_09c_res-2_dilated.nii')

        # clean the image
        cleaned_img = clean_img(func_file, confounds=confounds,
                                t_r=1, standardize=True,
                                mask_img=mask_brain_file)

        # save cleaned image
        cleaned_img.to_filename(cleaned_file)

    return cleaned_img
