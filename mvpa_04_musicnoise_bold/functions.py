#%% Imports
import pandas as pd
import os
import numpy as np
from nilearn.image import clean_img, mean_img, concat_imgs

def boldmeansegments_musicnoise(root_dir, output_dir, subj, task, run):
    #%% Settings

    # define paths
    fmriprep_dir = os.path.join(root_dir, 'derivatives','fmriprep23')

    # load data
    func_dir = os.path.join(fmriprep_dir, subj, 'ses-01', 'func')
    func_file = os.path.join(func_dir, subj + '_ses-01_task-' + task + '_run-' + run + '_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')

    # Load events file
    events_file = os.path.join(root_dir,subj,'ses-01','func',
                                subj + '_ses-01_task-' + task + '_run-' + run + '_events.tsv')
    events = pd.read_csv(events_file, sep='\t')

    # round onset and duration to integer
    events['onset'] = events['onset'].round(0).astype(int)
    events['duration'] = events['duration'].round(0).astype(int)

    # remove first row - first noise trial
    events = events[1:]

    # Identify all Noise trials which duration is 6 seconds
    intersong_trials = events.query("trial_type == 'Noise' and duration > 5.5 and duration < 6.5")

    # rename noise_trials to 'intersong'
    events.loc[intersong_trials.index, "trial_type"] = "Intersong"

    # remove all 'intersong' trials
    events = events[events.trial_type != 'Intersong']

    # rename all trial_types except 'Noise' to 'Music'
    events['trial_type'] = np.where(events['trial_type'] != 'Noise', 'Music', events['trial_type'])

    # %% Fetch confounds

    # fetch confounds to clean image
    confounds_file = os.path.join(fmriprep_dir,subj,'ses-01','func',
                            subj + '_ses-01_task-' + task + '_run-' + run + '_desc-confounds_timeseries.tsv')

    confounds = pd.read_csv(confounds_file, sep='\t')

    # only consider confounds that start with 'trans', 'rot', 'csf', 'cosine'
    confounds = confounds.filter(regex='trans|rot|csf|cosine')

    confounds = confounds.fillna(0)

    # %% Regress out the confounds from the data using nilearn
    cleaned_img = clean_img(func_file, confounds=confounds, t_r=1, standardize=True)

    # %% let's split the music trials into 4 segments of 6 seconds each and the noise trials into 3 segments of 6 seconds each
    # we will use the onsets and durations from the events file to do this

    # create a new dataframe to store the new events
    new_events = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])

    # loop through the events and split the trials
    for idx, row in events.iterrows():
        if row['trial_type'] == 'Music':
            for i in range(4):
                new_events = new_events.append({'onset': row['onset'] + i*6,
                                                'duration': 6,
                                                'trial_type': 'Music'},
                                            ignore_index=True)
        else:
            for i in range(3):
                new_events = new_events.append({'onset': row['onset'] + i*6,
                                                'duration': 6,
                                                'trial_type': 'Noise'},
                                            ignore_index=True)

    # %% Calculate the mean of the images in each segment
    bold_values = []

    for idx, row in new_events.iterrows():
        bold_values.append(
            mean_img(
                cleaned_img.slicer[..., row['onset']:row['onset']+row['duration']]
                )
            )

    # %% Concatenate into a single 4D image
    img_bold = concat_imgs(bold_values)

    # %% Labels for the segments
    labels = new_events['trial_type'].values

    # %% Save the 4D image and labels
    newimage = os.path.join(output_dir, subj + '_ses-01_task-' + task + '_run-' + run + '_musicnoise_bold_dataset.nii.gz')

    if os.path.exists(newimage):
        os.remove(newimage)
        print('Deleted existing image file')

    img_bold.to_filename(newimage)

    # save labels
    labels_file = os.path.join(output_dir, subj + '_ses-01_task-' + task + '_run-' + run + '_musicnoise_bold_labels.txt')

    if os.path.exists(labels_file):
        os.remove(labels_file)
        print('Deleted existing labels file')
        
    np.savetxt(labels_file, labels, fmt='%s')

    print('Done with ' + subj + ' ' + run)
