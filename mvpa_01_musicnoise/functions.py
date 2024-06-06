import os
import pandas as pd
import numpy as np
from nilearn.image import concat_imgs
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

def ls_a_musicnoise_confounds(root_dir, output_dir, subj, task, run):

    print('Extracting data for subject ' + subj + ' and run ' + run)

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

    # rename all trial_types except 'Noise' to 'Music'
    events.loc[~events["trial_type"].str.contains("Noise"), "trial_type"] = "Music"

    # Add counter to each trial_type in the format '01'
    events['trial_type'] = events['trial_type'] + events.groupby('trial_type').cumcount().add(1).astype(str).str.zfill(2)

    trialwise_conditions = events["trial_type"].unique()
    trialwise_conditions = [condition for condition in trialwise_conditions if 'InterSong' not in condition]

    # fetch confounds to clean image
    confounds_file = os.path.join(fmriprep_dir,subj,'ses-01','func',
                               subj + '_ses-01_task-' + task + '_run-' + run + '_desc-confounds_timeseries.tsv')

    confounds = pd.read_csv(confounds_file, sep='\t')

    # only consider confounds that start with 'trans', 'rot', and 'csf'
    confounds = confounds.filter(regex='csf|trans|rot')

    confounds = confounds.fillna(0)

    # Design matrix
    print('Creating design matrix')
    lsa_design = make_first_level_design_matrix(np.arange(660),
                                                events,
                                                add_regs=confounds,
                                                drift_model='cosine',
                                                high_pass=0.007,
                                                hrf_model='spm')

    # GLM
    print('Creating GLM')
    lsa_glm = FirstLevelModel(t_r=1,
                              standardize=True,
                              signal_scaling=False,
                              minimize_memory=True,
                              n_jobs=2)

    print('Fitting GLM')
    lsa_glm.fit(func_file, design_matrices = lsa_design)

    # Estimate statistical maps for each condition of interest and trial
    z_maps_lsa = []

    for contrast in trialwise_conditions:
        z_map = lsa_glm.compute_contrast(contrast, output_type='z_score')

        z_maps_lsa.append(z_map)

    # 4D image with all the beta maps
    img_lsa = concat_imgs(z_maps_lsa)   

    # check if image exists - if so delete it
    newimage = os.path.join(output_dir, subj + '_ses-01_task-' + task + '_run-' + run + '_musicnoise_confounds_dataset.nii.gz')
    
    if os.path.exists(newimage):
        os.remove(newimage)
        print('Deleted existing image file')

    # save concatenated images
    img_lsa.to_filename(newimage)

    # check if trial_types file exists - if so delete it
    newtrialtypes = os.path.join(output_dir, subj + '_ses-01_task-' + task + '_run-' + run + '_musicnoise_confounds_trial_types.txt')
    
    if os.path.exists(newtrialtypes):
        os.remove(newtrialtypes)
        print('Deleted existing trial_types file')

    # save trial_types
    np.savetxt(newtrialtypes, trialwise_conditions, fmt='%s')

    print('Done for subject ' + subj + ' and run ' + run + '\n')