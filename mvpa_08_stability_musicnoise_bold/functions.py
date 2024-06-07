import os
import pandas as pd
import nibabel as nb
import numpy as np
import itertools
from nibabel import Nifti1Image
from nilearn.image import clean_img

def get_brain_mask(data_root):
    """ Load MNI-152 template brain mask """
    print('Loading brain mask...')
    mask_file = os.path.join(data_root, 'derivatives', 'mni_icbm152_t1_tal_nlin_asym_09c_mask_dilate_resample_crop.nii.gz')
    img_mask = nb.load(mask_file)
    print('Brain mask loaded.')
    return img_mask

def clean_func_image(fmriprep_dir: str, output_func_dir: str, img_mask: str, subject: str, run: str, overwrite: bool = False) -> Nifti1Image:

    """
    Receives a single fmriprep preprocessed functional image and cleans it using nilearn.image.clean_img().
    It detrends, standardizes, high-pass filters, and removes confounds.

    Args:
        fmriprep_dir (str): Directory where the fmriprep preprocessed data is stored.
        output_func_dir (str): Directory where the cleaned functional image will be saved.
        img_mask (str): Path to the mask image to be used in cleaning.
        subject (str): Subject identifier.
        run (str): Run identifier.
        overwrite (bool): If True, overwrite existing cleaned image. Default is False.

    Returns:
        nibabel.Nifti1Image: The cleaned functional image.

    Raises:
        FileNotFoundError: If the required input files are not found.
        ValueError: If confounds DataFrame is empty or columns to be dropped are missing.
    """

    # Define the output path for the cleaned functional image
    func_clean_path = os.path.join(output_func_dir, 
                    f'sub-{subject}_ses-01_task-02a_run-{run}_cleaned.nii.gz')
    
    # Check if the cleaned image already exists and overwrite is set to False
    if os.path.exists(func_clean_path) and not overwrite:
        print(f'Functional image already cleaned for subject {subject}, run {run}.')
        return nb.load(func_clean_path)

    print(f'Cleaning functional image for subject {subject}, run {run}...')

    # Define the input path for the preprocessed functional image
    in_file = os.path.join(fmriprep_dir, f'sub-{subject}', 'ses-01',
                        'func', f'sub-{subject}_ses-01_task-02a_run-{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')

    # Raise an error if the input functional image does not exist
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"Functional image file not found: {in_file}")

    # Define the path for the confounds file
    confounds_file = os.path.join(fmriprep_dir, f'sub-{subject}', 'ses-01',
                        'func', f'sub-{subject}_ses-01_task-02a_run-{run}_desc-confounds_timeseries.tsv')
    
    # Raise an error if the confounds file does not exist
    if not os.path.exists(confounds_file):
        raise FileNotFoundError(f"Confounds file not found: {confounds_file}")

    # Load the confounds file into a DataFrame
    confounds = pd.read_csv(confounds_file, sep='\t')

    # Raise an error if the confounds DataFrame is empty
    if confounds.empty:
        raise ValueError("Confounds DataFrame is empty.")

    # Drop 'csf_wm' column if it exists
    if 'csf_wm' in confounds.columns:
        confounds.drop('csf_wm', axis=1, inplace=True)
    
    # Filter the confounds to include only 'csf', 'trans', and 'rot' related columns, and copy the DataFrame to avoid SettingWithCopyWarning
    confounds = confounds.filter(regex='csf|trans|rot').copy()
    
    # Fill any NaN values in the confounds DataFrame with 0
    confounds.fillna(0, inplace=True)

    # Clean the functional image using the provided mask, detrend, standardize, high-pass filter, and confounds
    img_clean = clean_img(in_file,
                          detrend=True,
                          standardize=True,
                          confounds=confounds,
                          high_pass=0.007,
                          t_r=1,
                          mask_img=img_mask)
    
    # Save the cleaned image to the specified output path
    print(f'Saving cleaned image for subject {subject}, run {run}...')
    img_clean.to_filename(func_clean_path)

    print(f'Functional image cleaned for subject {subject}, run {run}.')

    return img_clean

def edit_events_musicnoise_stability(root_dir, subject, run):
    """Edit events file."""

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
    events = events[events.trial_type != 'Noise_InterSong']

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
            t_name = f"{row['trial_type']}{i}"
            new_events = pd.concat([new_events, pd.DataFrame({'onset': row['onset'] + i*6,
                                                              'duration': 6,
                                                              'trial_type': t_name}, index=[0])], ignore_index=True)

    print(f'Events edited for subject {subject}, run {run}.')
    return new_events

def extract_features_for_stab(img_clean, events, output_feat_stab_dir, subject, run):

    print(f"Extracting features for sub {subject} run {run}...")

    # fetch unique conditions
    stab_cond_list = np.unique(events['trial_type'])
    n_stab_cond = len(stab_cond_list)

    stab_trial_counts = events["trial_type"].value_counts()
    max_stab_trial_counts = np.max(stab_trial_counts)

    # get functional data
    img_data = img_clean.get_fdata()

    # initialize outputs 
    FEAT_STAB = np.zeros((img_data.shape[0],img_data.shape[1],img_data.shape[2],n_stab_cond,max_stab_trial_counts))

    # iterate on the condition list
    for jj in range(len(stab_cond_list)):

        current_cond = stab_cond_list[jj]
        new_a = events[events['trial_type'] == current_cond].reset_index() # grab all events of that condition
        auxImg = np.zeros((img_data.shape[0],img_data.shape[1],img_data.shape[2],len(new_a))) # matrix to save features of current condition

        for zz in range(len(new_a)): #iterate on the trials
            auxImg[...,zz] = np.mean(img_data[..., new_a['onset'][zz]:new_a['onset'][zz]+new_a['duration'][zz]])
        
        if len(new_a) != max_stab_trial_counts: # some conditions have less trials, so we replicate them
            auxImg = np.concatenate([auxImg,auxImg,auxImg], axis=-1)
        
        FEAT_STAB[:,:,:,jj,:] = auxImg[:,:,:,:max_stab_trial_counts]

    # export
    np.save(os.path.join(output_feat_stab_dir, f'sub-{subject}_ses-01_task-02a_run-{run}_stab_features.npy'), FEAT_STAB)

    print(f"Done exporting features for subject {subject} run {run}.")

def estimate_stability(feat_dir, output_stab_dir, subject):

    print(f"Estimating stability for subject {subject}...")
    # find the files
    stab_feat_files = [os.path.join(feat_dir, f) for f in os.listdir(feat_dir) if f.endswith('_stab_features.npy') & f.startswith(f'sub-{subject}')]
    stab_feat_files.sort()

    # concatenating the runs
    stab_feat = np.concatenate([np.load(f) for f in stab_feat_files], axis=-1)

    # Generate a list of indexes
    indexes = list(range(stab_feat.shape[-1]))

    # Generate all combinations of the indexes taken two at a time
    combinations = list(itertools.combinations(indexes, 2))
    n_combinations = len(combinations)

    # to save stability per voxel and condition
    STAB = np.zeros((stab_feat.shape[0],stab_feat.shape[1],stab_feat.shape[2]))

    for i in range(stab_feat.shape[0]): # iterate on x

        print(f"X coordinate {i}/{stab_feat.shape[0]}...")

        for j in range(stab_feat.shape[1]): # iterate on y

            for k in range(stab_feat.shape[2]): # interate on z

                C = np.zeros((n_combinations, 1)) # initialize matrix for pairwise correlations

                for p in range(n_combinations): # iterate on the combinations

                    C[p] = np.corrcoef(stab_feat[i, j, k, :, combinations[p][0]], stab_feat[i, j, k, :, combinations[p][1]])[0,1]

                if np.isnan(C).any():
                    STAB[i, j, k] = 0
                else:
                    STAB[i, j, k] = np.mean(C)

    # save STAB
    np.save(os.path.join(output_stab_dir, f'sub-{subject}_STAB.npy'), STAB)

    print(f"Done stimating stability for subject {subject}.")