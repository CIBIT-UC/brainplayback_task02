import itertools
import os
from multiprocessing import Pool

import nibabel as nb
import numpy as np
import pandas as pd
from nibabel import Nifti1Image
from nilearn.image import clean_img
from nilearn.input_data import NiftiMasker

confounds_of_interest = [
    "csf",
    "white_matter",
    "trans_x",
    "trans_x_derivative1",
    "trans_x_power2",
    "trans_x_derivative1_power2",
    "trans_y",
    "trans_y_derivative1",
    "trans_y_power2",
    "trans_y_derivative1_power2",
    "trans_z",
    "trans_z_derivative1",
    "trans_z_derivative1_power2",
    "trans_z_power2",
    "rot_x",
    "rot_x_derivative1",
    "rot_x_power2",
    "rot_x_derivative1_power2",
    "rot_y",
    "rot_y_derivative1",
    "rot_y_power2",
    "rot_y_derivative1_power2",
    "rot_z",
    "rot_z_derivative1",
    "rot_z_power2",
    "rot_z_derivative1_power2",
]


def clean_func_image(fmriprep_dir: str, output_func_dir: str, mask_img: str, subject: str, run: str, overwrite: bool = False) -> Nifti1Image:
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
    func_clean_path = os.path.join(output_func_dir, f"sub-{subject}_ses-01_task-02a_run-{run}_cleaned.nii.gz")

    # Check if the cleaned image already exists and overwrite is set to False
    if os.path.exists(func_clean_path) and not overwrite:
        print(f"Functional image already cleaned for subject {subject}, run {run}.")
        return nb.load(func_clean_path)

    print(f"Cleaning functional image for subject {subject}, run {run}...")

    # Define the input path for the preprocessed functional image
    in_file = os.path.join(
        fmriprep_dir,
        f"sub-{subject}",
        "ses-01",
        "func",
        f"sub-{subject}_ses-01_task-02a_run-{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
    )

    # Raise an error if the input functional image does not exist
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"Functional image file not found: {in_file}")

    # Define the path for the confounds file
    confounds_file = os.path.join(
        fmriprep_dir, f"sub-{subject}", "ses-01", "func", f"sub-{subject}_ses-01_task-02a_run-{run}_desc-confounds_timeseries.tsv"
    )

    # Raise an error if the confounds file does not exist
    if not os.path.exists(confounds_file):
        raise FileNotFoundError(f"Confounds file not found: {confounds_file}")

    # Load the confounds file into a DataFrame
    confounds = pd.read_csv(confounds_file, sep="\t")

    # Raise an error if the confounds DataFrame is empty
    if confounds.empty:
        raise ValueError("Confounds DataFrame is empty.")

    # Filter the confounds to include only 'csf', 'trans', and 'rot' related columns, and copy the DataFrame to avoid SettingWithCopyWarning
    confounds = confounds[confounds_of_interest].copy()

    # Fill any NaN values in the confounds DataFrame with 0
    confounds.fillna(0, inplace=True)

    # Clean the functional image using the provided mask, detrend, standardize, high-pass filter, and confounds
    img_clean = clean_img(in_file, detrend=True, standardize=True, confounds=confounds, high_pass=0.007, t_r=1, mask_img=mask_img)

    # Save the cleaned image to the specified output path
    print(f"Saving cleaned image for subject {subject}, run {run}...")
    img_clean.to_filename(func_clean_path)

    print(f"Functional image cleaned for subject {subject}, run {run}.")

    return img_clean


# ===========================================
# MUSIC VS NOISE EVENT HANDLING
# ===========================================


def edit_events_musicnoise(root_dir, subject, run):
    """Edit events file."""

    print(f"Editing events for subject {subject}, run {run}...")
    events = pd.read_csv(
        os.path.join(root_dir, f"sub-{subject}", "ses-01", "func", f"sub-{subject}" + "_ses-01_task-02a_run-" + run + "_events.tsv"), sep="\t"
    )

    # round onset and duration to integer
    events["onset"] = events["onset"].round(0).astype(int)
    events["duration"] = events["duration"].round(0).astype(int)

    # remove first and last row - first and last noise trials
    events = events.iloc[1:-1]

    # remove all 'Noise_Intersong' trials
    events = events[events.trial_type != "Noise_InterSong"]

    # rename all trial_types except 'Noise' to 'Music'
    events.loc[events["trial_type"] != "Noise", "trial_type"] = "Music"

    # add the hemodynamic delay of 4 volumes to all onsets
    events.loc[:, "onset"] = events.loc[:, "onset"] + 4

    # let's split the music trials into 4 segments of 6 seconds each
    # and the noise trials into 3 segments of 6 seconds each
    # we will use the onsets and durations from the events file to do this

    # create a new dataframe to store the new events
    new_events = pd.DataFrame(columns=["onset", "duration", "trial_type"])

    # loop through the events and split the trials
    for _, row in events.iterrows():
        num_segments = 4 if row["trial_type"] == "Music" else 3
        for i in range(num_segments):
            t_name = f"{row['trial_type']}"
            new_events = pd.concat(
                [new_events, pd.DataFrame({"onset": row["onset"] + i * 6, "duration": 6, "trial_type": t_name}, index=[0])], ignore_index=True
            )

    print(f"Events edited for subject {subject}, run {run}.")
    return new_events


def edit_events_musicnoise_stability(root_dir, subject, run):
    """Edit events file."""

    print(f"Editing events for subject {subject}, run {run}...")
    events = pd.read_csv(
        os.path.join(root_dir, f"sub-{subject}", "ses-01", "func", f"sub-{subject}" + "_ses-01_task-02a_run-" + run + "_events.tsv"), sep="\t"
    )

    # round onset and duration to integer
    events["onset"] = events["onset"].round(0).astype(int)
    events["duration"] = events["duration"].round(0).astype(int)

    # remove first and last row - first and last noise trials
    events = events.iloc[1:-1]

    # remove all 'Noise_Intersong' trials
    events = events[events.trial_type != "Noise_InterSong"]

    # rename all trial_types except 'Noise' to 'Music'
    events.loc[events["trial_type"] != "Noise", "trial_type"] = "Music"

    # add the hemodynamic delay of 4 volumes to all onsets
    events.loc[:, "onset"] = events.loc[:, "onset"] + 4

    # let's get the middle 12 seconds of the music and noise trials
    # we will use the onsets and durations from the events file to do this

    # create a new dataframe to store the new events
    new_events = pd.DataFrame(columns=["onset", "duration", "trial_type"])

    # loop through the events and split the trials
    for _, row in events.iterrows():
        num_segments = 1
        for i in range(num_segments):
            t_name = f"{row['trial_type']}"
            new_events = pd.concat(
                [new_events, pd.DataFrame({"onset": row["onset"] + i * 6, "duration": 12, "trial_type": t_name}, index=[0])], ignore_index=True
            )

    print(f"Events edited for subject {subject}, run {run}.")
    return new_events


# ===========================================
# 9 EMOTIONS EVENT HANDLING
# ===========================================


def edit_events_full_stability(root_dir, subject, run):
    """Edit events file."""

    print(f"Editing events for subject {subject}, run {run}...")
    events = pd.read_csv(
        os.path.join(root_dir, f"sub-{subject}", "ses-01", "func", f"sub-{subject}" + "_ses-01_task-02a_run-" + run + "_events.tsv"), sep="\t"
    )

    # round onset and duration to integer
    events["onset"] = events["onset"].round(0).astype(int)
    events["duration"] = events["duration"].round(0).astype(int)

    # remove first and last row - first and last noise trials
    events = events.iloc[1:-1]

    # remove all Noise trials
    events = events[events.trial_type != "Noise"]
    events = events[events.trial_type != "Noise_InterSong"]

    # add the hemodynamic delay of 4 volumes to all onsets
    events.loc[:, "onset"] = events.loc[:, "onset"] + 4

    # let's split the music trials into 2 segments of 6 seconds each, centered around the middle of the block #FIXME

    # create a new dataframe to store the new events
    new_events = pd.DataFrame(columns=["onset", "duration", "trial_type"])

    # loop through the events and split the trials
    for _, row in events.iterrows():
        t_name = f"{row['trial_type']}"
        new_events = pd.concat(
            [new_events, pd.DataFrame({"onset": row["onset"] + 6, "duration": 12, "trial_type": t_name}, index=[0])], ignore_index=True
        )
        # new_events = pd.concat(
        #     [new_events, pd.DataFrame({"onset": row["onset"] + 6 + 6, "duration": 6, "trial_type": t_name}, index=[0])], ignore_index=True
        # )

    print(f"Events edited for subject {subject}, run {run}.")
    return new_events


def edit_events_full(root_dir, subject, run):
    """Edit events file."""

    print(f"Editing events for subject {subject}, run {run}...")
    events = pd.read_csv(
        os.path.join(root_dir, f"sub-{subject}", "ses-01", "func", f"sub-{subject}" + "_ses-01_task-02a_run-" + run + "_events.tsv"), sep="\t"
    )

    # round onset and duration to integer
    events["onset"] = events["onset"].round(0).astype(int)
    events["duration"] = events["duration"].round(0).astype(int)

    # remove first and last row - first and last noise trials
    events = events.iloc[1:-1]

    # remove all 'Noise' and 'Noise_Intersong' trials
    events = events[events.trial_type != "Noise"]
    events = events[events.trial_type != "Noise_InterSong"]

    # add the hemodynamic delay of 4 volumes to all onsets
    events.loc[:, "onset"] = events.loc[:, "onset"] + 4

    # let's split the music trials into 4 segments of 6 seconds each

    # create a new dataframe to store the new events
    new_events = pd.DataFrame(columns=["onset", "duration", "trial_type"])

    # loop through the events and split the trials
    for _, row in events.iterrows():
        num_segments = 4
        for i in range(num_segments):
            t_name = f"{row['trial_type']}"
            new_events = pd.concat(
                [new_events, pd.DataFrame({"onset": row["onset"] + i * 6, "duration": 6, "trial_type": t_name}, index=[0])], ignore_index=True
            )

    print(f"Events edited for subject {subject}, run {run}.")
    return new_events


# ===========================================
# 3 FACTORS EVENT HANDLING
# ===========================================


def edit_events_factors(root_dir, subject, run):
    """Edit events file."""

    print(f"Editing events for subject {subject}, run {run}...")
    events = pd.read_csv(
        os.path.join(root_dir, f"sub-{subject}", "ses-01", "func", f"sub-{subject}" + "_ses-01_task-02a_run-" + run + "_events.tsv"), sep="\t"
    )

    # round onset and duration to integer
    events["onset"] = events["onset"].round(0).astype(int)
    events["duration"] = events["duration"].round(0).astype(int)

    # remove first and last row - first and last noise trials
    events = events.iloc[1:-1]

    # remove all 'Noise' and 'Noise_Intersong' trials
    events = events[events.trial_type != "Noise"]
    events = events[events.trial_type != "Noise_InterSong"]

    # rename according to the factors
    # rename 'Wonder', 'Transcendence', 'Tenderness', 'Nostalgia' and 'Peacefulness' to 'Sublimity'
    events["trial_type"] = np.where(
        events["trial_type"].isin(["Wonder", "Transcendence", "Tenderness", "Nostalgia", "Peacefulness"]), "Sublimity", events["trial_type"]
    )

    # rename 'Power' and 'JoyfulActivation' to 'Vitality'
    events["trial_type"] = np.where(events["trial_type"].isin(["Power", "JoyfulActivation"]), "Vitality", events["trial_type"])

    # rename 'Tension' and 'Sadness' to 'Unease'
    events["trial_type"] = np.where(events["trial_type"].isin(["Tension", "Sadness"]), "Unease", events["trial_type"])

    # add the hemodynamic delay of 4 volumes to all onsets
    events.loc[:, "onset"] = events.loc[:, "onset"] + 4

    # let's split the music trials into 4 segments of 6 seconds each
    # and the noise trials into 3 segments of 6 seconds each
    # we will use the onsets and durations from the events file to do this

    # create a new dataframe to store the new events
    new_events = pd.DataFrame(columns=["onset", "duration", "trial_type"])

    # loop through the events and split the trials
    for _, row in events.iterrows():
        num_segments = 4
        for i in range(num_segments):
            t_name = f"{row['trial_type']}"
            new_events = pd.concat(
                [new_events, pd.DataFrame({"onset": row["onset"] + i * 6, "duration": 6, "trial_type": t_name}, index=[0])], ignore_index=True
            )

    print(f"Events edited for subject {subject}, run {run}.")
    return new_events


def edit_events_factors_stability(root_dir, subject, run):
    """Edit events file."""

    print(f"Editing events for subject {subject}, run {run}...")
    events = pd.read_csv(
        os.path.join(root_dir, f"sub-{subject}", "ses-01", "func", f"sub-{subject}" + "_ses-01_task-02a_run-" + run + "_events.tsv"), sep="\t"
    )

    # round onset and duration to integer
    events["onset"] = events["onset"].round(0).astype(int)
    events["duration"] = events["duration"].round(0).astype(int)

    # remove first and last row - first and last noise trials
    events = events.iloc[1:-1]

    # remove all 'Noise' and 'Noise_Intersong' trials
    events = events[events.trial_type != "Noise"]
    events = events[events.trial_type != "Noise_InterSong"]

    # rename according to the factors
    # rename 'Wonder', 'Transcendence', 'Tenderness', 'Nostalgia' and 'Peacefulness' to 'Sublimity'
    events["trial_type"] = np.where(
        events["trial_type"].isin(["Wonder", "Transcendence", "Tenderness", "Nostalgia", "Peacefulness"]), "Sublimity", events["trial_type"]
    )

    # rename 'Power' and 'JoyfulActivation' to 'Vitality'
    events["trial_type"] = np.where(events["trial_type"].isin(["Power", "JoyfulActivation"]), "Vitality", events["trial_type"])

    # rename 'Tension' and 'Sadness' to 'Unease'
    events["trial_type"] = np.where(events["trial_type"].isin(["Tension", "Sadness"]), "Unease", events["trial_type"])

    # add the hemodynamic delay of 4 volumes to all onsets
    events.loc[:, "onset"] = events.loc[:, "onset"] + 4

    # let's get the middle 12 seconds of the music blocks
    # we will use the onsets and durations from the events file to do this

    # create a new dataframe to store the new events
    new_events = pd.DataFrame(columns=["onset", "duration", "trial_type"])

    # loop through the events and split the trials
    for _, row in events.iterrows():
        num_segments = 1
        for i in range(num_segments):
            t_name = f"{row['trial_type']}"
            new_events = pd.concat(
                [new_events, pd.DataFrame({"onset": row["onset"] + i * 6, "duration": 12, "trial_type": t_name}, index=[0])], ignore_index=True
            )

    print(f"Events edited for subject {subject}, run {run}.")
    return new_events


# ===========================================
# STABILITY AND OTHER
# ===========================================


def extract_features_for_stab(img_clean, events, output_feat_stab_dir, subject, run):
    """
    Extracts and saves features for stability mask from functional imaging data.

    Parameters:
    - img_clean (Nifti1Image): A Nifti image containing the cleaned functional data.
    - events (DataFrame): A pandas DataFrame containing event-related information with at least 'trial_type', 'onset', and 'duration' columns.
    - output_feat_stab_dir (str): Directory where the output feature file will be saved.
    - subject (str): Subject identifier.
    - run (str): Run identifier.

    The function processes the input data to extract features for different trial types and saves the results in a .npy file.
    """
    print(f"Extracting features for subject {subject} run {run}...")

    # Fetch unique conditions
    stab_cond_list = np.unique(events["trial_type"])
    n_stab_cond = len(stab_cond_list)
    print(f"{n_stab_cond} conditions found per run.")

    # Count the number of trials for each condition
    stab_trial_counts = events["trial_type"].value_counts()
    max_stab_trial_counts = np.max(stab_trial_counts)
    print(f"Maximum number of trials per condition per run: {max_stab_trial_counts}")

    # Get functional data
    img_data = img_clean.get_fdata()

    # Initialize output array
    FEAT_STAB = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2], n_stab_cond, max_stab_trial_counts))

    # Iterate over the condition list
    for jj, current_cond in enumerate(stab_cond_list):
        # Grab all events of that condition
        new_a = events[events["trial_type"] == current_cond].reset_index()

        # Initialize matrix to save features of current condition
        auxImg = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2], len(new_a)))

        # Iterate over the trials
        for zz in range(len(new_a)):
            onset = new_a["onset"][zz]
            duration = new_a["duration"][zz]
            auxImg[..., zz] = np.mean(img_data[..., onset : onset + duration], axis=-1)

        # Handle conditions with fewer trials #TODO why is this needed?
        if len(new_a) < max_stab_trial_counts:
            # Calculate how many times to repeat the array to match max_stab_trial_counts
            print(f"Condition {current_cond} has {len(new_a)} trials. Repeating to match {max_stab_trial_counts} trials.")
            repeats = int(np.ceil(max_stab_trial_counts / len(new_a)))
            auxImg = np.tile(auxImg, (1, 1, 1, repeats))

        # Ensure the shape matches before assignment # TODO why is this needed?
        FEAT_STAB[:, :, :, jj, :] = auxImg[:, :, :, :max_stab_trial_counts]

    # Export the result
    output_file = os.path.join(output_feat_stab_dir, f"sub-{subject}_ses-01_task-02a_run-{run}_stab_features.npy")
    np.save(output_file, FEAT_STAB)

    print(f"Done exporting features for subject {subject} run {run}.")


def process_voxel(i, stab_feat, combinations):
    """
    Process a single x-coordinate slice of the voxel grid to estimate stability.

    Parameters:
    - i (int): The x-coordinate to process.
    - stab_feat (ndarray): The concatenated feature data.
    - combinations (ndarray): Array of index combinations for correlation calculation.

    Returns:
    - (int, ndarray): The x-coordinate and the calculated stability values for that slice.
    """
    print(f"X coordinate {i}/{stab_feat.shape[0]}...")
    STAB_slice = np.zeros((stab_feat.shape[1], stab_feat.shape[2]))

    for j in range(stab_feat.shape[1]):  # iterate on y
        for k in range(stab_feat.shape[2]):  # iterate on z
            # Extract the time series data for the current voxel
            voxel_data = stab_feat[i, j, k, :, :]

            # Check if voxel_data is all zeros
            if np.all(voxel_data == 0):
                STAB_slice[j, k] = 0
                continue

            # Calculate correlations for all pairs in combinations
            C = np.array([np.corrcoef(voxel_data[:, comb[0]], voxel_data[:, comb[1]])[0, 1] for comb in combinations])

            # Handle NaN values
            if np.isnan(C).any():
                STAB_slice[j, k] = 0
            else:
                STAB_slice[j, k] = np.mean(C)

    return i, STAB_slice


def estimate_stability(feat_dir, output_stab_dir, subject, n_jobs: int = 10):
    """
    Estimate the stability of voxel time series data for a given subject.

    Parameters:
    - feat_dir (str): Directory containing the feature files.
    - output_stab_dir (str): Directory to save the output stability file.
    - subject (str): Subject identifier.
    """
    print(f"Estimating stability for subject {subject}...")

    # Find the files
    stab_feat_files = [os.path.join(feat_dir, f) for f in os.listdir(feat_dir) if f.endswith("_stab_features.npy") and f.startswith(f"sub-{subject}")]
    stab_feat_files.sort()

    # Concatenating the runs
    stab_feat = np.concatenate([np.load(f) for f in stab_feat_files], axis=-1)

    print(f"Number of trials per condition per subject: {stab_feat.shape[-1]}")

    # Generate a list of indexes
    indexes = np.arange(stab_feat.shape[-1])
    combinations = np.array(list(itertools.combinations(indexes, 2)))
    n_combinations = len(combinations)
    print(f"Number of pairwise combinations: {n_combinations}")

    # Initialize STAB array
    STAB = np.zeros((stab_feat.shape[0], stab_feat.shape[1], stab_feat.shape[2]))

    # Use multiprocessing to process each x-coordinate slice in parallel
    with Pool(processes=n_jobs) as pool:
        results = pool.starmap(process_voxel, [(i, stab_feat, combinations) for i in range(stab_feat.shape[0])])

    # Collect the results
    for i, STAB_slice in results:
        STAB[i, :, :] = STAB_slice

    # Save STAB
    np.save(os.path.join(output_stab_dir, f"sub-{subject}_STAB.npy"), STAB)

    print(f"Done estimating stability for subject {subject}.")


def extract_samples(img_crop, img_mask, subject, run):
    print(f"Extracting samples for subject {subject}, run {run}...")

    masker = NiftiMasker(mask_img=img_mask, standardize=False, detrend=False)
    samples = masker.fit_transform(img_crop)

    print(f"Samples extracted for subject {subject}, run {run}.")
    return samples


def convert_samples_to_features(samples, output_func_dir, events_split, subject, run):
    print(f"Converting samples to features for subject {subject}, run {run}...")

    # Initialize features numpy array to store the mean of the samples in each segment
    features = np.zeros((len(events_split), samples.shape[1]))

    # Calculate the mean of the sample in each segment from events_split
    for i, row in events_split.iterrows():
        features[i, :] = np.mean(samples[row["onset"] : row["onset"] + row["duration"], :], axis=0)

    # save features
    np.save(os.path.join(output_func_dir, f"sub-{subject}_ses-01_task-02a_run-{run}_features.npy"), features)

    # save the labels ('trial_type' from events_split)
    labels = events_split["trial_type"].values
    np.save(os.path.join(output_func_dir, f"sub-{subject}_ses-01_task-02a_run-{run}_labels.npy"), labels)

    print(f"Features extracted and labels saved for subject {subject}, run {run}.")
    # return features
