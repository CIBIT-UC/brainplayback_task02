# %% [markdown]
# # GLM analysis on the main tasks
# It uses nilearn and performs the following steps:
# 1. Load the data from fmriPrep in BIDS format
# 2. Iterate on the subjects to:
#    1. Select the predictors and confounds for the design matrix
#    2. Generate 1st level model
#    3. Estimate contrast maps
# 3. Generate group level maps
# 4. Generate hMT+ mask

# %%
# Imports
import os
from nilearn.glm.first_level import first_level_from_bids
from nilearn.glm import threshold_stats_img
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn.reporting import get_clusters_table
from mni_to_atlas import AtlasBrowser
atlas = AtlasBrowser("AAL3")

# %%
# Settings
data_dir = '/users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/'
space_label = "MNI152NLin2009cAsym"
derivatives_folder = "derivatives/fmriprep23"
task_label = "02a"
#smoothing_fwhm = 4.0
high_pass_hz = 0.007
out_dir = os.path.join(data_dir,"derivatives","nilearn_glm")

# %%
# import first level data automatically from fmriPrep derivatives
(
    models,
    models_run_imgs,
    models_events,
    models_confounds,
) = first_level_from_bids(
    data_dir,
    task_label,
    space_label,
    hrf_model="spm",
    noise_model="ar2",
    #smoothing_fwhm=smoothing_fwhm,
    high_pass=high_pass_hz,
    drift_model='cosine',
    slice_time_ref=None,
    n_jobs=12,
    minimize_memory=True,
    derivatives_folder=derivatives_folder,
    #sub_labels=['16','17'], # !!
)

# %%
# condition names
condition_names = ['JoyfulActivation', 'Nostalgia', 'Peacefulness', 'Power', 'Sadness', 'Tenderness', 'Tension', 'Transcendence', 'Wonder']

# create strings for contrasts in the format of "condition_name - Noise"
contrasts = []

# add contrast all conditions vs. noise
contrasts.append("Peacefulness + Tenderness + Transcendence + Nostalgia + Power + JoyfulActivation + Tension + Sadness + Wonder - Noise*9")

# iterate to add the other contrasts
for condition in condition_names:
    contrasts.append(condition + " - Noise")

# add more contrasts based on the grouping variables Sublimity, Vitality, and Unease
contrasts.append("Wonder + Transcendence + Tenderness + Nostalgia + Peacefulness - Noise*5") # sublimity
contrasts.append("Power + JoyfulActivation - Noise*2") # vitality
contrasts.append("Tension + Sadness - Noise*2") # unease

# and between the grouping variables
contrasts.append("Wonder*0.2 + Transcendence*0.2 + Tenderness*0.2 + Nostalgia*0.2 + Peacefulness*0.2 - Power*0.5 - JoyfulActivation*0.5") # sublimity vs. vitality
contrasts.append("Power + JoyfulActivation - Tension - Sadness") # vitality vs. unease
contrasts.append("Tension*0.5 + Sadness*0.5 - Wonder*0.2 - Transcendence*0.2 - Tenderness*0.2 - Nostalgia*0.2 - Peacefulness*0.2") # unease vs. sublimity

contrasts

# %%
# Rename the contrasts list to remove white spaces and replace '-'

contrasts_renamed = ['All','JoyfulActivation', 'Nostalgia', 'Peacefulness', 'Power', 'Sadness', 'Tenderness', 'Tension', 'Transcendence', 'Wonder',
                     'Sublimity', 'Vitality', 'Unease', 'SublimityMinusVitality', 'VitalityMinusUnease', 'UneaseMinusSublimity']

contrasts_renamed

# %% Define function
def glm_function(model, imgs, events, confounds, contrasts, contrasts_renamed, out_dir):

    subject = f"sub-{model.subject_label}"

    print(f"Computing 1st level model for subject: {subject}")

    # trim confounds and replace NaNs with 0
    confounds = confounds.filter(regex='csf|trans|rot')
    confounds = confounds.fillna(0)
    
    # Fit and contrasts
    model.fit(imgs, events, confounds)

    # create and save z_map, t_map, and beta_map to nifti files for every contrast
    for ii in range(len(contrasts)):

        z_map = model.compute_contrast(contrasts[ii], output_type="z_score")
        #t_map = model.compute_contrast(contrasts[ii], output_type="stat")
        beta_map = model.compute_contrast(contrasts[ii], output_type="effect_size")

        z_map.to_filename(os.path.join(out_dir, f"{subject}_task-{task_label}_stat-z_con-{contrasts_renamed[ii]}.nii.gz"))
        #t_map.to_filename(os.path.join(out_dir, f"{subject}_task-{task_label}_stat-t_con-{contrasts_renamed[ii]}.nii.gz"))
        beta_map.to_filename(os.path.join(out_dir, f"{subject}_task-{task_label}_stat-beta_con-{contrasts_renamed[ii]}.nii.gz"))

        # create figure with thresholded map for fun
        clean_map, threshold = threshold_stats_img(
            z_map, alpha=0.05, height_control="fdr", cluster_threshold=25
        )

        plotting.plot_glass_brain(
            clean_map,
            colorbar=True,
            threshold=threshold,
            plot_abs=False,
            display_mode="ortho",
            figure=plt.figure(figsize=(10, 4)),
        )

        plt.savefig(os.path.join(
            out_dir,
            f"{subject}_task-{task_label}_plot-z_con-{contrasts_renamed[ii]}_c-fdr-0.05_clusterk-25.png"
            )
        )

        # cluster table
        table = get_clusters_table(z_map, threshold, 25)

        # AAL3 labelling
        coordinates = table[['X','Y','Z']].to_numpy()
        aal_labels = atlas.find_regions(coordinates, plot=False)
        table['AAL3'] = aal_labels

        # save table to tsv
        table.to_csv(os.path.join(
            out_dir,
            f"{subject}_task-{task_label}_table-clusters_con-{contrasts_renamed[ii]}_c-fdr-0.05_clusterk-25.tsv"
            ),
            sep='\t'
        )

# %%
# ## 2. Iterate on the subjects

for idx in range(len(models)):
    glm_function(models[idx], models_run_imgs[idx], models_events[idx], models_confounds[idx], contrasts, contrasts_renamed, out_dir)
