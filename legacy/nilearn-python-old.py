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
import glob
from nilearn.glm.first_level import first_level_from_bids
from nilearn.interfaces.bids import save_glm_to_bids
from nilearn.glm import threshold_stats_img
from nilearn import plotting
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nilearn.glm.second_level import SecondLevelModel
from nilearn.reporting import get_clusters_table
from nilearn.image import math_img
from nilearn.masking import apply_mask

# %%
# Settings
data_dir = '/users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/'
space_label = "MNI152NLin2009cAsym"
derivatives_folder = "derivatives/fmriprep23"
task_label = "02a"
smoothing_fwhm = 4.0
high_pass_hz = 0.007

# %% [markdown]
# ## 1. Load the data from fmriPrep in BIDS format

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
    smoothing_fwhm=smoothing_fwhm,
    high_pass=high_pass_hz,
    slice_time_ref=None,
    n_jobs=4,
    minimize_memory = False,
    derivatives_folder=derivatives_folder,
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


# %%
# create dictionary of contrasts in the format {contrast: "t"}
contrasts_dict = {}
for contrast in contrasts:
    contrasts_dict[contrast] = "t"

# %% [markdown]
# ## 2. Iterate on the subjects

# %%
for idx,model in enumerate(models):
    print(f"Model {model}: {idx}")

    subject = f"sub-{model.subject_label}"

    # trim confounds and replace NaNs with 0
    confounds = models_confounds[idx]

    for ii in range(len(confounds)):
        confounds[ii] = confounds[ii][['trans_x', 'trans_x_derivative1', 'trans_x_power2', 'trans_x_derivative1_power2',
                                            'trans_y', 'trans_y_derivative1', 'trans_y_power2', 'trans_y_derivative1_power2',
                                            'trans_z', 'trans_z_derivative1', 'trans_z_power2', 'trans_z_derivative1_power2',
                                            'rot_x', 'rot_x_derivative1', 'rot_x_power2', 'rot_x_derivative1_power2',
                                            'rot_y', 'rot_y_derivative1', 'rot_y_power2', 'rot_y_derivative1_power2',
                                            'rot_z', 'rot_z_derivative1', 'rot_z_power2', 'rot_z_derivative1_power2',
                                            ]]
    
        confounds[ii] = confounds[ii].fillna(0)
    
    # Fit and contrasts
    model.fit(models_run_imgs[idx], models_events[idx], confounds)

    # save model to disk
    save_glm_to_bids(
        model,
        contrasts=contrasts,
        contrast_types=contrasts_dict,
        out_dir=os.path.join(data_dir,"derivatives","nilearn_glm"),
        prefix=f"{subject}_task-{task_label}",
    )

    # create figure for all conditions vs. noise
    z_map = model.compute_contrast(contrasts[0])

    # create figure with thresholded map for fun
    clean_map, threshold = threshold_stats_img(
        z_map, alpha=0.05, height_control="bonferroni", cluster_threshold=50
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
        data_dir,'derivatives','nilearn_glm',
        f"{subject}_task-{task_label}_contrast-allMinusNoise_c-bonferroni_p-0.05_clusterk-50_plot.png"
        )
    )

    # Export cluster table
    table = get_clusters_table(z_map, threshold, 50)
    table.to_csv(os.path.join(data_dir,"derivatives","nilearn_glm",f"{subject}_task-{task_label}_contrast-allMinusNoise_c-bonferroni_p-0.05_clusterk-50_cluster-table.tsv"),sep='\t')

    # delete errorts file (very large and not useful)
    # check if file exists first
    if os.path.exists(os.path.join(data_dir,"derivatives","nilearn_glm",
                                    f"{subject}_task-{task_label}_stat-errorts_statmap.nii.gz")):
        
        os.remove(os.path.join(data_dir,"derivatives","nilearn_glm",
                            f"{subject}_task-{task_label}_stat-errorts_statmap.nii.gz"))
        print('deleted errorts file')
    
    # delete model
    

# %% [markdown]
# ## 3. Group level analysis

# %%
# List all tmap nii.gz files
tmap_files = glob.glob(
    os.path.join(data_dir,'derivatives','nilearn_glm',
        f"sub-*_task-{task_label}_contrast-allMinusNoise_stat-t_statmap.nii.gz"
    )
)
tmap_files.sort()

# List all zmap nii.gz files
zmap_files = glob.glob(
    os.path.join(data_dir,'derivatives','nilearn_glm',
        f"sub-*_task-{task_label}_contrast-allMinusNoise_stat-z_statmap.nii.gz"
    )
)
zmap_files.sort()

subject_list = [os.path.basename(f).split('_')[0] for f in tmap_files]

# %%
#| label: loc_singlesubject
# Plot all subjects
#subjects = data["ext_vars"]["participant_id"].tolist()

fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 10))
for cidx, tmap in enumerate(tmap_files):
    P = plotting.plot_glass_brain(
        tmap,
        colorbar=True,
        threshold=6.0,
        vmax=25,
        axes=axes[cidx % 5, int(cidx / 5)],
        plot_abs=False,
        display_mode="x",
    )
    P.title(subject_list[cidx], size=8)
fig.suptitle("subjects t_map")

plt.savefig(os.path.join(data_dir,"derivatives","nilearn_glm","group",
                         f"group_task-{task_label}_contrast-allMinusNoise_c-bonferroni_p-0.05_clusterk-50_persubject.png"))

# %%
# create design matrix for 2nd level
second_level_input = zmap_files
design_matrix_g = pd.DataFrame(
    [1] * len(second_level_input),
    columns=["intercept"],
)

# define 2nd level model
second_level_model = SecondLevelModel(smoothing_fwhm=6.0, n_jobs=4)
second_level_model.minimize_memory = False
second_level_model = second_level_model.fit(
    second_level_input,
    design_matrix=design_matrix_g,
)

# compute contrast (z score map)
z_map_g = second_level_model.compute_contrast(
    second_level_contrast="intercept",
    output_type="z_score",
)

# compute contrast (beta map)
beta_map_g = second_level_model.compute_contrast(
    second_level_contrast="intercept",
    output_type='effect_size',
)

# %%
# Threshold zmap and plot it
clean_map_g, threshold_g = threshold_stats_img(
    z_map_g, alpha=0.05, height_control="bonferroni", cluster_threshold=50
)

plotting.plot_glass_brain(
    clean_map_g,
    colorbar=True,
    threshold=threshold_g,
    plot_abs=False,
    display_mode="ortho",
    vmax=8,
    figure=plt.figure(figsize=(10, 4)),
    symmetric_cbar=False,
    cmap=nilearn_cmaps["cold_hot"],
)

plt.savefig(os.path.join(data_dir,"derivatives","nilearn_glm","group",
                         f"group_task-{task_label}_contrast-allMinusNoise_c-bonferroni_p-0.05_clusterk-50_plot.png"))

# %%
# Export cluster table
table,cluster_map_g = get_clusters_table(z_map_g, threshold_g+0.6, 50,
                                return_label_maps=True)

table.to_csv(os.path.join(data_dir,"derivatives","nilearn_glm","group",
                          f"group_task-{task_label}_contrast-allMinusNoise_c-bonferroni_p-0.05_clusterk-50_cluster-table.tsv"),sep='\t')

# %%
# Save group GLM
save_glm_to_bids(
    second_level_model,
    contrasts="intercept",
    contrast_types={"intercept": "t"},
    out_dir=os.path.join(data_dir,"derivatives","nilearn_glm","group"),
    prefix=f"group_task-{task_label}",
)
