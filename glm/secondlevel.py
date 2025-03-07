# %%
# Imports
import glob
import os

import pandas as pd
from joblib import Parallel, delayed
from mni_to_atlas import AtlasBrowser
from nilearn.glm.second_level import SecondLevelModel

atlas = AtlasBrowser("AAL3")

# %%
# Settings
# data_dir = '/users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/'
# data_dir = "/Volumes/T7/BIDS-BRAINPLAYBACK-TASK2"
data_dir = "/DATAPOOL/BRAINPLAYBACK/BIDS-BRAINPLAYBACK-TASK2"  # sim01 dir
out_dir = os.path.join(data_dir, "derivatives", "nilearn_glm")
out_dir_group = os.path.join(data_dir, "derivatives", "nilearn_glm", "group")

# %%
contrasts_renamed = [
    "All",
    "JoyfulActivation",
    "Nostalgia",
    "Peacefulness",
    "Power",
    "Sadness",
    "Tenderness",
    "Tension",
    "Transcendence",
    "Wonder",
    "Sublimity",
    "Vitality",
    "Unease",
    "SublimityMinusVitality",
    "VitalityMinusUnease",
    "UneaseMinusSublimity",
]

# %%
# check folder
if not os.path.exists(out_dir_group):
    os.makedirs(out_dir_group)


# %%
# define function to run the second-level analysis
def secondLevel(contrast_name):
    print(f"Running 2nd level for contrast {contrast_name}")

    # List all zmap nii.gz files
    zmap_files = glob.glob(
        os.path.join(out_dir, f"sub-*_task-02a_stat-z_con-{contrast_name}.nii.gz")
    )
    zmap_files.sort()

    # print number of zmap files
    print(f"Number of zmap files: {len(zmap_files)}")

    # create design matrix for 2nd level
    design_matrix_g = pd.DataFrame(
        [1] * len(zmap_files),
        columns=["intercept"],
    )

    # define 2nd level model
    second_level_model = SecondLevelModel(smoothing_fwhm=4, n_jobs=4)

    second_level_model = second_level_model.fit(
        zmap_files,
        design_matrix=design_matrix_g,
    )

    # compute contrast (z score map)
    z_map_g = second_level_model.compute_contrast(
        second_level_contrast="intercept",
        output_type="z_score",
    )

    # save group map
    z_map_g.to_filename(
        os.path.join(out_dir_group, f"group_task-02a_stat-z_con-{contrast_name}.nii.gz")
    )

    print(f"Finished 2nd level for contrast {contrast_name}")


# %%
# Iterate on the contrasts in parallel
Parallel(n_jobs=6)(delayed(secondLevel)(contrast) for contrast in contrasts_renamed)
