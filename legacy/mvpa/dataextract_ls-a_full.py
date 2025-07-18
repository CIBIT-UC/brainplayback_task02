import os
import nibabel as nb
import nibabel.processing as nbp
from joblib import Parallel, delayed
from de_functions import ls_a_full,ls_a_full_confounds

# define paths
root_dir = '/users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2'
fmriprep_dir = os.path.join(root_dir, 'derivatives/fmriprep23')
#output_dir = os.path.join(root_dir, 'derivatives/mvpa_ls_a_data')

# define subjects
subjectList = ['sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-07','sub-08','sub-09','sub-10','sub-11','sub-12','sub-13']
runList = ['1','2','3','4']

# create combinations of subjects and runs
combinations = [(subj, run) for subj in subjectList for run in runList]

# iterate on combinations in parallel
Parallel(n_jobs=13)(delayed(ls_a_full_confounds)(root_dir, subj, '02a', run) for subj, run in combinations)

print('All done!')
