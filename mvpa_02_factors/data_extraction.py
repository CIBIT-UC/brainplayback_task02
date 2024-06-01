import os
import nibabel as nb
import nibabel.processing as nbp
from joblib import Parallel, delayed
from functions import ls_a_factors

# define paths
root_dir = '/Volumes/T7/BIDS-BRAINPLAYBACK-TASK2'
fmriprep_dir = os.path.join(root_dir, 'derivatives','fmriprep23')
output_dir = os.path.join(root_dir, 'derivatives','mvpa_02_factors')

# define subjects
#subjectList = ['sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-07','sub-08','sub-09','sub-10',
#               'sub-11','sub-12','sub-13']
subjectList = ['sub-14','sub-15', 'sub-16', 'sub-17']
runList = ['1','2','3','4']

# create combinations of subjects and runs
combinations = [(subj, run) for subj in subjectList for run in runList]

# iterate on combinations in parallel
Parallel(n_jobs=2)(delayed(ls_a_factors)(root_dir, output_dir, subj, '02a', run) for subj, run in combinations)

print('All done!')
