#!/bin/bash

#SBATCH --job-name=fmriprep-2320-01
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=86G
#SBATCH --partition=hpc
#SBATCH --output=/users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/code/log/%x_%A_%a.out
#SBATCH --error=/users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/code/log/%x_%A_%a.err

# Used to guarantee that the environment does not have any other loaded module
module purge

# Load python
.  /users3/uccibit/alexsayal/miniconda3/etc/profile.d/conda.sh
conda activate fmriprep

# Do
SUBJECT=01

udocker run \
    -v /users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2:/data \
    -v /users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/derivatives/fmriprep2320:/out \
    -v /users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/derivatives/fmriprepwork:/work \
    -v /users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/derivatives/fmriprep2320/sourcedata/fmapreg:/fmapreg \
    -v /users3/uccibit/alexsayal/freesurfer_license.txt:/licensefile \
    fmriprep2320 \
    --participant_label $SUBJECT \
    --fs-license-file /licensefile \
    /data /out participant \
    -w /work \
    --stop-on-first-crash \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --mem=85G \
    --skip_bids_validation \
    --derivatives /fmapreg

echo "fmriPrep is done. Now deleting temporary files in work folder"

# Delete working folder for this subject
#rm -rf /users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/derivatives/fmriprepwork/fmriprep_23_2_wf/single_subject_${SUBJECT}_wf
#rm -rf /users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/derivatives/fmriprep2320/sourcedata/freesurfer/sub-${SUBJECT}

echo "Finished with job $SLURM_JOBID"