#!/bin/bash

#SBATCH --job-name=mvpa-lsa-musicnoise-spacenet
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256G
#SBATCH --partition=hpc
#SBATCH --output=/users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/code/log/%x_%A_%a.out
#SBATCH --error=/users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/code/log/%x_%A_%a.err

# Load conda environment
.  /users3/uccibit/alexsayal/miniconda3/etc/profile.d/conda.sh
conda activate mvpa

# Do
python /users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/code/mvpa/mvpa-lsa-musicnoise-spacenet.py

echo "Finished with job $SLURM_JOBID"