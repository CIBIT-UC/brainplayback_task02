#!/bin/bash

#SBATCH --job-name=firstlevel
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=hpc
#SBATCH --output=/users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/code/log/%x_%A_%a.out
#SBATCH --error=/users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/code/log/%x_%A_%a.err

# Load conda environment
.  /users3/uccibit/alexsayal/miniconda3/etc/profile.d/conda.sh
conda activate mvpa

# Do
python /users3/uccibit/alexsayal/BIDS-BRAINPLAYBACK-TASK2/code/glm/firstlevel.py

echo "Finished with job $SLURM_JOBID"