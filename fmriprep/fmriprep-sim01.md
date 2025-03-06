docker run \
    -v /DATAPOOL/BRAINPLAYBACK/BIDS-BRAINPLAYBACK-TASK2:/data \
    -v /DATAPOOL/BRAINPLAYBACK/BIDS-BRAINPLAYBACK-TASK2/derivatives/fmriprep23:/out \
    -v /DATAPOOL/BRAINPLAYBACK/BIDS-BRAINPLAYBACK-TASK2/derivatives/fmriprepwork:/work \
    -v /SCRATCH/software/freesurfer7/license.txt:/licensefile \
    nipreps/fmriprep:23.1.2 \
    --participant_label 20 \
    --fs-license-file /licensefile \
    /data /out participant \
    -w /work \
    --stop-on-first-crash \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --mem=85G \
    --skip_bids_validation