# TAl 2 MNI
    flirt -v -in Talairach_colin1.1.nii -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz -out Talairach_colin1.1_MNI -omat TAL2MNI_matrix
    flirt -in Koelsch_prob_TAL.nii -ref Talairach_colin1.1_MNI.nii.gz -applyxfm -init TAL2MNI_matrix -out Koelsch_prob_MNI.nii
    flirt -in Meta_analysis_C05_1k_clust.nii -ref Talairach_colin1.1_MNI.nii.gz -applyxfm -init TAL2MNI_matrix -out Meta_analysis_C05_1k_clust_MNI.nii
# Mask
    fslmaths Koelsch_prob_MNI.nii.gz -thr 0.01 Koelsch_mask_MNI
    fslmaths Koelsch_mask_MNI.nii.gz -bin Koelsch_mask_MNI
    fsleyes Koelsch_prob_MNI.nii.gz Koelsch_mask_MNI.nii.gz