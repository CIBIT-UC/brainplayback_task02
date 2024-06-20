# Imports
import os
import essentia
import essentia.standard as es
import numpy as np
#import pandas as pd

def extract_features_essentia(audio_name, audio_path):
    # Audio file
    #audio_name = 'sub-09_Joyfulactivation_70t7Q6AYG6ZgTYmJWcnkUM.wav'
    emotion_name = audio_name.split('_')[1]
    audiofile = os.path.join(audio_path, audio_name)

    # Compute all features.
    # Aggregate 'mean' and 'stdev' statistics for all low-level, rhythm, and tonal frame features.
    features, features_frames = es.MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                                rhythmStats=['mean', 'stdev'],
                                                tonalStats=['mean', 'stdev'])(audiofile)
    
    feat_names = sorted(features.descriptorNames())

    # fetch the features that have a single value
    new_feat_names = []

    for jj in range(len(feat_names)):
        if type(features[feat_names[jj]]) == float:
            new_feat_names.append(feat_names[jj])

    # Convert to numpy array
    M = np.zeros((1,len(new_feat_names)))

    for ff in range(len(new_feat_names)):
        M[0,ff] = features[new_feat_names[ff]]     

    return M, new_feat_names, emotion_name   