---
numbering:
  headings: true
---

Zoom Meeting - Results Discussion
====

April 23, 2025

Decoding music-evoked emotions from brain activity: a personalized experiment towards the understanding of the neural mechanisms of emotion in music

Alternative title: Listening to the Neural Echoes of Familiar Music: An fMRI Decoding Study

BrainPlayback project - Task02

# Emotion framework

To structure these emotions, we adopt the Geneva Emotional Music Scale (GEMS), a model specifically designed to capture the aesthetic emotions evoked by music, recognizing that musical experiences often elicit emotions distinct from those studied in general affective research [](doi:10.1037/1528-3542.8.4.494). This model organizes nine music-specific emotions into three clusters ([](#emotion_framework)): (1) Sublimity, which includes feelings of wonder, transcendence, nostalgia, and peacefulness; (2) Vitality, encompassing emotions of power, joy, and tension; and (3) Unease, which includes feelings of sadness and tenderness.

```{figure} ./emotion-framework-zentner.png
:label: emotion_framework
:alt: GEMS
:align: center

The nine aesthetic emotions framework (GEMS).
```

# Research questions

In this study, we focus on felt emotions, seeking to decode the neural representations of personal, aesthetically driven affective experiences. This perspective is important, as felt emotions are highly individualized, influenced by factors such as familiarity and personal meaning. Familiarity has been shown to enhance emotional engagement with music, leading to stronger activation in the brain’s limbic and reward systems [](doi:10.1371/journal.pone.0027241). By allowing participants to select familiar, personally meaningful musical stimuli, we aim to maximize the intensity and ecological validity of the emotions experienced during the study.


- **Q1**: What are the brain regions involved in the processing of music-evoked emotions?
- **Q2**: Can we decode the emotional content of music from brain activity?
- **Q3**: What are the specific brain regions contributing to this discrimination?
- (**Q4**: What are the activation patterns in these regions?)

# Participants and music selection

Twenty individuals (12 females; age range 22-41 years, M = 32 years, SD = 6 years) participated in the experiment. 

Participants were asked to select two songs they knew well and strongly associated with each of the nine emotions in the GEMS model. The model was briefly explained, with examples illustrating each emotional category. For instance, the ‘wonder’ factor included emotions such as happy, amazed, dazzled, allured, and moved. No restrictions were placed on music selection, except that the chosen tracks had to be available in Spotify’s catalog.

The specific instruction was as follows: “Please select, entirely at your discretion, two musical excerpts for each of the nine emotions. Each excerpt should clearly and consistently evoke the intended emotion throughout the 24-second duration. Use Spotify to identify and select the excerpts, indicating the song title and the exact start time in seconds in the table below. Only choose excerpts from music that you already know and/or are familiar with.”


# Experimental paradigm overview

The experimental paradigm followed a structured sequence to present the personalized musical stimuli interleaved with white noise periods [](#paradigm).

```{figure} ./paradigm_task02.png
:label: paradigm
:alt: Paradigm
:align: center

Diagram of the fMRI paradigm trial. Each trial was repeated twice for each of the nine emotions.
```


# Results

## Activation maps

Contrasting all emotions vs. noise.

:::{figure} #glm_2ndlevel_view_img
:label: fig-my-cell

GLM group activation map.
:::

:::{figure} #glm_2ndlevel_mosaic
:label: fig-my-cell

GLM group activation map.
:::

:::{table} Cluster table.
:label: mytable
![](#glm_2ndlevel_cluster_table)
:::

## ANOVA searching for any effect of emotion

:::{figure} #anova_2ndlevel_view_img
:label: fig-my-cell

ANOVA group activation map (any effect), z-scores. Bonferroni-corrected p < 0.05. Cluster size > 25 voxels.
:::

:::{figure} #anova_2ndlevel_mosaic
:label: fig-my-cell

ANOVA group activation map (any effect), z-scores. Bonferroni-corrected p < 0.05. Cluster size > 25 voxels.
:::

:::{table} Cluster table of the ANOVA.
:label: mytable
![](#anova_2ndlevel_cluster_table)
:::

## Decoding

### Feature selection - stability mask
:::{figure} #mvpa_stab_mask_mosaic
:label: stability_mask_mosaic

Stability mask.
:::

:::{figure} #mvpa_stab_mask_view_img
:label: stability_mask_view

Stability mask.
:::


:::{table} Cluster table of the stability mask.
:label: stability_mask_table
![](#mvpa_stab_mask_cluster_table)
:::

### Overall accuracy

Leave one run out (LORO) cross-validation was used to evaluate the classifier's performance. The classifier was trained on the data from all but one run and tested on the left-out run. This process was repeated for each run, and the overall accuracy was calculated as the average of the accuracies obtained from each run.

:::{figure} #mvpa_confusion_matrix
:label: confusion_matrix

Confusion matrix.
:::

### Predictive power map (classifier weights)
The predictive power map was generated by averaging the classifier weights across all runs. This map provides insights into the brain regions that contribute most significantly to the classification of music-evoked emotions.

:::{figure} #mvpa_weights_mosaic
:label: weights_mosaic

Classifier weights.
:::


:::{figure} #mvpa_weights_per_condition_mosaic
:label: weights_per_condition_mosaic

Classifier weights per condition.
:::





# Discussion topics
- **Paradigm**:
  -  Personalized music selection - pros and cons?
- **Activation maps**: 
  - Exploration of Z-scores/betas per emotion
- **Decoding**: 
  - Classifier performance
  - Predictive power map per emotion
- **Publication**:
  - Target journal
  - https://academic.oup.com/scan