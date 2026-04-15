# Diffusion-based Patient-Specific EEG Augmentation & Classification

## Overview

This project proposes a patient-specific EEG classification system enhanced by time-series diffusion-based data augmentation.
We address the critical challenge of data scarcity in epileptic seizure detection by generating realistic synthetic EEG signals and leveraging them to improve classification performance across 6 harmful brain activity types.

## Objectives

Patient-specific data augmentation

Generate high-quality synthetic EEG signals using limited seizure data

High-precision classification

Improve recall for seizure detection using deep learning models

Class imbalance mitigation

Reduce bias caused by rare pathological EEG patterns

## Dataset | Harmful Brain Activity Classification (~106,800 samples)
 
This project uses the HMS Harmful Brain Activity Classification dataset.

Download via Kaggle API:

```bash
kaggle competitions download -c hms-harmful-brain-activity-classification
```

Source: Harvard Medical School

### key Features

EEG Signals (Input, X)

**20-channel time-series data**:

Frontal: Fp1, Fp2, F3, F4

Central: C3, C4, Cz

Parietal: P3, P4, Pz

Occipital: O1, O2

Temporal: T3, T4, T5, T6

Others: Fz, EKG

Target Labels (Output, y)

**6-class expert-voted brain activity**:

Seizure

LPD (Lateralized Periodic Discharges)

GPD (Generalized Periodic Discharges)

LRDA (Lateralized Rhythmic Delta Activity)

GRDA (Generalized Rhythmic Delta Activity)

Other

## Motivation

1. Data Scarcity
Seizure events are rare compared to normal EEG signals, making it difficult to train robust models.

2. Patient Variability
EEG patterns vary significantly across patients, limiting generalization.

3. Opportunity with Generative AI
Recent advances in time-series diffusion models enable realistic signal generation, providing a solution to data imbalance.

## Key Features

Time-Series Diffusion Engine

Based on architectures like SSSD or DiffWave

Preserves temporal dynamics, frequency, and phase information

## Conditional Data Synthesis

Generates EEG signals conditioned on:

Patient ID

Brain activity type

## Evaluation Pipeline (TSTR)

Train on Synthetic, Test on Real

Quantitative validation of synthetic data quality

## Expected Impact

Enhanced rare disease research

Enables robust modeling even with limited seizure data

Improved diagnostic accuracy

Higher recall and reduced false negatives

Scalable medical AI

Reduces dependency on large-scale clinical datasets

## Conclusion

This project demonstrates the potential of a data augmentation engine that enables high-performance, patient-specific EEG classification using only limited real-world data.
