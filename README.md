# Diffusion-Based Patient-Specific EEG Augmentation for Harmful Brain Activity Classification

## Overview

This project focuses on improving EEG-based harmful brain activity classification by addressing data scarcity and class imbalance using **time-series diffusion models**.

We generate realistic synthetic EEG signals tailored to individual patients and use them to train a high-performance deep learning classifier for detecting 6 types of harmful brain activity.

---

## Objectives

* **Patient-Specific Data Augmentation**
  Generate synthetic EEG signals using limited seizure data per patient.

* **High-Recall Classification Model**
  Improve recall and F1-score for detecting harmful brain activity.

* **Class Imbalance Mitigation**
  Balance rare classes (e.g., seizure) using synthetic data.

---

## Dataset

* **Competition**: Harmful Brain Activity Classification
* **Source**: Harvard Medical School
* **Size**: ~106,800 samples

### Download via Kaggle API

```bash
kaggle competitions download -c hms-harmful-brain-activity-classification
```

---

## Data Description

### Input Features (EEG Signals)

* 20-channel EEG signals:

  * Frontal: Fp1, Fp2, F3, F4
  * Central: C3, C4, Cz
  * Parietal: P3, P4, Pz
  * Occipital: O1, O2
  * Temporal: T3, T4, T5, T6
  * Midline: Fz
  * EKG (heart signal)

### Target Labels (6 Classes)

* Seizure
* LPD (Lateralized Periodic Discharges)
* GPD (Generalized Periodic Discharges)
* LRDA (Lateralized Rhythmic Delta Activity)
* GRDA (Generalized Rhythmic Delta Activity)
* Other

Labels are derived from expert voting distributions.

---

## Methodology

### 1. Time-Series Diffusion Model

* Models: **SSSD / DiffWave**
* Generates realistic EEG signals preserving temporal and spectral characteristics

### 2. Conditional Generation

* Conditioned on:

  * Patient ID
  * Label type
* Enables targeted synthesis of rare patterns (e.g., seizure-specific EEG)

### 3. Training Strategy

* Combine:

  * Real EEG data
  * Synthetic EEG data
* Train deep learning classifier (e.g., CNN, Transformer-based models)

### 4. Evaluation (TSTR)

* **Train on Synthetic, Test on Real**
* Metrics:

  * Recall (primary)
  * F1-score
  * Accuracy

---

## Project Structure

```
.
├── data/
│   ├── raw/
│   ├── processed/
├── models/
│   ├── diffusion/
│   ├── classifier/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── train_diffusion.py
│   ├── generate_synthetic.py
│   ├── train_classifier.py
│   └── evaluate.py
├── results/
└── README.md
```

---

## Training Pipeline

1. **Data Profiling**

   * Extract patient-specific seizure segments

2. **Diffusion Model Training**

   * Train on real EEG signals

3. **Synthetic Data Generation**

   * Generate patient-conditioned EEG samples

4. **Classifier Training**

   * Train with augmented dataset

5. **Evaluation**

   * Compare performance before/after augmentation

---

## Expected Results

* Improved recall for seizure detection
* Better generalization across patients
* Reduced class imbalance impact

---

## Key Contributions

* Patient-specific EEG augmentation using diffusion models
* Practical application of generative AI in healthcare
* Validation via TSTR framework

---

## Future Work

* Real-time EEG monitoring system
* Deployment in clinical environments
* Extension to other neurological disorders

---

## References

* DiffWave: A Versatile Diffusion Model for Audio Synthesis
* SSSD: Structured State Space Diffusion Models
* EEG Signal Processing Literature

---

## Conclusion

This project demonstrates that **synthetic data generation using diffusion models** can effectively overcome data scarcity in EEG-based medical AI systems, enabling more accurate and personalized diagnosis.
