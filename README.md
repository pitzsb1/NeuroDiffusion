# Diffusion-Based EEG Augmentation for Harmful Brain Activity Classification

---

# 0. Background

Electroencephalography (EEG) is one of the most widely used biomedical signals for measuring brain activity and plays an essential role in diagnosing neurological disorders such as seizures, generalized periodic discharges (GPD), and generalized rhythmic delta activity (GRDA).

In recent years, deep learning-based automatic EEG diagnosis has gained significant attention. However, real-world medical datasets often suffer from severe **class imbalance**. Abnormal EEG patterns, especially seizures, are much more difficult to collect than normal recordings, resulting in limited training data.

Traditional data augmentation techniques, such as adding random noise, fail to preserve the complex temporal and frequency characteristics of EEG signals.

Recently, **Diffusion Models** have achieved remarkable performance not only in image generation but also in time-series data generation. Since diffusion models gradually learn the underlying data distribution, they have shown strong potential for generating high-dimensional sequential data such as EEG signals.

---

# 1. Project Overview

The objective of this project is to develop a diffusion-based generative model capable of synthesizing realistic EEG signals for future data augmentation tasks.

Instead of generating arbitrary signals, we designed a **Conv1D U-Net Diffusion Architecture** capable of learning both the temporal structure and statistical characteristics of EEG data. The generated signals were evaluated using multiple quantitative analyses.

---

# 2. Dataset

## Dataset

* **Source:** HMS Harmful Brain Activity Classification
* **Format:** EEG Parquet Files
* **Sampling Rate:** 200 Hz
* **Channels:** 20 EEG Channels

---

## Selected Channels

The EKG channel was excluded.

Final channels used:

```text
Fp1, F3, C3, P3, F7, T3, T5, O1,
Fz, Cz, Pz,
Fp2, F4, C4, P4, F8, T4, T6, O2
```

**Total:** 19 channels

---

# 3. Data Preprocessing

Since EEG recordings are long sequential signals, the data were divided into fixed-length segments using a sliding-window approach.

## Segmentation

```text
Segment Shape: (19 × 2000)

Sampling Rate: 200 Hz

Segment Length: 10 seconds

Stride: 500
```

---

## Normalization

Each EEG channel was standardized independently using:

```python
(signal - mean) / std
```

After normalization:

```text
Mean ≈ 0

Standard Deviation ≈ 1
```

---

## Result

```text
Total EEG Segments

≈ 6,085
```

These segments were used to train the diffusion model.

---

# 4. Diffusion-Based EEG Generation

## Forward Diffusion

During the forward process, Gaussian noise is gradually added to the original EEG signal.

```text
x₀ → x₁ → x₂ → ... → xₜ
```

As the diffusion step increases, the original EEG progressively becomes indistinguishable from random noise.

---

## Reverse Diffusion

The reverse process learns to recover the original EEG from noisy inputs.

```text
xₜ → xₜ₋₁ → ... → x₀
```

Through this denoising process, the model learns the underlying distribution of EEG signals.

---

## Why Diffusion?

Several generative models exist for data synthesis.

| Model     | Strength                | Weakness                 |
| --------- | ----------------------- | ------------------------ |
| GAN       | Fast generation         | Mode collapse            |
| VAE       | Stable training         | Lower generation quality |
| Diffusion | High-quality generation | Slow sampling            |

Considering the complex temporal characteristics of EEG signals, Diffusion Models provide more stable learning and higher-quality generation than conventional approaches.

---

# 5. Model Architecture

A **Conv1D-based U-Net** was adopted as the denoising network of the diffusion model.

---

<img width="1163" height="1352" alt="image" src="https://github.com/user-attachments/assets/b0945342-1f95-4216-8589-030aaf8176f7" />

---

## Encoder

The encoder extracts both local EEG patterns and long-term temporal features through repeated Conv1D and downsampling operations.

```text
Input EEG
↓
Conv1D
↓
Downsampling
↓
Conv1D
```

---

## Bottleneck

The bottleneck learns compressed global representations of EEG signals.

Diffusion timestep embeddings are incorporated so that the network is aware of the current noise level during denoising.

---

## Decoder

The decoder reconstructs EEG signals using upsampling together with skip connections.

```text
Upsampling
↓
Concatenate
↓
Conv1D
```

---

## Skip Connections

Skip connections directly transfer fine-grained information from the encoder to the decoder, helping preserve detailed EEG structures that would otherwise be lost during downsampling.

---

## Why U-Net?

A simple Conv1D network has limited ability to capture long-range temporal dependencies.

In contrast, U-Net simultaneously learns:

* Local patterns
* Long-term temporal structures
* Multi-scale features

making it particularly suitable for EEG signal generation.

---

# 6. Training Results

## Training Environment

* **Framework:** PyTorch
* **Device:** Apple Silicon GPU (MPS)
* **Batch Size:** 32
* **Epochs:** 50
* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE)

---

## Training Loss

<img width="691" height="391" alt="image" src="https://github.com/user-attachments/assets/5b8e64d3-a86f-40b3-86ae-5aa59a8c7793" />

---

## Result

Training initially relied on the CPU, resulting in slow execution and limited experimental efficiency.

After switching to the Apple Silicon GPU (MPS), training became significantly faster, and the loss decreased steadily.

```text
Epoch 1  : 0.8406
Epoch 5  : 0.3801
Epoch 10 : 0.3516
```

---

## Interpretation

At the beginning of training, the model struggled to predict noise accurately.

As training progressed, its denoising capability improved steadily.

Although the loss fluctuated slightly rather than decreasing monotonically, this behavior is expected because different random noise is sampled during each diffusion iteration. Overall, the model demonstrated stable convergence.

---

# 7. Evaluation of Generated EEG

Rather than relying solely on training loss, multiple quantitative analyses were conducted to evaluate the quality of generated EEG signals.

Evaluation metrics include:

* Waveform Comparison
* Statistical Feature Comparison
* FFT Analysis
* Band Power Analysis
* PCA Distribution Analysis

---

## 7.1 Generated EEG Example

<img width="999" height="391" alt="image" src="https://github.com/user-attachments/assets/f612e2ac-396e-418a-9d5e-ecc1f8d51100" />

### Result

The generated EEG exhibited continuous temporal structures similar to real EEG signals rather than appearing as pure random noise.

However, characteristic EEG rhythmic patterns and periodicity were not fully reproduced.

### Interpretation

The model successfully learned the general statistical characteristics of EEG but failed to completely capture long-range rhythmic structures.

---

## 7.2 Statistical Feature Comparison

| Feature            | Real EEG | Generated EEG |
| ------------------ | -------: | ------------: |
| Mean               |       ~0 |       -0.0026 |
| Standard Deviation |     1.00 |          0.48 |
| Minimum            |    -4.98 |         -3.68 |
| Maximum            |     4.15 |          3.12 |
| Range              |     9.13 |          6.80 |

### Key Findings

The mean value remained nearly identical to that of real EEG.

However, both the standard deviation and signal range were noticeably smaller.

### Interpretation

The generated EEG successfully reproduced the central distribution but failed to generate the large-amplitude fluctuations commonly observed in real EEG recordings.

---

## 7.3 FFT Analysis

Because EEG contains important frequency-domain information, Fast Fourier Transform (FFT) was performed to compare real and generated signals.

<img width="1014" height="468" alt="image" src="https://github.com/user-attachments/assets/04135444-39c0-4f22-8b8e-bd427568f84f" />

### Key Findings

The generated EEG covered a frequency range similar to that of real EEG.

However, energy in the low-frequency region was considerably weaker.

Strong low-frequency peaks observed in real EEG were not fully reproduced.

### Interpretation

The model partially learned the frequency distribution but did not adequately capture the dominant EEG rhythms.

---

## 7.4 Band Power Analysis

| Band  | Real EEG | Generated EEG |
| ----- | -------: | ------------: |
| Delta |    21815 |          4335 |
| Theta |    10530 |          1479 |
| Alpha |     3878 |           753 |
| Beta  |      462 |           248 |
| Gamma |      137 |           170 |

<img width="868" height="449" alt="image" src="https://github.com/user-attachments/assets/ab426697-f694-48d8-85f5-a8b2663a8f97" />

### Key Findings

The generated EEG contained substantially lower Delta, Theta, and Alpha band power than real EEG.

Delta and Theta bands, which account for a large proportion of EEG energy, were especially underrepresented.

Beta and Gamma bands remained relatively similar.

### Interpretation

The current model learned short-term signal structures but struggled to reproduce the dominant low-frequency rhythms of EEG.

These findings suggest that future models should explicitly incorporate frequency-domain information into the training objective.

---

## 7.5 PCA Distribution Analysis

Principal Component Analysis (PCA) was performed to compare the overall distributions of real and generated EEG samples.

<img width="698" height="545" alt="image" src="https://github.com/user-attachments/assets/ec5968d9-a624-49dc-a3a9-1ab4896b8f32" />

### Result

The PCA visualization showed a clear separation between real and generated EEG samples in feature space.

```text
Explained Variance Ratio

PC1 = 70.3%
PC2 = 4.4%
```

### Interpretation

Although the generated EEG captured certain statistical properties, it did not fully reproduce the overall distribution of real EEG data.

---

# 8. Limitations

Several important limitations were identified during this study.

## Insufficient Low-Frequency Rhythms

FFT and Band Power analyses revealed significantly weaker Delta and Theta activity in generated EEG.

This indicates that the current model has difficulty learning long-term temporal dependencies.

---

## Distribution Gap

PCA demonstrated a noticeable separation between real and generated EEG samples.

Therefore, the generated data only partially represents the true EEG distribution.

---

## Lack of Downstream Validation

Although the quality of generated EEG was evaluated, its effectiveness for data augmentation was not verified.

Future studies should investigate whether incorporating generated EEG improves classification performance.

---

# 9. Conclusion

In this project, a Conv1D U-Net Diffusion Model was developed using the HMS EEG dataset to generate realistic EEG time-series signals.

The model successfully learned the basic statistical characteristics and part of the frequency structure, demonstrating its ability to synthesize EEG signals from random noise.

However, FFT, Band Power, and PCA analyses showed that the generated EEG lacked dominant low-frequency rhythms and did not fully match the real EEG distribution.

Therefore, the proposed model should be regarded as a **baseline EEG generation model** that demonstrates the feasibility of diffusion-based EEG synthesis while providing a foundation for future improvements.

---

# 10. Contributions

This project extends beyond simple EEG generation by applying diffusion models to EEG time-series data and systematically analyzing their limitations.

Rather than relying solely on training loss, we evaluated generated EEG from multiple perspectives using FFT, Band Power, and PCA analyses.

These evaluations provide insights into both the strengths and weaknesses of the proposed model and establish a useful baseline framework for future EEG generation and data augmentation research.

---

# 11. Comparison with Previous Work

Previous studies have primarily focused on improving EEG classification performance, while quantitative evaluation of generated EEG quality has often been limited.

This project emphasizes comprehensive evaluation of generated EEG using FFT, Band Power, and PCA analyses.

Furthermore, instead of simply demonstrating successful signal generation, we identify the structural limitations of the proposed model and discuss possible directions for future improvements.

---

# 12. Future Work

## 12.1 Frequency-Aware Diffusion

The current model is trained only in the time domain.

Future work could introduce **Band Power Loss** or other frequency-aware objectives to directly preserve EEG frequency characteristics during training.

---

## 12.2 Architecture Improvement

The current U-Net architecture can be further improved by incorporating:

* Stronger residual connections
* A deeper U-Net architecture
* Temporal attention
* Channel attention

These enhancements are expected to improve the model's ability to learn long-term EEG temporal patterns.
