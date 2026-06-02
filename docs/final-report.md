# 프로젝트 최종보고 | Diffusion-Based EEG Augmentation for Harmful Brain Activity Classification

---

# 0. 연구 배경

EEG(Electroencephalography)는 뇌의 전기적 활동을 측정하는 대표적인 생체 신호로, 간질(Seizure), GPD, GRDA 등 다양한 신경학적 이상 패턴을 진단하는 데 활용된다.

최근에는 EEG 데이터를 활용한 딥러닝 기반 자동 진단 연구가 활발하게 진행되고 있으나, 실제 의료 데이터는 클래스 불균형(Class Imbalance) 문제가 심각하게 존재한다. 특히 발작(Seizure)과 같은 이상 신호는 정상 신호에 비해 수집이 어렵고 데이터 수가 제한적이다.

이러한 문제를 해결하기 위해 데이터 증강(Data Augmentation)이 활용되고 있으나, 기존의 단순 노이즈 추가 방식은 EEG의 복잡한 시간적 구조와 주파수 특성을 충분히 반영하지 못한다.

최근 생성형 AI 분야에서는 Diffusion Model이 이미지 생성뿐 아니라 시계열 데이터 생성 분야에서도 우수한 성능을 보이고 있다. Diffusion Model은 실제 데이터 분포를 점진적으로 학습하여 새로운 데이터를 생성할 수 있으며, EEG와 같은 고차원 시계열 데이터 생성에도 적용 가능성이 높다.

---

# 1. 프로젝트 개요

본 프로젝트는 EEG 시계열 데이터를 기반으로 Diffusion Model을 활용하여 새로운 EEG 신호를 생성하고, 이를 통해 향후 EEG 데이터 증강(Augmentation)에 활용 가능한 생성 모델을 구축하는 것을 목표로 한다.

특히 기존의 단순 신호 생성이 아닌 EEG의 시간적 구조와 통계적 특성을 학습할 수 있는 Conv1D 기반 U-Net Diffusion Architecture를 설계하였으며, 생성 결과를 다양한 정량적 평가 지표를 통해 분석하였다.

---

# 2. 데이터셋

## Dataset

* Source: HMS Harmful Brain Activity Classification
* Format: EEG Parquet Files
* Sampling Rate: 200 Hz
* Channels: 20 Channels

---

## 사용 채널

EKG 채널은 제외

최종 사용 채널:

```text
Fp1, F3, C3, P3, F7, T3, T5, O1,
Fz, Cz, Pz,
Fp2, F4, C4, P4, F8, T4, T6, O2
```

총 19채널

---

# 3. 데이터 전처리

EEG 데이터는 시계열 길이가 길고 학습 효율이 낮기 때문에, Sliding Window 기반 Segment 단위로 분할

## Segmentation

```text
Segment Shape: (19 × 2000)

Sampling Rate: 200 Hz

Segment Length: 10 seconds

Stride: 500
```

---

## Normalization

각 채널별로 Standardization을 수행

```python
(signal - mean) / std
```

결과적으로 모든 채널은 다음 조건을 만족하도록 변환

```text
Mean ≈ 0

Std ≈ 1
```

---

## 결과

```text
총 EEG Segment 수

≈ 6,085
```

이를 Diffusion 학습 데이터로 활용

---

# Diffusion Framework

<img width="1018" height="1176" alt="image" src="https://github.com/user-attachments/assets/5c9f2ae6-f593-4e97-b839-dce00b71974a" />

Diffusion-Based EEG Generation Framework

---

# 4. Diffusion 기반 EEG 생성

## Forward Diffusion

Diffusion Model은 실제 EEG 신호에 Gaussian Noise를 점진적으로 추가하는 과정(Forward Process)을 수행

```text
x₀ → x₁ → x₂ → ... → xₜ
```

→ 학습이 진행될수록 원본 EEG는 점차 랜덤 노이즈에 가까운 형태가 됨

---

## Reverse Diffusion

Reverse Process에서는 Noise가 포함된 EEG로부터 원래 EEG를 복원하도록 학습

```text
xₜ → xₜ₋₁ → ... → x₀
```

→ 이를 통해 모델은 EEG 데이터의 분포를 학습하게 됨

---

## 모델 선택 이유

생성형 모델에는 GAN, VAE, Diffusion 등이 존재

| 모델        | 장점       | 단점            |
| --------- | -------- | ------------- |
| GAN       | 빠른 생성    | Mode Collapse |
| VAE       | 안정적 학습   | 생성 품질 저하      |
| Diffusion | 높은 생성 품질 | 생성 속도 느림      |

본 프로젝트에서는 EEG의 복잡한 시계열 구조를 보다 안정적으로 학습하기 위해 Diffusion Model을 선택

---

# 5. 모델 아키텍처

본 프로젝트에서는 Conv1D 기반 U-Net을 Diffusion Model의 Denoising Network로 활용

---

<img width="1163" height="1352" alt="image" src="https://github.com/user-attachments/assets/8dd5d209-6cda-42d7-8a77-58411aa260d6" />

Conv1D U-Net Diffusion Architecture

---

## Encoder

Encoder는 EEG의 국소적 패턴과 장기적인 시간 구조를 학습하기 위해 Conv1D와 Downsampling을 반복 수행한다.

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

가장 압축된 표현 공간에서 EEG의 전반적인 구조 정보를 학습

또한 Diffusion Timestep 정보를 함께 입력받아 현재 Noise 수준을 인식하도록 설계

---

## Decoder

Upsampling과 Skip Connection을 이용하여 Encoder 단계의 정보를 복원

```text
Upsampling
↓
Concatenate
↓
Conv1D
```

---

## Skip Connection

U-Net의 Skip Connection은 Downsampling 과정에서 손실된 세부 EEG 정보를 Decoder에 직접 전달

이를 통해 단순 Conv1D 구조보다 더 정교한 신호 복원이 가능

---

## 모델 구조 선택 이유

단순 Conv1D 구조는 EEG의 장기적인 시간 패턴을 학습하는 데 한계 존재

반면 U-Net 구조는

* Local Pattern
* Long-term Temporal Structure
* Multi-scale Feature

를 동시에 학습할 수 있기 때문에 EEG 생성 문제에 적합

---

# 6. 학습 결과

## 학습 환경

* Framework: PyTorch
* Device: Apple Silicon GPU (MPS)
* Batch Size: 32
* Epoch: 50
* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)

---

## Training Loss

<img width="691" height="391" alt="image" src="https://github.com/user-attachments/assets/01c7c5ca-03f6-4874-b69d-6f3e61d53671" />

Training Loss Curve

---

## 결과

초기 학습에서는 CPU를 사용하였으나 학습 속도가 매우 느려 실험 효율이 낮았다.

이후 Apple Silicon GPU(MPS)를 활용하여 학습을 수행하였으며, Epoch가 증가함에 따라 Loss가 안정적으로 감소하는 것을 확인하였다.

```text
Epoch 1  : 0.8406
Epoch 5  : 0.3801
Epoch 10 : 0.3516
```

---

## 해석

초기에는 모델이 랜덤 노이즈를 거의 제거하지 못하였으나 학습이 진행됨에 따라 Noise Prediction 능력이 향상되었다.

또한 Loss는 완전한 단조 감소 형태가 아닌 진동을 동반하며 감소하는 양상을 보였다.

이는 Diffusion Model이 매 iteration마다 서로 다른 노이즈를 샘플링하여 학습하기 때문에 발생하는 자연스러운 현상이며, 전체적으로는 안정적인 수렴 경향을 확인하였다.

---

# 7. 생성 결과 평가

본 프로젝트에서는 생성 EEG의 품질을 평가하기 위해 단순 Loss뿐 아니라 다양한 정량적 평가 지표를 활용하였다.

평가 항목은 다음과 같다.

* Waveform Comparison
* Statistical Feature Comparison
* FFT Analysis
* Band Power Analysis
* PCA Distribution Analysis

---

## 7.1 Generated EEG Example

<img width="999" height="391" alt="image" src="https://github.com/user-attachments/assets/3fe57db9-397d-47d0-9e1e-8d916bdfbff6" />

Real EEG vs Generated EEG

---

### 결과

생성된 EEG는 완전한 랜덤 노이즈 형태에서 벗어나 실제 EEG와 유사한 형태의 연속적인 시계열 구조를 나타냈다.

그러나 실제 EEG에서 관찰되는 명확한 리듬 패턴 및 주기성은 충분히 재현되지 않았다.

---

### 해석

현재 모델은 EEG 신호의 기본적인 분포 특성은 학습하였으나, EEG 특유의 장기적인 리듬 구조를 완전히 복원하지는 못한 것으로 판단된다.

---

## 7.2 Statistical Feature Comparison

실제 EEG와 생성 EEG의 기본 통계량을 비교하였다.

| Feature | Real EEG | Generated EEG |
| ------- | -------: | ------------: |
| Mean    |       ~0 |       -0.0026 |
| Std     |     1.00 |          0.48 |
| Min     |    -4.98 |         -3.68 |
| Max     |     4.15 |          3.12 |
| Range   |     9.13 |          6.80 |

---

### 주요 발견

Mean 값은 실제 EEG와 매우 유사하게 유지되었다.

반면 Standard Deviation과 Range는 실제 EEG 대비 감소하였다.

---

### 해석

생성 EEG는 신호의 중심 분포는 학습하였으나 실제 EEG에서 나타나는 진폭 변동성을 충분히 재현하지 못하였다.

즉, 모델은 EEG의 평균적인 형태는 학습하였으나 강한 amplitude fluctuation은 충분히 생성하지 못하는 것으로 나타났다.

---

## 7.3 FFT Analysis

EEG는 시간 영역뿐 아니라 주파수 영역 특성도 매우 중요하다.

따라서 Fast Fourier Transform (FFT)을 이용하여 실제 EEG와 생성 EEG의 주파수 구조를 비교하였다.

---

<img width="1014" height="468" alt="image" src="https://github.com/user-attachments/assets/68bb3857-cf23-438e-88dd-9f000a69c70a" />

FFT Comparison

---

### 주요 발견

생성 EEG는 실제 EEG와 유사한 전체 주파수 범위를 포함하고 있었으나, 저주파 영역에서의 에너지가 상대적으로 부족하게 나타났다.

실제 EEG에서는 특정 저주파 대역에서 강한 Peak가 관찰된 반면, 생성 EEG에서는 해당 Peak가 충분히 형성되지 않았다.

---

### 해석

현재 모델은 주파수 분포의 일부를 학습하였으나 EEG의 핵심적인 리듬 구조를 충분히 재현하지 못하였다.

---

## 7.4 Band Power Analysis

EEG의 주요 주파수 대역에 대한 에너지를 비교하였다.

| Band  | Real EEG | Generated EEG |
| ----- | -------: | ------------: |
| Delta |    21815 |          4335 |
| Theta |    10530 |          1479 |
| Alpha |     3878 |           753 |
| Beta  |      462 |           248 |
| Gamma |      137 |           170 |

---

<img width="868" height="449" alt="image" src="https://github.com/user-attachments/assets/497d2c51-30b0-4193-b595-8f20572e6b5b" />

Band Power Comparison

---

### 주요 발견

생성 EEG는 실제 EEG 대비 Delta, Theta 및 Alpha 대역의 에너지가 크게 감소하였다.

특히 Delta 및 Theta 대역은 실제 EEG의 주요 에너지원임에도 불구하고 생성 EEG에서는 현저히 낮게 나타났다.

반면 Beta 및 Gamma 대역은 상대적으로 유사한 수준을 유지하였다.

---

### 해석

현재 모델은 단기적인 신호 구조는 일부 학습하였으나 EEG의 핵심적인 저주파 리듬을 충분히 생성하지 못하였다.

이는 EEG 생성 품질 향상을 위해 주파수 구조를 직접 반영하는 학습 전략이 필요함을 시사한다.

---

## 7.5 PCA Distribution Analysis

생성 EEG가 실제 EEG 데이터 분포를 얼마나 잘 따르는지 평가하기 위해 PCA를 수행하였다.

---

<img width="698" height="545" alt="image" src="https://github.com/user-attachments/assets/ff637c76-ca48-4663-8881-c24e86da8c64" />

PCA Comparison: Real vs Generated EEG

---

### 결과

PCA 결과 실제 EEG와 생성 EEG는 Feature Space 상에서 명확하게 분리되는 경향을 보였다.

```text
Explained Variance Ratio

PC1 = 70.3%
PC2 = 4.4%
```

---

### 해석

생성 EEG는 일부 통계적 특성을 학습하였으나 실제 EEG 데이터가 형성하는 전체 분포를 완전히 재현하지는 못하였다.

이는 현재 모델이 실제 EEG 분포를 부분적으로만 학습하고 있음을 의미한다.

---

# 8. 한계 분석

본 연구에서는 EEG 생성에 성공하였으나 몇 가지 중요한 한계를 확인하였다.

---

## 저주파 리듬 부족

FFT 및 Band Power 분석 결과 생성 EEG는 Delta 및 Theta 대역의 에너지가 실제 EEG보다 크게 감소하였다.

이는 현재 모델이 EEG의 장기적인 시간 구조(Long-Term Temporal Structure)를 충분히 학습하지 못했음을 의미한다.

---

## 분포 차이 존재

PCA 분석 결과 실제 EEG와 생성 EEG는 Feature Space에서 명확하게 분리되었다.

이는 생성 데이터가 실제 EEG 데이터의 전체 분포를 완전히 반영하지 못함을 시사한다.

---

## 평가 부족

본 연구에서는 생성 품질을 평가하였으나 실제 데이터 증강 효과는 검증하지 못하였다.

따라서 생성 EEG가 분류 성능 향상에 실제로 기여하는지는 추가 실험이 필요하다.

---

# 9. 최종 결론

본 연구에서는 HMS EEG 데이터를 활용하여 Conv1D 기반 U-Net Diffusion Model을 구축하고 EEG 시계열 생성 실험을 수행하였다.

학습 결과 모델은 EEG의 기본적인 통계 분포와 일부 주파수 구조를 학습할 수 있었으며, 랜덤 노이즈로부터 새로운 EEG 신호를 생성하는 데 성공하였다.

그러나 FFT, Band Power, PCA 분석 결과 생성 EEG는 실제 EEG 대비 저주파 리듬이 부족하였으며 데이터 분포 또한 완전히 일치하지 않는 것으로 나타났다.

따라서 현재 모델은 EEG 생성 가능성을 확인한 Baseline Model로서 의미를 가지며, 향후 보다 정교한 구조 개선을 통해 고품질 EEG 생성 모델로 발전시킬 수 있을 것으로 기대된다.

---

# 10. 프로젝트 의의

본 프로젝트는 단순한 EEG 생성 실험을 넘어 Diffusion Model을 EEG 시계열 데이터에 적용하고 그 한계를 정량적으로 분석하였다는 점에서 의의를 가진다.

특히 Loss뿐 아니라 FFT, Band Power, PCA를 활용하여 생성 품질을 다각적으로 평가하였으며, 이를 통해 현재 모델이 학습한 정보와 부족한 정보를 체계적으로 분석하였다.

이는 향후 EEG 생성 모델 연구 및 데이터 증강 연구를 위한 기초 프레임워크로 활용될 수 있다.

---

# 11. 참고 프로젝트와의 차별성

참고 프로젝트에서는 주로 EEG 분류(Classification) 성능 향상에 초점을 맞추고 있으며, 생성 모델을 활용하더라도 생성 품질 자체에 대한 정량적 분석은 제한적으로 수행되는 경우가 많다.

본 프로젝트는 생성 모델 구축뿐 아니라 FFT, Band Power, PCA를 이용하여 생성 EEG의 품질을 다각적으로 평가하였다.

또한 생성 성공 여부만을 보고하는 것이 아니라 생성 모델이 가지는 구조적 한계를 분석하고 향후 개선 방향을 제시하였다는 점에서 기존 연구와 차별성을 가진다.

---

# 12. 추후 발전 방향

## 12.1 Frequency-aware Diffusion

현재 모델은 시간 영역(Time Domain) 기반 학습을 수행한다.

향후에는 Band Power Loss를 추가하여 EEG의 주파수 구조를 직접 학습할 수 있도록 개선할 수 있다.

---

## 12.2 Architecture Improvement

현재 U-Net 구조를 개선하기 위해 다음과 같은 방법을 적용할 수 있다.

* Residual Connection 강화
* Deeper U-Net
* Temporal Attention
* Channel Attention

이를 통해 장기적인 EEG 패턴 학습 능력을 향상시킬 수 있다.

