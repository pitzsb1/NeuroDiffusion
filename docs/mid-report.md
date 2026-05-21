# 프로젝트 중간보고 | Diffusion-Based Patient-Specific EEG Augmentation for Harmful Brain Activity Classification

## 1. 프로젝트 개요

본 프로젝트는 EEG 시계열 데이터를 기반으로 diffusion 프레임워크를 이용하여, 실제 EEG 신호의 분포를 모사하고 희귀/이상 EEG 데이터를 생성(증강)하는 것을 목표로 함. 특히 EEG 데이터의 클래스 불균형 문제(이상 신호 부족)를 해결하기 위해 생성 모델 기반 데이터 증강 접근을 수행함.

## 2. 중간 결과

### 2.1 데이터 처리 및 세그먼트 구성

- HMS EEG parquet 데이터 사용

- 각 EEG 파일은 약 (10000 × 20) 시계열 데이터

- EKG 채널 제외 → 총 19채널 사용

- sliding window 기반 segmentation 수행

```text
segment shape: (19 × 2000)
stride: 500
```

- 각 segment별 normalization 수행 (channel-wise)

```text
(mean = 0, std = 1)
```

결과: 약 6,000개 이상의 EEG segment 확보

### 2.2 Diffusion 학습 구조

- Forward diffusion (noise 추가) 구현

- Reverse diffusion (denoising) 학습

### 2.3 모델 구조 변화

초기 모델: 단순 Conv1D Denoiser

문제:

- EEG의 시간적 구조 및 패턴 학습 부족

- 생성 결과가 random noise에 가까움

개선:

- **Conv1D 기반 U-Net Denoiser로 변경**

Insight:

단순 Conv1D 구조보다 장기적인 시간 패턴 학습이 가능한 U-Net이 시계열의 **multi-scale 구조 학습에 유리**함

WHY? EEG 데이터는 단순 신호가 아닌 여러 시간 스케일이 존재

### 2.4 학습 결과

Loss 변화:

```text
Epoch 1: ~0.84
Epoch 5: ~0.38
Epoch 10: ~0.35
```

Insight:

- 빠른 초기 수렴 확인

- 이후 loss 완만한 감소 및 진동 발생

→ diffusion 특성상 **완전 단조 감소가 아닌 stochastic 수렴 형태**

stochastic 수렴: 흔들리면서 수렴하는 것(결론적으로는 수렴)

### 2.5 생성 결과 분석

<img width="993" height="374" alt="image" src="https://github.com/user-attachments/assets/5312fc52-8bdd-4475-9d7e-5717bb179361" />

초기 결과:

- 완전 random noise에서 벗어남

- amplitude 분포는 실제 EEG와 유사

문제:

- 명확한 주기성 부족

- EEG 고유의 리듬 패턴 미흡

→ 즉, 모델이 신호의 통계적 특성은 학습했으나 구조적 패턴은 부족한 상태

## 3. 수정 목표

**초기 목표**

Diffusion 모델을 이용한 EEG 데이터 생성

**수정된 목표**

EEG의 구조적 패턴(리듬, 주기성)까지 반영하는 고품질 생성 모델 구축

**수정 이유**

단순 분포 학습만으로는 실제 EEG와 유사한 신호 생성이 어렵고, 의미 있는 데이터 증강을 위해서는 시간적 구조 학습이 필수적

## 4. 추후 일정

### 4.1 모델 구조 개선

- U-Net 구조 추가 개선

- Residual connection 강화

- deeper network 실험

- channel attention / temporal attention 도입 검토

### 4.2 diffusion 설계 개선

- beta schedule 개선 (linear → cosine 등)

- noise scaling 전략 변경

### 4.3 데이터 확장

* stride 감소 (500 → 250)

* segment 수 증가

* 다양한 EEG 패턴 확보

### 4.4 조건부 diffusion

- label 기반 생성 모델 설계

```text
Seizure / GPD / GRDA 등 조건별 EEG 생성
```

→ 클래스 불균형 해결 목적

### 4.5 생성 결과 정량 평가

- real vs generated 분포 비교

- 주파수 영역 분석 (FFT)

### 4.6 최종 코드 최적화 (현재 미완)

- 전체 pipeline 통합 코드 정리

- 학습 속도 개선 (GPU 최적화)

## 5. 현재 진행 상황 평가

✓ diffusion pipeline 구현 완료

✓ EEG segmentation 및 preprocessing 완료

✓ 학습 안정화 (loss 수렴 확인)

✓ 생성 결과 도출 성공
