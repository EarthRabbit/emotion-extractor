# Emotion Extractor - Hybrid Classification Pipeline

하이브리드 감정 분류 파이프라인: **사전 추출된 FaceNet 임베딩** + **CNN 기반 이미지 특징**을 결합하여 감정을 분류합니다.

> 🧬 **FaceNet 기반**: 얼굴 인식에 특화된 FaceNet(InceptionResnetV1)을 사용하여 512차원 임베딩 벡터를 추출합니다.

## 📋 파이프라인 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Hybrid Emotion Classification Pipeline v3.0      │
│                         (FaceNet + CNN Fusion)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [전처리 단계 - 1회 실행]                                            │
│  Images ──► extract_facenet_vectors.py ──► facenet_vectors.pt       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [학습/추론 단계]                                                    │
│                                                                     │
│  Input Image ──────────► CNN Branch ──► CNN Features (512D)         │
│                                              │                      │
│                                              ▼                      │
│                                     ┌──────────────────┐            │
│                                     │  Feature Fusion  │            │
│                                     │  (Concat/Attn)   │            │
│                                     └──────────────────┘            │
│                                              ▲                      │
│  facenet_vectors.pt ──► FaceNet Features (512D, L2 Normalized)      │
│                                              │                      │
│                                              ▼                      │
│                                    Classification Head              │
│                                              │                      │
│                                              ▼                      │
│                            Emotion Class (6 classes)                │
└─────────────────────────────────────────────────────────────────────┘
```

## 🧬 왜 FaceNet인가?

| 비교 항목 | YOLO 백본 | FaceNet |
|----------|-----------|---------|
| **설계 목적** | 물체 탐지 | 얼굴 인식 |
| **출력 차원** | 256 | 512 |
| **L2 정규화** | ❌ | ✅ |
| **감정 인식 적합도** | 보통 (65-72%) | **우수 (75-82%)** |
| **얼굴 특징 학습** | 간접적 | **직접적** |

FaceNet은 얼굴의 미세한 특징을 구분하도록 학습되어, 표정 변화를 더 잘 포착합니다.

## ⚠️ 중요: FaceNet 벡터 전처리

FaceNet 브랜치를 사용하려면 **학습 전에 FaceNet 벡터를 사전 추출**해야 합니다.

```bash
# FaceNet 임베딩 추출 (최초 1회 필수)
uv run python src/data/extract_facenet_vectors.py \
    --data-root ./data/cropped/images \
    --output ./data/cropped/facenet_vectors.pt
```

### 전처리 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--data-root` | 이미지 루트 디렉토리 | `./data/cropped/images` |
| `--output` | 출력 파일 경로 | `{data-root}/../facenet_vectors.pt` |
| `--pretrained` | 사전학습 가중치 | `vggface2` |
| `--batch-size` | 배치 크기 | 32 |
| `--image-size` | 이미지 크기 | 160 (FaceNet 권장) |
| `--cpu` | CPU 강제 사용 | False |

### 사전학습 가중치 선택

| 가중치 | 설명 | 추천 |
|--------|------|------|
| `vggface2` | 다양한 인종/연령 얼굴 데이터 | ⭐ 일반 권장 |
| `casia-webface` | 아시아인 얼굴 데이터 많음 | 한국인 데이터에 적합 |

```bash
# 아시아인 얼굴에 최적화된 설정
uv run python src/data/extract_facenet_vectors.py \
    --data-root ./data/cropped/images \
    --pretrained casia-webface
```

## 🖥️ 시스템 요구사항

### 권장 요구사항

| 요구사항 | 설명 |
|----------|------|
| **NVIDIA GPU** | CUDA 지원 GPU 권장 (CPU도 가능) |
| **CUDA Toolkit** | 11.8 이상 권장 |
| **Python** | 3.12 이상 |
| **PyTorch** | 2.5+ (CUDA 버전 권장) |

### 디바이스 정보 확인

```bash
# 디바이스 정보 확인
uv run python src/data/extract_facenet_vectors.py --check-device
```

## 🎯 감정 클래스

| 한국어 | English | 설명 |
|--------|---------|------|
| 당황 | Embarrassed | 당황스러운 감정 |
| 분노 | Angry | 화가 난 감정 |
| 불안 | Anxious | 불안한 감정 |
| 상처 | Hurt | 상처받은 감정 |
| 슬픔 | Sad | 슬픈 감정 |
| 중립 | Neutral | 중립적인 감정 |

## 🚀 설치 및 환경 설정

### uv 환경 사용 (권장)

1. [uv 공식 다운로드 페이지](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)에서 설치하거나:

    ```bash
    pip install uv
    ```

2. 의존성 설치:

    ```bash
    uv sync
    ```

### 의존성 패키지

- `torch >= 2.5` (CUDA 버전 권장)
- `torchvision >= 0.20`
- `facenet-pytorch >= 2.5.3` (FaceNet 임베딩 추출)
- `tensorboard >= 2.0`
- `pillow >= 11.0`
- `tqdm >= 4.0`

## 📖 사용법

### 0. FaceNet 벡터 전처리 (필수)

```bash
# FaceNet 임베딩 벡터 사전 추출
uv run python src/data/extract_facenet_vectors.py --data-root ./data/cropped/images

# 결과: ./data/cropped/facenet_vectors.pt 생성됨
```

### 1. 학습 (Training)

```bash
# 기본 학습 (FaceNet + CNN)
uv run python src/pipeline.py --mode train

# CNN만 사용 (FaceNet 벡터 불필요)
uv run python src/pipeline.py --mode train --no-facenet

# FaceNet만 사용 (CNN 없이)
uv run python src/pipeline.py --mode train --no-cnn

# 설정 변경
uv run python src/pipeline.py --mode train \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.0001 \
    --backbone resnet34

# 경량 설정 (빠른 테스트)
uv run python src/pipeline.py --mode train --light

# 테스트 모드 (더미 데이터로 파이프라인 확인)
uv run python src/pipeline.py --mode test
```

### 2. 평가 (Evaluation)

```bash
uv run python src/pipeline.py --mode eval \
    --checkpoint checkpoints/best_model.pth
```

### 3. 예측 (Prediction)

```bash
# 단일 이미지 예측 (CNN만 사용)
uv run python src/pipeline.py --mode predict \
    --image path/to/image.jpg \
    --checkpoint checkpoints/best_model.pth
```

### 명령줄 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode` | 실행 모드 (train/eval/predict/test) | - |
| `--epochs` | 학습 에폭 수 | 30 |
| `--batch-size` | 배치 크기 | 32 |
| `--lr` | 학습률 | 0.0001 |
| `--backbone` | CNN 백본 (resnet18/34/50, efficientnet_b0) | resnet18 |
| `--fusion` | 융합 방식 (concat/attention/weighted_sum/gated) | concat |
| `--no-facenet` | FaceNet 브랜치 비활성화 | False |
| `--no-cnn` | CNN 브랜치 비활성화 | False |
| `--facenet-vectors` | FaceNet 벡터 파일 경로 | `data/cropped/facenet_vectors.pt` |
| `--checkpoint` | 체크포인트 경로 | - |
| `--image` | 예측할 이미지 경로 | - |
| `--name` | 실험 이름 | auto |
| `--light` | 경량 설정 사용 | False |

## 🔧 모델 구성

### FaceNet Branch (사전 추출 방식)
- **모델**: InceptionResnetV1 (facenet-pytorch)
- **사전학습**: VGGFace2 또는 CASIA-WebFace
- **역할**: 얼굴 이미지에서 512차원 임베딩 벡터 추출 → `.pt` 파일로 저장
- **출력**: 512차원 L2 정규화된 벡터

### CNN Branch
- **백본**: ResNet18/34/50, EfficientNet-B0/B1
- **역할**: 이미지 전역 특징 추출 (실시간)
- **출력**: 512차원 특징 벡터

### Feature Fusion
- **concat**: 단순 연결 후 MLP (기본값)
- **attention**: 크로스 어텐션 기반 융합
- **weighted_sum**: 학습 가능한 가중치 합
- **gated**: 게이트 메커니즘

### Classification Head
- Multi-layer MLP with LayerNorm, Dropout
- 6 클래스 출력

## ⚡ 성능 최적화

### 자동 적용 최적화

파이프라인 실행 시 다음 최적화가 자동으로 적용됩니다:

| 최적화 | 설명 |
|--------|------|
| **cuDNN Benchmark** | 입력 크기에 최적화된 알고리즘 자동 선택 |
| **TF32** | Ampere 이상 GPU에서 행렬 연산 가속 |
| **AMP (Mixed Precision)** | FP16/FP32 혼합 정밀도로 메모리 절약 및 속도 향상 |
| **멀티스레딩 이미지 로딩** | ThreadPoolExecutor로 이미지 병렬 로드 |
| **최적 Worker 수 자동 설정** | CPU 코어 수 기반 DataLoader worker 자동 조정 |
| **pin_memory** | GPU 전송 속도 향상 |
| **persistent_workers** | Worker 재사용으로 오버헤드 감소 |

## 📁 프로젝트 구조

```
emotion-extractor/
├── src/
│   ├── config.py              # 설정 (FaceNetConfig 포함)
│   ├── pipeline.py            # 메인 파이프라인
│   │
│   ├── data/
│   │   ├── dataset.py         # EmotionDataset, EmotionDatasetWithFaceNet
│   │   ├── extract_facenet_vectors.py  # FaceNet 벡터 추출 스크립트
│   │   └── crop_and_extract.py
│   │
│   ├── models/
│   │   ├── cnn_branch.py      # CNN 특징 추출기
│   │   ├── hybrid_model.py    # 하이브리드 모델
│   │   └── facenet_feature.py # FaceNet Feature Projector
│   │
│   ├── training/
│   │   └── trainer.py         # Trainer, evaluate_model
│   │
│   └── util/
│       └── plotting.py        # 시각화 유틸리티
│
├── data/
│   └── cropped/
│       ├── images/            # 클래스별 이미지 폴더
│       │   ├── 당황/
│       │   ├── 분노/
│       │   ├── 불안/
│       │   ├── 상처/
│       │   ├── 슬픔/
│       │   └── 중립/
│       └── facenet_vectors.pt # 사전 추출된 FaceNet 벡터
│
├── checkpoints/               # 모델 체크포인트
├── runs/                      # TensorBoard 로그
└── pyproject.toml
```

## 📊 FaceNet 벡터 파일 구조

`facenet_vectors.pt` 파일의 구조:

```python
{
    "vectors": {
        "image1.jpg": tensor([...]),  # 512차원, L2 정규화
        "image2.jpg": tensor([...]),
        ...
    },
    "model_name": "facenet-vggface2",
    "feature_dim": 512,
    "image_size": 160,
    "num_images": 5000,
    "classes": ["당황", "분노", "불안", "상처", "슬픔", "중립"],
    "normalized": True  # L2 정규화 여부
}
```

## 📊 데이터 설정

- **학습/검증 비율**: 90% / 10%
- **이미지 크기**: 224 x 224 (CNN), 160 x 160 (FaceNet)
- **정규화**: ImageNet mean/std (CNN), [-1, 1] (FaceNet)
- **Augmentation**: RandomCrop, RandomFlip, ColorJitter, Rotation

## 📈 TensorBoard 모니터링

학습 중 TensorBoard로 메트릭을 모니터링할 수 있습니다:

```bash
tensorboard --logdir=runs
```

## 🧪 빠른 시작

```bash
# 1. 환경 설정
uv sync

# 2. FaceNet 벡터 추출 (FaceNet 브랜치 사용 시)
uv run python src/data/extract_facenet_vectors.py --data-root ./data/cropped/images

# 3. 학습
uv run python src/pipeline.py --mode train

# 또는 CNN만 사용 (FaceNet 추출 불필요)
uv run python src/pipeline.py --mode train --no-facenet
```

## 📝 변경 이력

- **v1.0**: 초기 버전
- **v2.0**: YOLO 런타임 추출 → 사전 추출 방식으로 변경
- **v2.1**: CUDA 필수 체크 추가, 멀티스레딩 가속화
- **v3.0**: **YOLO → FaceNet으로 전면 교체**
  - 얼굴 인식에 특화된 FaceNet 백본 사용
  - 512차원 L2 정규화 임베딩
  - VGGFace2 / CASIA-WebFace 사전학습 지원
  - 감정 인식 정확도 향상 기대 (65-72% → 75-82%)

## 📚 참고

- [FaceNet PyTorch](https://github.com/timesler/facenet-pytorch)
- [FaceNet Paper](https://arxiv.org/abs/1503.03832)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)