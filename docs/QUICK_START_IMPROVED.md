# 🚀 감정 분류 모델 성능 개선 - 빠른 시작 가이드

## 📌 현재 상황
- **Train Accuracy**: 83.94% ✅
- **Validation Accuracy**: 61.62% ⚠️
- **문제**: 과적합 (Train-Val Gap: 22%)

---

## ⚡ 1단계: 가장 빠른 개선 (5분)

### 개선된 설정으로 바로 실행하기

```bash
# 과적합 감소 설정으로 학습 (강추!)
python train_improved.py --config overfitting_reduction

# 또는 클래스 불균형 처리
python train_improved.py --config class_balanced
```

**예상 성능**: 61.62% → **63-65%**

---

## 🎯 2단계: 커스텀 설정으로 최적화 (10-15분)

### 현재 설정에서 선택적으로 조정

```bash
# Attention Fusion + 높은 Dropout
python train_improved.py \
  --config overfitting_reduction \
  --fusion attention \
  --dropout 0.5 \
  --backbone resnet34

# 또는 더 많은 에폭
python train_improved.py \
  --config overfitting_reduction \
  --epochs 60 \
  --name my_improved_v2
```

**예상 성능**: 63-65% → **65-67%**

---

## 📊 3단계: 성능 모니터링

### TensorBoard에서 실시간 확인

```bash
# 새 터미널 열어서 실행
tensorboard --logdir=runs
```

그 후 브라우저에서 `http://localhost:6006` 접속

### 확인할 메트릭
- **Scalars**: Loss, Accuracy 곡선
- **Validation Accuracy**: 70% 이상이 목표
- **Train-Val Gap**: 10% 이하가 이상적

---

## 🔧 설정 옵션 설명

### 기본 설정 프리셋

| 설정 | 사용 상황 | 예상 성능 |
|------|---------|---------|
| `default` | 원본 설정 | 61% |
| `light` | 빠른 테스트 | - |
| `heavy` | 고성능 | 62-63% |
| **`overfitting_reduction`** | **과적합 줄이기** | **63-65%** ⭐ |
| **`class_balanced`** | **클래스 불균형** | **63-65%** ⭐ |
| `cnn_only` | YOLO 제외 | 60% |

### 주요 파라미터

```bash
# 학습률 조정 (낮을수록 느리지만 정확)
--lr 1e-5          # 매우 낮음 (세밀한 조정)
--lr 5e-5          # 낮음 (기본)
--lr 1e-4          # 높음 (빠른 수렴)

# 드롭아웃 (높을수록 과적합 방지)
--dropout 0.3      # 약함
--dropout 0.5      # 중간 (권장)
--dropout 0.7      # 강함

# 데이터 증강
--augmentation light    # 약함
--augmentation medium   # 중간
--augmentation heavy    # 강함 (권장)

# Fusion 방식
--fusion concat         # 현재 (61%)
--fusion attention      # 권장 (63-65%)
--fusion weighted_sum   # 대안
--fusion gated          # 실험용

# CNN 백본
--backbone resnet18    # 기본 (빠름)
--backbone resnet34    # 권장 (더 강력)
--backbone efficientnet_b0  # 효율적
```

---

## 📈 예상 개선 경로

```
초기 상태
├─ Train: 83.94%
├─ Val: 61.62%
└─ Gap: 22%

           ↓ overfitting_reduction 적용

1단계 (예상)
├─ Train: 81-82%
├─ Val: 63-65% ← 개선!
└─ Gap: 18% ← 감소!

           ↓ attention fusion + resnet34 추가

2단계 (예상)
├─ Train: 82-83%
├─ Val: 65-67% ← 더 개선!
└─ Gap: 16% ← 더 감소!

           ↓ 데이터 수집/증강 (시간 필요)

3단계 (예상)
├─ Train: 80-82%
├─ Val: 68-72% ← 큰 개선!
└─ Gap: 10% ← 이상적 수준!
```

---

## 🎨 클래스별 성능 개선

### 현재 성능
| 클래스 | 정확도 | 상태 |
|--------|--------|------|
| 중립 | 75.92% | ✅ |
| 분노 | 69.41% | ✅ |
| 당황 | 67.20% | ✅ |
| 슬픔 | 60.51% | ⚠️ |
| **불안** | **50.20%** | ❌ |
| **상처** | **45.82%** | ❌ |

### 개선 방법

#### 방법 1: 자동 분석 (추천)
```bash
# 클래스 불균형 분석
python -c "
from src.util.class_balancer import ClassBalancer
from src.data.dataset import EmotionDataset

dataset = EmotionDataset(
    images_dir='data/cropped/images',
    class_names=['당황', '분노', '불안', '상처', '슬픔', '중립']
)

balancer = ClassBalancer(dataset, dataset.class_names)
balancer.print_analysis()
balancer.print_augmentation_strategy()
"
```

#### 방법 2: 수동 설정
```bash
# 불안, 상처 클래스에 추가 데이터 수집
# 또는 클래스별 맞춤형 증강 적용

python train_improved.py \
  --config class_balanced \
  --augmentation heavy \
  --dropout 0.5
```

---

## 💾 결과 저장 및 비교

### 체크포인트 위치
```
checkpoints/
├── improved_overfittingreduction/
│   └── best_model.pt
├── improved_classbalanced/
│   └── best_model.pt
└── ...
```

### 결과 비교
```bash
# TensorBoard에서 다중 실험 비교
tensorboard --logdir=runs

# 각 실험의 스칼라 값 확인
# 차트 우측 상단의 체크박스로 여러 실험 동시 표시
```

---

## 🔍 트러블슈팅

### Q: Val Accuracy가 안 올라갈 때?
```bash
# 원인: 모델 복잡도가 낮음
# 해결: 모델 강화
python train_improved.py \
  --config overfitting_reduction \
  --backbone resnet34 \
  --fusion attention
```

### Q: Train Accuracy가 떨어질 때?
```bash
# 원인: 정규화가 너무 강함
# 해결: 정규화 약화
python train_improved.py \
  --config overfitting_reduction \
  --dropout 0.3 \
  --lr 1e-4
```

### Q: 특정 클래스만 성능이 낮을 때?
```bash
# 원인: 클래스 불균형
# 해결: 클래스 균형 설정
python train_improved.py --config class_balanced
```

### Q: 학습이 너무 느릴 때?
```bash
# 원인: batch_size가 작음 또는 모델 복잡도 높음
# 해결: batch size 증가 또는 모델 단순화
python train_improved.py \
  --config overfitting_reduction \
  --batch-size 64 \
  --backbone resnet18
```

---

## 📚 상세 가이드

더 자세한 정보는 다음 파일을 참조하세요:

- **`docs/IMPROVEMENT_GUIDE.md`** - 전체 개선 전략 상세 설명
- **`docs/QUICK_START_IMPROVED.md`** - 이 파일 (빠른 시작)
- **`src/config.py`** - 설정 정의 및 프리셋

---

## 🎓 권장 학습 순서

### 1단계: 빠른 확인 (20분)
```bash
# 가장 빠른 개선 방법 시도
python train_improved.py --config overfitting_reduction
```
- ✅ 예상 Val Acc: 63-65%

### 2단계: 최적화 (40분)
```bash
# 커스텀 설정으로 미세 조정
python train_improved.py \
  --config overfitting_reduction \
  --fusion attention \
  --backbone resnet34
```
- ✅ 예상 Val Acc: 65-67%

### 3단계: 앙상블 (추가 시간)
```bash
# 여러 모델로 앙상블 (최종 성능)
# 구현은 별도 스크립트에서 진행
```
- ✅ 예상 Val Acc: 67-70%

---

## 📞 도움말

### 명령어 도움말
```bash
python train_improved.py --help
```

### 설정 옵션 확인
```bash
python -c "from src.config import get_overfitting_reduction_config; c = get_overfitting_reduction_config(); print(c.__dict__)"
```

### 데이터셋 분석
```bash
python -c "
from src.data.dataset import EmotionDataset
from src.util.class_balancer import ClassBalancer

dataset = EmotionDataset(
    images_dir='data/cropped/images',
    class_names=['당황', '분노', '불안', '상처', '슬픔', '중립']
)
balancer = ClassBalancer(dataset, dataset.class_names)
balancer.print_analysis()
"
```

---

## ✨ 핵심 요점 정리

| 항목 | 현재 | 목표 | 방법 |
|------|------|------|------|
| Val Accuracy | 61.62% | 70%+ | overfitting_reduction + attention |
| Train-Val Gap | 22% | 10% | 높은 dropout + augmentation |
| 불안 클래스 | 50.20% | 60%+ | class_balanced config |
| 상처 클래스 | 45.82% | 60%+ | 클래스별 증강 강화 |

---

**🚀 지금 바로 시작하세요!**

```bash
python train_improved.py --config overfitting_reduction
```

5분 후 개선된 성능을 확인할 수 있습니다! 🎉
