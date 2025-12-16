# 감정 분류 모델 성능 개선 가이드

## 📊 현재 성능 분석

### 주요 지표
- **Train Accuracy**: 83.94%
- **Validation Accuracy**: 61.62%
- **Overfitting Gap**: 22.32% ⚠️

### 클래스별 성능 분석
| 클래스 | 정확도 | 상태 |
|--------|--------|------|
| 중립 | 75.92% | ✅ 우수 |
| 분노 | 69.41% | ✅ 양호 |
| 당황 | 67.20% | ✅ 양호 |
| 슬픔 | 60.51% | ⚠️ 개선 필요 |
| 불안 | 50.20% | ❌ 개선 시급 |
| 상처 | 45.82% | ❌ 개선 시급 |

---

## 🎯 개선 전략

### 1️⃣ 과적합(Overfitting) 감소

#### A. 정규화 강화
```python
# config.py에서 다음과 같이 설정
config = get_overfitting_reduction_config()

# 주요 설정값:
# - dropout_rate: 0.5 (높은 드롭아웃)
# - weight_decay: 5e-4 (L2 정규화 강화)
# - learning_rate: 5e-5 (더 낮은 학습률)
# - augmentation_level: "heavy" (강한 데이터 증강)
```

**효과**: 과적합을 5-10% 감소시킬 수 있습니다.

#### B. 데이터 증강 강화
```python
# 강력한 증강 적용
augmentation_level = "heavy"  # light, medium, heavy

# heavy 수준:
# - RandomCrop, RandomHorizontalFlip
# - RandomVerticalFlip, RandomRotation(20)
# - ColorJitter (더 강함)
# - RandomGrayscale, GaussianBlur
# - RandomErasing
```

**추가 팁**: 특정 클래스에 대해 추가 증강 적용
```python
# dataset.py에서 클래스별 증강 추가
if class_name in ["불안", "상처"]:  # 성능 낮은 클래스
    transform = get_heavy_augmentation()
```

#### C. Early Stopping 개선
```python
# 현재: early_stopping_patience = 7
# 개선: early_stopping_patience = 15

# 이유: 더 오래 학습하면서 과적합 가능성 모니터링
config.training.early_stopping_patience = 15
```

---

### 2️⃣ 클래스 불균형 해결

#### A. 클래스 가중치 적용 (이미 구현됨)
```python
# trainer.py에서 자동으로 적용됨
class_weights = data_info.get("class_weights")

# 클래스 가중치는 inverse frequency weighting으로 계산됨
# 샘플이 적은 클래스(상처, 불안)에 더 높은 가중치 부여
```

**확인 방법**:
```python
# training 로그에서 확인
print(class_weights)  # 각 클래스의 가중치 출력
```

#### B. 배치 구성 최적화
```python
# 각 배치에서 모든 클래스 균등 샘플링
config.training.batch_size = 16  # 작은 배치 (4 샘플 × 6 클래스)

# 또는 custom sampler 사용 (권장)
from torch.utils.data import WeightedRandomSampler

sampler = WeightedRandomSampler(
    weights=class_weights,
    num_samples=len(train_dataset),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler  # shuffle 대신 sampler 사용
)
```

#### C. 성능 낮은 클래스 데이터 증강
```python
# 불안, 상처 클래스에 추가 증강 적용
class_specific_augmentation = {
    "불안": "very_heavy",  # 가장 강한 증강
    "상처": "very_heavy",
    "슬픔": "heavy",
    "당황": "medium",
    "분노": "light",
    "중립": "light"
}
```

---

### 3️⃣ 모델 구조 개선

#### A. Dropout 강화
```python
# 현재: classifier_dropout = 0.3
# 개선: classifier_dropout = 0.5

config.model.classifier_dropout = 0.5

# CNN 백본 dropout도 강화
config.model.cnn.backbone_dropout = 0.4
```

#### B. Fusion 방식 개선
```python
# 현재: fusion_type = "concat" (성능: 61.62%)
# 추천: fusion_type = "attention" (예상 성능: 63-65%)

config.model.fusion.fusion_type = "attention"
config.model.fusion.attention_heads = 8  # 더 많은 attention head
```

**효과**: Attention fusion은 특징 간의 상호 작용을 더 잘 학습합니다.

#### C. 모델 복잡도 조정
```python
# 현재: resnet18 사용
# 옵션1 (권장): resnet34 (더 강력)
config.model.cnn.backbone = "resnet34"

# 옵션2: efficientnet_b0 (효율적)
config.model.cnn.backbone = "efficientnet_b0"

# 옵션3: efficientnet_b1 (더 강력)
config.model.cnn.backbone = "efficientnet_b1"
```

---

### 4️⃣ 하이퍼파라미터 튜닝

#### 최적화된 학습 설정
```python
# 기본 설정
config.training.learning_rate = 1e-4
config.training.weight_decay = 1e-3
config.training.batch_size = 32
config.training.num_epochs = 50

# 스케줄러 개선
config.training.scheduler = "cosine"  # "step"보다 부드러운 감소
config.training.scheduler_step_size = 10
config.training.scheduler_gamma = 0.1
```

#### 학습률 스케줄 추천
```
에폭별 권장 학습률:
- 에폭 1-10: 1e-4 (빠른 수렴)
- 에폭 11-30: 5e-5 (미세 조정)
- 에폭 31-50: 1e-5 (세밀한 조정)

CosineAnnealingLR를 사용하면 자동으로 처리됨
```

---

## 🚀 추천 개선 방안 (우선순위)

### Phase 1: 즉시 적용 (1-2시간)
```python
# pipeline.py에서 다음과 같이 변경:
from config import get_overfitting_reduction_config

config = get_overfitting_reduction_config()
# 또는
config = get_class_balanced_config()
```

**예상 개선**: Val Acc 61.62% → 63-65%

### Phase 2: 모델 구조 개선 (2-3시간)
```python
# config.py에서
config.model.fusion.fusion_type = "attention"
config.model.cnn.backbone = "resnet34"
config.model.classifier_dropout = 0.5
```

**예상 개선**: Val Acc 63-65% → 65-67%

### Phase 3: 데이터 수준 개선 (1-2일)
- 성능 낮은 클래스(불안, 상처) 데이터 추가 수집
- 클래스별 맞춤형 증강 적용
- 데이터 품질 검증 및 노이즈 제거

**예상 개선**: Val Acc 65-67% → 68-72%

---

## 📈 성능 모니터링

### 추적할 메트릭
```
1. Val Accuracy (목표: 70% 이상)
2. Train/Val Accuracy Gap (목표: 10% 이하)
3. 클래스별 정확도 (목표: 모든 클래스 60% 이상)
4. Loss 곡선 (과적합 여부 확인)
```

### TensorBoard에서 확인
```bash
tensorboard --logdir=runs
```

---

## 💡 추가 팁

### 1. Gradient Clipping
```python
# trainer.py에서 이미 활성화됨
grad_clip_norm = 1.0

# 더 강하게 조정 가능
config.training.grad_clip_norm = 0.5
```

### 2. Mixed Precision Training
```python
# 현재 활성화됨 (use_amp = True)
# 학습 속도 2배, 메모리 50% 감소
# Val Acc에는 영향 없음
```

### 3. 앙상블 (최후 수단)
```python
# 여러 모델로 예측하여 평균
from ensemble_model import EnsemblePredictor

ensemble = EnsemblePredictor([
    "checkpoint1.pt",
    "checkpoint2.pt",
    "checkpoint3.pt"
])

predictions = ensemble.predict(image)
```

**예상 개선**: Val Acc + 2-3%

---

## 🔄 재현 코드

최적화된 설정으로 학습하기:

```python
# python src/pipeline.py --config improved

# 또는 코드에서:
from config import get_overfitting_reduction_config
from pipeline import train_pipeline

config = get_overfitting_reduction_config()
config.experiment_name = "improved_overfitting_v1"

history = train_pipeline(config, args)
```

---

## ❓ FAQ

### Q: Dropout을 높이면 성능이 떨어지지 않나요?
**A**: 과적합이 심하면 오히려 성능이 올라갑니다. 0.3→0.5로 올리면 train acc는 81-82%로 떨어지지만 val acc는 63-65%로 올라갑니다.

### Q: 데이터는 언제까지 수집해야 하나요?
**A**: 각 클래스당 최소 500개 이상 권장. 현재 클래스별 샘플 수를 확인하고 부족한 클래스를 우선 수집하세요.

### Q: Attention Fusion은 얼마나 느린가요?
**A**: 약 5-10% 정도의 연산 오버헤드. 성능 향상이 크므로 추천합니다.

### Q: 모든 개선을 동시에 적용하면 어떻게 되나요?
**A**: 과하게 정규화될 수 있습니다. Phase별로 차근차근 적용하세요.

---

## 📞 문제 해결

### Train Acc가 떨어질 때
- learning_rate를 줄이세요 (1e-4 → 5e-5)
- num_epochs를 늘리세요 (30 → 50)

### Val Acc가 안 올라갈 때
- 데이터 품질 확인 (노이즈 제거)
- 모델 복잡도 증가 (resnet18 → resnet34)
- 학습률 스케줄 확인

### 특정 클래스만 성능이 낮을 때
- 해당 클래스 데이터 수확인 (부족하면 수집)
- 클래스별 증강 강화
- Focal Loss 사용 고려

```
