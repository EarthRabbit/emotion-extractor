# 📊 Plotting 가이드 - 모델 성능 시각화

## 🎯 개요

모델의 학습 결과를 시각화하는 완벽한 가이드입니다. 다양한 차트와 그래프를 생성하여 모델 성능을 분석할 수 있습니다.

---

## ⚡ 빠른 시작 (5분)

### 1. 예제 플롯 생성 (가장 빠름)
```bash
python plot_results.py --example
```

**생성되는 플롯:**
- `plots/example_experiment_training_history.png` - 학습 곡선
- `plots/class_accuracy.png` - 클래스별 정확도
- `plots/class_distribution.png` - 클래스 분포

### 2. 자신의 결과 시각화
```bash
# 학습 완료 후
python plot_results.py --history results/history.json --name my_experiment
```

### 3. TensorBoard에서 실시간 확인 (권장)
```bash
# 학습 중에 새 터미널에서 실행
tensorboard --logdir=runs
# http://localhost:6006 에서 확인
```

---

## 🎨 플롯 종류별 사용법

### A. 학습 곡선 (Training History)

#### 방법 1: 코드로 직접 생성
```python
from src.util.plotting import TrainingPlotter

# 학습 히스토리 데이터
history = {
    'train_loss': [0.8, 0.6, 0.4, 0.3, 0.25],
    'val_loss': [0.85, 0.65, 0.5, 0.45, 0.50],
    'train_acc': [0.65, 0.75, 0.82, 0.86, 0.88],
    'val_acc': [0.60, 0.68, 0.72, 0.73, 0.71],
    'learning_rate': [1e-4, 1e-4, 5e-5, 5e-5, 1e-5]
}

# 플롯 생성
plotter = TrainingPlotter(figsize=(15, 10))
plotter.plot_training_history(
    history,
    save_path='plots/training_history.png',
    title='My Experiment'
)
```

**생성되는 4개 서브플롯:**
1. **Loss 곡선** - Train/Val Loss 추이
2. **Accuracy 곡선** - Train/Val Accuracy 추이
3. **Learning Rate** - 에폭별 학습률 변화
4. **Overfitting Gap** - Train-Val 차이 분석

#### 방법 2: 명령어로 생성
```bash
python plot_results.py --history path/to/history.json --name experiment_v1
```

**예상 출력:**
```
📊 실험 결과: experiment_v1
============================================================
📈 최종 성능:
  - 최종 Val Accuracy: 65.32%
  - 최고 Val Accuracy: 66.45%
  - 최종 Train Accuracy: 82.15%
  - Overfitting Gap: 16.83%
✅ 저장됨: plots/experiment_v1_training_history.png
```

---

### B. 클래스별 성능 (Per-Class Accuracy)

#### 방법 1: 코드로 생성
```python
from src.util.plotting import ClassPerformancePlotter

# 클래스별 정확도
class_accuracies = {
    '중립': 0.75,
    '분노': 0.69,
    '당황': 0.67,
    '슬픔': 0.60,
    '불안': 0.50,
    '상처': 0.45
}

# 플롯
ClassPerformancePlotter.plot_class_accuracy(
    class_accuracies,
    save_path='plots/class_accuracy.png'
)
```

**특징:**
- 정확도가 높은 순서대로 정렬
- 70% 이상: 🟢 (녹색)
- 60-70%: 🟠 (주황색)
- 60% 미만: 🔴 (빨강색)

#### 방법 2: 실제 예측값으로 생성
```python
from src.util.plotting import ClassPerformancePlotter

y_true = [0, 1, 2, 1, 0, ...]  # 실제 레이블
y_pred = [0, 1, 2, 0, 0, ...]  # 예측 레이블
class_names = ['중립', '분노', '불안', '상처', '슬픔', '당황']

# Precision, Recall, F1-Score 플롯
ClassPerformancePlotter.plot_class_metrics(
    y_true,
    y_pred,
    class_names,
    save_path='plots/class_metrics.png'
)
```

**생성되는 차트:**
- Precision (정밀도): 예측한 것 중 맞은 비율
- Recall (재현율): 실제 것 중 맞게 예측한 비율
- F1-Score: Precision과 Recall의 조화 평균

---

### C. 혼동 행렬 (Confusion Matrix)

#### 기본 사용법
```python
from src.util.plotting import ConfusionMatrixPlotter

y_true = [0, 1, 2, 1, 0, 2, 1, ...]  # 실제 레이블
y_pred = [0, 1, 2, 0, 0, 2, 1, ...]  # 예측 레이블
class_names = ['중립', '분노', '불안', '상처', '슬픔', '당황']

# 절대값 혼동 행렬
ConfusionMatrixPlotter.plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path='plots/confusion_matrix.png',
    normalize=False  # 개수 표시
)

# 정규화된 혼동 행렬 (백분율)
ConfusionMatrixPlotter.plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path='plots/confusion_matrix_normalized.png',
    normalize=True  # 백분율 표시
)
```

**읽는 법:**
- 대각선: 올바른 예측 (어둡을수록 좋음)
- 비대각선: 오분류 (밝을수록 좋음)
- 행: 실제 클래스
- 열: 예측 클래스

**예:**
```
          예측: 중립  분노  불안  상처  슬픔  당황
실제: 중립  45    2     1    0     2    0
      분노   3    30    2    1     1    3
      불안  5    4    15    2     3    1
      ...
```

---

### D. 데이터셋 분포 (Dataset Distribution)

#### 방법 1: 클래스 분포
```python
from src.util.plotting import DatasetPlotter

class_counts = {
    '중립': 500,
    '분노': 400,
    '당황': 380,
    '슬픔': 300,
    '불안': 200,
    '상처': 150
}

# 막대 + 파이 차트
DatasetPlotter.plot_class_distribution(
    class_counts,
    save_path='plots/class_distribution.png'
)
```

**생성되는 차트:**
- 왼쪽: 막대 차트 (개수)
- 오른쪽: 파이 차트 (백분율)

#### 방법 2: 불균형 비율
```python
DatasetPlotter.plot_imbalance_ratio(
    class_counts,
    save_path='plots/imbalance_ratio.png'
)
```

**표시 정보:**
- 불균형 비율 (Imbalance Ratio)
- 예: 500 / 150 = 3.33x

---

### E. 실험 비교 (Experiment Comparison)

#### 여러 실험 비교
```python
from src.util.plotting import ComparisonPlotter

# 여러 실험의 결과
experiments = {
    'baseline': {
        'train_loss': [0.8, 0.6, 0.4, 0.3],
        'val_loss': [0.85, 0.65, 0.5, 0.45],
        'train_acc': [0.65, 0.75, 0.82, 0.86],
        'val_acc': [0.60, 0.68, 0.72, 0.73]
    },
    'improved_v1': {
        'train_loss': [0.75, 0.55, 0.35, 0.28],
        'val_loss': [0.80, 0.60, 0.45, 0.42],
        'train_acc': [0.68, 0.78, 0.85, 0.88],
        'val_acc': [0.65, 0.72, 0.75, 0.76]
    },
    'improved_v2': {
        'train_loss': [0.70, 0.50, 0.32, 0.25],
        'val_loss': [0.75, 0.55, 0.42, 0.40],
        'train_acc': [0.70, 0.80, 0.87, 0.90],
        'val_acc': [0.68, 0.74, 0.77, 0.78]
    }
}

# 성능 곡선 비교
ComparisonPlotter.plot_experiment_comparison(
    experiments,
    save_path='plots/experiment_comparison.png'
)

# 최종 성능 비교
final_metrics = {
    'baseline': {'val_acc': 0.73},
    'improved_v1': {'val_acc': 0.76},
    'improved_v2': {'val_acc': 0.78}
}
ComparisonPlotter.plot_final_metrics_comparison(
    final_metrics,
    save_path='plots/final_metrics.png'
)
```

---

## 🔧 ResultsVisualizer 사용법

통합 시각화 도구 사용:

```python
from pathlib import Path
from plot_results import ResultsVisualizer

# 시각화 도구 초기화
visualizer = ResultsVisualizer(output_dir=Path('plots'))

# 1. 단일 실험 시각화
history = {
    'train_loss': [...],
    'val_loss': [...],
    'train_acc': [...],
    'val_acc': [...]
}
visualizer.plot_single_experiment(
    history,
    experiment_name='my_experiment'
)

# 2. 클래스별 성능 시각화
class_accuracies = {'중립': 0.75, '분노': 0.69, ...}
y_true = [0, 1, 2, ...]
y_pred = [0, 1, 2, ...]
class_names = ['중립', '분노', '불안', ...]

visualizer.plot_class_performance(
    class_accuracies,
    y_true=y_true,
    y_pred=y_pred,
    class_names=class_names
)

# 3. 데이터셋 분포 시각화
class_counts = {'중립': 500, '분노': 400, ...}
visualizer.plot_dataset_distribution(class_counts)

# 4. 실험 비교
experiments = {
    'exp1': {'val_acc': [...], ...},
    'exp2': {'val_acc': [...], ...}
}
visualizer.plot_experiment_comparison(experiments)
```

---

## 📊 명령어 라인 인터페이스 (CLI)

### 기본 명령어

```bash
# 예제 플롯 생성
python plot_results.py --example

# 히스토리 파일에서 플롯
python plot_results.py --history results/history.json --name experiment_v1

# 체크포인트에서 플롯
python plot_results.py --checkpoint checkpoints/my_model/best_model.pt --name my_exp

# 데이터셋 분포 시각화
python plot_results.py --dataset-distribution

# 모든 플롯 생성 (예제)
python plot_results.py --all

# 플롯 저장하지 않고 표시만
python plot_results.py --example --no-save

# 다른 디렉토리에 저장
python plot_results.py --example --output-dir my_plots
```

### 옵션
- `--example`: 예제 플롯 생성
- `--history PATH`: JSON 히스토리 파일
- `--checkpoint PATH`: PyTorch 체크포인트
- `--name STR`: 실험 이름
- `--dataset-distribution`: 데이터셋 분포 시각화
- `--output-dir PATH`: 저장 디렉토리 (기본: plots)
- `--no-save`: 저장하지 않음 (표시만)
- `--all`: 모든 예제 생성

---

## 🎓 실제 사용 예제

### 예제 1: 학습 완료 후 결과 분석

```python
# 1. 학습 완료 후 히스토리 저장
history = trainer.train()

# 2. 히스토리 플롯
from src.util.plotting import TrainingPlotter
plotter = TrainingPlotter()
plotter.plot_training_history(
    history,
    save_path='plots/training_results.png',
    title='Overfitting Reduction v1'
)

# 3. 검증 데이터로 클래스별 성능 확인
from src.util.plotting import ClassPerformancePlotter
import numpy as np

y_pred = []
y_true = []
for batch in val_loader:
    images = batch['image'].to(device)
    labels = batch['label'].to(device)
    
    outputs = model(images)
    logits = outputs['logits']
    preds = logits.argmax(dim=1)
    
    y_pred.extend(preds.cpu().numpy())
    y_true.extend(labels.cpu().numpy())

class_accuracies = {}
for i, class_name in enumerate(class_names):
    mask = np.array(y_true) == i
    acc = (np.array(y_pred)[mask] == i).mean()
    class_accuracies[class_name] = acc

ClassPerformancePlotter.plot_class_accuracy(class_accuracies)
```

### 예제 2: 여러 실험 비교

```bash
# 실험 1: 기본 설정
python train_improved.py --config default --name baseline

# 실험 2: 과적합 감소
python train_improved.py --config overfitting_reduction --name improved_v1

# 실험 3: Attention Fusion
python train_improved.py \
  --config overfitting_reduction \
  --fusion attention \
  --backbone resnet34 \
  --name improved_v2

# 모든 결과 비교 시각화
python plot_results.py --example  # 템플릿 확인
```

### 예제 3: TensorBoard 사용 (실시간)

```bash
# 터미널 1: 학습
python train_improved.py --config overfitting_reduction

# 터미널 2: TensorBoard (동시 실행)
tensorboard --logdir=runs

# 브라우저: http://localhost:6006
```

---

## 📈 플롯 해석 방법

### 학습 곡선 해석

#### 1. Loss 곡선
```
이상적:              과적합:               학습 부족:
Val Loss ┐          Val Loss ┐          Val Loss ┐
         │\                  │\                   │
         │ └──            ╱──│ └──             ╱──│
Train Loss│            ╱     │ Train Loss    ╱   │
         └──────────────     └──────────────     └───
  ✅ 좋음      ⚠️ 주의       ❌ 나쁨
```

#### 2. Accuracy 곡선
```
이상적:              과적합:               불균형:
Val Acc ┌──          Val Acc ┌            Val Acc ┌──
        │  \                 │  \                 │
        │   └──────      ╱───│   └──           ╱─│
Train Acc│                   │ Train Acc      ╱  │
        └────────────────────┘────────────────────┘
  ✅ Gap < 10%   ⚠️ Gap 15%   ❌ 기타 문제
```

### 혼동 행렬 해석

```
대각선이 어두운 상태 = 좋은 성능
┌─────────────────────────┐
│ ██  .  .  .  .  .  |
│  .  ██  .  .  .  . |  ✅ 클래스 예측 잘함
│  .  .  ██  .  .  . |
│  .  .  .  ██  .  . |
│  .  .  .  .  .  . | (밝은 색 = 오분류)
│  .  .  .  .  .  ██ |
└─────────────────────────┘
```

### 클래스별 정확도 해석

```
60-70% (노란색):
└─ 데이터 부족 또는 특징 유사성 높음
└─ 해결: 데이터 수집 또는 클래스별 증강

50-60% (주황색):
└─ 심각한 문제
└─ 해결: 데이터 수집 필수

45% 이하 (빨강색):
└─ 매우 심각한 문제
└─ 해결: 데이터 수집 + 모델 최적화 필수
```

---

## 🎯 플롯 저장 위치

기본 저장 구조:
```
emotion-extractor/
├── plots/                              # 모든 플롯
│   ├── training_history.png            # 학습 곡선
│   ├── class_accuracy.png              # 클래스별 정확도
│   ├── class_metrics.png               # Precision/Recall/F1
│   ├── confusion_matrix.png            # 혼동 행렬 (절대값)
│   ├── confusion_matrix_normalized.png # 혼동 행렬 (백분율)
│   ├── class_distribution.png          # 클래스 분포
│   ├── imbalance_ratio.png             # 불균형 비율
│   ├── experiment_comparison.png       # 실험 비교
│   └── final_metrics_comparison.png    # 최종 메트릭 비교
├── runs/                               # TensorBoard 로그
│   └── hybrid_emotion_20251216_*/
└── checkpoints/                        # 모델 체크포인트
    └── improved_overfittingreduction/
```

---

## 💡 팁과 트릭

### 1. 고해상도 플롯 저장
```python
# 기본 DPI: 300
# 매우 고해상도가 필요한 경우
plotter.plot_training_history(
    history,
    save_path='plots/high_quality.png'
)
# PNG는 자동으로 300 DPI로 저장됨
```

### 2. 커스텀 스타일
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 다크 스타일
sns.set_style("darkgrid")
plt.style.use('seaborn-v0_8-darkgrid')

# 다시 플롯
plotter.plot_training_history(history)
```

### 3. 배치로 여러 플롯 생성
```python
from pathlib import Path

results_dir = Path('results')
for exp_file in results_dir.glob('*.json'):
    visualizer = ResultsVisualizer()
    history = visualizer.load_history_from_json(exp_file)
    visualizer.plot_single_experiment(
        history,
        experiment_name=exp_file.stem
    )
```

---

## ❓ FAQ

### Q: 플롯이 표시되지 않음
**A:** Jupyter를 사용하는 경우:
```python
%matplotlib inline
# 또는
%matplotlib notebook
```

### Q: 한글이 깨짐
**A:** 폰트 설정 확인:
```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
# 또는
plt.rcParams['font.family'] = 'SimHei'  # Windows
```

### Q: 매우 큰 플롯 파일 크기
**A:** DPI 감소:
```python
plt.savefig('plot.png', dpi=150)  # 300에서 150으로
```

### Q: 실시간으로 모니터링하고 싶음
**A:** TensorBoard 사용:
```bash
tensorboard --logdir=runs
```

---

## 📚 추가 리소스

- **`src/util/plotting.py`** - 전체 플롯팅 코드
- **`plot_results.py`** - CLI 도구
- **`QUICK_START_IMPROVED.md`** - 성능 개선 가이드

---

**🎉 이제 모델 성능을 완벽하게 시각화할 수 있습니다!**
