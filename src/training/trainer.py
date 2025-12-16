"""
Trainer Module for Hybrid Emotion Classification
하이브리드 감정 분류 모델 학습 모듈

학습, 검증, 평가 로직을 담당합니다.

⚠️ CUDA 필수: 이 모듈은 CUDA가 있을 때만 최적 성능을 발휘합니다.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

# ============================================================
# CUDA 최적화 함수
# ============================================================


def check_cuda_available() -> bool:
    """CUDA 사용 가능 여부 확인"""
    return torch.cuda.is_available()


def setup_cuda_optimization():
    """CUDA 최적화 설정"""
    if not check_cuda_available():
        return

    # cuDNN 최적화 활성화
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # 입력 크기가 일정할 때 성능 향상

    # TF32 활성화 (Ampere 이상 GPU에서 성능 향상)
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

    # 메모리 단편화 방지
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def print_cuda_info():
    """CUDA 정보 출력"""
    if check_cuda_available():
        print(f"  - CUDA: ✓ ({torch.cuda.get_device_name(0)})")
        print(f"  - cuDNN: {torch.backends.cudnn.version()}")
        print("  - AMP 지원: ✓")
    else:
        print("  - CUDA: ✗ (CPU 모드)")
        print("  - AMP 지원: ✗")


class EarlyStopping:
    """
    조기 종료 클래스

    검증 손실이 개선되지 않으면 학습을 중단합니다.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Args:
            patience: 개선 없이 기다리는 에폭 수
            min_delta: 개선으로 인정할 최소 변화량
            mode: 'min' (손실) 또는 'max' (정확도)
            verbose: 출력 여부
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        스코어 확인 및 조기 종료 여부 결정

        Args:
            score: 현재 스코어 (손실 또는 정확도)

        Returns:
            개선 여부
        """
        if self.mode == "min":
            score = -score

        if self.best_score is None:
            self.best_score = score
            return True

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping: {self.counter}/{self.patience} "
                    f"(best: {-self.best_score if self.mode == 'min' else self.best_score:.4f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.counter = 0
            return True


class MetricTracker:
    """
    메트릭 추적 클래스

    학습 중 다양한 메트릭을 추적하고 기록합니다.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """메트릭 초기화"""
        self.loss = 0.0
        self.correct = 0
        self.total = 0
        self.batch_count = 0

        # 클래스별 통계
        self.class_correct = {}
        self.class_total = {}

    def update(
        self,
        loss: float,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        메트릭 업데이트

        Args:
            loss: 배치 손실
            predictions: 예측값 [B]
            labels: 정답 [B]
        """
        self.loss += loss
        self.batch_count += 1

        # 정확도 계산
        batch_correct = (predictions == labels).sum().item()
        self.correct += batch_correct
        self.total += labels.size(0)

        # 클래스별 정확도
        for pred, label in zip(predictions.cpu().numpy(), labels.cpu().numpy()):
            label = int(label)
            if label not in self.class_total:
                self.class_total[label] = 0
                self.class_correct[label] = 0

            self.class_total[label] += 1
            if pred == label:
                self.class_correct[label] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """현재 메트릭 반환"""
        avg_loss = self.loss / max(self.batch_count, 1)
        accuracy = 100.0 * self.correct / max(self.total, 1)

        # 클래스별 정확도
        class_acc: Dict[Any, float] = {}
        for label in self.class_total:
            class_acc[label] = (
                100.0
                * self.class_correct.get(label, 0)
                / max(self.class_total[label], 1)
            )

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "class_accuracy": class_acc,
        }


class Trainer:
    """
    모델 학습 클래스

    학습, 검증, 체크포인트 저장 등을 담당합니다.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        # 학습 설정
        num_epochs: int = 30,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        # AMP 설정
        use_amp: bool = True,
        # 로깅 설정
        log_interval: int = 10,
        tensorboard_dir: Optional[Path] = None,
        experiment_name: str = "experiment",
        # 체크포인트 설정
        checkpoint_dir: Optional[Path] = None,
        save_best_only: bool = True,
        # Early stopping
        early_stopping_patience: int = 7,
        # 클래스 가중치
        class_weights: Optional[torch.Tensor] = None,
        # 추가 정보
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model: 학습할 모델
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            optimizer: 옵티마이저 (None이면 AdamW 사용)
            scheduler: 학습률 스케줄러
            criterion: 손실 함수 (None이면 CrossEntropyLoss 사용)
            device: 연산 디바이스
            num_epochs: 학습 에폭 수
            learning_rate: 학습률
            weight_decay: 가중치 감쇠
            use_amp: Mixed Precision 사용 여부
            log_interval: 로그 출력 간격 (배치 단위)
            tensorboard_dir: TensorBoard 로그 디렉토리
            experiment_name: 실험 이름
            checkpoint_dir: 체크포인트 저장 디렉토리
            save_best_only: 최고 성능 모델만 저장 여부
            early_stopping_patience: 조기 종료 patience
            class_weights: 클래스 가중치 (불균형 처리)
            class_names: 클래스 이름 리스트
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.save_best_only = save_best_only
        self.class_names = class_names or []

        # 디바이스 설정
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # CUDA 최적화 적용
        if check_cuda_available():
            setup_cuda_optimization()

        # 옵티마이저
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optimizer

        # 스케줄러
        self.scheduler: Optional[LRScheduler]
        if scheduler is None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=learning_rate * 0.01,
            )
        else:
            self.scheduler = scheduler

        # 손실 함수
        if criterion is None:
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = criterion

        # AMP (Mixed Precision)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler: Optional[GradScaler] = GradScaler("cuda") if self.use_amp else None

        # TensorBoard
        if tensorboard_dir:
            log_dir = (
                Path(tensorboard_dir)
                / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.writer = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard 로그 경로: {log_dir}")
        else:
            self.writer = None

        # 체크포인트
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode="max",  # 정확도 기준
        )

        # 학습 기록
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

        self.best_val_acc = 0.0
        self.current_epoch = 0

        print("\nTrainer 초기화 완료:")
        print(f"  - 디바이스: {self.device}")
        print_cuda_info()
        print(f"  - 에폭 수: {num_epochs}")
        print(f"  - 학습률: {learning_rate}")
        print(f"  - AMP (Mixed Precision): {self.use_amp}")
        print(f"  - 학습 배치 수: {len(train_loader)}")
        print(f"  - 검증 배치 수: {len(val_loader)}")

    def train_one_epoch(self) -> Dict[str, float]:
        """
        1 에폭 학습

        Returns:
            학습 메트릭 딕셔너리
        """
        self.model.train()
        metrics = MetricTracker()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            # 데이터 로드
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # YOLO 벡터 (있는 경우)
            yolo_vector = None
            if "yolo_vector" in batch:
                yolo_vector = batch["yolo_vector"].to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                with torch.autocast(device_type="cuda"):
                    outputs = self.model(images, yolo_vector)
                    logits = outputs["logits"]
                    loss = self.criterion(logits, labels)

                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, yolo_vector)
                logits = outputs["logits"]
                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()

            # 메트릭 업데이트
            predictions = torch.argmax(logits, dim=1)
            metrics.update(loss.item(), predictions, labels)

            # 진행률 표시 업데이트
            current_metrics = metrics.get_metrics()
            pbar.set_postfix(
                {
                    "loss": f"{current_metrics['loss']:.4f}",
                    "acc": f"{current_metrics['accuracy']:.2f}%",
                }
            )

            # TensorBoard 로깅
            if self.writer and (batch_idx + 1) % self.log_interval == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar("Step/Train_Loss", loss.item(), global_step)
                self.writer.add_scalar(
                    "Step/Train_Accuracy", current_metrics["accuracy"], global_step
                )

        return metrics.get_metrics()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        검증 수행

        Returns:
            검증 메트릭 딕셔너리
        """
        self.model.eval()
        metrics = MetricTracker()

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1} [Val]",
            leave=False,
        )

        for batch in pbar:
            # 데이터 로드
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # YOLO 벡터 (있는 경우)
            yolo_vector = None
            if "yolo_vector" in batch:
                yolo_vector = batch["yolo_vector"].to(self.device, non_blocking=True)

            # Forward pass
            if self.use_amp:
                with torch.autocast(device_type="cuda"):
                    outputs = self.model(images, yolo_vector)
                    logits = outputs["logits"]
                    loss = self.criterion(logits, labels)
            else:
                outputs = self.model(images, yolo_vector)
                logits = outputs["logits"]
                loss = self.criterion(logits, labels)

            # 메트릭 업데이트
            predictions = torch.argmax(logits, dim=1)
            metrics.update(loss.item(), predictions, labels)

            # 진행률 표시 업데이트
            current_metrics = metrics.get_metrics()
            pbar.set_postfix(
                {
                    "loss": f"{current_metrics['loss']:.4f}",
                    "acc": f"{current_metrics['accuracy']:.2f}%",
                }
            )

        return metrics.get_metrics()

    def train(self) -> Dict[str, List[float]]:
        """
        전체 학습 수행

        Returns:
            학습 히스토리
        """
        print("\n" + "=" * 60)
        print("학습 시작")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            print(f"\nEpoch [{epoch + 1}/{self.num_epochs}]")
            print("-" * 40)

            # 학습
            train_metrics = self.train_one_epoch()

            # 검증
            val_metrics = self.validate()

            # 스케줄러 업데이트
            if self.scheduler is not None:
                self.scheduler.step()

            # 현재 학습률
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 히스토리 기록
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["learning_rate"].append(current_lr)

            # 에폭 시간
            epoch_time = time.time() - epoch_start

            # 결과 출력
            print(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}%"
            )
            print(
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )
            print(f"Learning Rate: {current_lr:.6f} | Time: {epoch_time:.1f}s")

            # 클래스별 정확도 출력
            class_accuracy = val_metrics.get("class_accuracy")
            if self.class_names and class_accuracy and isinstance(class_accuracy, dict):
                print("클래스별 Val 정확도:")
                for idx, acc in class_accuracy.items():
                    class_name = (
                        self.class_names[idx]
                        if idx < len(self.class_names)
                        else f"Class {idx}"
                    )
                    print(f"  {class_name}: {acc:.2f}%")

            # TensorBoard 에폭 로깅
            if self.writer:
                self.writer.add_scalars(
                    "Epoch/Loss",
                    {"train": train_metrics["loss"], "val": val_metrics["loss"]},
                    epoch + 1,
                )
                self.writer.add_scalars(
                    "Epoch/Accuracy",
                    {
                        "train": train_metrics["accuracy"],
                        "val": val_metrics["accuracy"],
                    },
                    epoch + 1,
                )
                self.writer.add_scalar("Epoch/Learning_Rate", current_lr, epoch + 1)

            # 최고 성능 체크포인트 저장
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                if self.checkpoint_dir:
                    self.save_checkpoint("best_model.pth")
                    print(f"최고 성능 모델 저장됨 (Val Acc: {self.best_val_acc:.2f}%)")

            # Early stopping 체크
            self.early_stopping(val_metrics["accuracy"])
            if self.early_stopping.early_stop:
                print(f"\n조기 종료: {epoch + 1} 에폭에서 학습 중단")
                break

        # 최종 모델 저장
        if self.checkpoint_dir and not self.save_best_only:
            self.save_checkpoint("final_model.pth")

        # 총 학습 시간
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("학습 완료!")
        print(f"  - 총 시간: {total_time / 60:.1f}분")
        print(f"  - 최고 검증 정확도: {self.best_val_acc:.2f}%")
        print("=" * 60)

        # TensorBoard 종료
        if self.writer:
            self.writer.close()

        return self.history

    def save_checkpoint(self, filename: str):
        """
        체크포인트 저장

        Args:
            filename: 저장 파일명
        """
        if not self.checkpoint_dir:
            return

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None  # type: ignore
            ),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: Path):
        """
        체크포인트 로드

        Args:
            checkpoint_path: 체크포인트 파일 경로
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # type: ignore

        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.history = checkpoint.get("history", self.history)

        print(f"체크포인트 로드 완료: {checkpoint_path}")
        print(f"  - 에폭: {self.current_epoch}")
        print(f"  - 최고 정확도: {self.best_val_acc:.2f}%")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    모델 평가

    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        device: 연산 디바이스
        class_names: 클래스 이름 리스트

    Returns:
        평가 결과 딕셔너리
    """
    model.eval()
    model.to(device)

    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            yolo_vector = None
            if "yolo_vector" in batch:
                yolo_vector = batch["yolo_vector"].to(device, non_blocking=True)

            outputs = model(images, yolo_vector)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 전체 정확도
    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)
    accuracy = (all_predictions == all_labels).float().mean().item() * 100

    # 클래스별 정확도
    class_accuracy = {}
    num_classes = len(class_names) if class_names else int(all_labels.max().item()) + 1

    for cls_idx in range(num_classes):
        mask = all_labels == cls_idx
        if mask.sum() > 0:
            cls_acc = (all_predictions[mask] == cls_idx).float().mean().item() * 100
            cls_name = class_names[cls_idx] if class_names else f"Class {cls_idx}"
            class_accuracy[cls_name] = {
                "accuracy": cls_acc,
                "count": int(mask.sum().item()),
            }

    # Confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for pred, label in zip(all_predictions, all_labels):
        confusion_matrix[label.long(), pred.long()] += 1

    return {
        "accuracy": accuracy,
        "class_accuracy": class_accuracy,
        "confusion_matrix": confusion_matrix.numpy(),
        "predictions": all_predictions.numpy(),
        "labels": all_labels.numpy(),
        "probabilities": all_probs,
    }


# 테스트 코드
if __name__ == "__main__":
    print("Trainer 모듈 테스트")

    # 더미 데이터로 테스트
    class DummyModel(nn.Module):
        def __init__(self, num_classes=6):
            super().__init__()
            self.fc = nn.Linear(3 * 224 * 224, num_classes)

        def forward(self, image, yolo_vector=None):
            x = image.view(image.size(0), -1)
            return {"logits": self.fc(x)}

    # 더미 데이터셋
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 224, 224),
                "label": torch.tensor(idx % 6),
            }

    # 데이터 로더 생성
    train_dataset = DummyDataset(100)
    val_dataset = DummyDataset(20)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 모델 생성
    model = DummyModel(num_classes=6)

    # 트레이너 생성
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        learning_rate=1e-3,
        use_amp=False,
        class_names=["당황", "분노", "불안", "상처", "슬픔", "중립"],
    )

    # 학습
    history = trainer.train()

    print("\n학습 히스토리:")
    print(f"  - Train Loss: {history['train_loss']}")
    print(f"  - Val Acc: {history['val_acc']}")

    print("\n테스트 완료!")
