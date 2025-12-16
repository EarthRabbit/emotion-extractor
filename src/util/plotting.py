"""
Plotting Utility for Model Performance Visualization
모델 성능 시각화 유틸리티

학습 곡선, 혼동 행렬, 클래스별 성능 등을 시각화합니다.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import font_manager as fm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

# 한글 폰트 설정
try:
    plt.rcParams["font.family"] = "DejaVu Sans"
    # Windows의 경우
    try:
        fm.fontManager.addfont("C:\\Windows\\Fonts\\malgun.ttf")
        plt.rcParams["font.family"] = "Malgun Gothic"
    except:
        pass
except:
    pass

# 스타일 설정
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 10


class TrainingPlotter:
    """학습 곡선 시각화"""

    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Args:
            figsize: 그림 크기
        """
        self.figsize = figsize

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[Path] = None,
        title: str = "Training History",
    ):
        """
        학습 히스토리 플롯 (Loss & Accuracy)

        Args:
            history: {'train_loss': [...], 'val_loss': [...], ...}
            save_path: 저장 경로
            title: 제목
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # 1. Training & Validation Loss
        ax = axes[0, 0]
        epochs = range(1, len(history.get("train_loss", [])) + 1)
        if "train_loss" in history:
            ax.plot(
                epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2
            )
        if "val_loss" in history:
            ax.plot(epochs, history["val_loss"], "r-s", label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Training & Validation Accuracy
        ax = axes[0, 1]
        if "train_acc" in history:
            ax.plot(epochs, history["train_acc"], "b-o", label="Train Acc", linewidth=2)
        if "val_acc" in history:
            ax.plot(epochs, history["val_acc"], "r-s", label="Val Acc", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training & Validation Accuracy")
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Learning Rate Schedule
        ax = axes[1, 0]
        if "learning_rate" in history:
            ax.plot(epochs, history["learning_rate"], "g-^", linewidth=2)
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate (log scale)")
            ax.set_title("Learning Rate Schedule")
            ax.grid(True, alpha=0.3)

        # 4. Overfitting Analysis (Gap)
        ax = axes[1, 1]
        if "train_acc" in history and "val_acc" in history:
            gap = [t - v for t, v in zip(history["train_acc"], history["val_acc"])]
            ax.bar(epochs, gap, color="orange", alpha=0.7, edgecolor="black")
            ax.axhline(
                y=0.1, color="g", linestyle="--", label="Ideal Gap (10%)", linewidth=2
            )
            ax.axhline(
                y=0.15, color="y", linestyle="--", label="Warning (15%)", linewidth=2
            )
            ax.axhline(
                y=0.20, color="r", linestyle="--", label="Critical (20%)", linewidth=2
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Gap (%)")
            ax.set_title("Overfitting Gap (Train - Val)")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 저장됨: {save_path}")

        plt.show()

    def plot_loss_only(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[Path] = None,
    ):
        """Loss만 플롯 (빠른 확인용)"""
        plt.figure(figsize=(10, 5))

        epochs = range(1, len(history.get("train_loss", [])) + 1)
        if "train_loss" in history:
            plt.plot(
                epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2
            )
        if "val_loss" in history:
            plt.plot(epochs, history["val_loss"], "r-s", label="Val Loss", linewidth=2)

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training & Validation Loss", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 저장됨: {save_path}")

        plt.show()

    def plot_accuracy_only(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[Path] = None,
    ):
        """Accuracy만 플롯 (빠른 확인용)"""
        plt.figure(figsize=(10, 5))

        epochs = range(1, len(history.get("train_acc", [])) + 1)
        if "train_acc" in history:
            plt.plot(
                epochs, history["train_acc"], "b-o", label="Train Acc", linewidth=2
            )
        if "val_acc" in history:
            plt.plot(epochs, history["val_acc"], "r-s", label="Val Acc", linewidth=2)

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
        plt.ylim([0, 1])
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 저장됨: {save_path}")

        plt.show()


class ConfusionMatrixPlotter:
    """혼동 행렬 시각화"""

    @staticmethod
    def plot_confusion_matrix(
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
        save_path: Optional[Path] = None,
        normalize: bool = False,
    ):
        """
        혼동 행렬 플롯

        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블
            class_names: 클래스 이름
            save_path: 저장 경로
            normalize: 정규화 여부
        """
        # 혼동 행렬 계산
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2%"
        else:
            fmt = "d"

        # 플롯
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Count" if not normalize else "Ratio"},
        )

        plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 저장됨: {save_path}")

        plt.show()


class ClassPerformancePlotter:
    """클래스별 성능 시각화"""

    @staticmethod
    def plot_class_accuracy(
        class_accuracies: Dict[str, float],
        save_path: Optional[Path] = None,
    ):
        """
        클래스별 정확도 플롯

        Args:
            class_accuracies: {'class_name': accuracy, ...}
            save_path: 저장 경로
        """
        classes = list(class_accuracies.keys())
        accuracies = list(class_accuracies.values())

        # 정렬 (내림차순)
        sorted_pairs = sorted(
            zip(classes, accuracies), key=lambda x: x[1], reverse=True
        )
        classes, accuracies = zip(*sorted_pairs)

        # 색상 (정확도에 따라)
        colors = [
            "green" if acc >= 0.7 else "orange" if acc >= 0.6 else "red"
            for acc in accuracies
        ]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            classes, accuracies, color=colors, edgecolor="black", linewidth=1.5
        )

        # 값 표시
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{acc:.2%}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.ylabel("Accuracy", fontsize=12)
        plt.title("Per-Class Accuracy", fontsize=14, fontweight="bold")
        plt.ylim([0, 1])
        plt.axhline(y=0.6, color="r", linestyle="--", alpha=0.5, label="60% threshold")
        plt.axhline(y=0.7, color="g", linestyle="--", alpha=0.5, label="70% threshold")
        plt.legend()
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 저장됨: {save_path}")

        plt.show()

    @staticmethod
    def plot_class_metrics(
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
        save_path: Optional[Path] = None,
    ):
        """
        클래스별 Precision, Recall, F1-Score 플롯

        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블
            class_names: 클래스 이름
            save_path: 저장 경로
        """
        # 분류 리포트 (딕셔너리 형식)
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        # 데이터 추출
        metrics = ["precision", "recall", "f1-score"]
        x = np.arange(len(class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, metric in enumerate(metrics):
            values = [report[class_name][metric] for class_name in class_names]
            ax.bar(x + i * width, values, width, label=metric.capitalize())

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Per-Class Performance Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 저장됨: {save_path}")

        plt.show()


class ComparisonPlotter:
    """실험 비교 시각화"""

    @staticmethod
    def plot_experiment_comparison(
        experiments: Dict[str, Dict[str, List[float]]],
        save_path: Optional[Path] = None,
    ):
        """
        여러 실험의 성능 비교

        Args:
            experiments: {
                'exp1': {'train_acc': [...], 'val_acc': [...]},
                'exp2': {'train_acc': [...], 'val_acc': [...]},
                ...
            }
            save_path: 저장 경로
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        colors = plt.cm.Set2(np.linspace(0, 1, len(experiments)))

        for (exp_name, history), color in zip(experiments.items(), colors):
            epochs = range(1, len(history.get("val_acc", [])) + 1)

            # Val Accuracy
            if "val_acc" in history:
                axes[0].plot(
                    epochs,
                    history["val_acc"],
                    label=exp_name,
                    linewidth=2,
                    color=color,
                    marker="o",
                )

            # Val Loss
            if "val_loss" in history:
                axes[1].plot(
                    epochs,
                    history["val_loss"],
                    label=exp_name,
                    linewidth=2,
                    color=color,
                    marker="s",
                )

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Validation Accuracy Comparison")
        axes[0].set_ylim([0, 1])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Validation Loss Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 저장됨: {save_path}")

        plt.show()

    @staticmethod
    def plot_final_metrics_comparison(
        experiments: Dict[str, Dict[str, float]],
        save_path: Optional[Path] = None,
    ):
        """
        최종 메트릭 비교 (막대 차트)

        Args:
            experiments: {
                'exp1': {'val_acc': 0.65, 'best_epoch': 25},
                'exp2': {'val_acc': 0.68, 'best_epoch': 30},
                ...
            }
            save_path: 저장 경로
        """
        exp_names = list(experiments.keys())
        val_accs = [experiments[name].get("val_acc", 0) for name in exp_names]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            exp_names, val_accs, color="skyblue", edgecolor="black", linewidth=1.5
        )

        # 값 표시
        for bar, acc in zip(bars, val_accs):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{acc:.2%}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_ylabel("Validation Accuracy", fontsize=12)
        ax.set_title(
            "Experiment Comparison (Final Metrics)", fontsize=14, fontweight="bold"
        )
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 저장됨: {save_path}")

        plt.show()


class DatasetPlotter:
    """데이터셋 시각화"""

    @staticmethod
    def plot_class_distribution(
        class_counts: Dict[str, int],
        save_path: Optional[Path] = None,
    ):
        """
        클래스 분포 시각화

        Args:
            class_counts: {'class_name': count, ...}
            save_path: 저장 경로
        """
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        total = sum(counts)

        # 정렬 (내림차순)
        sorted_pairs = sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_pairs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 막대 차트
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        bars = ax1.bar(classes, counts, color=colors, edgecolor="black", linewidth=1.5)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax1.set_ylabel("Number of Samples", fontsize=12)
        ax1.set_title("Class Distribution (Count)", fontsize=13, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.set_xticklabels(classes, rotation=45, ha="right")

        # 파이 차트
        percentages = [count / total * 100 for count in counts]
        wedges, texts, autotexts = ax2.pie(
            counts, labels=classes, autopct="%1.1f%%", colors=colors, startangle=90
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)
        ax2.set_title("Class Distribution (Percentage)", fontsize=13, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 저장됨: {save_path}")

        plt.show()

    @staticmethod
    def plot_imbalance_ratio(
        class_counts: Dict[str, int],
        save_path: Optional[Path] = None,
    ):
        """
        클래스 불균형 비율 시각화

        Args:
            class_counts: {'class_name': count, ...}
            save_path: 저장 경로
        """
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count

        # 정규화된 카운트
        normalized = [count / max_count for count in counts]

        plt.figure(figsize=(10, 6))

        bars = plt.barh(
            classes, normalized, color="steelblue", edgecolor="black", linewidth=1.5
        )

        # 값 표시
        for bar, count, norm in zip(bars, counts, normalized):
            width = bar.get_width()
            plt.text(
                width,
                bar.get_y() + bar.get_height() / 2.0,
                f"{int(count)} ({norm:.0%})",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.xlabel("Normalized Count (relative to max)", fontsize=12)
        plt.title(
            f"Class Imbalance Ratio: {imbalance_ratio:.2f}x",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlim([0, 1.3])
        plt.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 저장됨: {save_path}")

        plt.show()


# 테스트 코드
if __name__ == "__main__":
    # 더미 데이터 생성
    history = {
        "train_loss": [0.8, 0.6, 0.4, 0.3, 0.25, 0.22],
        "val_loss": [0.85, 0.65, 0.5, 0.45, 0.50, 0.55],
        "train_acc": [0.65, 0.75, 0.82, 0.86, 0.88, 0.89],
        "val_acc": [0.60, 0.68, 0.72, 0.73, 0.71, 0.70],
        "learning_rate": [1e-4, 1e-4, 5e-5, 5e-5, 1e-5, 1e-5],
    }

    class_accuracies = {
        "중립": 0.75,
        "분노": 0.69,
        "당황": 0.67,
        "슬픔": 0.60,
        "불안": 0.50,
        "상처": 0.45,
    }

    class_counts = {
        "중립": 500,
        "분노": 400,
        "당황": 380,
        "슬픔": 300,
        "불안": 200,
        "상처": 150,
    }

    # 플롯
    print("📊 Training History")
    plotter = TrainingPlotter()
    plotter.plot_training_history(history)

    print("\n📊 Class Accuracy")
    ClassPerformancePlotter.plot_class_accuracy(class_accuracies)

    print("\n📊 Dataset Distribution")
    DatasetPlotter.plot_class_distribution(class_counts)
