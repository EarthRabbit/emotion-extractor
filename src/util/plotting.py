"""
Live Plotting Module for Training Visualization
학습 과정 실시간 시각화 모듈

매 epoch마다 학습 메트릭을 실시간으로 시각화합니다.

주요 기능:
    - LivePlotter: 실시간 학습 모니터링
    - TrainingVisualizer: 학습 완료 후 종합 시각화
    - plot_training_history: 학습 히스토리 정적 플롯
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# 한글 폰트 설정 (Windows/Mac/Linux 호환)
try:
    import matplotlib.font_manager as fm

    # Windows
    if any("malgun" in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Malgun Gothic"
    # Mac
    elif any("apple" in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "AppleGothic"
    # Linux (나눔고딕)
    elif any("nanum" in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "NanumGothic"
except Exception:
    pass

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지


class LivePlotter:
    """
    실시간 학습 모니터링 플로터

    학습 중 매 epoch마다 그래프를 업데이트합니다.

    사용법:
        plotter = LivePlotter(num_epochs=30, class_names=['당황', '분노', ...])

        for epoch in range(num_epochs):
            train_metrics = trainer.train_one_epoch()
            val_metrics = trainer.validate()
            plotter.update(epoch, train_metrics, val_metrics)

        plotter.save('training_plot.png')
        plotter.close()
    """

    def __init__(
        self,
        num_epochs: int = 30,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_dir: Optional[Path] = None,
        experiment_name: str = "experiment",
        update_interval: int = 1,
        show_plot: bool = True,
        dark_mode: bool = False,
    ):
        """
        Args:
            num_epochs: 총 에폭 수
            class_names: 클래스 이름 리스트
            figsize: 그래프 크기
            save_dir: 그래프 저장 디렉토리
            experiment_name: 실험 이름
            update_interval: 그래프 업데이트 간격 (에폭 단위)
            show_plot: 그래프 표시 여부
            dark_mode: 다크 모드 사용
        """
        self.num_epochs = num_epochs
        self.class_names = class_names or []
        self.figsize = figsize
        self.save_dir = Path(save_dir) if save_dir else None
        self.experiment_name = experiment_name
        self.update_interval = update_interval
        self.show_plot = show_plot
        self.dark_mode = dark_mode

        # 히스토리 저장
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rate": [],
        }
        self.class_accuracy_history: Dict[str, List[float]] = {}

        # Figure 및 Axes 초기화
        self.fig: Optional[Figure] = None
        self.axes: Optional[np.ndarray] = None
        self._initialized = False

        # 저장 디렉토리 생성
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # 스타일 설정
        if dark_mode:
            plt.style.use("dark_background")
        else:
            plt.style.use("seaborn-v0_8-whitegrid")

    def _init_figure(self):
        """Figure 및 Axes 초기화"""
        if self._initialized:
            return

        # Interactive mode 활성화
        if self.show_plot:
            plt.ion()

        # 2x2 subplot 생성
        self.fig, self.axes = plt.subplots(2, 2, figsize=self.figsize)
        self.fig.suptitle(
            f"Training Monitor - {self.experiment_name}",
            fontsize=14,
            fontweight="bold",
        )

        # 축 참조
        self.ax_loss = self.axes[0, 0]
        self.ax_acc = self.axes[0, 1]
        self.ax_lr = self.axes[1, 0]
        self.ax_class = self.axes[1, 1]

        # 초기 설정
        self._setup_axes()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._initialized = True

        if self.show_plot:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _setup_axes(self):
        """축 초기 설정"""
        # Loss 축
        self.ax_loss.set_title("Loss", fontsize=12, fontweight="bold")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.set_xlim(0, self.num_epochs)
        self.ax_loss.legend(loc="upper right")
        self.ax_loss.grid(True, alpha=0.3)

        # Accuracy 축
        self.ax_acc.set_title("Accuracy", fontsize=12, fontweight="bold")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylabel("Accuracy (%)")
        self.ax_acc.set_xlim(0, self.num_epochs)
        self.ax_acc.set_ylim(0, 100)
        self.ax_acc.legend(loc="lower right")
        self.ax_acc.grid(True, alpha=0.3)

        # Learning Rate 축
        self.ax_lr.set_title("Learning Rate", fontsize=12, fontweight="bold")
        self.ax_lr.set_xlabel("Epoch")
        self.ax_lr.set_ylabel("LR")
        self.ax_lr.set_xlim(0, self.num_epochs)
        self.ax_lr.set_yscale("log")
        self.ax_lr.grid(True, alpha=0.3)

        # Class Accuracy 축
        self.ax_class.set_title("Class-wise Accuracy", fontsize=12, fontweight="bold")
        self.ax_class.set_xlabel("Class")
        self.ax_class.set_ylabel("Accuracy (%)")
        self.ax_class.set_ylim(0, 100)
        self.ax_class.grid(True, alpha=0.3, axis="y")

    def update(
        self,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        learning_rate: Optional[float] = None,
    ):
        """
        그래프 업데이트

        Args:
            epoch: 현재 에폭 (0-based)
            train_metrics: 학습 메트릭 {'loss': float, 'accuracy': float}
            val_metrics: 검증 메트릭 {'loss': float, 'accuracy': float, 'class_accuracy': dict}
            learning_rate: 현재 학습률
        """
        # Figure 초기화
        if not self._initialized:
            self._init_figure()

        # 히스토리 업데이트
        self.history["train_loss"].append(train_metrics.get("loss", 0))
        self.history["val_loss"].append(val_metrics.get("loss", 0))
        self.history["train_acc"].append(train_metrics.get("accuracy", 0))
        self.history["val_acc"].append(val_metrics.get("accuracy", 0))
        if learning_rate is not None:
            self.history["learning_rate"].append(learning_rate)

        # 클래스별 정확도 업데이트
        class_acc = val_metrics.get("class_accuracy", {})
        if isinstance(class_acc, dict):
            for class_name, acc_info in class_acc.items():
                if class_name not in self.class_accuracy_history:
                    self.class_accuracy_history[class_name] = []
                acc_value = (
                    acc_info.get("accuracy", 0)
                    if isinstance(acc_info, dict)
                    else acc_info
                )
                self.class_accuracy_history[class_name].append(acc_value)

        # 업데이트 간격 체크
        if (epoch + 1) % self.update_interval != 0:
            return

        # 그래프 업데이트
        self._update_plots(epoch)

        if self.show_plot:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)

    def _update_plots(self, epoch: int):
        """모든 플롯 업데이트"""
        epochs = list(range(1, len(self.history["train_loss"]) + 1))

        # Loss 플롯
        self.ax_loss.clear()
        self.ax_loss.plot(
            epochs,
            self.history["train_loss"],
            "b-",
            label="Train",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        self.ax_loss.plot(
            epochs,
            self.history["val_loss"],
            "r-",
            label="Validation",
            linewidth=2,
            marker="s",
            markersize=3,
        )
        self.ax_loss.set_title("Loss", fontsize=12, fontweight="bold")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.set_xlim(0, self.num_epochs + 1)
        self.ax_loss.legend(loc="upper right")
        self.ax_loss.grid(True, alpha=0.3)

        # 최저 손실 표시
        if self.history["val_loss"]:
            min_loss = min(self.history["val_loss"])
            min_epoch = self.history["val_loss"].index(min_loss) + 1
            self.ax_loss.axhline(y=min_loss, color="r", linestyle="--", alpha=0.5)
            self.ax_loss.annotate(
                f"Min: {min_loss:.4f}",
                xy=(min_epoch, min_loss),
                xytext=(min_epoch + 2, min_loss * 1.1),
                fontsize=9,
                color="red",
            )

        # Accuracy 플롯
        self.ax_acc.clear()
        self.ax_acc.plot(
            epochs,
            self.history["train_acc"],
            "b-",
            label="Train",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        self.ax_acc.plot(
            epochs,
            self.history["val_acc"],
            "r-",
            label="Validation",
            linewidth=2,
            marker="s",
            markersize=3,
        )
        self.ax_acc.set_title("Accuracy", fontsize=12, fontweight="bold")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylabel("Accuracy (%)")
        self.ax_acc.set_xlim(0, self.num_epochs + 1)
        self.ax_acc.set_ylim(0, 105)
        self.ax_acc.legend(loc="lower right")
        self.ax_acc.grid(True, alpha=0.3)

        # 최고 정확도 표시
        if self.history["val_acc"]:
            max_acc = max(self.history["val_acc"])
            max_epoch = self.history["val_acc"].index(max_acc) + 1
            self.ax_acc.axhline(y=max_acc, color="r", linestyle="--", alpha=0.5)
            self.ax_acc.annotate(
                f"Max: {max_acc:.2f}%",
                xy=(max_epoch, max_acc),
                xytext=(max_epoch + 2, max_acc - 5),
                fontsize=9,
                color="red",
            )

        # Learning Rate 플롯
        self.ax_lr.clear()
        if self.history["learning_rate"]:
            self.ax_lr.plot(
                epochs,
                self.history["learning_rate"],
                "g-",
                linewidth=2,
                marker="^",
                markersize=3,
            )
        self.ax_lr.set_title("Learning Rate", fontsize=12, fontweight="bold")
        self.ax_lr.set_xlabel("Epoch")
        self.ax_lr.set_ylabel("LR")
        self.ax_lr.set_xlim(0, self.num_epochs + 1)
        if self.history["learning_rate"]:
            self.ax_lr.set_yscale("log")
        self.ax_lr.grid(True, alpha=0.3)

        # Class Accuracy 바 차트
        self.ax_class.clear()
        if self.class_accuracy_history:
            class_names = list(self.class_accuracy_history.keys())
            current_acc = [
                self.class_accuracy_history[name][-1]
                if self.class_accuracy_history[name]
                else 0
                for name in class_names
            ]

            colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
            bars = self.ax_class.bar(
                class_names, current_acc, color=colors, edgecolor="black"
            )

            # 바 위에 값 표시
            for bar, acc in zip(bars, current_acc):
                self.ax_class.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{acc:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        self.ax_class.set_title(
            f"Class-wise Accuracy (Epoch {epoch + 1})", fontsize=12, fontweight="bold"
        )
        self.ax_class.set_xlabel("Class")
        self.ax_class.set_ylabel("Accuracy (%)")
        self.ax_class.set_ylim(0, 110)
        self.ax_class.grid(True, alpha=0.3, axis="y")

        # 레이블 회전
        if self.class_accuracy_history:
            self.ax_class.tick_params(axis="x", rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def save(
        self,
        filename: Optional[str] = None,
        dpi: int = 150,
        format: str = "png",
    ):
        """
        그래프 저장

        Args:
            filename: 저장 파일명 (None이면 자동 생성)
            dpi: 해상도
            format: 파일 포맷
        """
        if self.fig is None:
            warnings.warn("저장할 그래프가 없습니다.")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_training_{timestamp}.{format}"

        save_path = self.save_dir / filename if self.save_dir else Path(filename)
        self.fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=format)
        print(f"학습 그래프 저장됨: {save_path}")

    def close(self):
        """플로터 종료"""
        if self.show_plot:
            plt.ioff()
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
            self._initialized = False

    def get_history(self) -> Dict[str, List[float]]:
        """학습 히스토리 반환"""
        return self.history.copy()

    def __enter__(self):
        """Context manager 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.close()


class TrainingVisualizer:
    """
    학습 완료 후 종합 시각화 클래스

    학습 히스토리를 기반으로 다양한 분석 그래프를 생성합니다.
    """

    def __init__(
        self,
        history: Dict[str, List[float]],
        class_names: Optional[List[str]] = None,
        save_dir: Optional[Path] = None,
        experiment_name: str = "experiment",
    ):
        """
        Args:
            history: 학습 히스토리 딕셔너리
            class_names: 클래스 이름 리스트
            save_dir: 저장 디렉토리
            experiment_name: 실험 이름
        """
        self.history = history
        self.class_names = class_names or []
        self.save_dir = Path(save_dir) if save_dir else Path(".")
        self.experiment_name = experiment_name

        if save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_loss_curves(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save: bool = True,
    ) -> Figure:
        """Loss 곡선 플롯"""
        fig, ax = plt.subplots(figsize=figsize)

        epochs = range(1, len(self.history.get("train_loss", [])) + 1)

        ax.plot(
            epochs,
            self.history.get("train_loss", []),
            "b-",
            label="Train Loss",
            linewidth=2,
        )
        ax.plot(
            epochs,
            self.history.get("val_loss", []),
            "r-",
            label="Validation Loss",
            linewidth=2,
        )

        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 과적합 영역 표시
        train_loss = self.history.get("train_loss", [])
        val_loss = self.history.get("val_loss", [])
        if train_loss and val_loss:
            gap = [v - t for t, v in zip(train_loss, val_loss)]
            if max(gap) > 0.1:
                ax.fill_between(
                    epochs,
                    train_loss,
                    val_loss,
                    where=[v > t for t, v in zip(train_loss, val_loss)],
                    alpha=0.2,
                    color="red",
                    label="Overfitting Gap",
                )

        plt.tight_layout()

        if save:
            save_path = self.save_dir / f"{self.experiment_name}_loss_curves.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"저장됨: {save_path}")

        return fig

    def plot_accuracy_curves(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save: bool = True,
    ) -> Figure:
        """Accuracy 곡선 플롯"""
        fig, ax = plt.subplots(figsize=figsize)

        epochs = range(1, len(self.history.get("train_acc", [])) + 1)

        ax.plot(
            epochs,
            self.history.get("train_acc", []),
            "b-",
            label="Train Accuracy",
            linewidth=2,
        )
        ax.plot(
            epochs,
            self.history.get("val_acc", []),
            "r-",
            label="Validation Accuracy",
            linewidth=2,
        )

        ax.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 최고 정확도 마킹
        val_acc = self.history.get("val_acc", [])
        if val_acc:
            max_acc = max(val_acc)
            max_epoch = val_acc.index(max_acc) + 1
            ax.axhline(y=max_acc, color="green", linestyle="--", alpha=0.7)
            ax.scatter([max_epoch], [max_acc], color="green", s=100, zorder=5)
            ax.annotate(
                f"Best: {max_acc:.2f}%\n(Epoch {max_epoch})",
                xy=(max_epoch, max_acc),
                xytext=(max_epoch + len(epochs) * 0.1, max_acc - 5),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="green"),
            )

        plt.tight_layout()

        if save:
            save_path = self.save_dir / f"{self.experiment_name}_accuracy_curves.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"저장됨: {save_path}")

        return fig

    def plot_learning_rate(
        self,
        figsize: Tuple[int, int] = (10, 4),
        save: bool = True,
    ) -> Figure:
        """Learning Rate 스케줄 플롯"""
        fig, ax = plt.subplots(figsize=figsize)

        lr_history = self.history.get("learning_rate", [])
        if not lr_history:
            ax.text(0.5, 0.5, "No learning rate data", ha="center", va="center")
            return fig

        epochs = range(1, len(lr_history) + 1)
        ax.plot(epochs, lr_history, "g-", linewidth=2, marker="o", markersize=3)

        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.save_dir / f"{self.experiment_name}_lr_schedule.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"저장됨: {save_path}")

        return fig

    def plot_all(
        self,
        figsize: Tuple[int, int] = (15, 10),
        save: bool = True,
        show: bool = True,
    ) -> Figure:
        """모든 그래프 통합 플롯"""
        fig = plt.figure(figsize=figsize)

        # 2x2 그리드
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Loss
        ax1 = fig.add_subplot(gs[0, 0])
        epochs = range(1, len(self.history.get("train_loss", [])) + 1)
        ax1.plot(
            epochs, self.history.get("train_loss", []), "b-", label="Train", linewidth=2
        )
        ax1.plot(
            epochs, self.history.get("val_loss", []), "r-", label="Val", linewidth=2
        )
        ax1.set_title("Loss", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(
            epochs, self.history.get("train_acc", []), "b-", label="Train", linewidth=2
        )
        ax2.plot(
            epochs, self.history.get("val_acc", []), "r-", label="Val", linewidth=2
        )
        ax2.set_title("Accuracy", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_ylim(0, 105)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Learning Rate
        ax3 = fig.add_subplot(gs[1, 0])
        lr_history = self.history.get("learning_rate", [])
        if lr_history:
            ax3.plot(epochs, lr_history, "g-", linewidth=2)
            ax3.set_yscale("log")
        ax3.set_title("Learning Rate", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("LR")
        ax3.grid(True, alpha=0.3)

        # Train-Val Gap (과적합 모니터링)
        ax4 = fig.add_subplot(gs[1, 1])
        train_acc = self.history.get("train_acc", [])
        val_acc = self.history.get("val_acc", [])
        if train_acc and val_acc:
            gap = [t - v for t, v in zip(train_acc, val_acc)]
            colors = ["red" if g > 10 else "orange" if g > 5 else "green" for g in gap]
            ax4.bar(epochs, gap, color=colors, alpha=0.7)
            ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax4.axhline(
                y=5, color="orange", linestyle="--", alpha=0.5, label="Warning (5%)"
            )
            ax4.axhline(
                y=10, color="red", linestyle="--", alpha=0.5, label="Critical (10%)"
            )
        ax4.set_title("Overfitting Gap (Train - Val)", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Gap (%)")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        fig.suptitle(
            f"Training Summary - {self.experiment_name}",
            fontsize=14,
            fontweight="bold",
        )

        if save:
            save_path = self.save_dir / f"{self.experiment_name}_summary.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"저장됨: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    show: bool = True,
) -> Figure:
    """
    학습 히스토리 정적 플롯 (간단한 함수형 인터페이스)

    Args:
        history: 학습 히스토리 딕셔너리
        save_path: 저장 경로
        figsize: 그래프 크기
        show: 그래프 표시 여부

    Returns:
        matplotlib Figure 객체
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss
    axes[0, 0].plot(epochs, history.get("train_loss", []), "b-", label="Train")
    axes[0, 0].plot(epochs, history.get("val_loss", []), "r-", label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history.get("train_acc", []), "b-", label="Train")
    axes[0, 1].plot(epochs, history.get("val_acc", []), "r-", label="Validation")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning Rate
    lr_history = history.get("learning_rate", [])
    if lr_history:
        axes[1, 0].plot(epochs, lr_history, "g-")
        axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("LR")
    axes[1, 0].grid(True, alpha=0.3)

    # Gap
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])
    if train_acc and val_acc:
        gap = [t - v for t, v in zip(train_acc, val_acc)]
        axes[1, 1].fill_between(epochs, 0, gap, alpha=0.5, color="orange")
        axes[1, 1].plot(epochs, gap, "orange", linewidth=2)
    axes[1, 1].set_title("Overfitting Gap")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Train - Val (%)")
    axes[1, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장됨: {save_path}")

    if show:
        plt.show()

    return fig


# 테스트 코드
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Live Plotter 테스트")
    print("=" * 60)

    # 더미 데이터 생성
    num_epochs = 20
    class_names = ["기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립"]

    # LivePlotter 테스트
    print("\n[1] LivePlotter 테스트")
    plotter = LivePlotter(
        num_epochs=num_epochs,
        class_names=class_names,
        figsize=(14, 10),
        experiment_name="test_experiment",
        show_plot=True,
    )

    # 시뮬레이션 학습
    for epoch in range(num_epochs):
        # 더미 메트릭 생성
        train_loss = 2.0 * np.exp(-epoch / 10) + np.random.rand() * 0.1
        val_loss = 2.0 * np.exp(-epoch / 12) + np.random.rand() * 0.15
        train_acc = 100 * (1 - np.exp(-epoch / 8)) + np.random.rand() * 5
        val_acc = 100 * (1 - np.exp(-epoch / 10)) + np.random.rand() * 5 - 5

        train_metrics = {"loss": train_loss, "accuracy": min(train_acc, 100)}
        val_metrics = {
            "loss": val_loss,
            "accuracy": min(val_acc, 95),
            "class_accuracy": {
                name: min(100, val_acc + np.random.rand() * 20 - 10)
                for name in class_names
            },
        }
        lr = 0.001 * (0.95**epoch)

        plotter.update(epoch, train_metrics, val_metrics, learning_rate=lr)
        time.sleep(0.2)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss={train_loss:.4f}, Val Acc={val_acc:.2f}%"
        )

    # 저장 및 종료
    plotter.save("test_training_plot.png")
    plotter.close()

    # TrainingVisualizer 테스트
    print("\n[2] TrainingVisualizer 테스트")
    history = plotter.get_history()
    visualizer = TrainingVisualizer(
        history=history,
        class_names=class_names,
        experiment_name="test_visualizer",
    )
    visualizer.plot_all(save=False)
    plt.show()

    print("\n테스트 완료!")
