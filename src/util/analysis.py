"""
Analysis Module for Model Evaluation
모델 평가 및 분석 모듈

학습된 모델의 성능을 다양한 관점에서 분석합니다.

주요 기능:
    - compute_metrics: 분류 메트릭 계산 (Precision, Recall, F1, etc.)
    - plot_confusion_matrix: 혼동 행렬 시각화
    - plot_roc_curves: ROC 곡선 및 AUC
    - plot_precision_recall_curves: Precision-Recall 곡선
    - plot_class_distribution: 클래스 분포 시각화
    - analyze_misclassifications: 오분류 분석
    - AnalysisReport: 종합 분석 리포트 생성
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# scikit-learn imports
try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.preprocessing import label_binarize

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn이 설치되지 않았습니다. 일부 기능이 제한됩니다.")

# seaborn import
try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# 한글 폰트 설정
try:
    import matplotlib.font_manager as fm

    if any("malgun" in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif any("apple" in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "AppleGothic"
    elif any("nanum" in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "NanumGothic"
except Exception:
    pass

plt.rcParams["axes.unicode_minus"] = False


@dataclass
class MetricsResult:
    """메트릭 결과 데이터 클래스"""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "class_metrics": self.class_metrics,
        }

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "Classification Metrics",
            "=" * 50,
            f"Accuracy:           {self.accuracy:.4f} ({self.accuracy * 100:.2f}%)",
            "",
            "Macro Average:",
            f"  Precision:        {self.precision_macro:.4f}",
            f"  Recall:           {self.recall_macro:.4f}",
            f"  F1-Score:         {self.f1_macro:.4f}",
            "",
            "Weighted Average:",
            f"  Precision:        {self.precision_weighted:.4f}",
            f"  Recall:           {self.recall_weighted:.4f}",
            f"  F1-Score:         {self.f1_weighted:.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> MetricsResult:
    """
    분류 메트릭 계산

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
        verbose: 결과 출력 여부

    Returns:
        MetricsResult 객체
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("이 기능은 scikit-learn이 필요합니다.")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 기본 메트릭
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)

    # 클래스별 메트릭
    class_metrics = {}
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    for class_name in class_names:
        if class_name in report_dict:
            class_metrics[class_name] = {
                "precision": report_dict[class_name]["precision"],
                "recall": report_dict[class_name]["recall"],
                "f1-score": report_dict[class_name]["f1-score"],
                "support": report_dict[class_name]["support"],
            }

    result = MetricsResult(
        accuracy=accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        precision_weighted=precision_weighted,
        recall_weighted=recall_weighted,
        f1_weighted=f1_weighted,
        class_metrics=class_metrics,
        confusion_matrix=cm,
        classification_report=report_str,
    )

    if verbose:
        print(result)
        print("\nClassification Report:")
        print(report_str)

    return result


def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """
    혼동 행렬 시각화

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
        normalize: 정규화 여부
        figsize: 그래프 크기
        cmap: 컬러맵
        save_path: 저장 경로
        show: 그래프 표시 여부

    Returns:
        matplotlib Figure 객체
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("이 기능은 scikit-learn이 필요합니다.")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # NaN 처리
        fmt = ".2%"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]

    fig, ax = plt.subplots(figsize=figsize)

    if SEABORN_AVAILABLE:
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": "Ratio" if normalize else "Count"},
        )
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.get_cmap(cmap))
        plt.colorbar(im, ax=ax)

        # 텍스트 표시
        for i in range(len(cm)):
            for j in range(len(cm)):
                value = f"{cm[i, j]:.2%}" if normalize else f"{cm[i, j]}"
                text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, value, ha="center", va="center", color=text_color)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장됨: {save_path}")

    if show:
        plt.show()

    return fig


def plot_roc_curves(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[List[float]], np.ndarray],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """
    ROC 곡선 및 AUC 시각화

    Args:
        y_true: 실제 레이블
        y_prob: 예측 확률 [N, num_classes]
        class_names: 클래스 이름 리스트
        figsize: 그래프 크기
        save_path: 저장 경로
        show: 그래프 표시 여부

    Returns:
        matplotlib Figure 객체
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("이 기능은 scikit-learn이 필요합니다.")

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    num_classes = y_prob.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # One-hot 인코딩
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    fig, ax = plt.subplots(figsize=figsize)

    # 각 클래스에 대한 ROC 곡선
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))
    auc_scores = []

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        if y_true_bin.shape[1] > 1:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        else:
            # 이진 분류의 경우
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])

        auc_scores.append(roc_auc)
        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label=f"{class_name} (AUC = {roc_auc:.3f})",
        )

    # 대각선 (랜덤 분류기)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        f"ROC Curves (Macro AUC = {np.mean(auc_scores):.3f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장됨: {save_path}")

    if show:
        plt.show()

    return fig


def plot_precision_recall_curves(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[List[float]], np.ndarray],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """
    Precision-Recall 곡선 시각화

    Args:
        y_true: 실제 레이블
        y_prob: 예측 확률 [N, num_classes]
        class_names: 클래스 이름 리스트
        figsize: 그래프 크기
        save_path: 저장 경로
        show: 그래프 표시 여부

    Returns:
        matplotlib Figure 객체
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("이 기능은 scikit-learn이 필요합니다.")

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    num_classes = y_prob.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # One-hot 인코딩
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        if y_true_bin.shape[1] > 1:
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_prob[:, i]
            )
        else:
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])

        ax.plot(recall, precision, color=color, linewidth=2, label=class_name)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장됨: {save_path}")

    if show:
        plt.show()

    return fig


def plot_class_distribution(
    class_counts: Dict[str, int],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """
    클래스 분포 시각화

    Args:
        class_counts: {'class_name': count, ...}
        figsize: 그래프 크기
        save_path: 저장 경로
        show: 그래프 표시 여부

    Returns:
        matplotlib Figure 객체
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    total = sum(counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 막대 차트
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    bars = ax1.bar(classes, counts, color=colors, edgecolor="black")

    # 값 표시
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{count}\n({count / total * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax1.set_title("Class Distribution (Count)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3, axis="y")

    # 파이 차트
    _, _, autotexts = ax2.pie(
        counts,
        labels=classes,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=[0.02] * len(classes),
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")
    ax2.set_title("Class Distribution (Ratio)", fontsize=12, fontweight="bold")

    # 불균형 비율 표시
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    fig.suptitle(
        f"Class Distribution (Imbalance Ratio: {imbalance_ratio:.2f}x)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장됨: {save_path}")

    if show:
        plt.show()

    return fig


def analyze_misclassifications(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: Optional[List[str]] = None,
    top_k: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    오분류 패턴 분석

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
        top_k: 상위 K개 오분류 패턴
        verbose: 결과 출력 여부

    Returns:
        분석 결과 딕셔너리
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # 오분류 인덱스
    misclassified_idx = np.where(y_true != y_pred)[0]
    total_misclassified = len(misclassified_idx)
    total_samples = len(y_true)

    # 오분류 패턴 (true -> pred) 카운트
    misclassification_patterns = {}
    for idx in misclassified_idx:
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        pattern = f"{true_label} → {pred_label}"

        if pattern not in misclassification_patterns:
            misclassification_patterns[pattern] = {
                "count": 0,
                "indices": [],
                "true_class": true_label,
                "pred_class": pred_label,
            }
        misclassification_patterns[pattern]["count"] += 1
        misclassification_patterns[pattern]["indices"].append(idx)

    # 상위 K개 패턴
    sorted_patterns = sorted(
        misclassification_patterns.items(), key=lambda x: x[1]["count"], reverse=True
    )[:top_k]

    # 클래스별 오분류율
    class_misclassification_rate = {}
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_total = class_mask.sum()
        if class_total > 0:
            class_misclassified = ((y_true == i) & (y_pred != i)).sum()
            class_misclassification_rate[class_name] = {
                "rate": class_misclassified / class_total,
                "count": int(class_misclassified),
                "total": int(class_total),
            }

    result = {
        "total_samples": total_samples,
        "total_misclassified": total_misclassified,
        "misclassification_rate": total_misclassified / total_samples,
        "top_patterns": dict(sorted_patterns),
        "class_misclassification_rate": class_misclassification_rate,
        "misclassified_indices": misclassified_idx.tolist(),
    }

    if verbose:
        print("=" * 60)
        print("Misclassification Analysis")
        print("=" * 60)
        print(f"Total Samples:       {total_samples}")
        print(f"Total Misclassified: {total_misclassified}")
        print(f"Misclassification Rate: {total_misclassified / total_samples:.2%}")
        print()
        print(f"Top {top_k} Misclassification Patterns:")
        print("-" * 40)
        for pattern, info in sorted_patterns:
            print(f"  {pattern}: {info['count']} ({info['count'] / total_samples:.2%})")
        print()
        print("Class-wise Misclassification Rate:")
        print("-" * 40)
        for class_name, info in class_misclassification_rate.items():
            print(
                f"  {class_name}: {info['rate']:.2%} ({info['count']}/{info['total']})"
            )
        print("=" * 60)

    return result


@dataclass
class AnalysisReport:
    """
    종합 분석 리포트 클래스

    모델 평가 결과를 종합적으로 분석하고 리포트를 생성합니다.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None
    class_names: Optional[List[str]] = None
    experiment_name: str = "experiment"
    save_dir: Optional[Path] = None

    def __post_init__(self):
        self.y_true = np.array(self.y_true)
        self.y_pred = np.array(self.y_pred)
        if self.y_prob is not None:
            self.y_prob = np.array(self.y_prob)

        if self.class_names is None:
            num_classes = len(np.unique(self.y_true))
            self.class_names = [f"Class {i}" for i in range(num_classes)]

        if self.save_dir:
            self.save_dir = Path(self.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # 메트릭 계산
        self.metrics: Optional[MetricsResult] = None
        self.misclassification_analysis: Optional[Dict] = None

    def compute_all_metrics(self, verbose: bool = False) -> MetricsResult:
        """모든 메트릭 계산"""
        self.metrics = compute_metrics(
            self.y_true, self.y_pred, self.class_names, verbose=verbose
        )
        return self.metrics

    def analyze_errors(self, top_k: int = 5, verbose: bool = False) -> Dict:
        """오분류 분석"""
        self.misclassification_analysis = analyze_misclassifications(
            self.y_true, self.y_pred, self.class_names, top_k=top_k, verbose=verbose
        )
        return self.misclassification_analysis

    def plot_all(
        self,
        figsize: Tuple[int, int] = (15, 12),
        save: bool = True,
        show: bool = True,
    ) -> List[Figure]:
        """모든 분석 그래프 생성"""
        figures = []

        # 1. Confusion Matrix
        fig_cm = plot_confusion_matrix(
            self.y_true,
            self.y_pred,
            self.class_names,
            normalize=True,
            save_path=self.save_dir / f"{self.experiment_name}_confusion_matrix.png"
            if save and self.save_dir
            else None,
            show=show,
        )
        figures.append(fig_cm)

        # 2. ROC Curves (확률이 있는 경우)
        if self.y_prob is not None:
            try:
                fig_roc = plot_roc_curves(
                    self.y_true,
                    self.y_prob,
                    self.class_names,
                    save_path=self.save_dir / f"{self.experiment_name}_roc_curves.png"
                    if save and self.save_dir
                    else None,
                    show=show,
                )
                figures.append(fig_roc)

                # 3. Precision-Recall Curves
                fig_pr = plot_precision_recall_curves(
                    self.y_true,
                    self.y_prob,
                    self.class_names,
                    save_path=self.save_dir
                    / f"{self.experiment_name}_precision_recall_curves.png"
                    if save and self.save_dir
                    else None,
                    show=show,
                )
                figures.append(fig_pr)
            except Exception as e:
                print(f"ROC/PR 곡선 생성 실패: {e}")

        return figures

    def generate_report(self, save: bool = True) -> str:
        """텍스트 리포트 생성"""
        if self.metrics is None:
            self.compute_all_metrics()

        if self.misclassification_analysis is None:
            self.analyze_errors()

        lines = [
            "=" * 70,
            f"Analysis Report: {self.experiment_name}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "## Classification Metrics",
            "-" * 40,
            f"Accuracy:           {self.metrics.accuracy:.4f} ({self.metrics.accuracy * 100:.2f}%)",
            "",
            "Macro Average:",
            f"  Precision:        {self.metrics.precision_macro:.4f}",
            f"  Recall:           {self.metrics.recall_macro:.4f}",
            f"  F1-Score:         {self.metrics.f1_macro:.4f}",
            "",
            "Weighted Average:",
            f"  Precision:        {self.metrics.precision_weighted:.4f}",
            f"  Recall:           {self.metrics.recall_weighted:.4f}",
            f"  F1-Score:         {self.metrics.f1_weighted:.4f}",
            "",
            "## Class-wise Metrics",
            "-" * 40,
        ]

        for class_name, metrics in self.metrics.class_metrics.items():
            lines.append(
                f"  {class_name}: "
                f"P={metrics['precision']:.3f}, "
                f"R={metrics['recall']:.3f}, "
                f"F1={metrics['f1-score']:.3f}, "
                f"Support={int(metrics['support'])}"
            )

        lines.extend(
            [
                "",
                "## Misclassification Analysis",
                "-" * 40,
                f"Total Misclassified: {self.misclassification_analysis['total_misclassified']} "
                f"({self.misclassification_analysis['misclassification_rate']:.2%})",
                "",
                "Top Misclassification Patterns:",
            ]
        )

        for pattern, info in self.misclassification_analysis["top_patterns"].items():
            lines.append(f"  {pattern}: {info['count']}")

        lines.extend(["", "=" * 70])

        report = "\n".join(lines)

        if save and self.save_dir:
            report_path = self.save_dir / f"{self.experiment_name}_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"리포트 저장됨: {report_path}")

        return report


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("Analysis Module 테스트")
    print("=" * 60)

    # 더미 데이터 생성
    np.random.seed(42)
    num_samples = 200
    num_classes = 7
    class_names = ["기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립"]

    # 실제 레이블 (불균형 분포)
    y_true = np.concatenate(
        [
            np.zeros(45, dtype=int),  # 기쁨
            np.ones(40, dtype=int),  # 당황
            np.full(35, 2, dtype=int),  # 분노
            np.full(30, 3, dtype=int),  # 불안
            np.full(25, 4, dtype=int),  # 상처
            np.full(15, 5, dtype=int),  # 슬픔
            np.full(10, 6, dtype=int),  # 중립
        ]
    )

    # 예측 레이블 (약간의 오차 포함)
    y_pred = y_true.copy()
    error_idx = np.random.choice(len(y_true), size=40, replace=False)
    y_pred[error_idx] = np.random.randint(0, num_classes, size=40)

    # 예측 확률 (더미)
    y_prob = np.random.rand(num_samples, num_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    print("\n[1] compute_metrics 테스트")
    metrics = compute_metrics(y_true, y_pred, class_names)

    print("\n[2] plot_confusion_matrix 테스트")
    plot_confusion_matrix(y_true, y_pred, class_names, show=True)

    print("\n[3] analyze_misclassifications 테스트")
    analysis = analyze_misclassifications(y_true, y_pred, class_names)

    print("\n[4] plot_class_distribution 테스트")
    class_counts = {name: (y_true == i).sum() for i, name in enumerate(class_names)}
    plot_class_distribution(class_counts, show=True)

    print("\n[5] AnalysisReport 테스트")
    report = AnalysisReport(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        experiment_name="test_analysis",
    )
    report.compute_all_metrics(verbose=True)
    report.analyze_errors(verbose=True)
    print(report.generate_report(save=False))

    print("\n테스트 완료!")
