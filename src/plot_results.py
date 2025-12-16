"""
Plotting Script for Model Results
모델 결과 시각화 스크립트

학습 결과, 성능 비교, 혼동 행렬 등을 시각화합니다.

사용법:
    python plot_results.py --history runs/hybrid_emotion_*/events.*
    python plot_results.py --class-accuracy
    python plot_results.py --confusion-matrix
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from util.plotting import (
    ClassPerformancePlotter,
    ComparisonPlotter,
    ConfusionMatrixPlotter,
    DatasetPlotter,
    TrainingPlotter,
)


class ResultsVisualizer:
    """모델 결과 시각화"""

    def __init__(self, output_dir: Path = Path("plots")):
        """
        Args:
            output_dir: 플롯 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_history_from_json(self, json_path: Path) -> Dict:
        """JSON 파일에서 히스토리 로드"""
        if not json_path.exists():
            print(f"❌ 파일을 찾을 수 없음: {json_path}")
            return {}

        try:
            with open(json_path, "r") as f:
                history = json.load(f)
            print(f"✅ 로드됨: {json_path}")
            return history
        except Exception as e:
            print(f"❌ 로드 실패: {e}")
            return {}

    def load_history_from_checkpoint(self, checkpoint_path: Path) -> Dict:
        """체크포인트 파일에서 히스토리 로드"""
        if not checkpoint_path.exists():
            print(f"❌ 파일을 찾을 수 없음: {checkpoint_path}")
            return {}

        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            history = checkpoint.get("history", {})
            print(f"✅ 로드됨: {checkpoint_path}")
            return history
        except Exception as e:
            print(f"❌ 로드 실패: {e}")
            return {}

    def plot_single_experiment(
        self,
        history: Dict,
        experiment_name: str = "Experiment",
        save_plots: bool = True,
    ):
        """단일 실험 결과 시각화"""
        print(f"\n📊 실험 결과: {experiment_name}")
        print("=" * 60)

        plotter = TrainingPlotter(figsize=(15, 10))

        # 저장 경로
        save_path = (
            self.output_dir / f"{experiment_name}_training_history.png"
            if save_plots
            else None
        )

        # 플롯
        plotter.plot_training_history(
            history, save_path=save_path, title=f"Training History - {experiment_name}"
        )

        # 최종 성능 출력
        if "val_acc" in history:
            final_val_acc = history["val_acc"][-1]
            best_val_acc = max(history["val_acc"])
            print(f"\n📈 최종 성능:")
            print(f"  - 최종 Val Accuracy: {final_val_acc:.2%}")
            print(f"  - 최고 Val Accuracy: {best_val_acc:.2%}")

            if "train_acc" in history:
                final_train_acc = history["train_acc"][-1]
                gap = final_train_acc - final_val_acc
                print(f"  - 최종 Train Accuracy: {final_train_acc:.2%}")
                print(f"  - Overfitting Gap: {gap:.2%}")

    def plot_class_performance(
        self,
        class_accuracies: Dict[str, float],
        y_true: Optional[List[int]] = None,
        y_pred: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
        save_plots: bool = True,
    ):
        """클래스별 성능 시각화"""
        print(f"\n📊 클래스별 성능")
        print("=" * 60)

        # 1. 클래스별 정확도
        save_path = self.output_dir / "class_accuracy.png" if save_plots else None
        ClassPerformancePlotter.plot_class_accuracy(
            class_accuracies, save_path=save_path
        )

        # 2. Precision, Recall, F1-Score
        if y_true is not None and y_pred is not None and class_names is not None:
            save_path = self.output_dir / "class_metrics.png" if save_plots else None
            ClassPerformancePlotter.plot_class_metrics(
                y_true, y_pred, class_names, save_path=save_path
            )

            # 3. 혼동 행렬
            save_path = self.output_dir / "confusion_matrix.png" if save_plots else None
            ConfusionMatrixPlotter.plot_confusion_matrix(
                y_true, y_pred, class_names, save_path=save_path, normalize=False
            )

            save_path = (
                self.output_dir / "confusion_matrix_normalized.png"
                if save_plots
                else None
            )
            ConfusionMatrixPlotter.plot_confusion_matrix(
                y_true, y_pred, class_names, save_path=save_path, normalize=True
            )

    def plot_dataset_distribution(
        self,
        class_counts: Dict[str, int],
        save_plots: bool = True,
    ):
        """데이터셋 분포 시각화"""
        print(f"\n📊 데이터셋 분포")
        print("=" * 60)

        # 클래스 분포
        save_path = self.output_dir / "class_distribution.png" if save_plots else None
        DatasetPlotter.plot_class_distribution(class_counts, save_path=save_path)

        # 불균형 비율
        save_path = self.output_dir / "imbalance_ratio.png" if save_plots else None
        DatasetPlotter.plot_imbalance_ratio(class_counts, save_path=save_path)

    def plot_experiment_comparison(
        self,
        experiments: Dict[str, Dict],
        save_plots: bool = True,
    ):
        """여러 실험 비교"""
        print(f"\n📊 실험 비교")
        print("=" * 60)

        # 성능 곡선 비교
        save_path = (
            self.output_dir / "experiment_comparison.png" if save_plots else None
        )
        ComparisonPlotter.plot_experiment_comparison(experiments, save_path=save_path)

        # 최종 메트릭 비교
        final_metrics = {}
        for exp_name, history in experiments.items():
            final_metrics[exp_name] = {
                "val_acc": max(history.get("val_acc", [0])),
            }

        save_path = (
            self.output_dir / "final_metrics_comparison.png" if save_plots else None
        )
        ComparisonPlotter.plot_final_metrics_comparison(
            final_metrics, save_path=save_path
        )


def create_example_plots():
    """예제 플롯 생성"""
    print("\n" + "=" * 60)
    print("📊 예제 플롯 생성")
    print("=" * 60)

    visualizer = ResultsVisualizer(output_dir=Path("plots"))

    # 1. 학습 히스토리
    print("\n[1/3] 학습 곡선 플롯...")
    history = {
        "train_loss": [0.8, 0.6, 0.4, 0.3, 0.25, 0.22],
        "val_loss": [0.85, 0.65, 0.5, 0.45, 0.50, 0.55],
        "train_acc": [0.65, 0.75, 0.82, 0.86, 0.88, 0.89],
        "val_acc": [0.60, 0.68, 0.72, 0.73, 0.71, 0.70],
        "learning_rate": [1e-4, 1e-4, 5e-5, 5e-5, 1e-5, 1e-5],
    }
    visualizer.plot_single_experiment(history, "example_experiment")

    # 2. 클래스별 성능
    print("\n[2/3] 클래스별 성능 플롯...")
    class_accuracies = {
        "중립": 0.75,
        "분노": 0.69,
        "당황": 0.67,
        "슬픔": 0.60,
        "불안": 0.50,
        "상처": 0.45,
    }
    visualizer.plot_class_performance(class_accuracies)

    # 3. 데이터셋 분포
    print("\n[3/3] 데이터셋 분포 플롯...")
    class_counts = {
        "중립": 500,
        "분노": 400,
        "당황": 380,
        "슬픔": 300,
        "불안": 200,
        "상처": 150,
    }
    visualizer.plot_dataset_distribution(class_counts)

    print(f"\n✅ 플롯 저장됨: {visualizer.output_dir}/")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="모델 결과 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 예제 플롯 생성
  python plot_results.py --example

  # 학습 히스토리 플롯
  python plot_results.py --history path/to/history.json

  # 체크포인트에서 로드
  python plot_results.py --checkpoint path/to/checkpoint.pt

  # 클래스 분포 플롯
  python plot_results.py --dataset-distribution

  # 모든 플롯 생성
  python plot_results.py --all
        """,
    )

    parser.add_argument(
        "--example",
        action="store_true",
        help="예제 플롯 생성",
    )

    parser.add_argument(
        "--history",
        type=str,
        help="히스토리 JSON 파일 경로",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="체크포인트 파일 경로",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="experiment",
        help="실험 이름",
    )

    parser.add_argument(
        "--dataset-distribution",
        action="store_true",
        help="데이터셋 분포 시각화",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="플롯 저장 디렉토리",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="플롯 저장하지 않음",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="모든 플롯 생성 (예제)",
    )

    args = parser.parse_args()

    visualizer = ResultsVisualizer(output_dir=Path(args.output_dir))
    save_plots = not args.no_save

    # 예제 플롯
    if args.example or args.all:
        create_example_plots()
        return

    # 히스토리 플롯
    if args.history:
        print(f"\n📊 히스토리 로드: {args.history}")
        history = visualizer.load_history_from_json(Path(args.history))
        if history:
            visualizer.plot_single_experiment(
                history, experiment_name=args.name, save_plots=save_plots
            )
        return

    # 체크포인트에서 로드
    if args.checkpoint:
        print(f"\n📊 체크포인트 로드: {args.checkpoint}")
        history = visualizer.load_history_from_checkpoint(Path(args.checkpoint))
        if history:
            visualizer.plot_single_experiment(
                history, experiment_name=args.name, save_plots=save_plots
            )
        return

    # 데이터셋 분포
    if args.dataset_distribution:
        class_counts = {
            "중립": 500,
            "분노": 400,
            "당황": 380,
            "슬픔": 300,
            "불안": 200,
            "상처": 150,
        }
        visualizer.plot_dataset_distribution(class_counts, save_plots=save_plots)
        return

    # 도움말 출력
    parser.print_help()


if __name__ == "__main__":
    main()
