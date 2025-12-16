"""
개선된 설정으로 감정 분류 모델 학습
Improved Training Script for Emotion Classification

사용법:
    python train_improved.py --config overfitting_reduction
    python train_improved.py --config class_balanced
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from config import (
    Config,
    get_class_balanced_config,
    get_cnn_only_config,
    get_default_config,
    get_heavy_config,
    get_light_config,
    get_overfitting_reduction_config,
)
from pipeline import print_banner, train_pipeline


def main():
    """개선된 설정으로 학습 실행"""
    parser = argparse.ArgumentParser(
        description="감정 분류 모델 학습 (개선 버전)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 과적합 감소 설정으로 학습
  python train_improved.py --config overfitting_reduction

  # 클래스 불균형 처리 설정으로 학습
  python train_improved.py --config class_balanced

  # 고성능 설정으로 학습
  python train_improved.py --config heavy

  # 커스텀 이름으로 학습
  python train_improved.py --config overfitting_reduction --name my_experiment_v2

설정 옵션:
  default              기본 설정 (원본)
  light                경량 설정 (테스트용)
  heavy                고성능 설정
  cnn_only             CNN만 사용 (YOLO 제외)
  overfitting_reduction 과적합 감소 (강추)
  class_balanced       클래스 불균형 처리 (강추)
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="overfitting_reduction",
        choices=[
            "default",
            "light",
            "heavy",
            "cnn_only",
            "overfitting_reduction",
            "class_balanced",
        ],
        help="사용할 설정 프리셋",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="실험 이름 (생략 시 자동 생성)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="에폭 수 (설정을 덮어씀)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="배치 크기 (설정을 덮어씀)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="학습률 (설정을 덮어씀)",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="드롭아웃 비율 (설정을 덮어씀)",
    )

    parser.add_argument(
        "--augmentation",
        type=str,
        choices=["light", "medium", "heavy"],
        default=None,
        help="데이터 증강 수준 (설정을 덮어씀)",
    )

    parser.add_argument(
        "--fusion",
        type=str,
        choices=["concat", "attention", "weighted_sum", "gated"],
        default=None,
        help="Fusion 방식 (설정을 덮어씀)",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"],
        default=None,
        help="CNN 백본 (설정을 덮어씀)",
    )

    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="TensorBoard 로깅 비활성화",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드 (light 설정으로 빠르게 테스트)",
    )

    args = parser.parse_args()

    # 배너 출력
    print_banner()

    # 설정 선택
    print(f"\n📋 설정 선택: {args.config}")
    print("-" * 60)

    if args.config == "default":
        config = get_default_config()
        config_name = "기본 설정"
    elif args.config == "light":
        config = get_light_config()
        config_name = "경량 설정"
    elif args.config == "heavy":
        config = get_heavy_config()
        config_name = "고성능 설정"
    elif args.config == "cnn_only":
        config = get_cnn_only_config()
        config_name = "CNN 전용 설정"
    elif args.config == "overfitting_reduction":
        config = get_overfitting_reduction_config()
        config_name = "과적합 감소 설정 ⭐"
    elif args.config == "class_balanced":
        config = get_class_balanced_config()
        config_name = "클래스 불균형 처리 설정 ⭐"
    else:
        config = get_default_config()
        config_name = "기본 설정"

    print(f"사용 설정: {config_name}\n")

    # 디버그 모드
    if args.debug:
        print("🔧 디버그 모드 활성화")
        config = get_light_config()
        config_name = "디버그 모드 (경량)"

    # 커맨드라인 인자로 설정 덮어쓰기
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.dropout:
        config.model.classifier_dropout = args.dropout
    if args.augmentation:
        config.training.augmentation_level = args.augmentation
    if args.fusion:
        config.model.fusion.fusion_type = args.fusion
    if args.backbone:
        config.model.cnn.backbone = args.backbone

    # 실험 이름 설정
    if args.name:
        config.experiment_name = args.name
    else:
        # 설정 이름으로 자동 생성
        config_suffix = args.config.replace("_", "")
        config.experiment_name = f"improved_{config_suffix}"

    # 설정 출력
    print("\n⚙️  설정 상세 정보")
    print("-" * 60)
    print(f"  - 실험 이름: {config.experiment_name}")
    print(f"  - 에폭: {config.training.num_epochs}")
    print(f"  - 배치 크기: {config.training.batch_size}")
    print(f"  - 학습률: {config.training.learning_rate}")
    print(f"  - 드롭아웃: {config.model.classifier_dropout}")
    print(f"  - 데이터 증강: {config.training.augmentation_level}")
    print(f"  - Fusion: {config.model.fusion.fusion_type}")
    print(f"  - CNN 백본: {config.model.cnn.backbone}")
    print(f"  - Weight Decay: {config.training.weight_decay}")
    print(f"  - Early Stopping Patience: {config.training.early_stopping_patience}")

    # TensorBoard 설정
    if args.no_tensorboard:
        print(f"  - TensorBoard: 비활성화")
    else:
        print(f"  - TensorBoard: 활성화 → runs/")

    print("\n" + "=" * 60)

    # 학습 실행
    try:
        history = train_pipeline(config, args)

        # 결과 요약
        print("\n" + "=" * 60)
        print("✅ 학습 완료!")
        print("=" * 60)

        if history:
            final_val_acc = max(history.get("val_acc", [0]))
            final_train_acc = (
                history["train_acc"][-1] if history.get("train_acc") else 0
            )
            overfitting_gap = final_train_acc - final_val_acc

            print(f"\n📊 최종 결과:")
            print(f"  - Train Accuracy: {final_train_acc:.2%}")
            print(f"  - Val Accuracy: {final_val_acc:.2%}")
            print(f"  - Overfitting Gap: {overfitting_gap:.2%}")

            if overfitting_gap > 15:
                print("\n⚠️  과적합이 여전히 높습니다. 다음을 시도하세요:")
                print("  - dropout을 더 높이기 (0.5 → 0.6)")
                print("  - weight_decay를 높이기 (1e-3 → 5e-3)")
                print("  - augmentation_level을 'heavy'로 변경")
            elif overfitting_gap > 10:
                print("\n⚠️  여전히 과적합 경향이 있습니다.")
                print("  - 다른 설정을 시도해보세요")
            else:
                print("\n✅ 좋은 성능입니다!")

        # 체크포인트 경로
        checkpoint_dir = config.training.checkpoint_dir / config.experiment_name
        print(f"\n💾 체크포인트 저장 위치:")
        print(f"  {checkpoint_dir}/")

        print(f"\n📈 TensorBoard 확인:")
        print(f"  tensorboard --logdir=runs")

    except KeyboardInterrupt:
        print("\n\n⚠️  학습이 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
