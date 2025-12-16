"""
Hybrid Emotion Classification Pipeline (v2)
하이브리드 감정 분류 파이프라인

CNN + 사전 추출된 FaceNet 특징을 결합하여 감정을 분류합니다.

개선 사항 (v2):
    - 더 깊은 FaceNet Projector (4 블록)
    - Bilinear/Attention Fusion 지원
    - Label Smoothing 지원
    - Warmup Scheduler 지원
    - 다양한 프리셋 설정 지원
    - 예측 시 FaceNet 벡터 실시간 추출

사용법:
    python pipeline.py --mode train
    python pipeline.py --mode train --preset improved_v2
    python pipeline.py --mode train --live-plot
    python pipeline.py --mode eval --checkpoint checkpoints/best_model.pth
    python pipeline.py --mode predict --image path/to/image.jpg
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    Config,
    get_class_name,
    get_config_by_name,
    get_default_config,
    get_light_config,
    list_presets,
)
from data.dataset import create_dataloaders, get_transforms  # noqa: E402
from models.hybrid_model import HybridEmotionModel, HybridEmotionModelLite  # noqa: E402
from training.trainer import Trainer, evaluate_model  # noqa: E402
from util.cuda import (  # noqa: E402
    check_cuda_available,
    get_optimal_num_workers,
    print_cuda_info,
    print_system_info,
    require_cuda,
    setup_cuda_optimization,
    setup_multiprocessing,
)

# FaceNet 실시간 추출용 (MTCNN 얼굴 감지 포함)
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1

    FACENET_AVAILABLE = True
    MTCNN_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    MTCNN_AVAILABLE = False

# ============================================================
# 배너 및 설정
# ============================================================


def print_banner():
    """배너 출력"""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║       Hybrid Emotion Classification Pipeline v3.0                 ║
║       CNN + Precomputed FaceNet Features for Emotion Recognition  ║
╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def setup_config(args) -> Config:
    """설정 초기화"""
    # 프리셋 선택
    if hasattr(args, "preset") and args.preset:
        config = get_config_by_name(args.preset)
        print(f"[설정] 프리셋 '{args.preset}' 로드됨")
    elif args.light:
        config = get_light_config()
    else:
        config = get_default_config()

    # 명령줄 인자로 설정 오버라이드
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.backbone:
        config.model.cnn.backbone = args.backbone

    # FaceNet 브랜치 설정
    if args.no_facenet:
        config.model.use_facenet_branch = False
        config.model.facenet.enabled = False
    if args.no_cnn:
        config.model.use_cnn_branch = False

    # 융합 방식 설정
    if args.fusion:
        config.model.fusion.fusion_type = args.fusion

    # v2 추가: Label Smoothing
    if hasattr(args, "label_smoothing") and args.label_smoothing is not None:
        config.training.label_smoothing = args.label_smoothing

    # v2 추가: FaceNet 블록 수
    if hasattr(args, "facenet_blocks") and args.facenet_blocks:
        config.model.facenet.num_blocks = args.facenet_blocks

    # FaceNet 벡터 경로
    if args.facenet_vectors:
        config.model.facenet.vectors_path = Path(args.facenet_vectors)

    # 최적의 worker 수 자동 설정
    if check_cuda_available():
        optimal_workers = get_optimal_num_workers()
        config.training.num_workers = optimal_workers
    else:
        config.training.num_workers = 0

    # 실험 이름
    if args.name:
        config.experiment_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"hybrid_emotion_{timestamp}"

    return config


def create_model(config: Config) -> torch.nn.Module:
    """모델 생성 (v2 개선 옵션 지원)"""
    use_facenet = config.model.use_facenet_branch and config.model.facenet.enabled
    use_cnn = config.model.use_cnn_branch

    if use_facenet and use_cnn:
        # 하이브리드 모델 (FaceNet + CNN)
        model = HybridEmotionModel(
            # FaceNet 설정
            use_facenet_branch=True,
            facenet_input_dim=config.model.facenet.input_dim,
            facenet_feature_dim=config.model.facenet.feature_dim,
            facenet_num_blocks=config.model.facenet.num_blocks,  # v2: 블록 수
            # CNN 설정
            use_cnn_branch=True,
            cnn_backbone=config.model.cnn.backbone,
            cnn_feature_dim=config.model.cnn.feature_dim,
            cnn_pretrained=config.model.cnn.pretrained,
            freeze_cnn=config.model.cnn.freeze_backbone,
            use_multiscale_cnn=config.model.cnn.use_multiscale,
            # Fusion 설정
            fusion_type=config.model.fusion.fusion_type,
            fusion_output_dim=config.model.fusion.output_dim,
            attention_heads=config.model.fusion.attention_heads,
            use_se=config.model.fusion.use_se,  # v2: SE 블록
            # Classification 설정
            num_classes=config.data.num_classes,
            classifier_hidden_dims=config.model.classifier.hidden_dims,
            dropout=config.model.classifier.dropout,
            use_classifier_residual=config.model.classifier.use_residual,  # v2: Residual
            device=config.device,
        )
    elif use_facenet:
        # FaceNet만 사용
        model = HybridEmotionModel(
            use_facenet_branch=True,
            facenet_input_dim=config.model.facenet.input_dim,
            facenet_feature_dim=config.model.facenet.feature_dim,
            facenet_num_blocks=config.model.facenet.num_blocks,  # v2: 블록 수
            use_cnn_branch=False,
            num_classes=config.data.num_classes,
            classifier_hidden_dims=config.model.classifier.hidden_dims,
            dropout=config.model.classifier.dropout,
            use_classifier_residual=config.model.classifier.use_residual,  # v2: Residual
            device=config.device,
        )
    elif use_cnn:
        # CNN만 사용 (Lite 모델)
        model = HybridEmotionModelLite(
            backbone=config.model.cnn.backbone,
            feature_dim=config.model.cnn.feature_dim,
            num_classes=config.data.num_classes,
            pretrained=config.model.cnn.pretrained,
            dropout=config.model.classifier.dropout,
            classifier_hidden_dims=config.model.classifier.hidden_dims,
        )
    else:
        raise ValueError("최소 하나의 브랜치 (FaceNet 또는 CNN)가 필요합니다.")

    return model.to(config.device)


# ============================================================
# 파이프라인 함수
# ============================================================


@require_cuda
def train_pipeline(config: Config, args):
    """학습 파이프라인 (CUDA 필수)"""
    print("\n" + "=" * 60)
    print("학습 파이프라인 시작")
    print("=" * 60)

    # CUDA 최적화 적용
    setup_cuda_optimization()
    print_cuda_info()

    print("\n[설정]")
    print(f"  - 실험 이름: {config.experiment_name}")
    print(f"  - 디바이스: {config.device}")
    print(f"  - 배치 크기: {config.training.batch_size}")
    print(f"  - 에폭 수: {config.training.num_epochs}")
    print(f"  - 학습률: {config.training.learning_rate}")
    print(f"  - FaceNet 사용: {config.model.use_facenet_branch}")
    if config.model.use_facenet_branch:
        print(f"  - FaceNet 블록 수: {config.model.facenet.num_blocks}")
    print(f"  - CNN 백본: {config.model.cnn.backbone}")
    print(f"  - Fusion: {config.model.fusion.fusion_type}")
    print(f"  - Fusion SE: {config.model.fusion.use_se}")
    print(f"  - Label Smoothing: {config.training.label_smoothing}")
    print(f"  - Warmup 에폭: {config.training.warmup_epochs}")
    print(f"  - Num Workers: {config.training.num_workers}")
    print(f"  - AMP (Mixed Precision): {config.training.use_amp}")
    print(f"  - 실시간 Plotting: {args.live_plot}")

    # FaceNet 벡터 경로 확인
    use_facenet = config.model.use_facenet_branch and config.model.facenet.enabled
    facenet_vectors_path = None

    if use_facenet:
        facenet_vectors_path = config.model.facenet.vectors_path
        if not facenet_vectors_path.exists():
            print(f"\n[경고] FaceNet 벡터 파일이 없습니다: {facenet_vectors_path}")
            print("먼저 extract_facenet_vectors.py를 실행하여 벡터를 추출하세요.")
            print("또는 --no-facenet 옵션으로 CNN만 사용하세요.")
            sys.exit(1)
        print(f"  - FaceNet 벡터: {facenet_vectors_path}")

    # 데이터로더 생성
    print("\n[1/4] 데이터 로드 중...")
    if config.data.images_dir is None:
        raise ValueError("이미지 디렉토리가 설정되지 않았습니다.")

    train_loader, val_loader, data_info = create_dataloaders(
        images_dir=config.data.images_dir,
        class_names=config.data.class_names_kr,
        batch_size=config.training.batch_size,
        train_ratio=config.data.train_ratio,
        image_size=config.data.image_size,
        num_workers=config.training.num_workers,
        seed=config.data.seed,
        labels_dir=config.data.labels_dir,
        augmentation_level=config.training.augmentation_level,
        use_facenet=use_facenet,
        facenet_vectors_path=facenet_vectors_path,
    )

    # 모델 생성
    print("\n[2/4] 모델 생성 중...")
    model = create_model(config)

    # 플롯 저장 디렉토리
    plot_save_dir = config.training.checkpoint_dir / config.experiment_name / "plots"

    # Trainer 생성 (v2 옵션 추가)
    print("\n[3/4] Trainer 초기화 중...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.device,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        use_amp=config.training.use_amp,
        log_interval=config.training.log_interval,
        tensorboard_dir=config.training.tensorboard_dir,
        experiment_name=config.experiment_name,
        checkpoint_dir=config.training.checkpoint_dir / config.experiment_name,
        save_best_only=config.training.save_best_only,
        early_stopping_patience=config.training.early_stopping_patience,
        class_weights=data_info.get("class_weights"),
        class_names=config.data.class_names_kr,
        # 실시간 Plotting 설정
        live_plot=args.live_plot,
        plot_update_interval=1,
        plot_save_dir=plot_save_dir,
        # v2 추가 옵션
        label_smoothing=config.training.label_smoothing,
        warmup_epochs=config.training.warmup_epochs,
        scheduler_type=config.training.scheduler,
    )

    # 학습
    print("\n[4/4] 학습 시작...")
    history = trainer.train()

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"  - 최고 검증 정확도: {max(history.get('val_acc', [0])):.2f}%")
    print(f"  - 체크포인트: {config.training.checkpoint_dir / config.experiment_name}")

    # 학습 완료 후 분석 (옵션)
    if args.analysis:
        run_post_training_analysis(config, trainer, val_loader)

    return history


def run_post_training_analysis(config: Config, trainer: Trainer, val_loader):
    """학습 완료 후 분석 실행"""
    print("\n[분석] 학습 후 분석 시작...")

    try:
        from util.plotting import TrainingVisualizer

        # 학습 히스토리 시각화
        history = trainer.history
        save_dir = config.training.checkpoint_dir / config.experiment_name / "analysis"
        save_dir.mkdir(parents=True, exist_ok=True)

        visualizer = TrainingVisualizer(
            history=history,
            class_names=config.data.class_names_kr,
            save_dir=save_dir,
            experiment_name=config.experiment_name,
        )
        visualizer.plot_all(save=True, show=False)

        print(f"  - 학습 그래프 저장됨: {save_dir}")

    except ImportError as e:
        print(f"  - 분석 모듈 로드 실패: {e}")
        print("    matplotlib, seaborn, scikit-learn이 설치되어 있는지 확인하세요.")


@require_cuda
def eval_pipeline(config: Config, args):
    """평가 파이프라인 (CUDA 필수)"""
    print("\n" + "=" * 60)
    print("평가 파이프라인 시작")
    print("=" * 60)

    # CUDA 최적화 적용
    setup_cuda_optimization()
    print_cuda_info()

    if not args.checkpoint:
        print("Error: --checkpoint 옵션이 필요합니다.")
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        sys.exit(1)

    # FaceNet 설정
    use_facenet = config.model.use_facenet_branch and config.model.facenet.enabled
    facenet_vectors_path = config.model.facenet.vectors_path if use_facenet else None

    # 데이터로더 생성
    print("\n[1/3] 데이터 로드 중...")
    if config.data.images_dir is None:
        raise ValueError("이미지 디렉토리가 설정되지 않았습니다.")

    _, val_loader, _ = create_dataloaders(
        images_dir=config.data.images_dir,
        class_names=config.data.class_names_kr,
        batch_size=config.training.batch_size,
        train_ratio=config.data.train_ratio,
        image_size=config.data.image_size,
        num_workers=config.training.num_workers,
        seed=config.data.seed,
        use_facenet=use_facenet,
        facenet_vectors_path=facenet_vectors_path,
    )

    # 모델 로드
    print("\n[2/3] 모델 로드 중...")
    model = create_model(config)

    checkpoint = torch.load(
        checkpoint_path, map_location=config.device, weights_only=False
    )
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"  - 체크포인트: {checkpoint_path}")

    # 평가
    print("\n[3/3] 평가 중...")
    assert config.device is not None, "device must be set"
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=config.device,
        class_names=config.data.class_names_kr,
    )

    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(f"  - 정확도: {metrics['accuracy']:.2f}%")

    if "class_accuracy" in metrics:
        print("\n[클래스별 정확도]")
        for class_name, info in metrics["class_accuracy"].items():
            print(f"  - {class_name}: {info['accuracy']:.2f}% ({info['count']}개)")

    # 분석 리포트 생성 (옵션)
    if args.analysis:
        run_evaluation_analysis(config, metrics, checkpoint_path)

    return metrics


def run_evaluation_analysis(config: Config, metrics, checkpoint_path: Path):
    """평가 완료 후 분석 리포트 생성"""
    print("\n[분석] 평가 분석 리포트 생성 중...")

    try:
        from util.analysis import AnalysisReport, plot_confusion_matrix

        save_dir = checkpoint_path.parent / "analysis"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Confusion Matrix 시각화
        if "predictions" in metrics and "labels" in metrics:
            plot_confusion_matrix(
                y_true=metrics["labels"],
                y_pred=metrics["predictions"],
                class_names=config.data.class_names_kr,
                normalize=True,
                save_path=save_dir / "confusion_matrix.png",
                show=False,
            )
            print(f"  - Confusion Matrix 저장됨: {save_dir / 'confusion_matrix.png'}")

        # 종합 분석 리포트
        if "predictions" in metrics and "labels" in metrics:
            report = AnalysisReport(
                y_true=metrics["labels"],
                y_pred=metrics["predictions"],
                y_prob=metrics.get("probabilities"),
                class_names=config.data.class_names_kr,
                experiment_name=config.experiment_name,
                save_dir=save_dir,
            )
            report.compute_all_metrics(verbose=True)
            report.analyze_errors(verbose=True)
            report.generate_report(save=True)

    except ImportError as e:
        print(f"  - 분석 모듈 로드 실패: {e}")


def extract_facenet_vector_realtime(
    image: Image.Image,
    device: torch.device,
    pretrained: str = "vggface2",
) -> torch.Tensor:
    """
    단일 이미지에서 FaceNet 벡터를 실시간으로 추출

    Args:
        image: PIL 이미지
        device: 연산 디바이스
        pretrained: 사전학습 가중치 ('vggface2' 또는 'casia-webface')

    Returns:
        FaceNet 임베딩 벡터 [1, 512]
    """
    if not FACENET_AVAILABLE:
        raise ImportError(
            "facenet-pytorch 패키지가 필요합니다. 설치: pip install facenet-pytorch"
        )

    # FaceNet 모델 로드 (싱글톤 패턴으로 캐싱 가능하지만, 간단히 구현)
    facenet_model = InceptionResnetV1(pretrained=pretrained).eval().to(device)

    # FaceNet 입력 전처리 (160x160, 정규화)
    from torchvision import transforms as T

    facenet_transform = T.Compose(
        [
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # 이미지 전처리
    img_tensor = facenet_transform(image).unsqueeze(0).to(device)

    # FaceNet 임베딩 추출
    with torch.no_grad():
        embedding = facenet_model(img_tensor)

    return embedding


# FaceNet 모델 캐시 (전역 변수로 재사용)
_facenet_model_cache = None
_mtcnn_model_cache = None


def get_facenet_model(device: torch.device, pretrained: str = "vggface2"):
    """FaceNet 모델을 캐시하여 반환 (재사용)"""
    global _facenet_model_cache

    if _facenet_model_cache is None:
        if not FACENET_AVAILABLE:
            raise ImportError(
                "facenet-pytorch 패키지가 필요합니다. 설치: pip install facenet-pytorch"
            )
        print("  - FaceNet 모델 로드 중...")
        _facenet_model_cache = (
            InceptionResnetV1(pretrained=pretrained).eval().to(device)
        )
        print(f"  - FaceNet 사전학습: {pretrained}")

    return _facenet_model_cache


def get_mtcnn_model(device: torch.device):
    """MTCNN 얼굴 감지 모델을 캐시하여 반환 (재사용)"""
    global _mtcnn_model_cache

    if _mtcnn_model_cache is None:
        if not MTCNN_AVAILABLE:
            raise ImportError(
                "facenet-pytorch 패키지가 필요합니다. 설치: pip install facenet-pytorch"
            )
        print("  - MTCNN 얼굴 감지 모델 로드 중...")
        _mtcnn_model_cache = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,  # PIL 이미지 반환
            device=device,
            keep_all=True,  # 모든 얼굴 감지
        )

    return _mtcnn_model_cache


def detect_and_crop_face(
    image: Image.Image,
    device: torch.device,
    margin_ratio: float = 0.2,
) -> tuple:
    """
    MTCNN을 사용하여 얼굴을 감지하고 크롭

    Args:
        image: PIL 이미지
        device: 연산 디바이스
        margin_ratio: 바운딩 박스 주변 마진 비율

    Returns:
        (cropped_image, bbox, confidence) 또는 얼굴이 없으면 (None, None, None)
    """
    mtcnn = get_mtcnn_model(device)

    # 얼굴 감지 (bounding boxes, probabilities)
    boxes, probs = mtcnn.detect(image)

    if boxes is None or len(boxes) == 0:
        return None, None, None

    # 가장 큰 얼굴 또는 가장 신뢰도 높은 얼굴 선택
    if len(boxes) > 1:
        # 가장 큰 얼굴 선택 (면적 기준)
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        best_idx = max(range(len(areas)), key=lambda i: areas[i])
    else:
        best_idx = 0

    box = boxes[best_idx]
    prob = probs[best_idx] if probs is not None else 0.0

    # 바운딩 박스 좌표 추출
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    # 마진 추가
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio

    # 이미지 경계 내로 제한
    img_width, img_height = image.size
    x1 = max(0, int(x1 - margin_x))
    y1 = max(0, int(y1 - margin_y))
    x2 = min(img_width, int(x2 + margin_x))
    y2 = min(img_height, int(y2 + margin_y))

    # 크롭
    cropped = image.crop((x1, y1, x2, y2))
    bbox = (x1, y1, x2, y2)

    return cropped, bbox, float(prob)


@require_cuda
def predict_pipeline(config: Config, args):
    """예측 파이프라인 (CUDA 필수) - 얼굴 감지 + FaceNet 실시간 추출 지원"""
    print("\n" + "=" * 60)
    print("예측 파이프라인 시작")
    print("=" * 60)

    # CUDA 최적화 적용
    setup_cuda_optimization()
    print_cuda_info()

    if not args.image:
        print("Error: --image 옵션이 필요합니다.")
        sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: 이미지 파일을 찾을 수 없습니다: {image_path}")
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path and not checkpoint_path.exists():
        print(f"Error: 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        sys.exit(1)

    # FaceNet/MTCNN 사용 여부 결정
    use_facenet = config.model.use_facenet_branch and config.model.facenet.enabled
    use_face_detection = MTCNN_AVAILABLE and not getattr(
        args, "no_face_detection", False
    )

    if use_facenet and not FACENET_AVAILABLE:
        print(
            "\n[주의] facenet-pytorch가 설치되지 않아 FaceNet 브랜치를 비활성화합니다."
        )
        print("설치: pip install facenet-pytorch")
        use_facenet = False
        config.model.use_facenet_branch = False
        config.model.facenet.enabled = False

    if use_facenet:
        print("\n[FaceNet] 실시간 벡터 추출 모드 활성화")

    if use_face_detection:
        print("[MTCNN] 얼굴 감지 및 크롭 모드 활성화")

    # 모델 로드
    print("\n[1/5] 모델 로드 중...")
    model = create_model(config)

    if checkpoint_path:
        checkpoint = torch.load(
            checkpoint_path, map_location=config.device, weights_only=False
        )
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"  - 체크포인트: {checkpoint_path}")

    model.eval()

    # 이미지 로드
    print("\n[2/5] 이미지 로드 중...")
    original_image = Image.open(image_path).convert("RGB")
    print(f"  - 원본 이미지 크기: {original_image.size}")

    # 얼굴 감지 및 크롭
    face_bbox = None
    face_confidence = None
    if use_face_detection:
        print("\n[3/5] 얼굴 감지 중...")
        cropped_face, face_bbox, face_confidence = detect_and_crop_face(
            original_image, config.device, margin_ratio=0.2
        )

        if cropped_face is not None:
            print("  - 얼굴 감지 성공!")
            print(f"  - 바운딩 박스: {face_bbox}")
            print(f"  - 감지 신뢰도: {face_confidence:.4f}")
            print(f"  - 크롭된 크기: {cropped_face.size}")
            image_for_prediction = cropped_face
        else:
            print("  - [경고] 얼굴을 감지하지 못했습니다. 원본 이미지를 사용합니다.")
            image_for_prediction = original_image
    else:
        print("\n[3/5] 얼굴 감지 건너뜀 (--no-face-detection 옵션)")
        image_for_prediction = original_image

    # CNN용 이미지 전처리
    print("\n[4/5] 이미지 전처리 중...")
    transform = get_transforms(config.data.image_size, is_training=False)
    image_tensor: torch.Tensor = transform(image_for_prediction)  # type: ignore[assignment]
    image_tensor = image_tensor.unsqueeze(0).to(config.device)

    # FaceNet 벡터 추출 (실시간) - 크롭된 얼굴 이미지 사용
    facenet_vector = None
    if use_facenet:
        print("  - FaceNet 벡터 추출 중...")
        from torchvision import transforms as T

        facenet_transform = T.Compose(
            [
                T.Resize((160, 160)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        facenet_model = get_facenet_model(
            config.device, config.model.facenet.pretrained
        )
        # 크롭된 얼굴 이미지로 FaceNet 벡터 추출
        facenet_input = (
            facenet_transform(image_for_prediction).unsqueeze(0).to(config.device)
        )

        with torch.no_grad():
            facenet_vector = facenet_model(facenet_input)
        print(f"  - FaceNet 벡터 shape: {facenet_vector.shape}")

    # 예측
    print("\n[5/5] 예측 중...")
    with torch.no_grad():
        if use_facenet and facenet_vector is not None:
            outputs = model(image_tensor, facenet_vector=facenet_vector)
        else:
            outputs = model(image_tensor)

        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_class].item())

    # 결과 출력
    pred_name_kr = get_class_name(pred_class, "kr", config)
    pred_name_en = get_class_name(pred_class, "en", config)

    print("\n" + "=" * 60)
    print("예측 결과")
    print("=" * 60)
    print(f"  - 이미지: {image_path}")
    if face_bbox:
        print(f"  - 얼굴 위치: {face_bbox}")
        print(f"  - 얼굴 감지 신뢰도: {face_confidence:.4f}")
    print(f"  - 예측 클래스: {pred_name_kr} ({pred_name_en})")
    print(f"  - 예측 신뢰도: {confidence:.4f}")
    print(f"  - 얼굴 감지 사용: {'✓' if use_face_detection and face_bbox else '✗'}")
    print(f"  - FaceNet 사용: {'✓' if use_facenet else '✗'}")

    print("\n[전체 확률]")
    for i, prob in enumerate(probs[0].tolist()):
        name_kr = get_class_name(i, "kr", config)
        name_en = get_class_name(i, "en", config)
        bar = "█" * int(prob * 20)
        print(f"  {name_kr:4s} ({name_en:12s}): {prob:.4f} {bar}")

    return {
        "class_idx": pred_class,
        "class_name_kr": pred_name_kr,
        "class_name_en": pred_name_en,
        "confidence": confidence,
        "probabilities": probs[0].tolist(),
        "facenet_used": use_facenet,
        "face_detected": face_bbox is not None,
        "face_bbox": face_bbox,
        "face_detection_confidence": face_confidence,
    }


@require_cuda
def run_test_pipeline(config: Config, args):
    """테스트 파이프라인 (더미 데이터로 동작 확인, CUDA 필수)"""
    print("\n" + "=" * 60)
    print("테스트 파이프라인 시작")
    print("=" * 60)

    # CUDA 최적화 적용
    setup_cuda_optimization()
    print_cuda_info()

    # 더미 데이터셋
    class DummyDataset(Dataset):
        def __init__(self, size: int = 100, use_facenet: bool = False):
            self.size = size
            self.use_facenet = use_facenet
            self.facenet_dim = config.model.facenet.input_dim

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            result = {
                "image": torch.randn(3, 224, 224),
                "label": torch.tensor(idx % 7),
                "path": f"dummy_{idx}.jpg",
            }
            if self.use_facenet:
                result["facenet_vector"] = torch.randn(self.facenet_dim)
            return result

    def collate_fn(batch):
        result = {
            "image": torch.stack([item["image"] for item in batch]),
            "label": torch.stack([item["label"] for item in batch]),
            "path": [item["path"] for item in batch],
        }
        if "facenet_vector" in batch[0]:
            result["facenet_vector"] = torch.stack(
                [item["facenet_vector"] for item in batch]
            )
        return result

    # 설정 조정
    config.training.num_epochs = 2
    config.training.batch_size = 8

    use_facenet = config.model.use_facenet_branch and config.model.facenet.enabled

    # 더미 데이터로더 (멀티스레딩 적용)
    num_workers = get_optimal_num_workers()
    print(f"\n[멀티스레딩] num_workers = {num_workers}")

    train_dataset = DummyDataset(64, use_facenet=use_facenet)
    val_dataset = DummyDataset(16, use_facenet=use_facenet)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )

    # 모델 생성
    print("\n[1/3] 모델 생성 중...")
    model = create_model(config)

    # Trainer 생성
    print("\n[2/3] Trainer 초기화 중...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.device,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        use_amp=config.training.use_amp,
        class_names=config.data.class_names_kr,
        live_plot=args.live_plot,
    )

    # 학습
    print("\n[3/3] 테스트 학습 중...")
    history = trainer.train()

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

    return history


# ============================================================
# 메인 함수
# ============================================================


def main():
    # 멀티프로세싱 설정 (Windows 호환성)
    setup_multiprocessing()

    parser = argparse.ArgumentParser(
        description="Hybrid Emotion Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    # 학습 (CUDA 필수)
    python pipeline.py --mode train

    # 실시간 플롯과 함께 학습
    python pipeline.py --mode train --live-plot

    # 학습 후 분석 리포트 생성
    python pipeline.py --mode train --analysis

    # 경량 모드로 학습
    python pipeline.py --mode train --light

    # CNN만 사용 (FaceNet 없이)
    python pipeline.py --mode train --no-facenet

    # FaceNet만 사용 (CNN 없이)
    python pipeline.py --mode train --no-cnn

    # 평가
    python pipeline.py --mode eval --checkpoint checkpoints/best_model.pth

    # 평가 + 분석 리포트
    python pipeline.py --mode eval --checkpoint checkpoints/best_model.pth --analysis

    # 예측
    python pipeline.py --mode predict --image test.jpg --checkpoint checkpoints/best_model.pth

    # 테스트 (더미 데이터)
    python pipeline.py --mode test

    # CUDA 정보 확인
    python pipeline.py --mode check-cuda
        """,
    )

    # 필수 인자
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval", "predict", "test", "check-cuda"],
        help="실행 모드",
    )

    # 선택 인자
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(list_presets().keys()),
        help="설정 프리셋 선택 (예: improved_v2, max_performance)",
    )
    parser.add_argument("--light", action="store_true", help="경량 설정 사용")
    parser.add_argument("--name", type=str, help="실험 이름")
    parser.add_argument("--batch-size", type=int, help="배치 크기")
    parser.add_argument("--epochs", type=int, help="에폭 수")
    parser.add_argument("--lr", type=float, help="학습률")
    parser.add_argument(
        "--backbone", type=str, help="CNN 백본 (resnet18, resnet34, ...)"
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["concat", "attention", "weighted_sum", "gated", "bilinear"],
        help="특징 융합 방식",
    )
    parser.add_argument(
        "--no-facenet", action="store_true", help="FaceNet 브랜치 비활성화"
    )
    parser.add_argument("--no-cnn", action="store_true", help="CNN 브랜치 비활성화")
    parser.add_argument("--facenet-vectors", type=str, help="FaceNet 벡터 파일 경로")

    # v2 추가 옵션
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label Smoothing 값 (0.0 ~ 0.2 권장)",
    )
    parser.add_argument(
        "--facenet-blocks",
        type=int,
        help="FaceNet Projector 블록 수 (기본값: 4)",
    )
    parser.add_argument(
        "--no-face-detection",
        action="store_true",
        help="얼굴 감지 비활성화 (원본 이미지 그대로 사용)",
    )
    parser.add_argument("--checkpoint", type=str, help="체크포인트 파일 경로")
    parser.add_argument("--image", type=str, help="예측할 이미지 경로")

    # 실시간 Plotting 및 분석 옵션
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="학습 중 실시간 그래프 표시",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="학습/평가 후 분석 리포트 생성",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="사용 가능한 프리셋 목록 출력",
    )

    args = parser.parse_args()

    # 프리셋 목록 출력
    if hasattr(args, "list_presets") and args.list_presets:
        print_banner()
        print("\n[사용 가능한 프리셋]")
        print("-" * 50)
        for name, desc in list_presets().items():
            print(f"  {name:20s} : {desc}")
        print("-" * 50)
        print("\n사용 예시: python pipeline.py --mode train --preset improved_v2")
        return

    # CUDA 정보 확인 모드
    if args.mode == "check-cuda":
        print_banner()
        print_cuda_info()
        print_system_info()
        return

    print_banner()
    config = setup_config(args)

    if args.mode == "train":
        train_pipeline(config, args)
    elif args.mode == "eval":
        eval_pipeline(config, args)
    elif args.mode == "predict":
        predict_pipeline(config, args)
    elif args.mode == "test":
        run_test_pipeline(config, args)


if __name__ == "__main__":
    main()
