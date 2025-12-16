"""
Configuration file for Hybrid Emotion Classification Pipeline
하이브리드 감정 분류 파이프라인 설정

FaceNet 기반 얼굴 임베딩 + CNN 특징 융합
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch


@dataclass
class DataConfig:
    """데이터 관련 설정"""

    # 데이터 경로
    data_root: Path = Path(__file__).parent.parent / "data" / "cropped"
    images_dir: Optional[Path] = field(default=None)
    labels_dir: Optional[Path] = field(default=None)

    # 클래스 정의 (한국어 폴더명) - 기쁨 추가
    class_names_kr: List[str] = field(
        default_factory=lambda: ["기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립"]
    )

    # 클래스 정의 (영어 표시명) - Joy 추가
    class_names_en: List[str] = field(
        default_factory=lambda: [
            "Joy",
            "Embarrassed",
            "Angry",
            "Anxious",
            "Hurt",
            "Sad",
            "Neutral",
        ]
    )

    num_classes: int = 7  # 6 -> 7로 변경

    # 데이터 분할 비율
    train_ratio: float = 0.9  # 90% 학습, 10% 검증

    # 이미지 크기
    image_size: Tuple[int, int] = (224, 224)

    # 랜덤 시드
    seed: int = 42

    def __post_init__(self):
        if self.images_dir is None:
            self.images_dir = self.data_root / "images"
        if self.labels_dir is None:
            self.labels_dir = self.data_root / "labels"


@dataclass
class FaceNetConfig:
    """FaceNet Feature 설정 (사전 추출된 임베딩용)"""

    # 사전 추출된 벡터 파일 경로
    vectors_path: Path = (
        Path(__file__).parent.parent / "data" / "cropped" / "facenet_vectors.pt"
    )

    # FaceNet 임베딩 차원 (고정값)
    input_dim: int = 512

    # Projection 출력 차원
    feature_dim: int = 512

    # 사전학습 가중치: "vggface2" 또는 "casia-webface"
    pretrained: str = "vggface2"

    # FaceNet 입력 이미지 크기 (고정값)
    image_size: int = 160

    # FaceNet 브랜치 사용 여부
    enabled: bool = True


@dataclass
class CNNConfig:
    """CNN Branch 설정"""

    # 백본 모델
    backbone: str = "resnet18"  # resnet18, resnet34, resnet50, efficientnet_b0

    # 사전학습 가중치 사용
    pretrained: bool = True

    # Feature 차원
    feature_dim: int = 512

    # Fine-tuning 설정
    freeze_backbone: bool = False  # 전체 학습
    unfreeze_layers: int = -1  # -1: 전체, 양수: 마지막 N개 레이어만

    # 멀티스케일 CNN 사용
    use_multiscale: bool = False


@dataclass
class FusionConfig:
    """Feature Fusion 설정"""

    # Fusion 방식: "concat", "attention", "weighted_sum", "gated"
    fusion_type: str = "concat"

    # Attention 설정 (fusion_type="attention"일 때)
    attention_heads: int = 4
    attention_dropout: float = 0.1

    # 출력 차원
    output_dim: int = 256


@dataclass
class PlottingConfig:
    """실시간 Plotting 설정"""

    # 실시간 플롯 활성화
    enabled: bool = False

    # 플롯 업데이트 간격 (에폭 단위)
    update_interval: int = 1

    # 그래프 크기
    figsize: Tuple[int, int] = (14, 10)

    # 그래프 저장 디렉토리 (None이면 checkpoint_dir 사용)
    save_dir: Optional[Path] = None

    # 다크 모드
    dark_mode: bool = False

    # 그래프 표시 여부 (False면 저장만)
    show_plot: bool = True


@dataclass
class TrainingConfig:
    """학습 관련 설정"""

    # 배치 크기
    batch_size: int = 32

    # 에폭 수
    num_epochs: int = 30

    # 학습률
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # 학습률 스케줄러
    scheduler: str = "cosine"  # "step", "cosine", "plateau"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1

    # Early stopping
    early_stopping_patience: int = 7

    # Gradient clipping
    grad_clip_norm: float = 1.0

    # Mixed precision training
    use_amp: bool = True

    # DataLoader 설정
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation 수준: "light", "medium", "heavy"
    augmentation_level: str = "medium"

    # Regularization
    dropout_rate: float = 0.5  # 높은 dropout으로 과적합 방지
    l2_weight_decay: float = 1e-3  # L2 정규화

    # 체크포인트
    save_best_only: bool = True
    checkpoint_dir: Path = Path(__file__).parent.parent / "checkpoints"

    # 로깅
    log_interval: int = 10  # 배치 단위
    tensorboard_dir: Path = Path(__file__).parent.parent / "runs"

    # Plotting 설정
    plotting: PlottingConfig = field(default_factory=PlottingConfig)


@dataclass
class ModelConfig:
    """전체 모델 설정"""

    facenet: FaceNetConfig = field(default_factory=FaceNetConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Classification Head
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    classifier_dropout: float = 0.3

    # 브랜치 사용 여부
    use_facenet_branch: bool = True
    use_cnn_branch: bool = True


@dataclass
class Config:
    """전체 설정"""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # 디바이스
    device: Optional[torch.device] = field(default=None)

    # 실험 이름
    experiment_name: str = "hybrid_emotion_facenet_v1"

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """딕셔너리에서 Config 생성"""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Config를 딕셔너리로 변환"""
        return {
            "data": {
                "data_root": str(self.data.data_root),
                "num_classes": self.data.num_classes,
                "train_ratio": self.data.train_ratio,
                "image_size": self.data.image_size,
            },
            "model": {
                "facenet_enabled": self.model.facenet.enabled,
                "facenet_input_dim": self.model.facenet.input_dim,
                "facenet_feature_dim": self.model.facenet.feature_dim,
                "facenet_pretrained": self.model.facenet.pretrained,
                "cnn_backbone": self.model.cnn.backbone,
                "cnn_feature_dim": self.model.cnn.feature_dim,
                "fusion_type": self.model.fusion.fusion_type,
                "use_facenet_branch": self.model.use_facenet_branch,
                "use_cnn_branch": self.model.use_cnn_branch,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "num_epochs": self.training.num_epochs,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
            },
            "device": str(self.device),
            "experiment_name": self.experiment_name,
        }


# ============================================================
# 설정 프리셋
# ============================================================


def get_default_config() -> Config:
    """기본 설정 반환"""
    return Config()


def get_light_config() -> Config:
    """경량 설정 (테스트/디버깅용)"""
    config = Config()
    config.training.batch_size = 16
    config.training.num_epochs = 5
    config.training.num_workers = 2
    config.model.cnn.backbone = "resnet18"
    return config


def get_heavy_config() -> Config:
    """고성능 설정"""
    config = Config()
    config.training.batch_size = 64
    config.training.num_epochs = 50
    config.model.cnn.backbone = "resnet34"
    config.model.fusion.fusion_type = "attention"
    return config


def get_overfitting_reduction_config() -> Config:
    """과적합 감소 설정 (높은 정규화)"""
    config = Config()
    config.training.num_epochs = 50
    config.training.early_stopping_patience = 10
    config.training.learning_rate = 5e-5  # 더 낮은 학습률
    config.training.weight_decay = 5e-4  # 더 높은 L2 정규화
    config.training.augmentation_level = "heavy"  # 강한 augmentation
    config.training.batch_size = 32
    config.model.classifier_dropout = 0.5  # 높은 dropout
    config.model.cnn.backbone = "resnet34"
    config.model.fusion.fusion_type = "attention"  # 더 강력한 fusion
    return config


def get_class_balanced_config() -> Config:
    """클래스 불균형 처리 설정"""
    config = Config()
    config.training.num_epochs = 40
    config.training.batch_size = 16  # 작은 배치로 각 클래스 샘플 균등 분배
    config.training.learning_rate = 1e-4
    config.training.augmentation_level = "heavy"
    config.training.early_stopping_patience = 10
    config.model.classifier_dropout = 0.4
    return config


def get_cnn_only_config() -> Config:
    """CNN만 사용하는 설정 (FaceNet 없음)"""
    config = Config()
    config.model.use_facenet_branch = False
    config.model.facenet.enabled = False
    return config


def get_facenet_only_config() -> Config:
    """FaceNet만 사용하는 설정 (CNN 없음)"""
    config = Config()
    config.model.use_cnn_branch = False
    config.model.use_facenet_branch = True
    config.model.facenet.enabled = True
    return config


def get_asian_face_config() -> Config:
    """아시아인 얼굴 최적화 설정 (CASIA-WebFace 사전학습)"""
    config = Config()
    config.model.facenet.pretrained = "casia-webface"
    config.training.num_epochs = 40
    config.training.learning_rate = 1e-4
    return config


def get_live_plot_config() -> Config:
    """실시간 플롯 활성화 설정"""
    config = Config()
    config.training.plotting.enabled = True
    config.training.plotting.update_interval = 1
    config.training.plotting.show_plot = True
    return config


# ============================================================
# 유틸리티 함수
# ============================================================


def get_class_mapping(config: Optional[Config] = None) -> dict:
    """클래스 인덱스 매핑 반환"""
    if config is None:
        config = get_default_config()

    return {name: idx for idx, name in enumerate(config.data.class_names_kr)}


def get_class_name(idx: int, lang: str = "kr", config: Optional[Config] = None) -> str:
    """인덱스에서 클래스 이름 반환"""
    if config is None:
        config = get_default_config()

    if lang == "kr":
        return config.data.class_names_kr[idx]
    else:
        return config.data.class_names_en[idx]


# 테스트 코드
if __name__ == "__main__":
    print("=" * 50)
    print("Config 테스트")
    print("=" * 50)

    config = get_default_config()
    print("\n[기본 설정]")
    print(f"  - 데이터 경로: {config.data.data_root}")
    print(f"  - 클래스 수: {config.data.num_classes}")
    print(f"  - FaceNet 사용: {config.model.use_facenet_branch}")
    print(f"  - FaceNet 사전학습: {config.model.facenet.pretrained}")
    print(f"  - FaceNet 임베딩 차원: {config.model.facenet.input_dim}")
    print(f"  - CNN 백본: {config.model.cnn.backbone}")
    print(f"  - Fusion: {config.model.fusion.fusion_type}")
    print(f"  - 배치 크기: {config.training.batch_size}")
    print(f"  - 디바이스: {config.device}")

    print("\n[클래스 매핑]")
    mapping = get_class_mapping()
    for name, idx in mapping.items():
        en_name = get_class_name(idx, "en")
        print(f"  {idx}: {name} ({en_name})")

    print("\n[설정 딕셔너리]")
    config_dict = config.to_dict()
    for key, value in config_dict.items():
        print(f"  {key}: {value}")

    print("\n[FaceNet 전용 설정]")
    facenet_config = get_facenet_only_config()
    print(f"  - FaceNet 사용: {facenet_config.model.use_facenet_branch}")
    print(f"  - CNN 사용: {facenet_config.model.use_cnn_branch}")

    print("\n[아시아인 얼굴 설정]")
    asian_config = get_asian_face_config()
    print(f"  - 사전학습: {asian_config.model.facenet.pretrained}")

    print("\n[실시간 플롯 설정]")
    plot_config = get_live_plot_config()
    print(f"  - 플롯 활성화: {plot_config.training.plotting.enabled}")
    print(f"  - 업데이트 간격: {plot_config.training.plotting.update_interval}")

    print("\n테스트 완료!")
