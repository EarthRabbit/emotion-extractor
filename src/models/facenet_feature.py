"""
FaceNet Feature Projector Module
FaceNet 특징 벡터 처리 모듈

사전 추출된 FaceNet 임베딩 벡터(512차원)를 처리합니다.

FaceNet 임베딩 특징:
    - 512차원 벡터
    - L2 정규화됨 (Norm = 1.0)
    - 얼굴 인식에 특화된 특징
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class FaceNetFeatureProjector(nn.Module):
    """
    FaceNet 임베딩 벡터 프로젝터

    사전 추출된 FaceNet 임베딩 벡터를 받아
    분류에 적합한 차원으로 변환합니다.

    FaceNet은 이미 L2 정규화된 512차원 벡터를 출력하므로,
    추가적인 정규화 없이 projection만 수행합니다.
    """

    def __init__(
        self,
        input_dim: int = 512,
        feature_dim: int = 512,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            input_dim: 입력 FaceNet 임베딩 차원 (기본 512)
            feature_dim: 출력 특징 차원
            dropout: 드롭아웃 비율
            use_layer_norm: LayerNorm 사용 여부
        """
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim

        # Projection layers
        if use_layer_norm:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim),
            )

        print("FaceNetFeatureProjector 초기화:")
        print(f"  - 입력 차원: {input_dim}")
        print(f"  - 출력 차원: {feature_dim}")
        print(f"  - LayerNorm: {'사용' if use_layer_norm else '미사용'}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FaceNet 임베딩 벡터를 projection

        Args:
            x: FaceNet 임베딩 벡터 [B, input_dim]

        Returns:
            변환된 특징 벡터 [B, feature_dim]
        """
        return self.projection(x)


def load_facenet_vectors(
    vectors_path: Path,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    사전 추출된 FaceNet 임베딩 벡터 로드

    Args:
        vectors_path: .pt 파일 경로
        device: 로드할 디바이스

    Returns:
        파일명을 키로 하는 벡터 딕셔너리
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not vectors_path.exists():
        raise FileNotFoundError(f"FaceNet 벡터 파일을 찾을 수 없습니다: {vectors_path}")

    data = torch.load(vectors_path, map_location=device, weights_only=False)

    vectors = data.get("vectors", {})
    model_name = data.get("model_name", "unknown")
    feature_dim = data.get("feature_dim", 512)
    num_images = data.get("num_images", len(vectors))
    is_normalized = data.get("normalized", True)

    print("FaceNet 벡터 로드 완료:")
    print(f"  - 파일: {vectors_path}")
    print(f"  - 모델: {model_name}")
    print(f"  - 벡터 수: {num_images}개")
    print(f"  - 차원: {feature_dim}")
    print(f"  - L2 정규화: {'예' if is_normalized else '아니오'}")

    return vectors


# 테스트 코드
if __name__ == "__main__":
    print("=" * 50)
    print("FaceNet Feature Projector 테스트")
    print("=" * 50)

    # 더미 입력 (FaceNet 임베딩 시뮬레이션)
    batch_size = 4
    dummy_facenet_embedding = torch.randn(batch_size, 512)
    # L2 정규화 (실제 FaceNet 출력처럼)
    dummy_facenet_embedding = torch.nn.functional.normalize(
        dummy_facenet_embedding, p=2, dim=1
    )

    print("\n[1] 기본 FaceNetFeatureProjector 테스트")
    projector = FaceNetFeatureProjector(
        input_dim=512,
        feature_dim=512,
        dropout=0.1,
    )
    output = projector(dummy_facenet_embedding)
    print(f"  - 입력: {dummy_facenet_embedding.shape}")
    print(f"  - 출력: {output.shape}")

    print("\n[2] 차원 축소 테스트")
    projector_256 = FaceNetFeatureProjector(
        input_dim=512,
        feature_dim=256,
        dropout=0.1,
    )
    output_256 = projector_256(dummy_facenet_embedding)
    print(f"  - 출력: {output_256.shape}")

    print("\n테스트 완료!")
