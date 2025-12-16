"""
YOLO Feature Module
사전 추출된 YOLO 특징 벡터를 처리하는 모듈

사용법:
    1. extract_yolo_vectors.py로 이미지에서 YOLO 벡터 추출 및 저장
    2. 학습 시 YOLOFeatureProjector로 벡터를 projection하여 사용
"""

import torch
import torch.nn as nn


class YOLOFeatureProjector(nn.Module):
    """
    사전 추출된 YOLO 특징 벡터를 위한 Projection 레이어

    YOLO 백본에서 추출된 raw 벡터를 받아서
    학습 가능한 projection layer를 통과시킵니다.
    """

    def __init__(
        self,
        input_dim: int = 256,  # YOLO 백본 출력 차원 (모델에 따라 다름)
        feature_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: 입력 특징 차원 (사전 추출된 YOLO 벡터 차원)
            feature_dim: 출력 특징 차원
            dropout: 드롭아웃 비율
        """
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 사전 추출된 YOLO 특징 벡터 [B, input_dim]

        Returns:
            투영된 특징 벡터 [B, feature_dim]
        """
        return self.projection(x)


def load_yolo_vectors(vectors_path: str) -> dict:
    """
    저장된 YOLO 벡터 파일 로드

    Args:
        vectors_path: .pt 파일 경로

    Returns:
        {
            'vectors': {filename: tensor},
            'feature_dim': int,
            'model_name': str,
            ...
        }
    """
    data = torch.load(vectors_path, weights_only=False)

    print("YOLO 벡터 로드됨:")
    print(f"  - 파일: {vectors_path}")
    print(f"  - 벡터 수: {data.get('num_images', len(data.get('vectors', {})))}")
    print(f"  - 특징 차원: {data.get('feature_dim', 'unknown')}")
    print(f"  - 모델: {data.get('model_name', 'unknown')}")

    return data  # type: ignore[return-value]


# 테스트 코드
if __name__ == "__main__":
    print("=" * 50)
    print("YOLO Feature Projector 테스트")
    print("=" * 50)

    # 더미 입력 (배치 크기 4, 특징 차원 256)
    dummy_input = torch.randn(4, 256)

    # Projector 테스트
    projector = YOLOFeatureProjector(
        input_dim=256,
        feature_dim=512,
    )

    output = projector(dummy_input)
    print(f"입력: {dummy_input.shape}")
    print(f"출력: {output.shape}")

    # 파라미터 수
    num_params = sum(p.numel() for p in projector.parameters())
    print(f"파라미터 수: {num_params:,}")

    print("\n테스트 완료!")
