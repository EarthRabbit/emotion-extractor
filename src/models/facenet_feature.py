"""
FaceNet Feature Projector Module
FaceNet 특징 벡터 처리 모듈

사전 추출된 FaceNet 임베딩 벡터(512차원)를 처리합니다.

FaceNet 임베딩 특징:
    - 512차원 벡터
    - L2 정규화됨 (Norm = 1.0)
    - 얼굴 인식에 특화된 특징

개선 사항 (v2):
    - 2 레이어 → 4 레이어로 깊이 증가
    - Residual 연결 추가로 학습 안정성 향상
    - 더 풍부한 특징 변환 가능
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block for FaceNet Projector

    입력과 출력을 더하는 skip connection을 포함합니다.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            dim: 입력/출력 차원
            hidden_dim: 히든 레이어 차원 (None이면 dim과 동일)
            dropout: 드롭아웃 비율
            use_layer_norm: LayerNorm 사용 여부
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim

        layers = [
            nn.Linear(dim, hidden_dim),
        ]

        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))

        layers.extend(
            [
                nn.GELU(),  # ReLU 대신 GELU 사용 (더 부드러운 활성화)
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            ]
        )

        if use_layer_norm:
            layers.append(nn.LayerNorm(dim))

        self.block = nn.Sequential(*layers)

        # Dropout for residual connection
        self.res_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 [B, dim]

        Returns:
            출력 텐서 [B, dim]
        """
        residual = x
        out = self.block(x)
        out = self.res_dropout(out)
        return out + residual  # Residual connection


class FaceNetFeatureProjector(nn.Module):
    """
    FaceNet 임베딩 벡터 프로젝터 (개선된 버전)

    사전 추출된 FaceNet 임베딩 벡터를 받아
    분류에 적합한 차원으로 변환합니다.

    개선 사항:
    - 4개의 레이어 블록 (기존 2개)
    - Residual 연결로 gradient flow 개선
    - GELU 활성화 함수
    - 더 깊은 네트워크로 표현력 증가
    """

    def __init__(
        self,
        input_dim: int = 512,
        feature_dim: int = 512,
        hidden_dim: Optional[int] = None,
        num_blocks: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            input_dim: 입력 FaceNet 임베딩 차원 (기본 512)
            feature_dim: 출력 특징 차원
            hidden_dim: 히든 레이어 차원 (None이면 feature_dim과 동일)
            num_blocks: Residual 블록 수 (기본 4)
            dropout: 드롭아웃 비율
            use_layer_norm: LayerNorm 사용 여부
        """
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_blocks = num_blocks

        if hidden_dim is None:
            hidden_dim = feature_dim

        # 입력 프로젝션 (차원 변환)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual 블록들
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    dim=feature_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_blocks)
            ]
        )

        # 출력 프로젝션
        self.output_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim) if use_layer_norm else nn.Identity(),
        )

        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("FaceNetFeatureProjector 초기화 (개선 버전):")
        print(f"  - 입력 차원: {input_dim}")
        print(f"  - 출력 차원: {feature_dim}")
        print(f"  - 히든 차원: {hidden_dim}")
        print(f"  - Residual 블록 수: {num_blocks}")
        print(f"  - LayerNorm: {'사용' if use_layer_norm else '미사용'}")
        print(f"  - 총 파라미터: {total_params:,}")
        print(f"  - 학습 가능 파라미터: {trainable_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FaceNet 임베딩 벡터를 projection

        Args:
            x: FaceNet 임베딩 벡터 [B, input_dim]

        Returns:
            변환된 특징 벡터 [B, feature_dim]
        """
        # 입력 프로젝션
        x = self.input_projection(x)

        # Residual 블록 통과
        for block in self.residual_blocks:
            x = block(x)

        # 출력 프로젝션
        x = self.output_projection(x)

        return x


class FaceNetFeatureProjectorLite(nn.Module):
    """
    FaceNet 임베딩 벡터 프로젝터 (경량 버전)

    기존 2-layer 구조를 유지하되, 개선된 활성화 함수 사용
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

        # Projection layers (기존 구조 유지, 활성화 함수만 변경)
        if use_layer_norm:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim),
            )

        print("FaceNetFeatureProjectorLite 초기화:")
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
    print("=" * 60)
    print("FaceNet Feature Projector 테스트 (개선 버전)")
    print("=" * 60)

    # 더미 입력 (FaceNet 임베딩 시뮬레이션)
    batch_size = 4
    dummy_facenet_embedding = torch.randn(batch_size, 512)
    # L2 정규화 (실제 FaceNet 출력처럼)
    dummy_facenet_embedding = torch.nn.functional.normalize(
        dummy_facenet_embedding, p=2, dim=1
    )

    print("\n[1] 개선된 FaceNetFeatureProjector 테스트 (4 블록)")
    projector = FaceNetFeatureProjector(
        input_dim=512,
        feature_dim=512,
        num_blocks=4,
        dropout=0.1,
    )
    output = projector(dummy_facenet_embedding)
    print(f"  - 입력: {dummy_facenet_embedding.shape}")
    print(f"  - 출력: {output.shape}")

    print("\n[2] 6 블록 테스트")
    projector_deep = FaceNetFeatureProjector(
        input_dim=512,
        feature_dim=512,
        num_blocks=6,
        dropout=0.1,
    )
    output_deep = projector_deep(dummy_facenet_embedding)
    print(f"  - 출력: {output_deep.shape}")

    print("\n[3] 차원 축소 테스트")
    projector_256 = FaceNetFeatureProjector(
        input_dim=512,
        feature_dim=256,
        num_blocks=4,
        dropout=0.1,
    )
    output_256 = projector_256(dummy_facenet_embedding)
    print(f"  - 출력: {output_256.shape}")

    print("\n[4] Lite 버전 테스트")
    projector_lite = FaceNetFeatureProjectorLite(
        input_dim=512,
        feature_dim=512,
        dropout=0.1,
    )
    output_lite = projector_lite(dummy_facenet_embedding)
    print(f"  - 출력: {output_lite.shape}")

    print("\n[5] Gradient 테스트")
    projector.train()
    output = projector(dummy_facenet_embedding)
    loss = output.sum()
    loss.backward()
    print("  - Gradient 계산 성공")

    print("\n테스트 완료!")
