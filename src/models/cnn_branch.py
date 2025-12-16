"""
CNN Branch Feature Extractor Module
CNN 기반 이미지 특징 추출기

ResNet, EfficientNet 등 다양한 백본을 지원합니다.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportAssignmentType=false

from typing import Any, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)


class CNNFeatureExtractor(nn.Module):
    """
    CNN 백본을 사용한 특징 추출기

    다양한 사전학습 모델을 백본으로 사용하여
    이미지에서 고정 크기 특징 벡터를 추출합니다.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        feature_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_layers: int = -1,
        dropout: float = 0.1,
    ):
        """
        Args:
            backbone: 백본 모델 이름 (resnet18, resnet34, resnet50, efficientnet_b0, efficientnet_b1)
            feature_dim: 출력 특징 벡터 차원
            pretrained: 사전학습 가중치 사용 여부
            freeze_backbone: 백본 가중치 고정 여부
            unfreeze_layers: 학습 가능한 레이어 수 (-1: 전체, 양수: 마지막 N개)
            dropout: 드롭아웃 비율
        """
        super().__init__()

        self.backbone_name = backbone
        self.feature_dim = feature_dim
        self.pretrained = pretrained

        # 백본 모델 로드
        self.backbone, self.backbone_out_features = self._create_backbone(
            backbone, pretrained
        )

        # 백본 가중치 고정/해제 설정
        self._configure_freezing(freeze_backbone, unfreeze_layers)

        # Feature projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_out_features, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )

        # Global Average Pooling (백본에 없는 경우 사용)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        print("CNNFeatureExtractor 초기화:")
        print(f"  - 백본: {backbone}")
        print(f"  - 백본 출력 차원: {self.backbone_out_features}")
        print(f"  - 출력 차원: {feature_dim}")
        print(f"  - 사전학습: {pretrained}")
        print(f"  - 학습 가능 파라미터: {self._count_trainable_params():,}")

    def _create_backbone(self, backbone_name: str, pretrained: bool) -> Tuple[Any, Any]:
        """
        백본 모델 생성

        Returns:
            (백본 모델, 출력 특징 차원)
        """
        backbone_name = backbone_name.lower()

        if backbone_name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet18(weights=weights)
            out_features = model.fc.in_features
            # FC 레이어 제거
            model.fc = nn.Identity()  # type: ignore

        elif backbone_name == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet34(weights=weights)
            out_features = model.fc.in_features
            model.fc = nn.Identity()  # type: ignore

        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
            out_features = model.fc.in_features
            model.fc = nn.Identity()  # type: ignore

        elif backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            out_features = model.classifier[1].in_features
            model.classifier = nn.Identity()  # type: ignore

        elif backbone_name == "efficientnet_b1":
            weights = EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b1(weights=weights)
            out_features = model.classifier[1].in_features
            model.classifier = nn.Identity()  # type: ignore

        else:
            raise ValueError(
                f"지원하지 않는 백본: {backbone_name}. "
                f"지원 목록: resnet18, resnet34, resnet50, efficientnet_b0, efficientnet_b1"
            )

        return model, out_features

    def _configure_freezing(self, freeze_backbone: bool, unfreeze_layers: int):
        """백본 레이어 고정/해제 설정"""
        if freeze_backbone:
            # 전체 백본 고정
            for param in self.backbone.parameters():  # type: ignore
                param.requires_grad = False

            # 마지막 N개 레이어만 해제
            if unfreeze_layers > 0:
                # ResNet의 경우
                if self.backbone_name.startswith("resnet"):
                    layers = [
                        self.backbone.layer4,  # type: ignore
                        self.backbone.layer3,  # type: ignore
                        self.backbone.layer2,  # type: ignore
                        self.backbone.layer1,  # type: ignore
                    ]
                    for layer in layers[:unfreeze_layers]:
                        for param in layer.parameters():
                            param.requires_grad = True

                # EfficientNet의 경우
                elif self.backbone_name.startswith("efficientnet"):
                    features = list(self.backbone.features.children())  # type: ignore
                    for layer in features[-unfreeze_layers:]:
                        for param in layer.parameters():
                            param.requires_grad = True

    def _count_trainable_params(self) -> int:
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        백본에서 특징 추출

        Args:
            x: 입력 이미지 텐서 [B, C, H, W]

        Returns:
            특징 벡터 [B, backbone_out_features]
        """
        features = self.backbone(x)

        # 출력이 4D인 경우 (일부 백본)
        if len(features.shape) == 4:
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        이미지에서 특징 벡터 추출

        Args:
            x: 입력 이미지 텐서 [B, C, H, W] - 정규화된 이미지

        Returns:
            특징 벡터 [B, feature_dim]
        """
        # 백본 특징 추출
        features = self.extract_features(x)

        # 프로젝션
        output = self.projection(features)

        return output

    def get_backbone_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        프로젝션 없이 백본 출력만 반환

        Args:
            x: 입력 이미지 텐서 [B, C, H, W]

        Returns:
            백본 특징 벡터 [B, backbone_out_features]
        """
        return self.extract_features(x)


class CNNFeatureExtractorMultiScale(nn.Module):
    """
    멀티스케일 CNN 특징 추출기

    여러 레이어에서 특징을 추출하여 결합합니다.
    더 풍부한 특징 표현을 제공합니다.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        feature_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            backbone: 백본 모델 이름
            feature_dim: 출력 특징 벡터 차원
            pretrained: 사전학습 가중치 사용 여부
            freeze_backbone: 백본 가중치 고정 여부
        """
        super().__init__()

        self.feature_dim = feature_dim

        # ResNet 백본만 지원
        if not backbone.startswith("resnet"):
            raise ValueError("멀티스케일 추출은 ResNet 백본만 지원합니다.")

        # 백본 로드
        if backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            layer_dims = [64, 128, 256, 512]
        elif backbone == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            layer_dims = [64, 128, 256, 512]
        elif backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            layer_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"지원하지 않는 백본: {backbone}")

        # FC 레이어 제거
        self.backbone.fc = nn.Identity()  # type: ignore

        # 백본 고정
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 각 레이어 출력을 위한 Pooling
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)) for _ in range(4)])

        # 각 레이어 출력 프로젝션
        proj_dim = feature_dim // 4
        self.projections = nn.ModuleList(
            [nn.Linear(dim, proj_dim) for dim in layer_dims]
        )

        # 최종 프로젝션
        self.final_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

        print("CNNFeatureExtractorMultiScale 초기화:")
        print(f"  - 백본: {backbone}")
        print(f"  - 레이어 차원: {layer_dims}")
        print(f"  - 출력 차원: {feature_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        멀티스케일 특징 추출

        Args:
            x: 입력 이미지 텐서 [B, C, H, W]

        Returns:
            특징 벡터 [B, feature_dim]
        """
        # 초기 레이어
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # 각 레이어에서 특징 추출
        features = []

        x = self.backbone.layer1(x)
        f1 = self.pools[0](x).view(x.size(0), -1)
        f1 = self.projections[0](f1)
        features.append(f1)

        x = self.backbone.layer2(x)
        f2 = self.pools[1](x).view(x.size(0), -1)
        f2 = self.projections[1](f2)
        features.append(f2)

        x = self.backbone.layer3(x)
        f3 = self.pools[2](x).view(x.size(0), -1)
        f3 = self.projections[2](f3)
        features.append(f3)

        x = self.backbone.layer4(x)
        f4 = self.pools[3](x).view(x.size(0), -1)
        f4 = self.projections[3](f4)
        features.append(f4)

        # 특징 결합
        combined = torch.cat(features, dim=1)

        # 최종 프로젝션
        output = self.final_projection(combined)

        return output


# 테스트 코드
if __name__ == "__main__":
    print("=" * 50)
    print("CNN Feature Extractor 테스트")
    print("=" * 50)

    # 더미 입력
    dummy_input = torch.randn(4, 3, 224, 224)

    # 기본 추출기 테스트
    print("\n[1] 기본 CNNFeatureExtractor 테스트 (ResNet18)")
    extractor = CNNFeatureExtractor(
        backbone="resnet18",
        feature_dim=512,
        pretrained=False,  # 테스트용으로 False
    )
    output = extractor(dummy_input)
    print(f"  - 입력: {dummy_input.shape}")
    print(f"  - 출력: {output.shape}")

    # ResNet34 테스트
    print("\n[2] CNNFeatureExtractor 테스트 (ResNet34)")
    extractor34 = CNNFeatureExtractor(
        backbone="resnet34",
        feature_dim=512,
        pretrained=False,
    )
    output34 = extractor34(dummy_input)
    print(f"  - 출력: {output34.shape}")

    # 멀티스케일 테스트
    print("\n[3] CNNFeatureExtractorMultiScale 테스트")
    ms_extractor = CNNFeatureExtractorMultiScale(
        backbone="resnet18",
        feature_dim=512,
        pretrained=False,
    )
    ms_output = ms_extractor(dummy_input)
    print(f"  - 출력: {ms_output.shape}")

    # Frozen backbone 테스트
    print("\n[4] Frozen backbone 테스트")
    frozen_extractor = CNNFeatureExtractor(
        backbone="resnet18",
        feature_dim=512,
        pretrained=False,
        freeze_backbone=True,
        unfreeze_layers=1,
    )
    frozen_output = frozen_extractor(dummy_input)
    print(f"  - 출력: {frozen_output.shape}")
    print(f"  - 학습 가능 파라미터: {frozen_extractor._count_trainable_params():,}")

    print("\n테스트 완료!")
