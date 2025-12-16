"""
Hybrid Emotion Classification Model
하이브리드 감정 분류 모델

CNN 기반 이미지 특징 + 사전 추출된 YOLO 특징을 결합하여
감정을 분류하는 하이브리드 모델입니다.

사용법:
    1. extract_yolo_vectors.py로 YOLO 벡터 사전 추출
    2. 학습 시 사전 추출된 벡터와 이미지를 함께 사용
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_branch import CNNFeatureExtractor, CNNFeatureExtractorMultiScale
from .yolo_feature import YOLOFeatureProjector


class FeatureFusion(nn.Module):
    """
    특징 융합 모듈

    다양한 융합 방식을 지원합니다:
    - concat: 단순 연결
    - attention: 크로스 어텐션
    - weighted_sum: 가중 합
    - gated: 게이트 메커니즘
    """

    def __init__(
        self,
        yolo_dim: int = 512,
        cnn_dim: int = 512,
        output_dim: int = 256,
        fusion_type: str = "concat",
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            yolo_dim: YOLO 특징 차원
            cnn_dim: CNN 특징 차원
            output_dim: 출력 차원
            fusion_type: 융합 방식 ("concat", "attention", "weighted_sum", "gated")
            attention_heads: 어텐션 헤드 수 (attention 방식일 때)
            dropout: 드롭아웃 비율
        """
        super().__init__()

        self.fusion_type = fusion_type
        self.output_dim = output_dim

        if fusion_type == "concat":
            # 단순 연결 후 프로젝션
            self.projection = nn.Sequential(
                nn.Linear(yolo_dim + cnn_dim, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim),
            )

        elif fusion_type == "attention":
            # 크로스 어텐션
            self.yolo_proj = nn.Linear(yolo_dim, output_dim)
            self.cnn_proj = nn.Linear(cnn_dim, output_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(output_dim)
            self.ffn = nn.Sequential(
                nn.Linear(output_dim, output_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim),
            )

        elif fusion_type == "weighted_sum":
            # 학습 가능한 가중치로 합
            self.yolo_proj = nn.Linear(yolo_dim, output_dim)
            self.cnn_proj = nn.Linear(cnn_dim, output_dim)
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.norm = nn.LayerNorm(output_dim)

        elif fusion_type == "gated":
            # 게이트 메커니즘
            self.yolo_proj = nn.Linear(yolo_dim, output_dim)
            self.cnn_proj = nn.Linear(cnn_dim, output_dim)
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid(),
            )
            self.norm = nn.LayerNorm(output_dim)

        else:
            raise ValueError(f"지원하지 않는 fusion_type: {fusion_type}")

        print(f"FeatureFusion 초기화: {fusion_type}")
        print(f"  - YOLO 차원: {yolo_dim}")
        print(f"  - CNN 차원: {cnn_dim}")
        print(f"  - 출력 차원: {output_dim}")

    def forward(
        self, yolo_features: torch.Tensor, cnn_features: torch.Tensor
    ) -> torch.Tensor:
        """
        특징 융합

        Args:
            yolo_features: YOLO 특징 [B, yolo_dim]
            cnn_features: CNN 특징 [B, cnn_dim]

        Returns:
            융합된 특징 [B, output_dim]
        """
        if self.fusion_type == "concat":
            combined = torch.cat([yolo_features, cnn_features], dim=1)
            output = self.projection(combined)

        elif self.fusion_type == "attention":
            yolo_proj = self.yolo_proj(yolo_features).unsqueeze(1)
            cnn_proj = self.cnn_proj(cnn_features).unsqueeze(1)

            # YOLO를 query, CNN을 key/value로 사용
            attn_out, _ = self.attention(yolo_proj, cnn_proj, cnn_proj)
            attn_out = attn_out.squeeze(1)
            attn_out = self.norm(attn_out + yolo_proj.squeeze(1))
            output = self.ffn(attn_out) + attn_out

        elif self.fusion_type == "weighted_sum":
            yolo_proj = self.yolo_proj(yolo_features)
            cnn_proj = self.cnn_proj(cnn_features)
            weights = F.softmax(self.weights, dim=0)
            output = weights[0] * yolo_proj + weights[1] * cnn_proj
            output = self.norm(output)

        elif self.fusion_type == "gated":
            yolo_proj = self.yolo_proj(yolo_features)
            cnn_proj = self.cnn_proj(cnn_features)
            combined = torch.cat([yolo_proj, cnn_proj], dim=1)
            gate = self.gate(combined)
            output = gate * yolo_proj + (1 - gate) * cnn_proj
            output = self.norm(output)

        else:
            raise ValueError(f"지원하지 않는 fusion_type: {self.fusion_type}")

        return output


class ClassificationHead(nn.Module):
    """
    분류 헤드

    특징 벡터를 받아 클래스 확률을 출력합니다.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: 입력 특징 차원
            hidden_dims: 히든 레이어 차원 리스트
            num_classes: 출력 클래스 수
            dropout: 드롭아웃 비율
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # 최종 분류 레이어
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        print("ClassificationHead 초기화:")
        print(f"  - 입력 차원: {input_dim}")
        print(f"  - 히든 차원: {hidden_dims}")
        print(f"  - 클래스 수: {num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 특징 [B, input_dim]

        Returns:
            로짓 [B, num_classes]
        """
        return self.classifier(x)


class HybridEmotionModel(nn.Module):
    """
    하이브리드 감정 분류 모델

    사전 추출된 YOLO 특징과 CNN 기반 이미지 특징을 결합하여
    감정을 분류합니다.

    구조:
        Input Image ──────► CNN Branch ──► CNN Features (512D)
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │   Feature    │
                                        │   Fusion     │
                                        └──────────────┘
                                                ▲
        Precomputed YOLO Vector ──► YOLO Projector ──► YOLO Features (512D)
                                                │
                                                ▼
                                      Classification Head
                                                │
                                                ▼
                                        Emotion Class (6)
    """

    def __init__(
        self,
        # YOLO 설정 (사전 추출된 벡터용)
        use_yolo_branch: bool = True,
        yolo_input_dim: int = 256,  # 사전 추출된 벡터 차원
        yolo_feature_dim: int = 512,
        # CNN 설정
        use_cnn_branch: bool = True,
        cnn_backbone: str = "resnet18",
        cnn_feature_dim: int = 512,
        cnn_pretrained: bool = True,
        freeze_cnn: bool = False,
        use_multiscale_cnn: bool = False,
        # Fusion 설정
        fusion_type: str = "concat",
        fusion_output_dim: int = 256,
        attention_heads: int = 4,
        # Classification 설정
        num_classes: int = 6,
        classifier_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        # 기타
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            use_yolo_branch: YOLO 브랜치 사용 여부
            yolo_input_dim: 사전 추출된 YOLO 벡터 차원
            yolo_feature_dim: YOLO projection 출력 차원

            use_cnn_branch: CNN 브랜치 사용 여부
            cnn_backbone: CNN 백본 이름
            cnn_feature_dim: CNN 특징 차원
            cnn_pretrained: CNN 사전학습 가중치 사용
            freeze_cnn: CNN 백본 고정 여부
            use_multiscale_cnn: 멀티스케일 CNN 사용 여부

            fusion_type: 특징 융합 방식
            fusion_output_dim: 융합 출력 차원
            attention_heads: 어텐션 헤드 수

            num_classes: 클래스 수
            classifier_hidden_dims: 분류기 히든 레이어 차원
            dropout: 드롭아웃 비율
            device: 연산 디바이스
        """
        super().__init__()

        self.use_yolo_branch = use_yolo_branch
        self.use_cnn_branch = use_cnn_branch
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if classifier_hidden_dims is None:
            classifier_hidden_dims = [256, 128]

        # 최소 하나의 브랜치 필요
        if not use_yolo_branch and not use_cnn_branch:
            raise ValueError("최소 하나의 브랜치 (YOLO 또는 CNN)가 필요합니다.")

        print("=" * 60)
        print("HybridEmotionModel 초기화")
        print("=" * 60)

        # YOLO Branch (사전 추출된 벡터를 projection)
        self.yolo_branch: Optional[YOLOFeatureProjector] = None
        if use_yolo_branch:
            self.yolo_branch = YOLOFeatureProjector(
                input_dim=yolo_input_dim,
                feature_dim=yolo_feature_dim,
                dropout=dropout,
            )
            print("\nYOLO Branch: YOLOFeatureProjector")
            print(f"  - 입력 차원: {yolo_input_dim}")
            print(f"  - 출력 차원: {yolo_feature_dim}")
        else:
            yolo_feature_dim = 0

        # CNN Branch
        self.cnn_branch: Optional[nn.Module] = None
        if use_cnn_branch:
            if use_multiscale_cnn:
                self.cnn_branch = CNNFeatureExtractorMultiScale(
                    backbone=cnn_backbone,
                    feature_dim=cnn_feature_dim,
                    pretrained=cnn_pretrained,
                    freeze_backbone=freeze_cnn,
                )
            else:
                self.cnn_branch = CNNFeatureExtractor(
                    backbone=cnn_backbone,
                    feature_dim=cnn_feature_dim,
                    pretrained=cnn_pretrained,
                    freeze_backbone=freeze_cnn,
                    dropout=dropout,
                )
        else:
            cnn_feature_dim = 0

        # Feature Fusion
        self.fusion: Optional[FeatureFusion] = None
        if use_yolo_branch and use_cnn_branch:
            self.fusion = FeatureFusion(
                yolo_dim=yolo_feature_dim,
                cnn_dim=cnn_feature_dim,
                output_dim=fusion_output_dim,
                fusion_type=fusion_type,
                attention_heads=attention_heads,
                dropout=dropout,
            )
            classifier_input_dim = fusion_output_dim
        else:
            # 단일 브랜치만 사용
            classifier_input_dim = (
                yolo_feature_dim if use_yolo_branch else cnn_feature_dim
            )

        # Classification Head
        self.classifier = ClassificationHead(
            input_dim=classifier_input_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        )

        # 모델 정보 저장
        self.model_info = {
            "use_yolo_branch": use_yolo_branch,
            "use_cnn_branch": use_cnn_branch,
            "yolo_input_dim": yolo_input_dim if use_yolo_branch else 0,
            "yolo_feature_dim": yolo_feature_dim,
            "cnn_feature_dim": cnn_feature_dim,
            "fusion_type": fusion_type if self.fusion else "none",
            "num_classes": num_classes,
        }

        print("\n모델 구성 완료:")
        print(f"  - YOLO 브랜치: {'활성' if use_yolo_branch else '비활성'}")
        print(f"  - CNN 브랜치: {'활성' if use_cnn_branch else '비활성'}")
        print(f"  - 융합 방식: {fusion_type if self.fusion else 'N/A'}")
        print(f"  - 클래스 수: {num_classes}")
        print(f"  - 총 파라미터: {self._count_params():,}")
        print(f"  - 학습 가능 파라미터: {self._count_trainable_params():,}")
        print("=" * 60)

    def _count_params(self) -> int:
        """전체 파라미터 수"""
        return sum(p.numel() for p in self.parameters())

    def _count_trainable_params(self) -> int:
        """학습 가능 파라미터 수"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        image: torch.Tensor,
        yolo_vector: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        순전파

        Args:
            image: 입력 이미지 [B, C, H, W]
            yolo_vector: 사전 추출된 YOLO 특징 [B, yolo_input_dim]

        Returns:
            dict with keys:
                - 'logits': 분류 로짓 [B, num_classes]
                - 'yolo_features': YOLO 특징 [B, D] (있는 경우)
                - 'cnn_features': CNN 특징 [B, D] (있는 경우)
                - 'fused_features': 융합된 특징 [B, D] (있는 경우)
        """
        outputs: Dict[str, torch.Tensor] = {}

        # YOLO Branch (사전 추출된 벡터 projection)
        yolo_features: Optional[torch.Tensor] = None
        if self.use_yolo_branch and self.yolo_branch is not None:
            if yolo_vector is not None:
                yolo_feat = self.yolo_branch(yolo_vector)
                yolo_features = yolo_feat
                outputs["yolo_features"] = yolo_feat
            else:
                raise ValueError(
                    "YOLO 브랜치가 활성화되어 있지만 yolo_vector가 제공되지 않았습니다."
                )

        # CNN Branch
        cnn_features: Optional[torch.Tensor] = None
        if self.use_cnn_branch and self.cnn_branch is not None:
            cnn_feat = self.cnn_branch(image)
            cnn_features = cnn_feat
            outputs["cnn_features"] = cnn_feat

        # Feature Fusion
        classifier_input: torch.Tensor
        if (
            self.fusion is not None
            and yolo_features is not None
            and cnn_features is not None
        ):
            fused = self.fusion(yolo_features, cnn_features)
            outputs["fused_features"] = fused
            classifier_input = fused
        elif self.use_yolo_branch and yolo_features is not None:
            classifier_input = yolo_features
        elif self.use_cnn_branch and cnn_features is not None:
            classifier_input = cnn_features
        else:
            raise ValueError("특징 추출 실패: 유효한 특징이 없습니다.")

        # Classification
        logits = self.classifier(classifier_input)
        outputs["logits"] = logits

        return outputs

    def predict(
        self, image: torch.Tensor, yolo_vector: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        예측 수행

        Args:
            image: 입력 이미지 [B, C, H, W]
            yolo_vector: 사전 추출된 YOLO 특징

        Returns:
            (예측 클래스 [B], 확률 [B, num_classes])
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(image, yolo_vector)
            logits = outputs["logits"]
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)

        return predictions, probs

    def get_features(
        self, image: torch.Tensor, yolo_vector: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        특징만 추출 (분류 제외)

        Args:
            image: 입력 이미지 [B, C, H, W]
            yolo_vector: 사전 추출된 YOLO 특징

        Returns:
            특징 딕셔너리
        """
        outputs: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            if self.use_yolo_branch and self.yolo_branch is not None:
                if yolo_vector is not None:
                    outputs["yolo_features"] = self.yolo_branch(yolo_vector)

            if self.use_cnn_branch and self.cnn_branch is not None:
                outputs["cnn_features"] = self.cnn_branch(image)

            if (
                self.fusion is not None
                and "yolo_features" in outputs
                and "cnn_features" in outputs
            ):
                outputs["fused_features"] = self.fusion(
                    outputs["yolo_features"], outputs["cnn_features"]
                )

        return outputs


class HybridEmotionModelLite(nn.Module):
    """
    경량 하이브리드 모델

    CNN만 사용하는 간소화된 버전입니다.
    YOLO 벡터 없이 이미지만으로 학습/추론합니다.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        feature_dim: int = 512,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """
        Args:
            backbone: CNN 백본
            feature_dim: 특징 차원
            num_classes: 클래스 수
            pretrained: 사전학습 사용
            dropout: 드롭아웃 비율
        """
        super().__init__()

        print("=" * 60)
        print("HybridEmotionModelLite 초기화")
        print("=" * 60)

        # CNN 백본
        self.feature_extractor = CNNFeatureExtractor(
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained,
            dropout=dropout,
        )

        # 분류기
        self.classifier = ClassificationHead(
            input_dim=feature_dim,
            hidden_dims=[256, 128],
            num_classes=num_classes,
            dropout=dropout,
        )

        print(f"\n총 파라미터: {sum(p.numel() for p in self.parameters()):,}")
        print("=" * 60)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """순전파"""
        features = self.feature_extractor(image)
        logits = self.classifier(features)

        return {
            "logits": logits,
            "features": features,
        }

    def predict(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """예측"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(image)
            probs = F.softmax(outputs["logits"], dim=1)
            predictions = torch.argmax(probs, dim=1)

        return predictions, probs


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("Hybrid Model 테스트")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n디바이스: {device}")

    # 더미 입력
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_yolo_vector = torch.randn(batch_size, 256).to(device)

    # 테스트 1: 하이브리드 모델 (YOLO + CNN)
    print("\n[1] HybridEmotionModel (YOLO + CNN) 테스트")
    model = HybridEmotionModel(
        use_yolo_branch=True,
        yolo_input_dim=256,
        yolo_feature_dim=512,
        use_cnn_branch=True,
        cnn_backbone="resnet18",
        cnn_feature_dim=512,
        cnn_pretrained=False,
        fusion_type="concat",
        num_classes=6,
        device=device,
    ).to(device)

    outputs = model(dummy_image, dummy_yolo_vector)
    print(f"  - logits: {outputs['logits'].shape}")
    print(f"  - yolo_features: {outputs.get('yolo_features', 'N/A')}")
    print(f"  - cnn_features: {outputs.get('cnn_features', 'N/A')}")

    # 테스트 2: CNN만 사용
    print("\n[2] HybridEmotionModel (CNN only) 테스트")
    model_cnn = HybridEmotionModel(
        use_yolo_branch=False,
        use_cnn_branch=True,
        cnn_backbone="resnet18",
        cnn_pretrained=False,
        num_classes=6,
        device=device,
    ).to(device)

    outputs_cnn = model_cnn(dummy_image)
    print(f"  - logits: {outputs_cnn['logits'].shape}")

    # 테스트 3: Lite 모델
    print("\n[3] HybridEmotionModelLite 테스트")
    model_lite = HybridEmotionModelLite(
        backbone="resnet18",
        pretrained=False,
        num_classes=6,
    ).to(device)

    outputs_lite = model_lite(dummy_image)
    print(f"  - logits: {outputs_lite['logits'].shape}")

    # 테스트 4: 예측
    print("\n[4] 예측 테스트")
    predictions, probs = model.predict(dummy_image, dummy_yolo_vector)
    print(f"  - predictions: {predictions}")
    print(f"  - probs shape: {probs.shape}")

    print("\n테스트 완료!")
