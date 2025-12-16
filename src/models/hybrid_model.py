"""
Hybrid Emotion Classification Model
하이브리드 감정 분류 모델

CNN 기반 이미지 특징 + 사전 추출된 FaceNet 특징을 결합하여
감정을 분류하는 하이브리드 모델입니다.

사용법:
    1. extract_facenet_vectors.py로 FaceNet 벡터 사전 추출
    2. 학습 시 사전 추출된 벡터와 이미지를 함께 사용

개선 사항 (v2):
    - Bilinear Fusion 추가
    - Multi-Head Cross Attention 강화
    - Squeeze-and-Excitation 기반 채널 가중치
    - 더 깊은 Classification Head
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_branch import CNNFeatureExtractor, CNNFeatureExtractorMultiScale
from .facenet_feature import FaceNetFeatureProjector


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation 블록

    채널별 중요도를 학습하여 가중치를 적용합니다.
    """

    def __init__(self, dim: int, reduction: int = 4):
        """
        Args:
            dim: 입력 차원
            reduction: 축소 비율
        """
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서 [B, dim]

        Returns:
            가중치가 적용된 텐서 [B, dim]
        """
        # [B, dim] -> [B, dim, 1] for pooling
        b, d = x.shape
        scale = self.excitation(x)  # [B, dim]
        return x * scale


class FeatureFusion(nn.Module):
    """
    특징 융합 모듈 (개선된 버전)

    다양한 융합 방식을 지원합니다:
    - concat: 단순 연결
    - attention: 크로스 어텐션 (개선됨)
    - weighted_sum: 가중 합
    - gated: 게이트 메커니즘
    - bilinear: Bilinear Fusion (새로 추가)
    """

    def __init__(
        self,
        facenet_dim: int = 512,
        cnn_dim: int = 512,
        output_dim: int = 256,
        fusion_type: str = "concat",
        attention_heads: int = 4,
        dropout: float = 0.1,
        use_se: bool = True,
    ):
        """
        Args:
            facenet_dim: FaceNet 특징 차원
            cnn_dim: CNN 특징 차원
            output_dim: 출력 차원
            fusion_type: 융합 방식 ("concat", "attention", "weighted_sum", "gated", "bilinear")
            attention_heads: 어텐션 헤드 수 (attention 방식일 때)
            dropout: 드롭아웃 비율
            use_se: Squeeze-and-Excitation 사용 여부
        """
        super().__init__()

        self.fusion_type = fusion_type
        self.output_dim = output_dim
        self.use_se = use_se

        if fusion_type == "concat":
            # 단순 연결 후 프로젝션 (개선: 3-layer로 확장)
            self.projection = nn.Sequential(
                nn.Linear(facenet_dim + cnn_dim, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
            )

            if use_se:
                self.se = SqueezeExcitation(output_dim)

        elif fusion_type == "attention":
            # 개선된 크로스 어텐션
            self.facenet_proj = nn.Sequential(
                nn.Linear(facenet_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.cnn_proj = nn.Sequential(
                nn.Linear(cnn_dim, output_dim),
                nn.LayerNorm(output_dim),
            )

            # Multi-head Cross Attention
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )

            # Self Attention (추가)
            self.self_attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )

            self.norm1 = nn.LayerNorm(output_dim)
            self.norm2 = nn.LayerNorm(output_dim)
            self.norm3 = nn.LayerNorm(output_dim)

            # FFN (개선: 더 깊게)
            self.ffn = nn.Sequential(
                nn.Linear(output_dim, output_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 4, output_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim),
            )

            if use_se:
                self.se = SqueezeExcitation(output_dim)

        elif fusion_type == "weighted_sum":
            # 학습 가능한 가중치로 합
            self.facenet_proj = nn.Sequential(
                nn.Linear(facenet_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.cnn_proj = nn.Sequential(
                nn.Linear(cnn_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.norm = nn.LayerNorm(output_dim)

            if use_se:
                self.se = SqueezeExcitation(output_dim)

        elif fusion_type == "gated":
            # 게이트 메커니즘 (개선)
            self.facenet_proj = nn.Sequential(
                nn.Linear(facenet_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.cnn_proj = nn.Sequential(
                nn.Linear(cnn_dim, output_dim),
                nn.LayerNorm(output_dim),
            )

            # 개선된 게이트 네트워크
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
                nn.Sigmoid(),
            )
            self.norm = nn.LayerNorm(output_dim)

            if use_se:
                self.se = SqueezeExcitation(output_dim)

        elif fusion_type == "bilinear":
            # 새로 추가: Bilinear Fusion
            # 두 특징 간의 곱셈적 상호작용을 캡처
            self.facenet_proj = nn.Sequential(
                nn.Linear(facenet_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.cnn_proj = nn.Sequential(
                nn.Linear(cnn_dim, output_dim),
                nn.LayerNorm(output_dim),
            )

            # Bilinear 레이어
            self.bilinear = nn.Bilinear(output_dim, output_dim, output_dim)

            # 추가 프로젝션
            self.output_proj = nn.Sequential(
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
            )

            if use_se:
                self.se = SqueezeExcitation(output_dim)

        else:
            raise ValueError(f"지원하지 않는 fusion_type: {fusion_type}")

        print(f"FeatureFusion 초기화: {fusion_type}")
        print(f"  - FaceNet 차원: {facenet_dim}")
        print(f"  - CNN 차원: {cnn_dim}")
        print(f"  - 출력 차원: {output_dim}")
        print(f"  - SE 블록: {'사용' if use_se else '미사용'}")

    def forward(
        self, facenet_features: torch.Tensor, cnn_features: torch.Tensor
    ) -> torch.Tensor:
        """
        특징 융합

        Args:
            facenet_features: FaceNet 특징 [B, facenet_dim]
            cnn_features: CNN 특징 [B, cnn_dim]

        Returns:
            융합된 특징 [B, output_dim]
        """
        if self.fusion_type == "concat":
            combined = torch.cat([facenet_features, cnn_features], dim=1)
            output = self.projection(combined)
            if self.use_se:
                output = self.se(output)

        elif self.fusion_type == "attention":
            facenet_proj = self.facenet_proj(facenet_features).unsqueeze(1)  # [B, 1, D]
            cnn_proj = self.cnn_proj(cnn_features).unsqueeze(1)  # [B, 1, D]

            # Concat for sequence
            seq = torch.cat([facenet_proj, cnn_proj], dim=1)  # [B, 2, D]

            # Self-attention
            self_attn_out, _ = self.self_attention(seq, seq, seq)
            seq = self.norm1(seq + self_attn_out)

            # Cross attention: FaceNet attends to CNN
            cross_attn_out, _ = self.cross_attention(facenet_proj, cnn_proj, cnn_proj)
            attn_out = self.norm2(facenet_proj + cross_attn_out)

            # FFN
            ffn_out = self.ffn(attn_out.squeeze(1))
            output = self.norm3(attn_out.squeeze(1) + ffn_out)

            if self.use_se:
                output = self.se(output)

        elif self.fusion_type == "weighted_sum":
            facenet_proj = self.facenet_proj(facenet_features)
            cnn_proj = self.cnn_proj(cnn_features)
            weights = F.softmax(self.weights, dim=0)
            output = weights[0] * facenet_proj + weights[1] * cnn_proj
            output = self.norm(output)
            if self.use_se:
                output = self.se(output)

        elif self.fusion_type == "gated":
            facenet_proj = self.facenet_proj(facenet_features)
            cnn_proj = self.cnn_proj(cnn_features)
            combined = torch.cat([facenet_proj, cnn_proj], dim=1)
            gate = self.gate(combined)
            output = gate * facenet_proj + (1 - gate) * cnn_proj
            output = self.norm(output)
            if self.use_se:
                output = self.se(output)

        elif self.fusion_type == "bilinear":
            facenet_proj = self.facenet_proj(facenet_features)
            cnn_proj = self.cnn_proj(cnn_features)

            # Bilinear interaction
            bilinear_out = self.bilinear(facenet_proj, cnn_proj)

            # Output projection with residual
            output = self.output_proj(bilinear_out)

            if self.use_se:
                output = self.se(output)

        else:
            raise ValueError(f"지원하지 않는 fusion_type: {self.fusion_type}")

        return output


class ClassificationHead(nn.Module):
    """
    분류 헤드 (개선된 버전)

    특징 벡터를 받아 클래스 확률을 출력합니다.
    더 깊은 네트워크와 Residual 연결을 지원합니다.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 7,
        dropout: float = 0.3,
        use_residual: bool = True,
    ):
        """
        Args:
            input_dim: 입력 특징 차원
            hidden_dims: 히든 레이어 차원 리스트
            num_classes: 출력 클래스 수
            dropout: 드롭아웃 비율
            use_residual: Residual 연결 사용 여부
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.use_residual = use_residual
        self.hidden_dims = hidden_dims

        # 레이어 구성
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Residual projection (차원이 다를 때)
        self.residual_projs = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

            # Residual projection
            if use_residual and prev_dim != hidden_dim:
                self.residual_projs.append(nn.Linear(prev_dim, hidden_dim))
            else:
                self.residual_projs.append(None)

            prev_dim = hidden_dim

        # 최종 분류 레이어
        self.classifier = nn.Linear(prev_dim, num_classes)

        print("ClassificationHead 초기화 (개선 버전):")
        print(f"  - 입력 차원: {input_dim}")
        print(f"  - 히든 차원: {hidden_dims}")
        print(f"  - 클래스 수: {num_classes}")
        print(f"  - Residual 연결: {'사용' if use_residual else '미사용'}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 특징 [B, input_dim]

        Returns:
            로짓 [B, num_classes]
        """
        for i, (layer, norm, drop) in enumerate(
            zip(self.layers, self.norms, self.dropouts)
        ):
            residual = x

            # Forward
            x = layer(x)
            x = norm(x)
            x = F.gelu(x)
            x = drop(x)

            # Residual connection
            if self.use_residual:
                if self.residual_projs[i] is not None:
                    residual = self.residual_projs[i](residual)
                x = x + residual * 0.1  # Scaled residual

        return self.classifier(x)


class HybridEmotionModel(nn.Module):
    """
    하이브리드 감정 분류 모델

    사전 추출된 FaceNet 특징과 CNN 기반 이미지 특징을 결합하여
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
        Precomputed FaceNet Vector ──► FaceNet Projector ──► FaceNet Features (512D)
                                                │
                                                ▼
                                      Classification Head
                                                │
                                                ▼
                                        Emotion Class (7)
    """

    def __init__(
        self,
        # FaceNet 설정 (사전 추출된 벡터용)
        use_facenet_branch: bool = True,
        facenet_input_dim: int = 512,  # 사전 추출된 벡터 차원
        facenet_feature_dim: int = 512,
        facenet_num_blocks: int = 4,  # FaceNet Projector 블록 수
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
        use_se: bool = True,
        # Classification 설정
        num_classes: int = 7,
        classifier_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        use_classifier_residual: bool = True,
        # 기타
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            use_facenet_branch: FaceNet 브랜치 사용 여부
            facenet_input_dim: 사전 추출된 FaceNet 벡터 차원
            facenet_feature_dim: FaceNet projection 출력 차원
            facenet_num_blocks: FaceNet Projector의 Residual 블록 수

            use_cnn_branch: CNN 브랜치 사용 여부
            cnn_backbone: CNN 백본 이름
            cnn_feature_dim: CNN 특징 차원
            cnn_pretrained: CNN 사전학습 가중치 사용
            freeze_cnn: CNN 백본 고정 여부
            use_multiscale_cnn: 멀티스케일 CNN 사용 여부

            fusion_type: 특징 융합 방식
            fusion_output_dim: 융합 출력 차원
            attention_heads: 어텐션 헤드 수
            use_se: Squeeze-and-Excitation 사용 여부

            num_classes: 클래스 수
            classifier_hidden_dims: 분류기 히든 레이어 차원
            dropout: 드롭아웃 비율
            use_classifier_residual: 분류기에서 Residual 연결 사용
            device: 연산 디바이스
        """
        super().__init__()

        self.use_facenet_branch = use_facenet_branch
        self.use_cnn_branch = use_cnn_branch
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if classifier_hidden_dims is None:
            classifier_hidden_dims = [256, 128]

        # 최소 하나의 브랜치 필요
        if not use_facenet_branch and not use_cnn_branch:
            raise ValueError("최소 하나의 브랜치 (FaceNet 또는 CNN)가 필요합니다.")

        print("=" * 60)
        print("HybridEmotionModel 초기화 (개선 버전)")
        print("=" * 60)

        # FaceNet Branch (사전 추출된 벡터를 projection)
        self.facenet_branch: Optional[FaceNetFeatureProjector] = None
        if use_facenet_branch:
            self.facenet_branch = FaceNetFeatureProjector(
                input_dim=facenet_input_dim,
                feature_dim=facenet_feature_dim,
                num_blocks=facenet_num_blocks,
                dropout=dropout,
            )
            print(
                f"\nFaceNet Branch: FaceNetFeatureProjector ({facenet_num_blocks} blocks)"
            )
            print(f"  - 입력 차원: {facenet_input_dim}")
            print(f"  - 출력 차원: {facenet_feature_dim}")
        else:
            facenet_feature_dim = 0

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
        if use_facenet_branch and use_cnn_branch:
            self.fusion = FeatureFusion(
                facenet_dim=facenet_feature_dim,
                cnn_dim=cnn_feature_dim,
                output_dim=fusion_output_dim,
                fusion_type=fusion_type,
                attention_heads=attention_heads,
                dropout=dropout,
                use_se=use_se,
            )
            classifier_input_dim = fusion_output_dim
        else:
            # 단일 브랜치만 사용
            classifier_input_dim = (
                facenet_feature_dim if use_facenet_branch else cnn_feature_dim
            )

        # Classification Head
        self.classifier = ClassificationHead(
            input_dim=classifier_input_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            use_residual=use_classifier_residual,
        )

        # 모델 정보 저장
        self.model_info = {
            "use_facenet_branch": use_facenet_branch,
            "use_cnn_branch": use_cnn_branch,
            "facenet_input_dim": facenet_input_dim if use_facenet_branch else 0,
            "facenet_feature_dim": facenet_feature_dim,
            "facenet_num_blocks": facenet_num_blocks if use_facenet_branch else 0,
            "cnn_feature_dim": cnn_feature_dim,
            "fusion_type": fusion_type if self.fusion else "none",
            "num_classes": num_classes,
        }

        print("\n모델 구성 완료:")
        print(f"  - FaceNet 브랜치: {'활성' if use_facenet_branch else '비활성'}")
        print(f"  - CNN 브랜치: {'활성' if use_cnn_branch else '비활성'}")
        print(f"  - 융합 방식: {fusion_type if self.fusion else 'N/A'}")
        print(f"  - SE 블록: {'사용' if use_se else '미사용'}")
        print(f"  - 클래스 수: {num_classes}")
        print(f"  - 총 파라미터: {self._count_params():,}")
        print(f"  - 학습 가능 파라미터: {self._count_trainable_params():,}")
        print("=" * 60)

    def _count_params(self) -> int:
        """총 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters())

    def _count_trainable_params(self) -> int:
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        image: torch.Tensor,
        facenet_vector: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        순전파

        Args:
            image: 입력 이미지 [B, C, H, W]
            facenet_vector: 사전 추출된 FaceNet 특징 [B, facenet_input_dim]

        Returns:
            dict with keys:
                - 'logits': 분류 로짓 [B, num_classes]
                - 'facenet_features': FaceNet 특징 [B, D] (있는 경우)
                - 'cnn_features': CNN 특징 [B, D] (있는 경우)
                - 'fused_features': 융합된 특징 [B, D] (있는 경우)
        """
        outputs: Dict[str, torch.Tensor] = {}

        # FaceNet Branch (사전 추출된 벡터 projection)
        facenet_features: Optional[torch.Tensor] = None
        if self.use_facenet_branch and self.facenet_branch is not None:
            if facenet_vector is not None:
                facenet_feat = self.facenet_branch(facenet_vector)
                facenet_features = facenet_feat
                outputs["facenet_features"] = facenet_feat
            else:
                raise ValueError(
                    "FaceNet 브랜치가 활성화되어 있지만 facenet_vector가 제공되지 않았습니다."
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
            and facenet_features is not None
            and cnn_features is not None
        ):
            fused = self.fusion(facenet_features, cnn_features)
            outputs["fused_features"] = fused
            classifier_input = fused
        elif self.use_facenet_branch and facenet_features is not None:
            classifier_input = facenet_features
        elif self.use_cnn_branch and cnn_features is not None:
            classifier_input = cnn_features
        else:
            raise ValueError("특징 추출 실패: 유효한 특징이 없습니다.")

        # Classification
        logits = self.classifier(classifier_input)
        outputs["logits"] = logits

        return outputs

    def predict(
        self,
        image: torch.Tensor,
        facenet_vector: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        예측 수행

        Args:
            image: 입력 이미지 [B, C, H, W]
            facenet_vector: FaceNet 벡터 [B, facenet_input_dim]

        Returns:
            (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(image, facenet_vector)
            logits = outputs["logits"]
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs

    def get_features(
        self,
        image: torch.Tensor,
        facenet_vector: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        중간 특징 추출

        Args:
            image: 입력 이미지 [B, C, H, W]
            facenet_vector: FaceNet 벡터 [B, facenet_input_dim]

        Returns:
            각 브랜치의 특징을 포함하는 딕셔너리
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(image, facenet_vector)

        features = {}
        if "facenet_features" in outputs:
            features["facenet"] = outputs["facenet_features"]
        if "cnn_features" in outputs:
            features["cnn"] = outputs["cnn_features"]
        if "fused_features" in outputs:
            features["fused"] = outputs["fused_features"]

        return features


class HybridEmotionModelLite(nn.Module):
    """
    경량 하이브리드 감정 분류 모델 (CNN 전용)

    FaceNet 벡터 없이 CNN만으로 감정을 분류합니다.
    단일 이미지 예측이나 빠른 테스트에 적합합니다.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        feature_dim: int = 512,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
        classifier_hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        if classifier_hidden_dims is None:
            classifier_hidden_dims = [256, 128]

        # CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained,
            dropout=dropout,
        )

        # Classification Head
        self.classifier = ClassificationHead(
            input_dim=feature_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            use_residual=True,
        )

        # 호환성을 위한 플래그
        self.use_facenet_branch = False
        self.use_cnn_branch = True

        print("HybridEmotionModelLite 초기화 (CNN 전용):")
        print(f"  - 백본: {backbone}")
        print(f"  - 특징 차원: {feature_dim}")
        print(f"  - 클래스 수: {num_classes}")

    def forward(
        self,
        image: torch.Tensor,
        facenet_vector: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        순전파 (facenet_vector는 무시됨)

        Args:
            image: 입력 이미지 [B, C, H, W]
            facenet_vector: 무시됨 (호환성용)

        Returns:
            dict with 'logits' and 'cnn_features'
        """
        features = self.cnn(image)
        logits = self.classifier(features)
        return {"logits": logits, "cnn_features": features}

    def predict(
        self,
        image: torch.Tensor,
        facenet_vector: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """예측 수행"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(image)
            logits = outputs["logits"]
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs


# 테스트 코드
if __name__ == "__main__":
    print("=" * 60)
    print("Hybrid Emotion Model 테스트 (개선 버전)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 더미 입력
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_facenet = torch.randn(batch_size, 512).to(device)
    dummy_facenet = F.normalize(dummy_facenet, p=2, dim=1)

    print("\n[1] 기본 HybridEmotionModel 테스트 (concat)")
    model = HybridEmotionModel(
        use_facenet_branch=True,
        use_cnn_branch=True,
        fusion_type="concat",
        cnn_backbone="resnet18",
        cnn_pretrained=False,
    ).to(device)

    outputs = model(dummy_image, dummy_facenet)
    print(f"  - Logits shape: {outputs['logits'].shape}")

    print("\n[2] Attention Fusion 테스트")
    model_attn = HybridEmotionModel(
        use_facenet_branch=True,
        use_cnn_branch=True,
        fusion_type="attention",
        cnn_backbone="resnet18",
        cnn_pretrained=False,
    ).to(device)

    outputs_attn = model_attn(dummy_image, dummy_facenet)
    print(f"  - Logits shape: {outputs_attn['logits'].shape}")

    print("\n[3] Bilinear Fusion 테스트")
    model_bilinear = HybridEmotionModel(
        use_facenet_branch=True,
        use_cnn_branch=True,
        fusion_type="bilinear",
        cnn_backbone="resnet18",
        cnn_pretrained=False,
    ).to(device)

    outputs_bilinear = model_bilinear(dummy_image, dummy_facenet)
    print(f"  - Logits shape: {outputs_bilinear['logits'].shape}")

    print("\n[4] Gated Fusion 테스트")
    model_gated = HybridEmotionModel(
        use_facenet_branch=True,
        use_cnn_branch=True,
        fusion_type="gated",
        cnn_backbone="resnet18",
        cnn_pretrained=False,
    ).to(device)

    outputs_gated = model_gated(dummy_image, dummy_facenet)
    print(f"  - Logits shape: {outputs_gated['logits'].shape}")

    print("\n[5] FaceNet Only 테스트")
    model_facenet = HybridEmotionModel(
        use_facenet_branch=True,
        use_cnn_branch=False,
        cnn_pretrained=False,
    ).to(device)

    outputs_facenet = model_facenet(dummy_image, dummy_facenet)
    print(f"  - Logits shape: {outputs_facenet['logits'].shape}")

    print("\n[6] Prediction 테스트")
    preds, probs = model.predict(dummy_image, dummy_facenet)
    print(f"  - Predictions: {preds}")
    print(f"  - Probabilities shape: {probs.shape}")

    print("\n[7] Lite Model 테스트")
    model_lite = HybridEmotionModelLite().to(device)
    preds_lite, probs_lite = model_lite.predict(dummy_facenet)
    print(f"  - Predictions: {preds_lite}")

    print("\n테스트 완료!")
