"""
Models module for Hybrid Emotion Classification
모델 관련 모듈

- facenet_feature.py: FaceNet 특징 벡터 처리 (사전 추출된 벡터용)
- cnn_branch.py: CNN 기반 특징 추출기
- hybrid_model.py: 하이브리드 모델
"""

from .cnn_branch import CNNFeatureExtractor, CNNFeatureExtractorMultiScale
from .facenet_feature import (
    FaceNetFeatureProjector,
    load_facenet_vectors,
)
from .hybrid_model import (
    ClassificationHead,
    FeatureFusion,
    HybridEmotionModel,
    HybridEmotionModelLite,
)

__all__ = [
    # FaceNet (사전 추출된 벡터용)
    "FaceNetFeatureProjector",
    "load_facenet_vectors",
    # CNN
    "CNNFeatureExtractor",
    "CNNFeatureExtractorMultiScale",
    # Hybrid
    "HybridEmotionModel",
    "HybridEmotionModelLite",
    "FeatureFusion",
    "ClassificationHead",
]
