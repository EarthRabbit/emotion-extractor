"""
Models module for Hybrid Emotion Classification
모델 관련 모듈

- yolo_feature.py: YOLO 특징 벡터 처리 (사전 추출된 벡터용)
- cnn_branch.py: CNN 기반 특징 추출기
- hybrid_model.py: 하이브리드 모델
"""

from .cnn_branch import CNNFeatureExtractor, CNNFeatureExtractorMultiScale
from .hybrid_model import (
    ClassificationHead,
    FeatureFusion,
    HybridEmotionModel,
    HybridEmotionModelLite,
)
from .yolo_feature import YOLOFeatureProjector, load_yolo_vectors

__all__ = [
    # YOLO (사전 추출된 벡터용)
    "YOLOFeatureProjector",
    "load_yolo_vectors",
    # CNN
    "CNNFeatureExtractor",
    "CNNFeatureExtractorMultiScale",
    # Hybrid
    "HybridEmotionModel",
    "HybridEmotionModelLite",
    "FeatureFusion",
    "ClassificationHead",
]
