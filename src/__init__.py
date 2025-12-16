"""
Emotion Extractor - Main package
감정 분류 시스템 패키지

Subpackages:
- data: 데이터 로딩 및 전처리
- models: 모델 정의 (CNN, FaceNet, Hybrid)
- training: 학습 및 평가
- util: 유틸리티 함수 (CUDA, 시각화, 분석)
"""

__version__ = "0.1.0"
__all__ = ["data", "models", "training", "util"]
```

Now update your `pyproject.toml` to specify the package root:
