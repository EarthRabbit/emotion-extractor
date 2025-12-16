"""
유틸리티 모듈

이미지 처리, YOLO 벡터 추출 등의 유틸리티 함수들을 제공합니다.

Note: extract_yolo_vectors 모듈은 ultralytics 패키지가 필요합니다.
      설치: pip install ultralytics
"""

__all__ = [
    "YOLOBackboneExtractor",
    "extract_features",
    "get_image_paths",
]

# ultralytics가 설치된 경우에만 import
try:
    from .extract_yolo_vectors import (
        YOLOBackboneExtractor,
        extract_features,
        get_image_paths,
    )
except ImportError:
    # ultralytics가 설치되지 않은 경우 placeholder
    YOLOBackboneExtractor = None  # type: ignore
    extract_features = None  # type: ignore
    get_image_paths = None  # type: ignore
