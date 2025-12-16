"""
Utility module for Hybrid Emotion Classification
유틸리티 모듈

- cuda.py: CUDA 관련 유틸리티 함수
- plotting.py: 실시간 학습 시각화
- analysis.py: 모델 분석 및 평가 도구
"""

from .analysis import (
    AnalysisReport,
    analyze_misclassifications,
    compute_metrics,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from .cuda import (
    check_cuda_available,
    clear_cuda_cache,
    get_device,
    get_gpu_memory_info,
    get_optimal_num_workers,
    print_cuda_info,
    print_device_info,
    print_gpu_memory_info,
    print_system_info,
    require_cuda,
    setup_cuda_optimization,
    setup_multiprocessing,
)
from .plotting import LivePlotter, TrainingVisualizer, plot_training_history

__all__ = [
    # CUDA utilities
    "check_cuda_available",
    "clear_cuda_cache",
    "get_device",
    "get_gpu_memory_info",
    "get_optimal_num_workers",
    "print_cuda_info",
    "print_device_info",
    "print_gpu_memory_info",
    "print_system_info",
    "require_cuda",
    "setup_cuda_optimization",
    "setup_multiprocessing",
    # Plotting
    "LivePlotter",
    "TrainingVisualizer",
    "plot_training_history",
    # Analysis
    "AnalysisReport",
    "compute_metrics",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_class_distribution",
    "analyze_misclassifications",
]
