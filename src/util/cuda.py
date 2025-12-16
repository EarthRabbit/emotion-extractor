"""
CUDA 유틸리티 모듈
CUDA Utility Module

CUDA 관련 공통 유틸리티 함수들을 제공합니다.
- CUDA 사용 가능 여부 확인
- CUDA 최적화 설정
- 디바이스 정보 출력
- 최적의 DataLoader worker 수 계산

여러 모듈에서 공통으로 사용되는 CUDA 관련 함수들을 통합합니다.
"""

import multiprocessing
import os
import sys
from functools import wraps
from typing import Callable, TypeVar

import torch

F = TypeVar("F", bound=Callable)


def check_cuda_available() -> bool:
    """
    CUDA 사용 가능 여부 확인

    Returns:
        CUDA 사용 가능 여부
    """
    return torch.cuda.is_available()


def get_device(force_cpu: bool = False) -> torch.device:
    """
    사용할 디바이스 반환

    Args:
        force_cpu: 강제로 CPU 사용

    Returns:
        torch.device 객체
    """
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if check_cuda_available() else "cpu")


def get_optimal_num_workers() -> int:
    """
    최적의 DataLoader worker 수 반환

    CPU 코어 수를 기반으로 최적의 worker 수를 계산합니다.
    CUDA를 사용하지 않는 경우 0을 반환합니다.

    Returns:
        최적의 worker 수
    """
    if not check_cuda_available():
        return 0

    cpu_count = multiprocessing.cpu_count()
    # 일반적으로 CPU 코어 수의 절반 정도가 적절
    optimal = min(cpu_count // 2, 8)
    return max(optimal, 2)


def setup_cuda_optimization() -> None:
    """
    CUDA 최적화 설정

    cuDNN 벤치마크, TF32, 메모리 설정 등을 최적화합니다.
    CUDA를 사용할 수 없는 경우 아무 작업도 수행하지 않습니다.
    """
    if not check_cuda_available():
        return

    # cuDNN 최적화 활성화
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # 입력 크기가 일정할 때 성능 향상

    # TF32 활성화 (Ampere 이상 GPU에서 성능 향상)
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

    # 메모리 단편화 방지
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def print_cuda_info() -> None:
    """
    CUDA 정보 출력

    사용 가능한 GPU 정보, CUDA 버전, cuDNN 버전 등을 출력합니다.
    """
    print("\n[CUDA 정보]")
    if check_cuda_available():
        print("  - CUDA 사용 가능: ✓")
        print(f"  - CUDA 버전: {torch.version.cuda}")
        print(f"  - GPU 이름: {torch.cuda.get_device_name(0)}")
        print(
            f"  - GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
        print(f"  - cuDNN 버전: {torch.backends.cudnn.version()}")
        print(f"  - cuDNN 활성화: {torch.backends.cudnn.enabled}")
        print("  - AMP 지원: ✓")
    else:
        print("  - CUDA 사용 가능: ✗")
        print("  - CPU 모드로 실행됩니다.")
        print("  - AMP 지원: ✗")


def print_device_info(device: torch.device) -> None:
    """
    디바이스 정보 출력

    Args:
        device: 출력할 디바이스
    """
    print("\n[디바이스 정보]")
    if device.type == "cuda":
        print("  - CUDA 사용 가능: ✓")
        print(f"  - CUDA 버전: {torch.version.cuda}")
        print(f"  - GPU 이름: {torch.cuda.get_device_name(0)}")
        print(
            f"  - GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
        print(f"  - cuDNN 버전: {torch.backends.cudnn.version()}")
    else:
        print("  - CUDA 사용 가능: ✗")
        print("  - CPU 모드로 실행됩니다 (속도가 느릴 수 있습니다)")


def print_system_info() -> None:
    """
    시스템 정보 출력

    CPU, PyTorch 버전 등 시스템 정보를 출력합니다.
    """
    print("\n[시스템 정보]")
    print(f"  - CPU 코어 수: {multiprocessing.cpu_count()}")
    print(f"  - 최적 Worker 수: {get_optimal_num_workers()}")
    print(f"  - PyTorch 버전: {torch.__version__}")
    print(f"  - Python 버전: {sys.version.split()[0]}")


def require_cuda(func: F) -> F:
    """
    CUDA 필수 데코레이터

    CUDA가 없으면 오류 메시지를 출력하고 프로그램을 종료합니다.

    Args:
        func: 래핑할 함수

    Returns:
        래핑된 함수
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not check_cuda_available():
            print("\n" + "=" * 60)
            print("⚠️  CUDA를 사용할 수 없습니다!")
            print("=" * 60)
            print("\n이 작업은 CUDA(GPU)가 필요합니다.")
            print("다음 사항을 확인해주세요:")
            print("  1. NVIDIA GPU가 설치되어 있는지 확인")
            print("  2. CUDA Toolkit이 설치되어 있는지 확인")
            print("  3. PyTorch가 CUDA 버전으로 설치되어 있는지 확인")
            print("\nPyTorch CUDA 설치 확인:")
            print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
            print(f"  torch.version.cuda = {torch.version.cuda}")
            print("\n작업이 취소되었습니다.")
            sys.exit(1)
        return func(*args, **kwargs)

    return wrapper  # type: ignore


def setup_multiprocessing() -> None:
    """
    멀티프로세싱 최적화 설정

    Windows에서의 멀티프로세싱 문제를 방지하고
    OpenMP/MKL 스레드 수를 최적화합니다.
    """
    # Windows에서 멀티프로세싱 문제 방지
    if sys.platform == "win32":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # 이미 설정된 경우

    # OpenMP 스레드 수 설정
    cpu_count = multiprocessing.cpu_count()
    os.environ.setdefault("OMP_NUM_THREADS", str(max(cpu_count // 2, 1)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(cpu_count // 2, 1)))


def clear_cuda_cache() -> None:
    """
    CUDA 메모리 캐시 정리

    GPU 메모리를 정리하여 메모리 부족 문제를 방지합니다.
    """
    if check_cuda_available():
        torch.cuda.empty_cache()


def get_gpu_memory_info() -> dict:
    """
    GPU 메모리 정보 반환

    Returns:
        GPU 메모리 정보 딕셔너리 (total, allocated, cached, free)
        CUDA를 사용할 수 없는 경우 빈 딕셔너리 반환
    """
    if not check_cuda_available():
        return {}

    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    cached = torch.cuda.memory_reserved(0)
    free = total - allocated

    return {
        "total": total,
        "total_gb": total / 1024**3,
        "allocated": allocated,
        "allocated_gb": allocated / 1024**3,
        "cached": cached,
        "cached_gb": cached / 1024**3,
        "free": free,
        "free_gb": free / 1024**3,
    }


def print_gpu_memory_info() -> None:
    """
    GPU 메모리 정보 출력
    """
    info = get_gpu_memory_info()
    if not info:
        print("GPU 메모리 정보를 가져올 수 없습니다 (CUDA 미사용)")
        return

    print("\n[GPU 메모리 정보]")
    print(f"  - 전체: {info['total_gb']:.2f} GB")
    print(f"  - 할당됨: {info['allocated_gb']:.2f} GB")
    print(f"  - 캐시됨: {info['cached_gb']:.2f} GB")
    print(f"  - 여유: {info['free_gb']:.2f} GB")


# ============================================================
# 테스트 코드
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("CUDA 유틸리티 모듈 테스트")
    print("=" * 50)

    print(f"\n[1] CUDA 사용 가능: {check_cuda_available()}")

    print("\n[2] 디바이스 정보")
    device = get_device()
    print(f"  - 현재 디바이스: {device}")

    print(f"\n[3] 최적 Worker 수: {get_optimal_num_workers()}")

    print("\n[4] CUDA 정보 출력")
    print_cuda_info()

    print("\n[5] 시스템 정보 출력")
    print_system_info()

    if check_cuda_available():
        print("\n[6] CUDA 최적화 설정")
        setup_cuda_optimization()
        print("  - 최적화 설정 완료")

        print("\n[7] GPU 메모리 정보")
        print_gpu_memory_info()

        print("\n[8] 캐시 정리")
        clear_cuda_cache()
        print("  - 캐시 정리 완료")

    print("\n테스트 완료!")
