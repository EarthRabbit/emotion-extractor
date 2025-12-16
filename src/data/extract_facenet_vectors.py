"""
FaceNet 특징 벡터 추출 및 저장 스크립트

이미지 폴더에서 FaceNet(InceptionResnetV1)을 사용하여
얼굴 임베딩 벡터(512차원)를 추출하고 .pt 파일로 저장합니다.

FaceNet은 얼굴 인식에 특화된 모델로, YOLO보다 감정 인식에 더 적합합니다.

⚠️ CUDA(GPU) 권장: GPU가 있으면 훨씬 빠르게 처리됩니다.

사용법:
    python extract_facenet_vectors.py --data-root path/to/images --output path/to/output.pt
    python extract_facenet_vectors.py --data-root d:/emotion-extractor/Data/cropped/images

설치 필요:
    pip install facenet-pytorch
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# CUDA 유틸리티 import
from util.cuda import (
    get_device,
    get_optimal_num_workers,
    print_device_info,
    print_system_info,
    setup_cuda_optimization,
)

# ============================================================
# FaceNet import
# ============================================================

try:
    from facenet_pytorch import InceptionResnetV1

    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("=" * 60)
    print("Error: facenet-pytorch 패키지가 필요합니다.")
    print("설치: pip install facenet-pytorch")
    print("=" * 60)
    sys.exit(1)


# ============================================================
# FaceNet 임베딩 추출기
# ============================================================


class FaceNetExtractor(nn.Module):
    """FaceNet(InceptionResnetV1)에서 얼굴 임베딩을 추출하는 클래스

    출력:
        512차원 L2 정규화된 임베딩 벡터
    """

    def __init__(
        self,
        pretrained: str = "vggface2",
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            pretrained: 사전학습 가중치 ('vggface2' 또는 'casia-webface')
                - vggface2: 더 다양한 얼굴 데이터로 학습 (권장)
                - casia-webface: 아시아인 얼굴 데이터 많음
            device: 연산 디바이스
        """
        super().__init__()
        self.device: torch.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # FaceNet 모델 로드
        print(f"FaceNet 모델 로드 중 (pretrained={pretrained})...")
        self.model = InceptionResnetV1(pretrained=pretrained)

        # 모델 고정 (학습 안 함)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)
        self.model.eval()

        # 모델 정보 출력
        print(f"  - 디바이스: {self.device}")
        print("  - 출력 차원: 512")
        print("  - 정규화: L2 (Norm = 1.0)")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        이미지에서 얼굴 임베딩 벡터 추출

        Args:
            x: 입력 이미지 [B, 3, 160, 160] (정규화된 텐서)

        Returns:
            임베딩 벡터 [B, 512] (L2 정규화됨)
        """
        embeddings = self.model(x)
        return embeddings


# ============================================================
# 이미지 로딩 (멀티스레딩)
# ============================================================


def load_single_image(
    path: Path, transform: transforms.Compose
) -> Optional[torch.Tensor]:
    """단일 이미지 로드 및 전처리"""
    try:
        img = Image.open(path).convert("RGB")
        img_tensor: torch.Tensor = transform(img)
        return img_tensor
    except Exception as e:
        print(f"\nWarning: 이미지 로드 실패 - {path}: {e}")
        return None


def load_images_parallel(
    paths: List[Path],
    transform: transforms.Compose,
    num_workers: int = 4,
) -> List[tuple]:
    """멀티스레딩으로 이미지 병렬 로드"""
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(load_single_image, path, transform): path for path in paths
        }

        for future in futures:
            path = futures[future]
            try:
                img_tensor = future.result()
                if img_tensor is not None:
                    results.append((path, img_tensor))
            except Exception as e:
                print(f"\nWarning: 이미지 처리 실패 - {path}: {e}")

    return results


# ============================================================
# 유틸리티 함수
# ============================================================


def get_image_paths(data_root: Path) -> Dict[str, List[Path]]:
    """
    데이터 루트에서 이미지 경로 수집

    Args:
        data_root: 이미지 루트 디렉토리

    Returns:
        {class_name: [image_paths]} 딕셔너리
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_paths: Dict[str, List[Path]] = {}

    # 클래스별 폴더 구조 확인
    for class_dir in data_root.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            paths = []
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    paths.append(img_path)
            if paths:
                image_paths[class_name] = sorted(paths)
                print(f"  - {class_name}: {len(paths)}개 이미지")

    # 폴더 구조가 아닌 경우 (평면 구조)
    if not image_paths:
        paths = []
        for img_path in data_root.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                paths.append(img_path)
        if paths:
            image_paths["all"] = sorted(paths)
            print(f"  - 전체: {len(paths)}개 이미지")

    return image_paths


def extract_features(
    extractor: FaceNetExtractor,
    image_paths: List[Path],
    batch_size: int = 32,
    image_size: int = 160,
    num_workers: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    이미지들에서 FaceNet 임베딩 추출 (멀티스레딩 가속화)

    Args:
        extractor: FaceNet 특징 추출기
        image_paths: 이미지 경로 리스트
        batch_size: 배치 크기
        image_size: 이미지 크기 (FaceNet 기본: 160)
        num_workers: 이미지 로딩 worker 수

    Returns:
        {filename: embedding_tensor} 딕셔너리
    """
    # FaceNet 전용 전처리
    # FaceNet은 [-1, 1] 범위로 정규화
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    features_dict: Dict[str, torch.Tensor] = {}
    device = extractor.device

    # 배치 처리
    num_batches = (len(image_paths) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="FaceNet 임베딩 추출 중"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]

        # 멀티스레딩으로 이미지 로드
        loaded_images = load_images_parallel(batch_paths, transform, num_workers)

        if not loaded_images:
            continue

        # 배치 텐서 생성
        valid_paths = [item[0] for item in loaded_images]
        images = [item[1] for item in loaded_images]
        batch_tensor = torch.stack(images).to(device, non_blocking=True)

        # 임베딩 추출
        embeddings = extractor(batch_tensor)

        # 결과 저장
        for path, emb in zip(valid_paths, embeddings):
            filename = path.name
            features_dict[filename] = emb.cpu()

    return features_dict


# ============================================================
# 메인 함수
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="FaceNet 얼굴 임베딩 벡터 추출 및 저장",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FaceNet은 얼굴 인식에 특화된 모델로, 감정 인식에 더 적합합니다.

특징:
  - 출력 차원: 512 (L2 정규화됨)
  - 사전학습: VGGFace2 (다양한 얼굴) 또는 CASIA-WebFace (아시아인 많음)
  - GPU 권장 (CPU도 가능하지만 느림)

예시:
    # 기본 사용법
    python extract_facenet_vectors.py --data-root ./data/cropped/images

    # 출력 경로 지정
    python extract_facenet_vectors.py --data-root ./data/cropped/images --output ./data/facenet_vectors.pt

    # 배치 크기 및 사전학습 모델 지정
    python extract_facenet_vectors.py --data-root ./data/cropped/images --batch-size 64 --pretrained casia-webface

    # 디바이스 정보 확인
    python extract_facenet_vectors.py --check-device
        """,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="d:/emotion-extractor/Data/cropped/images",
        help="이미지 루트 디렉토리 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 파일 경로 (.pt). 기본값: data-root/../facenet_vectors.pt",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="vggface2",
        choices=["vggface2", "casia-webface"],
        help="사전학습 가중치 (기본: vggface2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="배치 크기 (기본: 32)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=160,
        help="이미지 크기 (기본: 160, FaceNet 권장)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="이미지 로딩 worker 수 (기본: 자동)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="강제로 CPU 사용",
    )
    parser.add_argument(
        "--check-device",
        action="store_true",
        help="디바이스 정보만 확인하고 종료",
    )

    args = parser.parse_args()

    # 디바이스 설정
    device = get_device(force_cpu=args.cpu)

    # 디바이스 정보 확인 모드
    if args.check_device:
        print("=" * 60)
        print("디바이스 정보 확인")
        print("=" * 60)
        print_device_info(device)
        print_system_info()
        print(f"  - FaceNet 사용 가능: {'✓' if FACENET_AVAILABLE else '✗'}")
        return

    # CUDA 최적화 적용
    if device.type == "cuda":
        setup_cuda_optimization()

    # 경로 설정
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: 데이터 경로가 존재하지 않습니다: {data_root}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_root.parent / "facenet_vectors.pt"

    # Worker 수 설정
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = get_optimal_num_workers()

    print("=" * 60)
    print("🧬 FaceNet 얼굴 임베딩 추출")
    print("=" * 60)
    print_device_info(device)
    print("\n[설정]")
    print(f"  - 데이터 경로: {data_root}")
    print(f"  - 출력 경로: {output_path}")
    print(f"  - 사전학습: {args.pretrained}")
    print(f"  - 배치 크기: {args.batch_size}")
    print(f"  - 이미지 크기: {args.image_size}")
    print(f"  - 디바이스: {device}")
    print(f"  - 이미지 로딩 Workers: {num_workers}")

    # 이미지 경로 수집
    print("\n[1/3] 이미지 경로 수집 중...")
    image_paths_dict = get_image_paths(data_root)

    if not image_paths_dict:
        print("Error: 이미지를 찾을 수 없습니다.")
        sys.exit(1)

    # 전체 이미지 경로 리스트
    all_paths: List[Path] = []
    for paths in image_paths_dict.values():
        all_paths.extend(paths)

    total_images = len(all_paths)
    print(f"  - 총 이미지 수: {total_images}")

    # FaceNet 추출기 초기화
    print("\n[2/3] FaceNet 모델 초기화 중...")
    extractor = FaceNetExtractor(
        pretrained=args.pretrained,
        device=device,
    )

    # 임베딩 추출 (멀티스레딩 가속화)
    print("\n[3/3] 임베딩 추출 중 (멀티스레딩 가속화)...")
    features_dict = extract_features(
        extractor=extractor,
        image_paths=all_paths,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=num_workers,
    )

    # 특징 차원 확인
    sample_feature = next(iter(features_dict.values()))
    feature_dim = sample_feature.shape[0]

    # L2 정규화 확인
    sample_norm = torch.norm(sample_feature).item()

    # 저장
    print("\n저장 중...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "vectors": features_dict,
        "model_name": f"facenet-{args.pretrained}",
        "feature_dim": feature_dim,
        "image_size": args.image_size,
        "num_images": len(features_dict),
        "classes": list(image_paths_dict.keys()),
        "normalized": True,  # L2 정규화 여부
    }

    torch.save(save_data, output_path)

    # GPU 메모리 정리
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)
    print(f"  - 저장 경로: {output_path}")
    print(f"  - 총 벡터 수: {len(features_dict)}")
    print(f"  - 특징 차원: {feature_dim}")
    print(f"  - L2 Norm (샘플): {sample_norm:.4f}")
    print(f"  - 파일 크기: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
