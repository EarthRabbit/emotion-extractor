"""
Custom Dataset for Hybrid Emotion Classification
하이브리드 감정 분류를 위한 커스텀 데이터셋

FaceNet 임베딩 벡터 + CNN 특징 융합 지원
"""

# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def collate_fn(batch):
    """
    배치 데이터를 텐서로 변환하는 함수

    Multiprocessing 호환성을 위해 모듈 수준에서 정의됨
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    paths = [item["path"] for item in batch]

    result = {
        "image": images,
        "label": labels,
        "path": paths,
    }

    # FaceNet 벡터가 있는 경우
    if "facenet_vector" in batch[0]:
        facenet_vectors = torch.stack([item["facenet_vector"] for item in batch])
        result["facenet_vector"] = facenet_vectors

    return result


class TransformSubset(Dataset):
    """
    Transform이 적용된 데이터셋 서브셋

    Multiprocessing 호환성을 위해 모듈 수준에서 정의됨
    """

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 원본 데이터셋에서 가져오기
        original_idx = self.indices[idx]
        image_path, label = self.dataset.samples[original_idx]

        # 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # Transform 적용
        if self.transform:
            image = self.transform(image)

        result = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(image_path),
        }

        # FaceNet 벡터 추가 (있는 경우)
        if hasattr(self.dataset, "facenet_vectors"):
            filename = image_path.name
            if filename in self.dataset.facenet_vectors:
                result["facenet_vector"] = self.dataset.facenet_vectors[filename]
            else:
                result["facenet_vector"] = torch.zeros(512)

        return result


class EmotionDataset(Dataset):
    """
    감정 분류 데이터셋

    폴더 구조:
        data/cropped/
        ├── images/
        │   ├── 기쁨/
        │   ├── 당황/
        │   ├── 분노/
        │   ├── 불안/
        │   ├── 상처/
        │   ├── 슬픔/
        │   └── 중립/
        └── labels/
            ├── 기쁨_metadata.json
            ├── 당황_metadata.json
            └── ...
    """

    def __init__(
        self,
        images_dir: Path,
        class_names: List[str],
        transform: Optional[Callable] = None,
        labels_dir: Optional[Path] = None,
        load_metadata: bool = False,
    ):
        """
        Args:
            images_dir: 이미지가 있는 루트 디렉토리
            class_names: 클래스 이름 리스트 (폴더명과 일치해야 함)
            transform: 이미지 변환 함수
            labels_dir: 메타데이터 JSON이 있는 디렉토리 (선택)
            load_metadata: 메타데이터 로드 여부
        """
        self.images_dir = Path(images_dir)
        self.class_names = class_names
        self.transform = transform
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.load_metadata = load_metadata

        # 클래스 이름 -> 인덱스 매핑
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(class_names)}

        # 메타데이터 로드 (선택적)
        self.metadata = {}
        if self.load_metadata and self.labels_dir:
            self._load_metadata()

        # 이미지 경로 및 레이블 수집
        self.samples: List[Tuple[Path, int]] = []
        self._collect_samples()

        print("EmotionDataset 초기화 완료:")
        print(f"  - 총 샘플 수: {len(self.samples)}")
        print(f"  - 클래스 수: {len(self.class_names)}")
        for class_name in self.class_names:
            count = sum(
                1 for _, label in self.samples if label == self.class_to_idx[class_name]
            )
            print(f"    - {class_name}: {count}개")

    def _load_metadata(self):
        """메타데이터 JSON 파일 로드"""
        if not self.labels_dir or not self.labels_dir.exists():
            return

        for class_name in self.class_names:
            json_path = self.labels_dir / f"{class_name}_metadata.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # filename을 키로 사용
                    for item in data:
                        if "filename" in item:
                            self.metadata[item["filename"]] = item

    def _collect_samples(self):
        """이미지 경로와 레이블 수집"""
        for class_name in self.class_names:
            class_dir = self.images_dir / class_name
            if not class_dir.exists():
                print(f"Warning: 클래스 폴더가 존재하지 않음: {class_dir}")
                continue

            class_idx = self.class_to_idx[class_name]

            # 지원하는 이미지 확장자
            valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in valid_extensions:
                    self.samples.append((image_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - 'image': 변환된 이미지 텐서
                - 'label': 클래스 레이블
                - 'path': 이미지 경로 (문자열)
                - 'metadata': 메타데이터 (선택적)
        """
        image_path, label = self.samples[idx]

        # 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # 변환 적용
        if self.transform:
            image = self.transform(image)

        result = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(image_path),
        }

        # 메타데이터 추가 (선택적)
        if self.load_metadata:
            filename = image_path.name
            result["metadata"] = self.metadata.get(filename, {})

        return result

    def get_class_weights(self) -> torch.Tensor:
        """클래스 불균형 처리를 위한 가중치 계산"""
        class_counts = torch.zeros(len(self.class_names))
        for _, label in self.samples:
            class_counts[label] += 1

        # Inverse frequency weighting
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(self.class_names)

        return weights


class EmotionDatasetWithFaceNet(EmotionDataset):
    """
    FaceNet 임베딩 벡터를 포함하는 감정 분류 데이터셋

    사전 추출된 FaceNet 임베딩 벡터(512차원)를 로드하여
    이미지와 함께 반환합니다.

    FaceNet 임베딩 특징:
        - 512차원 벡터
        - L2 정규화됨 (Norm = 1.0)
        - 얼굴 인식에 특화된 특징
    """

    def __init__(
        self,
        images_dir: Path,
        class_names: List[str],
        transform: Optional[Callable] = None,
        labels_dir: Optional[Path] = None,
        load_metadata: bool = False,
        facenet_vectors_path: Optional[Path] = None,
    ):
        """
        Args:
            images_dir: 이미지 디렉토리
            class_names: 클래스 이름 리스트
            transform: 이미지 변환 함수
            labels_dir: 메타데이터 디렉토리
            load_metadata: 메타데이터 로드 여부
            facenet_vectors_path: 사전 추출된 FaceNet 벡터 파일 경로 (.pt)
        """
        super().__init__(
            images_dir=images_dir,
            class_names=class_names,
            transform=transform,
            labels_dir=labels_dir,
            load_metadata=load_metadata,
        )

        self.facenet_vectors_path = facenet_vectors_path
        self.facenet_vectors: Dict[str, torch.Tensor] = {}
        self.feature_dim = 512  # FaceNet 고정 차원

        # 사전 추출된 벡터 로드
        if facenet_vectors_path is not None and Path(facenet_vectors_path).exists():
            self._load_facenet_vectors()

    def _load_facenet_vectors(self):
        """사전 추출된 FaceNet 임베딩 벡터 로드"""
        if self.facenet_vectors_path is None:
            return

        print(f"FaceNet 벡터 로드 중: {self.facenet_vectors_path}")
        data = torch.load(self.facenet_vectors_path, weights_only=False)
        self.facenet_vectors = data.get("vectors", {})

        # 메타데이터 출력
        model_name = data.get("model_name", "unknown")
        feature_dim = data.get("feature_dim", 512)
        num_images = data.get("num_images", len(self.facenet_vectors))
        is_normalized = data.get("normalized", True)

        print("  ✓ FaceNet 벡터 로드 완료:")
        print(f"    - 모델: {model_name}")
        print(f"    - 벡터 수: {num_images}개")
        print(f"    - 차원: {feature_dim}")
        normalized_str = "예" if is_normalized else "아니오"
        print(f"    - L2 정규화: {normalized_str}")

        # 매칭 통계
        matched = sum(
            1 for path, _ in self.samples if path.name in self.facenet_vectors
        )
        print(
            f"    - 매칭된 이미지: {matched}/{len(self.samples)} ({matched / len(self.samples) * 100:.1f}%)"
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = super().__getitem__(idx)

        image_path = str(result["path"])
        filename = Path(image_path).name

        # FaceNet 벡터 추가
        if filename in self.facenet_vectors:
            result["facenet_vector"] = self.facenet_vectors[filename]
        else:
            # 벡터가 없는 경우 더미 벡터 (경고 출력은 하지 않음 - 너무 많을 수 있음)
            result["facenet_vector"] = torch.zeros(self.feature_dim)

        return result


def get_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = True,
    augmentation_level: str = "medium",
) -> transforms.Compose:
    """
    데이터 변환 파이프라인 생성

    Args:
        image_size: 출력 이미지 크기
        is_training: 학습용 여부 (augmentation 적용)
        augmentation_level: "light", "medium", "heavy"

    Returns:
        transforms.Compose 객체
    """
    # ImageNet 정규화 값
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_training:
        if augmentation_level == "light":
            transform_list = [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        elif augmentation_level == "medium":
            transform_list = [
                transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        else:  # heavy
            transform_list = [
                transforms.Resize((int(image_size[0] * 1.2), int(image_size[1] * 1.2))),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.2),
            ]
    else:
        # Validation/Test - augmentation 없음
        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    return transforms.Compose(transform_list)


def create_dataloaders(
    images_dir: Path,
    class_names: List[str],
    batch_size: int = 32,
    train_ratio: float = 0.9,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    seed: int = 42,
    labels_dir: Optional[Path] = None,
    augmentation_level: str = "medium",
    use_facenet: bool = False,
    facenet_vectors_path: Optional[Path] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """
    Train/Validation DataLoader 생성

    Args:
        images_dir: 이미지 디렉토리
        class_names: 클래스 이름 리스트
        batch_size: 배치 크기
        train_ratio: 학습 데이터 비율 (기본 0.9)
        image_size: 이미지 크기
        num_workers: DataLoader worker 수
        seed: 랜덤 시드
        labels_dir: 레이블 디렉토리
        augmentation_level: Augmentation 수준
        use_facenet: FaceNet 벡터 사용 여부
        facenet_vectors_path: 사전 추출된 FaceNet 벡터 파일 경로 (.pt)

    Returns:
        (train_loader, val_loader, info_dict)
    """
    # Transform 생성
    train_transform = get_transforms(
        image_size, is_training=True, augmentation_level=augmentation_level
    )
    val_transform = get_transforms(image_size, is_training=False)

    # 데이터셋 생성
    full_dataset: EmotionDataset
    if use_facenet:
        full_dataset = EmotionDatasetWithFaceNet(
            images_dir=images_dir,
            class_names=class_names,
            transform=None,  # 분할 후 적용
            labels_dir=labels_dir,
            facenet_vectors_path=facenet_vectors_path,
        )
    else:
        full_dataset = EmotionDataset(
            images_dir=images_dir,
            class_names=class_names,
            transform=None,  # 분할 후 적용
            labels_dir=labels_dir,
        )

    # Train/Val 분할
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    # 인덱스 분할
    generator = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]

    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_transform)

    # DataLoader 생성
    use_cuda = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if use_cuda else 0,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
        persistent_workers=True if (use_cuda and num_workers > 0) else False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if use_cuda else 0,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
        persistent_workers=True if (use_cuda and num_workers > 0) else False,
    )

    # 정보 딕셔너리
    info = {
        "total_samples": total_size,
        "train_samples": train_size,
        "val_samples": val_size,
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_to_idx": full_dataset.class_to_idx,
        "class_weights": full_dataset.get_class_weights(),
        "use_facenet": use_facenet,
    }

    print("\n데이터로더 생성 완료:")
    print(f"  - 전체: {total_size}개")
    print(f"  - 학습: {train_size}개 ({train_ratio * 100:.0f}%)")
    print(f"  - 검증: {val_size}개 ({(1 - train_ratio) * 100:.0f}%)")
    print(f"  - 배치 크기: {batch_size}")
    print(f"  - Worker 수: {num_workers if use_cuda else 0}")
    print(f"  - FaceNet 사용: {'예' if use_facenet else '아니오'}")

    return train_loader, val_loader, info
