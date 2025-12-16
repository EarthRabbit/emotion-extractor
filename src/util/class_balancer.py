"""
클래스 불균형 처리 유틸리티
Class Imbalance Handling Utility

불균형한 클래스 분포를 개선하기 위한 도구들을 제공합니다.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class ClassBalancer:
    """클래스 불균형 분석 및 처리"""

    def __init__(self, dataset: Dataset, class_names: List[str]):
        """
        Args:
            dataset: 데이터셋 (samples 속성 필요)
            class_names: 클래스 이름 리스트
        """
        self.dataset = dataset
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.class_counts = self._count_classes()

    def _count_classes(self) -> Dict[str, int]:
        """각 클래스의 샘플 수 계산"""
        counts = {name: 0 for name in self.class_names}

        for _, label in self.dataset.samples:
            class_name = self.class_names[label]
            counts[class_name] += 1

        return counts

    def analyze_imbalance(self) -> Dict[str, any]:
        """클래스 불균형 분석"""
        counts = self.class_counts
        total = sum(counts.values())

        # 통계 계산
        max_count = max(counts.values())
        min_count = min(counts.values())
        imbalance_ratio = max_count / min_count

        # 각 클래스의 비율
        percentages = {name: (count / total * 100) for name, count in counts.items()}

        return {
            "total_samples": total,
            "class_counts": counts,
            "class_percentages": percentages,
            "max_count": max_count,
            "min_count": min_count,
            "imbalance_ratio": imbalance_ratio,
        }

    def print_analysis(self):
        """클래스 불균형 분석 결과 출력"""
        analysis = self.analyze_imbalance()

        print("\n📊 클래스 불균형 분석")
        print("=" * 60)
        print(f"전체 샘플: {analysis['total_samples']}")
        print(f"불균형 비율: {analysis['imbalance_ratio']:.2f}x")
        print(f"  (최다: {analysis['max_count']}, 최소: {analysis['min_count']})\n")

        print("클래스별 분포:")
        for class_name in self.class_names:
            count = analysis["class_counts"][class_name]
            percent = analysis["class_percentages"][class_name]
            bar_length = int(percent / 5)
            bar = "█" * bar_length
            print(f"  {class_name:6s}: {count:4d} ({percent:5.1f}%) {bar}")

        print("=" * 60)

    def get_class_weights(self) -> torch.Tensor:
        """역 빈도 가중치 계산 (inverse frequency weighting)"""
        counts = list(self.class_counts.values())
        counts = np.array(counts, dtype=np.float32)

        # 역 빈도 가중치
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(weights)

        return torch.from_numpy(weights)

    def get_balanced_weights(self) -> torch.Tensor:
        """균형 가중치 계산 (모든 클래스를 동등하게)"""
        weights = torch.ones(self.num_classes) / self.num_classes
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """각 샘플의 가중치 계산 (WeightedRandomSampler용)"""
        class_weights = self.get_class_weights()

        sample_weights = []
        for _, label in self.dataset.samples:
            sample_weights.append(class_weights[label].item())

        return torch.tensor(sample_weights, dtype=torch.float32)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """불균형 처리 샘플러 생성"""
        sample_weights = self.get_sample_weights()

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.dataset),
            replacement=True,
        )

        return sampler

    def get_augmentation_strategy(self) -> Dict[str, str]:
        """클래스별 증강 전략 추천"""
        analysis = self.analyze_imbalance()
        percentages = analysis["class_percentages"]
        mean_percent = 100 / self.num_classes

        strategy = {}
        for class_name, percent in percentages.items():
            if percent < mean_percent * 0.5:
                # 매우 부족 (평균의 50% 미만)
                strategy[class_name] = "very_heavy"
            elif percent < mean_percent * 0.8:
                # 부족 (평균의 80% 미만)
                strategy[class_name] = "heavy"
            elif percent < mean_percent * 1.2:
                # 적절 (평균의 80-120%)
                strategy[class_name] = "medium"
            else:
                # 충분 (평균의 120% 이상)
                strategy[class_name] = "light"

        return strategy

    def print_augmentation_strategy(self):
        """증강 전략 출력"""
        strategy = self.get_augmentation_strategy()

        print("\n🔄 클래스별 권장 증강 수준")
        print("=" * 60)

        for class_name in self.class_names:
            aug_level = strategy[class_name]

            # 이모지 매핑
            emoji_map = {
                "very_heavy": "🔥",
                "heavy": "⚠️ ",
                "medium": "✅",
                "light": "😊",
            }
            emoji = emoji_map.get(aug_level, "")

            level_desc_map = {
                "very_heavy": "매우 강함 (RandomCrop, Rotation(25), ColorJitter, Grayscale, Blur, Erasing)",
                "heavy": "강함 (RandomCrop, Rotation(20), ColorJitter, Blur)",
                "medium": "중간 (RandomCrop, Rotation(15), ColorJitter)",
                "light": "약함 (Resize, RandomFlip)",
            }
            desc = level_desc_map.get(aug_level, "")

            print(f"  {emoji} {class_name:6s}: {aug_level:12s} - {desc}")

        print("=" * 60)


class StratifiedDataSplitter:
    """층화 샘플링을 이용한 train/val 분할"""

    @staticmethod
    def split_stratified(
        dataset: Dataset,
        class_names: List[str],
        train_ratio: float = 0.9,
        seed: int = 42,
    ) -> Tuple[List[int], List[int]]:
        """
        층화 샘플링으로 train/val 분할

        각 클래스의 비율을 유지하면서 분할합니다.

        Args:
            dataset: 데이터셋
            class_names: 클래스 이름
            train_ratio: 학습 데이터 비율
            seed: 랜덤 시드

        Returns:
            (train_indices, val_indices)
        """
        np.random.seed(seed)

        # 클래스별 인덱스
        class_indices = {name: [] for name in class_names}

        for idx, (_, label) in enumerate(dataset.samples):
            class_name = class_names[label]
            class_indices[class_name].append(idx)

        train_indices = []
        val_indices = []

        # 각 클래스별로 분할
        for class_name in class_names:
            indices = class_indices[class_name]
            np.random.shuffle(indices)

            split_point = int(len(indices) * train_ratio)
            train_indices.extend(indices[:split_point])
            val_indices.extend(indices[split_point:])

        return train_indices, val_indices

    @staticmethod
    def print_split_info(
        dataset: Dataset,
        class_names: List[str],
        train_indices: List[int],
        val_indices: List[int],
    ):
        """분할 정보 출력"""
        # 각 분할의 클래스 분포 계산
        train_counts = {name: 0 for name in class_names}
        val_counts = {name: 0 for name in class_names}

        for idx in train_indices:
            _, label = dataset.samples[idx]
            train_counts[class_names[label]] += 1

        for idx in val_indices:
            _, label = dataset.samples[idx]
            val_counts[class_names[label]] += 1

        print("\n📊 데이터셋 분할 정보")
        print("=" * 60)
        print(
            f"전체: {len(dataset.samples)} | 학습: {len(train_indices)} | 검증: {len(val_indices)}\n"
        )

        print("학습 데이터 분포:")
        for class_name in class_names:
            count = train_counts[class_name]
            percent = count / len(train_indices) * 100
            print(f"  {class_name:6s}: {count:4d} ({percent:5.1f}%)")

        print("\n검증 데이터 분포:")
        for class_name in class_names:
            count = val_counts[class_name]
            percent = count / len(val_indices) * 100
            print(f"  {class_name:6s}: {count:4d} ({percent:5.1f}%)")

        print("=" * 60)


class FocalLossWeight:
    """Focal Loss 계산 (어려운 샘플에 가중치 부여)"""

    @staticmethod
    def compute_focal_weights(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """
        Focal Loss 가중치 계산

        어려운 샘플(낮은 confidence)에 더 높은 가중치를 부여합니다.

        Args:
            predictions: 모델 예측 확률 [N, C]
            targets: 실제 레이블 [N]
            alpha: 클래스 가중치
            gamma: 포커싱 파라미터 (높을수록 어려운 샘플에 더 집중)

        Returns:
            샘플별 가중치 [N]
        """
        # 정확한 클래스의 확률 추출
        batch_size = predictions.shape[0]
        p = predictions[torch.arange(batch_size), targets]

        # Focal weight: (1-p)^gamma
        focal_weight = (1 - p) ** gamma

        return focal_weight


# 테스트 코드
if __name__ == "__main__":
    # 더미 데이터셋 생성
    class DummyDataset:
        def __init__(self):
            self.samples = [
                (Path("img1.jpg"), 0),  # 중립: 100개
                (Path("img2.jpg"), 0),
                *[(Path(f"img{i}.jpg"), 1) for i in range(50)],  # 분노: 50개
                *[(Path(f"img{i}.jpg"), 2) for i in range(30)],  # 불안: 30개
                *[(Path(f"img{i}.jpg"), 3) for i in range(20)],  # 상처: 20개
            ]

    class_names = ["중립", "분노", "불안", "상처"]
    dataset = DummyDataset()

    # 불균형 분석
    balancer = ClassBalancer(dataset, class_names)
    balancer.print_analysis()
    balancer.print_augmentation_strategy()

    # 가중치 확인
    print("\n⚖️  클래스 가중치:")
    weights = balancer.get_class_weights()
    for name, weight in zip(class_names, weights):
        print(f"  {name}: {weight:.4f}")
