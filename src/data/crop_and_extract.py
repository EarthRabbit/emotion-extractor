"""
이미지 크롭핑 및 메타데이터 추출 스크립트

1. 바운딩 박스를 이용해 이미지를 크롭하여 새로 저장
2. 바운딩 박스를 제외한 데이터만 추출해서 새로운 JSON으로 저장
3. annot_A, B, C 중 유효한 바운딩 박스를 가진 것 중 랜덤 선택
4. 멀티쓰레드를 사용하여 빠른 처리
5. EXIF 정보를 기반으로 이미지 자동 회전 처리
"""

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm import tqdm


# 스레드 안전한 카운터
class Counter:
    def __init__(self):
        self.success = 0
        self.fail = 0
        self.lock = threading.Lock()

    def add_success(self):
        with self.lock:
            self.success += 1

    def add_fail(self):
        with self.lock:
            self.fail += 1


def load_json_data(json_path: str) -> list:
    """JSON 파일 로드"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def is_valid_bbox(bbox: dict) -> bool:
    """바운딩 박스가 유효한지 검사"""
    if bbox is None:
        return False

    required_keys = ["minX", "minY", "maxX", "maxY"]
    if not all(key in bbox for key in required_keys):
        return False

    try:
        min_x = float(bbox["minX"])
        min_y = float(bbox["minY"])
        max_x = float(bbox["maxX"])
        max_y = float(bbox["maxY"])

        if min_x < max_x and min_y < max_y:
            return True
        return False

    except (ValueError, TypeError):
        return False


def get_random_valid_bbox(item: dict) -> tuple:
    """유효한 바운딩 박스 중 랜덤으로 하나 선택하여 반환 (bbox, annotator_name)"""
    annotators = ["annot_A", "annot_B", "annot_C"]
    valid_bboxes = []

    for annotator in annotators:
        if annotator in item and "boxes" in item[annotator]:
            bbox = item[annotator]["boxes"]
            if is_valid_bbox(bbox):
                valid_bboxes.append((bbox, annotator))

    if not valid_bboxes:
        return None, None

    return random.choice(valid_bboxes)


def crop_image(image_path: str, bbox: dict, output_path: str) -> bool:
    """
    바운딩 박스를 이용해 이미지 크롭 및 저장

    - EXIF 정보를 기반으로 이미지 자동 회전
    - 좌표 범위 검증 및 클리핑
    - 정규화된 좌표(0-1) 지원
    """
    try:
        with Image.open(image_path) as img:
            # EXIF 정보를 기반으로 이미지 자동 회전 처리
            img = exif_transpose(img)

            img_width, img_height = img.size

            # 바운딩 박스 좌표 추출 (정수 변환)
            minX = float(bbox["minX"])
            minY = float(bbox["minY"])
            maxX = float(bbox["maxX"])
            maxY = float(bbox["maxY"])

            # 정규화된 좌표인 경우 (0-1 범위) → 픽셀 좌표로 변환
            # 만약 모든 좌표가 0-1 사이면 정규화된 좌표로 판단
            if 0 <= minX <= 1 and 0 <= maxX <= 1 and 0 <= minY <= 1 and 0 <= maxY <= 1:
                minX = minX * img_width
                maxX = maxX * img_width
                minY = minY * img_height
                maxY = maxY * img_height

            # 정수로 변환
            left = max(0, int(minX))
            top = max(0, int(minY))
            right = min(img_width, int(maxX))
            bottom = min(img_height, int(maxY))

            # 유효성 검사
            if left >= right or top >= bottom:
                print(f"⚠️ 경고: 잘못된 바운딩 박스 좌표 - {image_path}")
                print(
                    f"   원본 좌표: minX={minX}, minY={minY}, maxX={maxX}, maxY={maxY}"
                )
                print(f"   이미지 크기: {img_width}x{img_height}")
                return False

            # 크롭 및 저장
            cropped = img.crop((left, top, right, bottom))
            cropped.save(output_path)
            return True
    except Exception as e:
        print(f"❌ 크롭 오류 ({image_path}): {e}")
        return False


def process_single_item(args: tuple) -> tuple:
    """단일 이미지 항목 처리 (멀티스레드용)"""
    item, source_folder, output_image_folder = args

    filename = item.get("filename")
    if not filename:
        return None, False

    # 원본 이미지 경로
    source_image_path = source_folder / filename

    # 유효한 바운딩 박스 중 랜덤 선택
    bbox, _ = get_random_valid_bbox(item)
    if not bbox:
        return item, False

    # 크롭된 이미지 저장 경로
    output_image_path = output_image_folder / filename

    # 이미지 크롭
    success = False
    if source_image_path.exists():
        success = crop_image(str(source_image_path), bbox, str(output_image_path))

    return item, success


def process_emotion_folder(
    label_folder: Path,
    source_folder: Path,
    output_image_folder: Path,
    output_json_path: Path,
    num_workers: int = 8,
):
    """하나의 감정 폴더 처리 (멀티스레드)"""

    # JSON 파일 찾기
    json_files = list(label_folder.glob("*.json"))
    if not json_files:
        print(f"JSON 파일을 찾을 수 없음: {label_folder}")
        return

    json_path = json_files[0]
    print(f"\n처리 중: {json_path.name}")

    # 데이터 로드
    data = load_json_data(str(json_path))
    print(f"총 {len(data)}개의 데이터 항목 (스레드: {num_workers})")

    # 출력 폴더 생성
    output_image_folder.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # 처리된 데이터 저장 리스트
    processed_data = []
    success_count = 0
    fail_count = 0

    # 작업 인자 준비
    work_items = [(item, source_folder, output_image_folder) for item in data]

    # 멀티스레드로 처리
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_item, args): args for args in work_items
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="이미지 크롭 중"
        ):
            try:
                new_item, success = future.result()
                if new_item:
                    processed_data.append(new_item)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception:
                fail_count += 1

    # 새 JSON 저장
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"완료 - 성공: {success_count}, 실패: {fail_count}")
    print(f"JSON 저장: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="이미지 크롭 및 메타데이터 추출")
    parser.add_argument(
        "--data-root",
        type=str,
        default=r"D:\한국인 감정인식을 위한 복합 영상\Validation",
        help="데이터 루트 경로",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=r"d:\emotion-extractor\Data\cropped",
        help="출력 루트 경로",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="사용할 스레드 수 (기본: 8)"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    # 감정 목록
    emotions = ["기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립"]

    for emotion in emotions:
        # 라벨 폴더 (JSON 파일 위치)
        label_folder = data_root / f"[라벨]EMOIMG_{emotion}_VALID"
        # 원천 폴더 (원본 이미지 위치)
        source_folder = data_root / f"[원천]EMOIMG_{emotion}_VALID"

        # 출력 경로
        output_image_folder = output_root / "images" / emotion
        output_json_path = output_root / "labels" / f"{emotion}_metadata.json"

        if label_folder.exists() and source_folder.exists():
            process_emotion_folder(
                label_folder=label_folder,
                source_folder=source_folder,
                output_image_folder=output_image_folder,
                output_json_path=output_json_path,
                num_workers=args.workers,
            )
        else:
            print(f"폴더를 찾을 수 없음: {emotion}")
            if not label_folder.exists():
                print(f"  - 라벨 폴더 없음: {label_folder}")
            if not source_folder.exists():
                print(f"  - 원천 폴더 없음: {source_folder}")


if __name__ == "__main__":
    main()
