"""
바운딩 박스 검증 및 이미지 크롭 테스트 디버그 스크립트

사용법:
    python debug_crop.py --image "path/to/image.jpg" --bbox '{"minX": 892.6, "minY": 675.8, "maxX": 1810.2, "maxY": 1845.6}'
"""

import argparse
import json
from pathlib import Path

from PIL import Image
from PIL.ImageOps import exif_transpose


def load_image_with_exif(image_path: str) -> Image.Image:
    """EXIF 정보를 기반으로 이미지 로드 및 자동 회전"""
    with Image.open(image_path) as img:
        img = exif_transpose(img)
        return img.copy()


def debug_bbox_and_crop(image_path: str, bbox: dict, output_path: str = None):
    """바운딩 박스 검증 및 크롭 테스트"""

    print("=" * 80)
    print("🔍 바운딩 박스 검증 및 크롭 테스트")
    print("=" * 80)

    # 이미지 로드
    print(f"\n1️⃣  이미지 로드")
    print(f"   경로: {image_path}")

    if not Path(image_path).exists():
        print(f"   ❌ 파일을 찾을 수 없습니다!")
        return False

    try:
        img = load_image_with_exif(image_path)
        img_width, img_height = img.size
        print(f"   ✅ 로드 성공 (크기: {img_width}x{img_height})")
    except Exception as e:
        print(f"   ❌ 로드 실패: {e}")
        return False

    # EXIF 정보 확인
    print(f"\n2️⃣  EXIF 정보")
    try:
        with Image.open(image_path) as img_original:
            if hasattr(img_original, "_getexif") and img_original._getexif():
                exif = img_original._getexif()
                if exif:
                    orientation_tag = 274  # EXIF Orientation tag
                    if orientation_tag in exif:
                        orientation = exif[orientation_tag]
                        orientation_map = {
                            1: "일반 (정상)",
                            2: "좌우 반전",
                            3: "180도 회전",
                            4: "좌우 반전 + 180도",
                            5: "시계 반대 90도 + 좌우 반전",
                            6: "시계 90도 회전",
                            7: "시계 90도 + 좌우 반전",
                            8: "시계 반대 90도 회전",
                        }
                        print(
                            f"   Orientation: {orientation} ({orientation_map.get(orientation, '알 수 없음')})"
                        )
                    else:
                        print(f"   Orientation: 없음 (정상)")
                else:
                    print(f"   EXIF 정보: 없음")
            else:
                print(f"   EXIF 정보: 없음")
    except Exception as e:
        print(f"   ⚠️  EXIF 읽기 실패: {e}")

    # 바운딩 박스 정보
    print(f"\n3️⃣  바운딩 박스 좌표")
    minX = float(bbox.get("minX", 0))
    minY = float(bbox.get("minY", 0))
    maxX = float(bbox.get("maxX", 0))
    maxY = float(bbox.get("maxY", 0))

    print(f"   minX: {minX}")
    print(f"   minY: {minY}")
    print(f"   maxX: {maxX}")
    print(f"   maxY: {maxY}")

    # 좌표 검증
    print(f"\n4️⃣  좌표 검증")

    # 정규화 여부 확인
    is_normalized = (
        0 <= minX <= 1 and 0 <= maxX <= 1 and 0 <= minY <= 1 and 0 <= maxY <= 1
    )
    print(f"   정규화된 좌표 (0-1 범위)? {is_normalized}")

    if is_normalized:
        print(f"   ✅ 정규화된 좌표 감지 - 픽셀 좌표로 변환")
        minX_px = minX * img_width
        minY_px = minY * img_height
        maxX_px = maxX * img_width
        maxY_px = maxY * img_height
    else:
        minX_px = minX
        minY_px = minY
        maxX_px = maxX
        maxY_px = maxY
        print(f"   ℹ️  픽셀 좌표로 해석")

    print(f"\n   변환 후 좌표:")
    print(f"   left: {minX_px:.2f} → {int(minX_px)}")
    print(f"   top: {minY_px:.2f} → {int(minY_px)}")
    print(f"   right: {maxX_px:.2f} → {int(maxX_px)}")
    print(f"   bottom: {maxY_px:.2f} → {int(maxY_px)}")

    # 클리핑
    left = max(0, int(minX_px))
    top = max(0, int(minY_px))
    right = min(img_width, int(maxX_px))
    bottom = min(img_height, int(maxY_px))

    print(f"\n   클리핑 후 좌표:")
    print(f"   left: {left}, top: {top}, right: {right}, bottom: {bottom}")

    # 유효성 검사
    print(f"\n5️⃣  유효성 검사")
    width = right - left
    height = bottom - top

    print(f"   너비: {width}px")
    print(f"   높이: {height}px")
    print(f"   면적: {width * height}px²")

    if left >= right or top >= bottom:
        print(f"   ❌ 유효하지 않은 바운딩 박스!")
        return False

    if width < 10 or height < 10:
        print(f"   ⚠️  경고: 너무 작은 크롭 영역입니다.")

    print(f"   ✅ 유효한 바운딩 박스")

    # 크롭
    print(f"\n6️⃣  이미지 크롭")
    try:
        cropped = img.crop((left, top, right, bottom))
        print(f"   ✅ 크롭 성공 (크기: {cropped.size[0]}x{cropped.size[1]})")

        # 저장
        if output_path:
            cropped.save(output_path)
            print(f"   ✅ 저장 성공: {output_path}")
            return True
        else:
            return cropped
    except Exception as e:
        print(f"   ❌ 크롭 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="바운딩 박스 검증 및 이미지 크롭 테스트"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="검증할 이미지 경로",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        required=True,
        help='바운딩 박스 JSON 문자열 (예: \'{"minX": 100, "minY": 200, "maxX": 300, "maxY": 400}\')',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="크롭된 이미지 저장 경로 (선택사항)",
    )

    args = parser.parse_args()

    # JSON 파싱
    try:
        bbox = json.loads(args.bbox)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        return

    # 디버그 실행
    debug_bbox_and_crop(args.image, bbox, args.output)


if __name__ == "__main__":
    main()
