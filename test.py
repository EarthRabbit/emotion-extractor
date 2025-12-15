import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from pathlib import Path
import argparse

# ============================================================
# 설정
# ============================================================
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Surprise"]

# 이미지 전처리 (학습 시 val_transform과 동일)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# 모델 로드
# ============================================================
def load_model(model_path: str):
    """학습된 모델 로드"""
    # ResNet-18 구조 생성
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    
    # 학습된 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    print(f"모델 로드 완료: {model_path}")
    print(f"Device: {DEVICE}")
    
    return model

# ============================================================
# 추론 함수
# ============================================================
def predict(model, image_path: str) -> dict:
    """단일 이미지 감정 예측"""
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert("L").convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # 추론
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item() * 100
    
    # 결과 딕셔너리 생성
    result = {
        "predicted_class": predicted_class,
        "confidence": confidence_score,
        "probabilities": {
            name: prob.item() * 100 
            for name, prob in zip(CLASS_NAMES, probabilities[0])
        }
    }
    
    return result

def print_result(image_path: str, result: dict):
    """결과 출력"""
    print("\n" + "="*50)
    print(f"이미지: {image_path}")
    print("="*50)
    print(f"예측 감정: {result['predicted_class']}")
    print(f"신뢰도: {result['confidence']:.2f}%")
    print("\n클래스별 확률:")
    for name, prob in result['probabilities'].items():
        bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
        print(f"  {name:10s} [{bar}] {prob:6.2f}%")
    print("="*50)

# ============================================================
# 메인 실행
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="감정 분류 추론")
    parser.add_argument("image", type=str, help="분류할 이미지 경로")
    parser.add_argument(
        "--model", 
        type=str, 
        default="best_model.pth",
        help="모델 파일 경로 (기본값: best_model.pth)"
    )
    args = parser.parse_args()
    
    # 경로 처리
    model_path = Path(__file__).parent / args.model
    image_path = Path(args.image)
    
    if not model_path.exists():
        print(f"오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    if not image_path.exists():
        print(f"오류: 이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    # 모델 로드
    model = load_model(str(model_path))
    
    # 추론
    result = predict(model, str(image_path))
    
    # 결과 출력
    print_result(str(image_path), result)

if __name__ == "__main__":
    main()