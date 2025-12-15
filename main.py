import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ============================================================
# 설정
# ============================================================
DATA_DIR = Path(__file__).parent / "Data"
NUM_CLASSES = 5
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 클래스 이름 매핑 (폴더명 -> 표시명)
CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Surprise"]

# TensorBoard 로그 디렉토리
LOG_DIR = Path(__file__).parent / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# 데이터 전처리
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(15),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# 데이터셋 로드 및 분할
# ============================================================
def load_datasets():
    """데이터셋을 로드하고 train/validation으로 분할"""
    # ImageFolder로 전체 데이터셋 로드
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    
    # 클래스 이름 출력 (폴더명 기준)
    print(f"감지된 클래스: {full_dataset.classes}")
    print(f"클래스 매핑: {full_dataset.class_to_idx}")
    
    # Train/Validation 분할 (80:20)
    total_size = len(full_dataset)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Validation 데이터셋에는 augmentation 없는 transform 적용
    # (random_split은 원본 transform을 유지하므로 별도 처리 필요시 커스텀 필요)
    
    print(f"전체 데이터: {total_size}개")
    print(f"학습 데이터: {train_size}개 ({TRAIN_RATIO*100:.0f}%)")
    print(f"검증 데이터: {val_size}개 ({(1-TRAIN_RATIO)*100:.0f}%)")
    
    return train_dataset, val_dataset, full_dataset.class_to_idx

def create_dataloaders(train_dataset, val_dataset):
    """DataLoader 생성"""
    # GPU 사용 시 worker 설정
    use_cuda = torch.cuda.is_available()
    num_workers = 4 if use_cuda else 0
    prefetch_factor = 2 if num_workers > 0 else None
    persistent_workers = True if num_workers > 0 else False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=use_cuda,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=use_cuda,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )
    
    if use_cuda:
        print(f"DataLoader: {num_workers} workers, prefetch_factor={prefetch_factor}, pin_memory=True")
    
    return train_loader, val_loader

# ============================================================
# 모델 정의
# ============================================================
def create_model():
    """ResNet-18 모델 생성 (Fine-tuning용)"""
    # 사전학습된 ResNet-18 로드
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Fine-tuning: 모든 레이어 학습 가능하게 설정
    for param in model.parameters():
        param.requires_grad = True
    
    # 마지막 FC 레이어를 5개 클래스로 수정
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    
    model = model.to(DEVICE)
    print(f"모델이 {DEVICE}에 로드되었습니다.")
    print(f"총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"학습 가능 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model

# ============================================================
# 학습 함수
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer, writer=None, epoch=0):
    """1 에폭 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Global step 계산용
    steps_per_epoch = len(train_loader)
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        # non_blocking=True로 비동기 GPU 전송 (배치 미리 로드)
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 현재 배치의 accuracy
        batch_acc = 100. * correct / total
        
        # TensorBoard에 step 단위로 기록
        if writer is not None:
            global_step = epoch * steps_per_epoch + batch_idx
            writer.add_scalar("Step/Train_Loss", loss.item(), global_step)
            writer.add_scalar("Step/Train_Accuracy", batch_acc, global_step)
        
        # tqdm 진행률 바 업데이트
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{batch_acc:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, writer=None, epoch=0):
    """검증"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Global step 계산용
    steps_per_epoch = len(val_loader)
    
    pbar = tqdm(val_loader, desc="Validating")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            # non_blocking=True로 비동기 GPU 전송
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 현재 배치의 accuracy
            batch_acc = 100. * correct / total
            
            # TensorBoard에 step 단위로 기록
            if writer is not None:
                global_step = epoch * steps_per_epoch + batch_idx
                writer.add_scalar("Step/Val_Loss", loss.item(), global_step)
                writer.add_scalar("Step/Val_Accuracy", batch_acc, global_step)
            
            # tqdm 진행률 바 업데이트
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_acc:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def train(model, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    """전체 학습 루프"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_val_acc = 0.0
    
    # TensorBoard writer 초기화
    writer = SummaryWriter(log_dir=LOG_DIR)
    print(f"TensorBoard 로그 저장 경로: {LOG_DIR}")
    
    # 하이퍼파라미터 기록
    writer.add_text("Hyperparameters", 
        f"Batch Size: {BATCH_SIZE}\n"
        f"Learning Rate: {LEARNING_RATE}\n"
        f"Epochs: {NUM_EPOCHS}\n"
        f"Train Ratio: {TRAIN_RATIO}\n"
        f"Device: {DEVICE}")
    
    print("\n" + "="*60)
    print("학습 시작")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 40)
        
        # 학습 (writer와 epoch 전달)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, writer, epoch)
        
        # 검증 (writer와 epoch 전달)
        val_loss, val_acc = validate(model, val_loader, criterion, writer, epoch)
        
        # Learning rate 스케줄러 업데이트
        scheduler.step()
        
        # TensorBoard에 로그 기록
        writer.add_scalars("Loss", {
            "train": train_loss,
            "val": val_loss
        }, epoch + 1)
        
        writer.add_scalars("Accuracy", {
            "train": train_acc,
            "val": val_acc
        }, epoch + 1)
        
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch + 1)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, "best_model.pth")
            print(f"최고 성능 모델 저장됨 (Val Acc: {val_acc:.2f}%)")
    
    # TensorBoard writer 종료
    writer.close()
    
    print("\n" + "="*60)
    print(f"학습 완료! 최고 검증 정확도: {best_val_acc:.2f}%")
    print(f"TensorBoard 실행: tensorboard --logdir={LOG_DIR.parent}")
    print("="*60)
    
    return model

# ============================================================
# 모델 저장/로드
# ============================================================
def save_model(model, filename="emotion_model.pth"):
    """모델 저장"""
    save_path = Path(__file__).parent / filename
    torch.save(model.state_dict(), save_path)
    print(f"모델 저장됨: {save_path}")

def load_model(filename="emotion_model.pth"):
    """모델 로드"""
    model = create_model()
    load_path = Path(__file__).parent / filename
    model.load_state_dict(torch.load(load_path, map_location=DEVICE))
    model.eval()
    print(f"모델 로드됨: {load_path}")
    return model

# ============================================================
# 추론 함수
# ============================================================
def predict(model, image_path, class_to_idx=None):
    """단일 이미지 감정 예측"""
    model.eval()
    
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0).to(DEVICE)
    
    # 예측
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item() * 100
    
    print(f"이미지: {image_path}")
    print(f"예측 감정: {predicted_class}")
    print(f"신뢰도: {confidence_score:.2f}%")
    print("\n클래스별 확률:")
    for i, class_name in enumerate(CLASS_NAMES):
        prob = probabilities[0][i].item() * 100
        print(f"  {class_name}: {prob:.2f}%")
    
    return predicted_class, confidence_score

# ============================================================
# 메인 실행
# ============================================================
def main():
    print("="*60)
    print("ResNet-18 감정 분류 모델 (Fine-tuning)")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"클래스: {CLASS_NAMES}")
    print(f"배치 크기: {BATCH_SIZE}")
    print(f"에폭 수: {NUM_EPOCHS}")
    print(f"학습률: {LEARNING_RATE}")
    print(f"Train/Val 비율: {TRAIN_RATIO*100:.0f}:{(1-TRAIN_RATIO)*100:.0f}")
    
    # 데이터 로드
    print("\n[1/3] 데이터 로드 중...")
    train_dataset, val_dataset, class_to_idx = load_datasets()
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)
    
    # 모델 생성
    print("\n[2/3] 모델 생성 중...")
    model = create_model()
    
    # 학습
    print("\n[3/3] 학습 시작...")
    model = train(model, train_loader, val_loader)
    
    # 최종 모델 저장
    save_model(model, "final_model.pth")
    
    return model

if __name__ == "__main__":
    main() 