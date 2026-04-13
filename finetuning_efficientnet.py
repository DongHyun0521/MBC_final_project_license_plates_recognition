import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import json
from tqdm import tqdm  # 💡 진행률 바 라이브러리 추가!

def get_model(model_name, num_classes):
    """이름에 따라 구조가 다른 모델의 마지막 출력층을 맞춰주는 함수"""
    if model_name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    
    elif model_name == 'mobilenet_v3':
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    
    elif model_name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    
    return m

def run_classification_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = './country_classification_data_v1'
    batch_size = 32
    num_epochs = 50  
    patience = 7     

    # 1. 데이터로더 준비
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}
    num_classes = len(image_datasets['train'].classes)

    experiments = ['efficientnet_b0', 'mobilenet_v3', 'resnet18']

    print("🚀 국가 판별 3대장 모델 비교 실험을 시작합니다...\n")

    for exp_name in experiments:
        print("\n" + "="*60)
        print(f"🔥 [실험 시작] 모델: {exp_name.upper()}")
        print("="*60)

        model = get_model(exp_name, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_acc = 0.0
        early_stop_counter = 0

        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }

        for epoch in range(num_epochs):
            print(f"\n[Epoch {epoch+1}/{num_epochs}]") # 에폭 시작 알림
            
            for phase in ['train', 'val']:
                if phase == 'train': model.train()
                else: model.eval()

                running_loss = 0.0
                running_corrects = 0

                # 💡 tqdm 적용: dataloaders[phase]를 tqdm으로 감쌉니다!
                # desc를 통해 바 앞에 'Train'인지 'Val'인지 표시합니다.
                pbar = tqdm(dataloaders[phase], desc=f"  {phase.capitalize():>5}", leave=False)

                for inputs, labels in pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # 💡 진행률 바 오른쪽에 현재 배치까지의 임시 Loss 보여주기
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])
                
                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.item())
                    # Train 끝난 후 깔끔하게 결과 출력
                    print(f"  👉 Train 결과 | Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc.item())
                    # Val 끝난 후 결과 출력 및 모델 저장 확인
                    print(f"  👉 Val   결과 | Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}", end="")

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), f'best_{exp_name}.pt')
                        early_stop_counter = 0
                        print(f"  🌟 최고 기록 갱신! (저장완료)")
                    else:
                        early_stop_counter += 1
                        print() # 줄바꿈

            if early_stop_counter >= patience:
                print(f"\n🛑 [조기 종료 발동] {patience}번 동안 개선이 없었습니다. (최고 정확도: {best_acc:.4f})")
                break
        
        with open(f'history_{exp_name}.json', 'w') as f:
            json.dump(history, f)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss', marker='.')
        plt.plot(history['val_loss'], label='Val Loss', marker='.')
        plt.title(f'{exp_name} - Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc', marker='.')
        plt.plot(history['val_acc'], label='Val Acc', marker='.')
        plt.title(f'{exp_name} - Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'plot_{exp_name}.png')
        plt.close()

        print(f"\n✅ [{exp_name}] 실험 종료! 그래프 생성 완료. 최고 정확도: {best_acc:.4f}\n")

if __name__ == '__main__':
    run_classification_experiments()