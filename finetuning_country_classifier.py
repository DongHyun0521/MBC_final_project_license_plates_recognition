"""
국가 번호판 분류 파인튜닝 (5-class 분류: brazil, china, europe, india, korea)

데이터 클래스 불균형이 심하므로 (예: china 248k vs europe 588 → 422배):
  1. WeightedRandomSampler로 batch마다 5개국이 균등하게 등장하도록 샘플링
  2. CrossEntropyLoss에 클래스 가중치를 줘서 minority 오분류에 더 큰 페널티
  3. balanced_acc(클래스별 recall 평균)을 best 판단 기준으로 사용 (전체 정확도는 china에 끌려감)

실험 모델: EfficientNet-B0, MobileNet-V3, ResNet18
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json
from tqdm import tqdm


def get_model(model_name, num_classes):
    """이름에 따라 구조가 다른 모델의 마지막 출력층을 맞춰주는 함수"""
    if model_name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenet_v3':
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    # elif model_name == 'resnet18':
    #     m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    #     m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"알 수 없는 모델 이름: {model_name}")
    return m


def make_balanced_sampler(dataset, samples_per_class_multiplier=20):
    """클래스 불균형 해결: 각 샘플 가중치를 1/(클래스 샘플 수)로 설정.
    소수 클래스도 다수 클래스와 동일 확률로 뽑힘.

    samples_per_class_multiplier: 한 epoch당 num_samples를 min × 이 값으로 설정.
      값이 클수록 minority 한 epoch 안에 여러 번 등장 가능 (속도/다양성 trade-off).
    """
    targets = [label for _, label in dataset.samples]
    class_counts = [targets.count(i) for i in range(len(dataset.classes))]
    print(f"  클래스별 샘플 수: {dict(zip(dataset.classes, class_counts))}")

    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[label] for label in targets]

    num_samples = min(class_counts) * samples_per_class_multiplier * len(class_counts)
    print(f"  → 한 epoch당 {num_samples}장 샘플링 (배치 내 {len(class_counts)}개국 균형)")

    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True), class_counts


def make_class_weights(class_counts, device):
    """sklearn 'balanced' 방식: weight_c = total / (num_classes * count_c).
    minority 클래스에 더 큰 손실 가중치 부여."""
    total = sum(class_counts)
    n_classes = len(class_counts)
    weights = [total / (n_classes * c) for c in class_counts]
    print(f"  Loss 클래스 가중치: {[f'{w:.3f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def compute_balanced_metrics(model, dataloader, device, num_classes):
    """클래스별 recall 평균(balanced_acc) + 일반 정확도 계산.
    불균형 데이터에서 일반 정확도는 다수 클래스에 휘둘리므로 balanced_acc가 신뢰성 높음."""
    model.eval()
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes
    correct_total = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, preds = torch.max(model(inputs), 1)
            correct_total += (preds == labels).sum().item()
            total += labels.size(0)
            for c in range(num_classes):
                mask = (labels == c)
                correct_per_class[c] += (preds[mask] == c).sum().item()
                total_per_class[c] += mask.sum().item()

    per_class_recall = [correct_per_class[c] / max(total_per_class[c], 1)
                        for c in range(num_classes)]
    balanced_acc = sum(per_class_recall) / num_classes
    overall_acc = correct_total / max(total, 1)
    return overall_acc, balanced_acc, per_class_recall


def format_elapsed(seconds: float) -> str:
    """초 단위 시간을 사람이 읽기 쉬운 형식(Hh Mm Ss)으로 변환"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def export_to_onnx(model, model_name, save_path, device, input_size=(1, 3, 224, 224)):
    """학습된 PyTorch 모델을 ONNX 포맷으로 변환.
    onnx 패키지 미설치 등 실패 케이스는 경고만 출력하고 학습 결과는 보존."""
    try:
        model.eval()
        dummy_input = torch.randn(*input_size).to(device)
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        )
        print(f"  [{model_name}] ONNX 변환 완료 → {save_path}")
        return True
    except ModuleNotFoundError as e:
        print(f"  [{model_name}] ONNX 변환 실패 (모듈 누락): {e}")
        print(f"      → 'pip install onnx onnxruntime' 후 재시도하세요.")
        return False
    except Exception as e:
        print(f"  [{model_name}] ONNX 변환 중 에러: {e}")
        return False


def run_classification_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = './country_classification_data_v1'
    batch_size = 32
    num_epochs = 50
    patience = 10

    # ── 데이터 전처리 ──
    # train: 데이터 증강 적용 (minority 클래스가 같은 이미지 반복 노출되어도 다양성 확보)
    #   - hue=0.02로 매우 약하게 (한국 EV 하늘색 등 색상 정보 보존)
    # val: 리사이즈만 (평가 일관성)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # ── 데이터셋 로드 ──
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    num_classes = len(image_datasets['train'].classes)
    class_names = image_datasets['train'].classes
    print(f"클래스: {class_names} (총 {num_classes}개)")

    # ── DataLoader: train은 균형 샘플링, val은 순차 ──
    print("\n[train] 균형 샘플러 설정")
    train_sampler, class_counts = make_balanced_sampler(
        image_datasets['train'],
        samples_per_class_multiplier=20  # min × 5(클래스수) × 20 = epoch당 샘플 수
    )

    print("\n[val] 클래스별 샘플 수 확인")
    val_targets = [label for _, label in image_datasets['val'].samples]
    val_counts = [val_targets.count(i) for i in range(num_classes)]
    print(f"  {dict(zip(class_names, val_counts))}")

    # DataLoader 병렬 처리 설정
    #   num_workers: CPU 코어 수만큼 병렬로 데이터 전처리 (증강이 무거우면 효과 큼)
    #   pin_memory:  GPU 사용 시 CPU→GPU 전송 속도 ↑
    #   persistent_workers: 매 epoch마다 worker 재생성 비용 절감
    use_cuda = torch.cuda.is_available()
    num_workers = 4   # PaddleOCR/다른 프로세스와 CPU 경쟁 시 2로 줄이세요
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'], batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0),
        ),
        'val': DataLoader(
            image_datasets['val'], batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0),
        ),
    }

    # ── Loss 클래스 가중치 (minority 오분류 페널티 증가) ──
    class_weights_tensor = make_class_weights(class_counts, device)

    experiments = [
        'efficientnet_b0',
        'mobilenet_v3',
        # 'resnet18',
    ]

    time_summary = {}
    overall_start = time.time()

    print("\n국가 판별 모델 비교 실험을 시작합니다 (균형 샘플링 + class weights)...\n")

    for exp_name in experiments:
        print("\n" + "=" * 60)
        print(f"[실험 시작] 모델: {exp_name.upper()}")
        print("=" * 60)

        model_start = time.time()

        model = get_model(exp_name, num_classes).to(device)
        # 클래스 가중치를 적용한 CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        # fine-tuning용 낮은 lr (사전학습 가중치 보존). 0.001은 scratch 수준이라 폭주 위험.
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # CosineAnnealingLR: 1e-4 → 1e-6까지 코사인 곡선 따라 부드럽게 감소
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_balanced_acc = 0.0   # 일반 정확도 대신 balanced_acc 기준
        early_stop_counter = 0

        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_balanced_acc': [],     # 클래스별 recall 평균 (불균형 데이터 표준)
            'val_per_class_recall': [], # 클래스별 recall 추적
            'epoch_seconds': [],
            'lr': [],                   # epoch별 학습률 추적 (스케줄러 동작 확인용)
        }

        for epoch in range(num_epochs):
            print(f"\n[Epoch {epoch+1}/{num_epochs}]")
            epoch_start = time.time()

            # ── Train ──
            model.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            pbar = tqdm(dataloaders['train'], desc="  Train", leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total += inputs.size(0)
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            train_loss = running_loss / total
            train_acc = running_corrects / total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            print(f"  Train | Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            # ── Val ──
            model.eval()
            val_loss = 0.0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in tqdm(dataloaders['val'], desc="  Val  ", leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    val_total += inputs.size(0)

            val_loss /= val_total

            # 클래스별 recall + balanced_acc + 전체 acc
            overall_acc, balanced_acc, per_class_recall = compute_balanced_metrics(
                model, dataloaders['val'], device, num_classes
            )

            history['val_loss'].append(val_loss)
            history['val_acc'].append(overall_acc)
            history['val_balanced_acc'].append(balanced_acc)
            history['val_per_class_recall'].append(per_class_recall)

            print(f"  Val   | Loss: {val_loss:.4f}, Overall Acc: {overall_acc:.4f}, "
                  f"Balanced Acc: {balanced_acc:.4f}")
            recall_str = ", ".join(f"{cls}={r:.3f}"
                                   for cls, r in zip(class_names, per_class_recall))
            print(f"        | Recall: {recall_str}")

            # 현재 lr 기록 후 다음 epoch을 위해 스케줄러 step
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            print(f"        | LR: {current_lr:.6f}")
            scheduler.step()

            epoch_elapsed = time.time() - epoch_start
            history['epoch_seconds'].append(epoch_elapsed)
            print(f"  Epoch 시간: {format_elapsed(epoch_elapsed)} ({epoch_elapsed:.1f}s)")

            # balanced_acc 기준으로 베스트 모델 저장
            if balanced_acc > best_balanced_acc:
                best_balanced_acc = balanced_acc
                torch.save(model.state_dict(), f'best_{exp_name}.pt')
                early_stop_counter = 0
                print(f"  최고 기록 갱신! (Balanced Acc: {balanced_acc:.4f}) → 저장 완료")
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"\n[조기 종료 발동] {patience}번 동안 개선 없음. "
                      f"(최고 Balanced Acc: {best_balanced_acc:.4f})")
                break

        # ── 모델별 학습 종료 ──
        train_elapsed = time.time() - model_start
        history['total_train_seconds'] = train_elapsed
        history['epochs_run'] = len(history['train_loss'])
        time_summary[exp_name] = {
            'train_seconds': train_elapsed,
            'train_time_str': format_elapsed(train_elapsed),
            'epochs_run': history['epochs_run'],
            'best_balanced_acc': float(best_balanced_acc),
        }
        print(f"\n[{exp_name}] 학습 소요 시간: {format_elapsed(train_elapsed)} "
              f"({train_elapsed:.1f}s) | {history['epochs_run']} epochs")

        with open(f'history_{exp_name}.json', 'w') as f:
            json.dump(history, f, indent=2)

        # ── 그래프 저장 ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(history['train_loss'], label='Train Loss', marker='.')
        axes[0].plot(history['val_loss'], label='Val Loss', marker='.')
        axes[0].set_title(f'{exp_name} - Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].legend()

        axes[1].plot(history['train_acc'], label='Train Acc', marker='.')
        axes[1].plot(history['val_acc'], label='Val Overall Acc', marker='.')
        axes[1].plot(history['val_balanced_acc'], label='Val Balanced Acc', marker='.', linewidth=2)
        axes[1].set_title(f'{exp_name} - Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend()

        # 클래스별 recall 추이
        per_class_arr = np.array(history['val_per_class_recall'])  # shape (epochs, classes)
        for i, cls in enumerate(class_names):
            axes[2].plot(per_class_arr[:, i], label=cls, marker='.')
        axes[2].set_title(f'{exp_name} - Per-Class Recall')
        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('Recall')
        axes[2].grid(True, linestyle='--', alpha=0.5)
        axes[2].legend(loc='lower right', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'plot_{exp_name}.png')
        plt.close()

        # ── ONNX 변환 ──
        print(f"  [{exp_name}] 베스트 모델 로드해서 ONNX 변환 중...")
        model.load_state_dict(torch.load(f'best_{exp_name}.pt'))
        export_to_onnx(model, exp_name, f'best_{exp_name}.onnx', device)

        print(f"\n[{exp_name}] 실험 종료! 최고 Balanced Acc: {best_balanced_acc:.4f}\n")

    # ═══ 전체 실험 시간 요약 ═══
    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 75)
    print("전체 실험 종료 - 시간 / 성능 요약")
    print("=" * 75)
    print(f"{'Model':<20} {'Best Balanced Acc':<20} {'Epochs':<8} {'학습시간':<15}")
    print("-" * 75)
    for name, info in time_summary.items():
        print(f"{name:<20} {info['best_balanced_acc']:<20.4f} {info['epochs_run']:<8} "
              f"{info['train_time_str']:<15}")
    print("-" * 75)
    print(f"전체 실험 소요 시간: {format_elapsed(overall_elapsed)} ({overall_elapsed:.1f}s)")
    print("=" * 75)

    time_summary['_meta'] = {
        'total_seconds': overall_elapsed,
        'total_time_str': format_elapsed(overall_elapsed),
    }
    with open('time_summary_country_classifier.json', 'w', encoding='utf-8') as f:
        json.dump(time_summary, f, indent=2, ensure_ascii=False)
    print("시간 정보 저장: time_summary_country_classifier.json")


if __name__ == '__main__':
    run_classification_experiments()
