"""
한국 전기차 번호판 판별 모델 파인튜닝 (2-class binary classification)

데이터 구조:
  korea_ev/
    train/
      ev_true/   (전기차 번호판 - 하늘색 배경)
      ev_false/  (일반 번호판 - 흰색/노란색 배경)
    val/
      ev_true/
      ev_false/

클래스 불균형이 극심하므로 (ev_true:ev_false = 1:250) WeightedRandomSampler로
배치마다 1:1 균형을 유지합니다.

실험 모델: EfficientNet-B0, MobileNet-V3, ResNet18
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def get_model(model_name: str, num_classes: int):
    """모델 이름에 따라 마지막 층을 binary 분류용으로 교체"""
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


def make_balanced_sampler(dataset):
    """클래스 불균형 해결: 각 샘플 가중치를 1/(클래스 샘플 수)로 설정.
    소수 클래스(ev_true) 샘플이 다수 클래스와 동일 확률로 뽑힘."""
    targets = [label for _, label in dataset.samples]
    class_counts = [targets.count(i) for i in range(len(dataset.classes))]
    print(f"  클래스별 샘플 수: {dict(zip(dataset.classes, class_counts))}")

    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[label] for label in targets]

    # 한 epoch당 소수 클래스의 4배 크기만큼 샘플링 (속도/다양성 균형)
    num_samples = min(class_counts) * 4
    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)


def compute_per_class_metrics(model, dataloader, device, num_classes=2, return_confusion=False):
    """클래스별 정확도, Precision, Recall, F1 계산 (불균형 데이터용).
    return_confusion=True면 혼동 행렬도 반환."""
    model.eval()
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    confusion = np.zeros((num_classes, num_classes), dtype=int)  # [실제][예측]

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="  평가", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            _, preds = torch.max(model(inputs), 1)

            for c in range(num_classes):
                mask_label = (labels == c)
                correct_per_class[c] += (preds[mask_label] == c).sum().item()
                total_per_class[c] += mask_label.sum().item()
                tp[c] += ((preds == c) & (labels == c)).sum().item()
                fp[c] += ((preds == c) & (labels != c)).sum().item()
                fn[c] += ((preds != c) & (labels == c)).sum().item()

            for l, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[l][p] += 1

    metrics = {}
    for c in range(num_classes):
        acc = correct_per_class[c] / max(total_per_class[c], 1)
        prec = tp[c] / max(tp[c] + fp[c], 1)
        rec = tp[c] / max(tp[c] + fn[c], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        metrics[c] = {'acc': acc, 'precision': prec, 'recall': rec, 'f1': f1}

    # balanced accuracy = 클래스별 recall 평균 (불균형 데이터 표준 지표)
    metrics['balanced_acc'] = sum(metrics[c]['recall'] for c in range(num_classes)) / num_classes

    if return_confusion:
        return metrics, confusion
    return metrics


def plot_confusion_matrix(confusion, class_names, save_path, title='Confusion Matrix'):
    """혼동 행렬 시각화 (발표용)"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap='Blues')

    # 각 셀에 숫자 표시
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            color = 'white' if confusion[i][j] > confusion.max() / 2 else 'black'
            ax.text(j, i, str(confusion[i][j]), ha='center', va='center',
                    color=color, fontsize=14, fontweight='bold')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_model_comparison(summary_dict, save_path):
    """3개 모델 성능 비교 막대차트 (발표용)"""
    models_list = list(summary_dict.keys())
    metrics_to_plot = ['balanced_acc', 'ev_recall', 'ev_precision', 'ev_f1']
    metric_labels = ['Balanced Acc', 'EV Recall', 'EV Precision', 'EV F1']

    x = np.arange(len(models_list))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (m_key, m_label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [summary_dict[model][m_key] for model in models_list]
        bars = ax.bar(x + i * width, values, width, label=m_label)
        # 막대 위에 수치 표시
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('EV Classifier - Model Comparison', fontsize=13)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models_list)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def save_summary_table(summary_dict, save_path):
    """3개 모델 최종 성능 요약 (텍스트 표)"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 95 + "\n")
        f.write("EV Classifier 파인튜닝 최종 결과\n")
        f.write("=" * 95 + "\n\n")
        f.write(f"{'Model':<20} {'Balanced Acc':<14} {'EV Recall':<12} {'EV Precision':<14} "
                f"{'EV F1':<10} {'학습시간':<12}\n")
        f.write("-" * 95 + "\n")
        for model, m in summary_dict.items():
            elapsed_str = format_elapsed(m.get('train_seconds', 0))
            f.write(f"{model:<20} {m['balanced_acc']:<14.4f} {m['ev_recall']:<12.4f} "
                    f"{m['ev_precision']:<14.4f} {m['ev_f1']:<10.4f} {elapsed_str:<12}\n")
        f.write("=" * 95 + "\n")


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
        print(f"  → 'pip install onnx onnxruntime' 후 재시도하세요.")
        return False
    except Exception as e:
        print(f"  [{model_name}] ONNX 변환 중 에러: {e}")
        return False


def run_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")

    data_dir = './korea_ev'
    batch_size = 32
    num_epochs = 30
    patience = 5

    # ── 데이터 전처리 ──
    # train: 데이터 증강 적용 (하늘색 보존 위해 hue는 아주 약하게)
    # val: 리사이즈만
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
    print(f"클래스: {image_datasets['train'].classes} (총 {num_classes}개)")

    # ── DataLoader: train은 균형 샘플링, val은 순차 ──
    print("\n[train] 균형 샘플링 준비")
    train_sampler = make_balanced_sampler(image_datasets['train'])

    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0
        ),
        'val': DataLoader(
            image_datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        ),
    }

    # 실험할 모델들
    experiments = ['efficientnet_b0', 'mobilenet_v3', 'resnet18']

    # 3개 모델 최종 성능 저장 (발표용 비교표)
    final_summary = {}

    # 전체 실험 시작 시각 기록 (모든 모델 학습 + 변환 + 비교 차트까지 포함)
    overall_start = time.time()

    print("\n" + "=" * 60)
    print("전기차 여부 판별 모델 파인튜닝 시작")
    print("=" * 60)

    for exp_name in experiments:
        print("\n" + "=" * 60)
        print(f"[실험 시작] 모델: {exp_name.upper()}")
        print("=" * 60)

        # 모델별 학습 시작 시각 (epoch 루프 + 베스트 평가까지 포함)
        model_start = time.time()

        model = get_model(exp_name, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_balanced_acc = 0.0
        early_stop_counter = 0

        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_balanced_acc': [],
            'val_ev_recall': [],    # EV 탐지율 (진짜 중요한 지표)
            'val_ev_precision': [],
            'epoch_seconds': [],    # epoch별 소요 시간(초)
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
            val_corrects = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in tqdm(dataloaders['val'], desc="  Val  ", leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data).item()
                    val_total += inputs.size(0)

            val_loss /= val_total
            val_acc = val_corrects / val_total

            # 클래스별 상세 지표 계산
            metrics = compute_per_class_metrics(model, dataloaders['val'], device, num_classes)
            balanced_acc = metrics['balanced_acc']

            # ev_true는 클래스 인덱스 1 (알파벳 순서: ev_false=0, ev_true=1)
            ev_true_idx = image_datasets['val'].classes.index('ev_true')
            ev_recall = metrics[ev_true_idx]['recall']
            ev_precision = metrics[ev_true_idx]['precision']
            ev_f1 = metrics[ev_true_idx]['f1']

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_balanced_acc'].append(balanced_acc)
            history['val_ev_recall'].append(ev_recall)
            history['val_ev_precision'].append(ev_precision)

            print(f"  Val   | Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  Val   | Balanced Acc: {balanced_acc:.4f}")
            print(f"  Val   | EV Recall: {ev_recall:.4f}, Precision: {ev_precision:.4f}, F1: {ev_f1:.4f}")

            # epoch 종료 시각 기록 (Train + Val + 클래스별 평가 모두 포함)
            epoch_elapsed = time.time() - epoch_start
            history['epoch_seconds'].append(epoch_elapsed)
            print(f"  Epoch 시간: {format_elapsed(epoch_elapsed)} ({epoch_elapsed:.1f}s)")

            # Balanced Accuracy 기준으로 베스트 모델 저장 (정확도는 불균형 데이터에 왜곡됨)
            if balanced_acc > best_balanced_acc:
                best_balanced_acc = balanced_acc
                torch.save(model.state_dict(), f'best_ev_{exp_name}.pt')
                early_stop_counter = 0
                print(f"  최고 기록 갱신! (Balanced Acc: {balanced_acc:.4f}) → 저장 완료")
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"\n[조기 종료] {patience} epoch 동안 개선 없음. 최고 Balanced Acc: {best_balanced_acc:.4f}")
                break

        # ── 모델별 학습 종료 시각 기록 (epoch 루프 끝난 직후) ──
        train_elapsed = time.time() - model_start
        history['total_train_seconds'] = train_elapsed
        print(f"\n[{exp_name}] 학습 소요 시간: {format_elapsed(train_elapsed)} ({train_elapsed:.1f}s)")

        # ── 학습 이력 저장 ──
        with open(f'history_ev_{exp_name}.json', 'w') as f:
            json.dump(history, f, indent=2)

        # ── 그래프 저장 ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(history['train_loss'], label='Train', marker='.')
        axes[0].plot(history['val_loss'], label='Val', marker='.')
        axes[0].set_title(f'{exp_name} - Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].legend()

        axes[1].plot(history['train_acc'], label='Train Acc', marker='.')
        axes[1].plot(history['val_acc'], label='Val Acc', marker='.')
        axes[1].plot(history['val_balanced_acc'], label='Val Balanced Acc', marker='.')
        axes[1].set_title(f'{exp_name} - Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend()

        axes[2].plot(history['val_ev_recall'], label='EV Recall', marker='.', color='tab:red')
        axes[2].plot(history['val_ev_precision'], label='EV Precision', marker='.', color='tab:green')
        axes[2].set_title(f'{exp_name} - EV Detection Quality')
        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('Score')
        axes[2].grid(True, linestyle='--', alpha=0.5)
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(f'plot_ev_{exp_name}.png')
        plt.close()

        print(f"\n[{exp_name}] 완료 | 최고 Balanced Acc: {best_balanced_acc:.4f}")

        # ── 베스트 모델 로드 후 최종 평가 (혼동 행렬 생성) ──
        print(f"  [{exp_name}] 베스트 모델 로드해서 최종 평가 중...")
        model.load_state_dict(torch.load(f'best_ev_{exp_name}.pt'))
        final_metrics, confusion = compute_per_class_metrics(
            model, dataloaders['val'], device, num_classes, return_confusion=True
        )

        class_names = image_datasets['val'].classes
        ev_true_idx = class_names.index('ev_true')

        # 혼동 행렬 이미지 저장
        plot_confusion_matrix(
            confusion, class_names,
            f'confusion_ev_{exp_name}.png',
            title=f'{exp_name} - Confusion Matrix'
        )

        # ── ONNX 변환 (베스트 모델이 이미 model에 로드된 상태) ──
        onnx_path = f'best_ev_{exp_name}.onnx'
        export_to_onnx(model, exp_name, onnx_path, device)

        # 최종 성능 기록 (3개 모델 비교용)
        final_summary[exp_name] = {
            'balanced_acc': final_metrics['balanced_acc'],
            'ev_recall': final_metrics[ev_true_idx]['recall'],
            'ev_precision': final_metrics[ev_true_idx]['precision'],
            'ev_f1': final_metrics[ev_true_idx]['f1'],
            'confusion': confusion.tolist(),
            'train_seconds': train_elapsed,                # 모델 학습 소요(초)
            'train_time_str': format_elapsed(train_elapsed),  # 사람이 읽기 쉬운 형식
            'epochs_run': len(history['train_loss']),      # 실제 돌린 epoch 수 (조기종료 반영)
        }

        print(f"  [{exp_name}] 혼동 행렬 저장: confusion_ev_{exp_name}.png")

    # ═══ 3개 모델 비교 (발표용) ═══
    print("\n" + "=" * 60)
    print("전체 모델 비교 결과 생성")
    print("=" * 60)

    # 비교 막대차트
    plot_model_comparison(final_summary, 'comparison_ev_models.png')
    print("  모델 비교 차트 저장: comparison_ev_models.png")

    # 요약 표 저장
    save_summary_table(final_summary, 'summary_ev_models.txt')
    print("  최종 성능 요약 저장: summary_ev_models.txt")

    # 콘솔에 최종 결과 출력
    print("\n=== 최종 결과 ===")
    print(f"{'Model':<20} {'Balanced Acc':<14} {'EV Recall':<12} {'EV Precision':<14} "
          f"{'EV F1':<10} {'학습시간':<12}")
    print("-" * 90)
    for model_name, m in final_summary.items():
        print(f"{model_name:<20} {m['balanced_acc']:<14.4f} {m['ev_recall']:<12.4f} "
              f"{m['ev_precision']:<14.4f} {m['ev_f1']:<10.4f} {m['train_time_str']:<12}")

    # 최고 모델 선정
    best_model = max(final_summary.items(), key=lambda x: x[1]['balanced_acc'])
    print(f"\n최고 성능 모델: {best_model[0]} (Balanced Acc: {best_model[1]['balanced_acc']:.4f})")

    # 전체 실험 소요 시간 출력 + summary JSON에 모든 정보 한 번에 저장
    overall_elapsed = time.time() - overall_start
    print(f"\n전체 실험 소요 시간: {format_elapsed(overall_elapsed)} ({overall_elapsed:.1f}s)")

    # _meta 키에 전체 시간 / 모델별 시간 모음을 저장 (다른 분석/리포트에 활용 가능)
    final_summary['_meta'] = {
        'total_seconds': overall_elapsed,
        'total_time_str': format_elapsed(overall_elapsed),
        'per_model_seconds': {k: v['train_seconds'] for k, v in final_summary.items()
                              if k != '_meta'},
    }
    with open('summary_ev_models.json', 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2)
    print("  전체 결과 JSON 저장: summary_ev_models.json (시간 정보 포함)")


if __name__ == '__main__':
    run_experiments()
