"""
PaddleOCR 5개국 번호판 인식 모델 파인튜닝 스크립트 (GPU 2개 활용)
=================================================================
사전 준비:
  1) conda create -n paddle python=3.10 -y && conda activate paddle
  2) pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
  3) pip install matplotlib pyyaml
  4) git clone https://github.com/PaddlePaddle/PaddleOCR.git
  5) cd PaddleOCR && pip install -r requirements.txt
  6) 이 스크립트를 PaddleOCR, ocr_train_data_v1과 같은 위치에 놓기

폴더 구조:
  parent/
  ├── PaddleOCR/
  │   ├── tools/train.py
  │   ├── tools/export_model.py
  │   └── pretrained_models/       ← 자동 다운로드
  ├── ocr_train_data_v1/
  │   └── brazil/ china/ europe/ india/ korea/
  ├── training_report/             ← 자동 생성 (모든 기록)
  └── finetuning_ocr.py            ← 이 파일

실행:
  python finetuning_ocr.py                     # 전체 파이프라인
  python finetuning_ocr.py --step train        # 학습만
  python finetuning_ocr.py --step analyze      # 그래프/리포트만 재생성
"""
import os
import sys
import re
import json
import yaml
import shutil
import platform
import subprocess
import time
import threading
from datetime import datetime
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("  [!] matplotlib 없음 → pip install matplotlib (그래프 생성 건너뜀)")


# ════════════════════════════════════════════════════════
#  설정
# ════════════════════════════════════════════════════════
PARENT_DIR     = os.path.dirname(os.path.abspath(__file__))
PADDLE_OCR_DIR = os.path.join(PARENT_DIR, 'PaddleOCR')
DATA_DIR       = os.path.join(PARENT_DIR, 'ocr_train_data_v1')
REPORT_DIR     = os.path.join(PARENT_DIR, 'training_report')

COUNTRIES = {
    'korea': (
        'korean_PP-OCRv4_rec_train',
        'korean',
        'https://paddleocr.bj.bcebos.com/PP-OCRv4/korean/korean_PP-OCRv4_rec_train.tar'
    ),
    'china': (
        'ch_PP-OCRv4_rec_train',
        'ch',
        'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar'
    ),
    'brazil': (
        'en_PP-OCRv4_rec_train',
        'en',
        'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar'
    ),
    'europe': (
        'en_PP-OCRv4_rec_train',
        'en',
        'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar'
    ),
    'india': (
        'en_PP-OCRv4_rec_train',
        'en',
        'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar'
    ),
}

# 학습 하이퍼파라미터
EPOCH_NUM      = 100
BATCH_SIZE     = 64
LEARNING_RATE  = 0.0005
WARMUP_EPOCH   = 5
NUM_WORKERS    = 0
EVAL_STEP      = 500

# Early Stopping
EARLY_STOP_PATIENCE = 10   # 평가 N번 연속 개선 없으면 학습 중단 (0 = 비활성화)

# GPU 모니터링
GPU_LOG_INTERVAL = 30       # nvidia-smi 기록 간격 (초)

# GPU 스케줄
GPU_SCHEDULE = [
    [('0', 'korea'), ('1', 'china')],
    [('0', 'brazil'), ('1', 'india')],
    [('0', 'europe')],
]

# 로그 파싱용 정규식
RE_EPOCH    = re.compile(r'epoch:\s*\[(\d+)/(\d+)\]')
RE_STEP     = re.compile(r'(?:global_step|iter):\s*(\d+)')
RE_LR       = re.compile(r'\blr:\s*([\d.eE+-]+)')
RE_LOSS     = re.compile(r'(?<!_)loss:\s*([\d.]+)')           # loss: (not loss_ctc:)
RE_LOSS_CTC = re.compile(r'loss_ctc:\s*([\d.]+)')
RE_LOSS_NRTR= re.compile(r'loss_nrtr:\s*([\d.]+)')
RE_ACC      = re.compile(r'\bacc:\s*([\d.]+)')
RE_NED      = re.compile(r'norm_edit_dis:\s*([\d.]+)')
RE_ETA      = re.compile(r'eta:\s*([\d:]+)')
RE_IPS      = re.compile(r'ips:\s*([\d.]+)')


# ════════════════════════════════════════════════════════
#  유틸리티
# ════════════════════════════════════════════════════════
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def file_size_mb(path):
    """파일 크기를 MB로 반환"""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def dir_size_mb(path):
    """폴더 전체 크기를 MB로 반환"""
    total = 0
    if os.path.exists(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


_MPL_SETUP_DONE = False

def setup_matplotlib():
    """matplotlib 한글 폰트 설정 (sudo 없이 유저 폴더에 자동 다운로드)"""
    global _MPL_SETUP_DONE
    if _MPL_SETUP_DONE or not HAS_MPL:
        return
    _MPL_SETUP_DONE = True

    import matplotlib.font_manager as fm

    # 1) 시스템 폰트 확인
    font_paths = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/nanum/NanumGothic.ttf',
        'C:/Windows/Fonts/malgun.ttf',
    ]
    # 2) 유저 로컬 폰트
    local_font_dir = os.path.join(os.path.expanduser('~'), '.local', 'share', 'fonts')
    local_nanum = os.path.join(local_font_dir, 'NanumGothic-Regular.ttf')
    font_paths.append(local_nanum)

    for fp in font_paths:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['figure.dpi'] = 150
            return

    # 3) 없으면 직접 다운로드 (sudo 불필요)
    os.makedirs(local_font_dir, exist_ok=True)
    font_urls = [
        'https://raw.githubusercontent.com/googlefonts/nanum/main/fonts/NanumGothic-Regular.ttf',
        'https://github.com/googlefonts/nanum/raw/main/fonts/NanumGothic-Regular.ttf',
    ]
    print("  [DL] NanumGothic 폰트 다운로드 중 (sudo 불필요)...")
    for url in font_urls:
        if _download_file(url, local_nanum):
            try:
                subprocess.run(['fc-cache', '-f', local_font_dir],
                               capture_output=True, timeout=30)
            except Exception:
                pass
            fm.fontManager.addfont(local_nanum)
            plt.rcParams['font.family'] = fm.FontProperties(fname=local_nanum).get_name()
            print(f"  [OK] NanumGothic -> {local_nanum}")
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['figure.dpi'] = 150
            return
        if os.path.exists(local_nanum):
            os.remove(local_nanum)

    # 4) 폰트 없어도 동작은 함 (한글만 네모로 표시)
    import warnings
    warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')
    print("  [!] 한글 폰트 설치 실패 (차트의 한글이 깨질 수 있음)")
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150


def _download_file(url, dest):
    """파일 다운로드 (wget -> curl -> Python urllib 순서로 시도)"""
    # 1) wget
    try:
        r = subprocess.run(['wget', '-q', '--show-progress', '-O', dest, url], timeout=600)
        if r.returncode == 0:
            return True
    except Exception:
        pass

    # 2) curl
    try:
        r = subprocess.run(['curl', '-fSL', '-o', dest, url], timeout=600)
        if r.returncode == 0:
            return True
    except Exception:
        pass

    # 3) Python urllib (진행률 표시 없지만 가장 호환성 높음)
    try:
        import urllib.request
        print(f"       (urllib fallback)")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"       download failed: {e}")

    return False


# ════════════════════════════════════════════════════════
#  STEP 0-a. 환경 정보 기록
# ════════════════════════════════════════════════════════
def record_environment():
    print("\n" + "=" * 60)
    print("  STEP 0-a. 환경 정보 기록")
    print("=" * 60)

    ensure_dir(REPORT_DIR)
    lines = [f"기록 시각: {now_str()}", ""]

    # Python
    lines.append(f"Python: {sys.version}")
    lines.append(f"OS: {platform.platform()}")
    lines.append(f"CPU: {platform.processor() or platform.machine()}")
    lines.append("")

    # PaddlePaddle
    try:
        import paddle
        lines.append(f"PaddlePaddle: {paddle.__version__}")
        lines.append(f"CUDA compiled: {paddle.is_compiled_with_cuda()}")
        if paddle.is_compiled_with_cuda():
            lines.append(f"GPU count: {paddle.device.cuda.device_count()}")
    except Exception as e:
        lines.append(f"PaddlePaddle: import 실패 ({e})")
    lines.append("")

    # GPU 상세
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10)
        lines.append("GPU 정보:")
        for gpu_line in r.stdout.strip().split('\n'):
            lines.append(f"  {gpu_line.strip()}")
    except Exception:
        lines.append("nvidia-smi: 실행 불가")
    lines.append("")

    # CUDA 버전
    try:
        r = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        m = re.search(r'release ([\d.]+)', r.stdout)
        if m:
            lines.append(f"CUDA (nvcc): {m.group(1)}")
    except Exception:
        pass

    # PaddleOCR git commit
    try:
        r = subprocess.run(['git', 'log', '-1', '--format=%H %ad %s', '--date=short'],
                           capture_output=True, text=True, cwd=PADDLE_OCR_DIR, timeout=10)
        lines.append(f"PaddleOCR commit: {r.stdout.strip()}")
    except Exception:
        pass

    # pip 패키 목록
    try:
        r = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=freeze'],
                           capture_output=True, text=True, timeout=30)
        lines.append("\n설치된 패키지:")
        for pkg in r.stdout.strip().split('\n'):
            if any(kw in pkg.lower() for kw in ['paddle', 'torch', 'onnx', 'opencv',
                                                'matplotlib', 'numpy', 'pillow', 'yaml']):
                lines.append(f"  {pkg}")
    except Exception:
        pass

    # 하이퍼파라미터
    lines.append(f"\n학습 하이퍼파라미터:")
    lines.append(f"  EPOCH_NUM: {EPOCH_NUM}")
    lines.append(f"  BATCH_SIZE: {BATCH_SIZE}")
    lines.append(f"  LEARNING_RATE: {LEARNING_RATE}")
    lines.append(f"  WARMUP_EPOCH: {WARMUP_EPOCH}")
    lines.append(f"  EVAL_STEP: {EVAL_STEP}")
    lines.append(f"  EARLY_STOP_PATIENCE: {EARLY_STOP_PATIENCE}")
    lines.append(f"  NUM_WORKERS: {NUM_WORKERS}")
    lines.append(f"  Architecture: SVTR_LCNet (PPLCNetV3 + MultiHead CTC+NRTR)")

    env_path = os.path.join(REPORT_DIR, 'environment.txt')
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  -> {env_path}")


# ════════════════════════════════════════════════════════
#  STEP 0-b. 데이터셋 통계 기록
# ════════════════════════════════════════════════════════
def record_dataset_stats():
    setup_matplotlib()  # 차트 생성 전 폰트 설정
    print("\n" + "=" * 60)
    print("  STEP 0-b. 데이터셋 통계 기록")
    print("=" * 60)

    chart_dir = ensure_dir(os.path.join(REPORT_DIR, 'charts'))
    all_stats = {}
    report_lines = []

    for country in COUNTRIES:
        train_path = os.path.join(DATA_DIR, country, 'ocr_dataset', 'rec_gt_train.txt')
        val_path   = os.path.join(DATA_DIR, country, 'ocr_dataset', 'rec_gt_val.txt')

        chars = Counter()
        label_lengths = []
        samples = []
        train_count = 0

        if os.path.exists(train_path):
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        train_count += 1
                        label = parts[1]
                        label_lengths.append(len(label))
                        for ch in label:
                            chars[ch] += 1
                        if len(samples) < 10:
                            samples.append(label)

        val_count = 0
        if os.path.exists(val_path):
            with open(val_path, 'r', encoding='utf-8') as f:
                val_count = sum(1 for ln in f if '\t' in ln)

        avg_len = sum(label_lengths) / len(label_lengths) if label_lengths else 0
        stats = {
            'train_count': train_count,
            'val_count': val_count,
            'total_chars': len(chars),
            'avg_label_length': round(avg_len, 1),
            'max_label_length': max(label_lengths) if label_lengths else 0,
            'min_label_length': min(label_lengths) if label_lengths else 0,
            'sample_labels': samples,
            'top_chars': [[ch, cnt] for ch, cnt in chars.most_common(30)],
        }
        all_stats[country] = stats

        report_lines.append(f"\n{'='*50}")
        report_lines.append(f"  {country.upper()}")
        report_lines.append(f"{'='*50}")
        report_lines.append(f"  Train: {train_count:,}  |  Val: {val_count:,}")
        report_lines.append(f"  고유 문자: {len(chars)}개")
        report_lines.append(f"  라벨 길이: avg {stats['avg_label_length']} / "
                            f"min {stats['min_label_length']} / max {stats['max_label_length']}")
        report_lines.append(f"  샘플: {', '.join(samples[:5])}")
        report_lines.append(f"  빈도 상위: {''.join(ch for ch, _ in chars.most_common(20))}")
        print(f"  {country}: train={train_count:,}  val={val_count:,}  chars={len(chars)}")

        # 문자 분포 차트
        if HAS_MPL and chars:
            top_n = min(30, len(chars))
            top = chars.most_common(top_n)
            fig, ax = plt.subplots(figsize=(max(8, top_n * 0.4), 4))
            ax.bar([c for c, _ in top], [n for _, n in top], color='#42A5F5')
            ax.set_title(f'{country.upper()} - Character Frequency (Top {top_n})')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, f'{country}_char_dist.png'))
            plt.close()

    # 데이터셋 크기 비교 차트
    if HAS_MPL and all_stats:
        fig, ax = plt.subplots(figsize=(10, 5))
        countries = list(all_stats.keys())
        trains = [all_stats[c]['train_count'] for c in countries]
        vals   = [all_stats[c]['val_count'] for c in countries]
        x = range(len(countries))
        w = 0.35
        b1 = ax.bar([i - w/2 for i in x], trains, w, label='Train', color='#4CAF50')
        b2 = ax.bar([i + w/2 for i in x], vals,   w, label='Val',   color='#2196F3')
        ax.set_ylabel('Sample Count')
        ax.set_title('Dataset Size by Country')
        ax.set_xticks(list(x))
        ax.set_xticklabels([c.upper() for c in countries])
        ax.legend()
        ax.bar_label(b1, fmt='{:,.0f}', fontsize=7, padding=2)
        ax.bar_label(b2, fmt='{:,.0f}', fontsize=7, padding=2)
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, 'dataset_distribution.png'))
        plt.close()

    # 저장
    with open(os.path.join(REPORT_DIR, 'dataset_stats.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    with open(os.path.join(REPORT_DIR, 'dataset_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"  -> {REPORT_DIR}/dataset_stats.*")


# ════════════════════════════════════════════════════════
#  STEP 1. 문자 사전 생성
# ════════════════════════════════════════════════════════
def generate_dicts():
    print("\n" + "=" * 60)
    print("  STEP 1. 국가별 문자 사전(dict) 생성")
    print("=" * 60)

    for country in COUNTRIES:
        label_path = os.path.join(DATA_DIR, country, 'ocr_dataset', 'rec_gt_train.txt')
        if not os.path.exists(label_path):
            print(f"  [!] {country}: 라벨 파일 없음 -> {label_path}")
            continue

        chars = set()
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    for ch in parts[1]:
                        chars.add(ch)

        sorted_chars = sorted(chars)
        dict_path = os.path.join(DATA_DIR, country, 'ocr_dataset', f'{country}_dict.txt')
        with open(dict_path, 'w', encoding='utf-8') as f:
            for ch in sorted_chars:
                f.write(ch + '\n')

        print(f"  {country}: {len(sorted_chars)}개 문자  "
              f"[{' '.join(sorted_chars[:10])}{'...' if len(sorted_chars) > 10 else ''}]")


# ════════════════════════════════════════════════════════
#  STEP 2. 사전학습 모델 다운로드
# ════════════════════════════════════════════════════════
def download_pretrained():
    print("\n" + "=" * 60)
    print("  STEP 2. 사전학습 모델 다운로드")
    print("=" * 60)

    model_dir = os.path.join(PADDLE_OCR_DIR, 'pretrained_models')
    os.makedirs(model_dir, exist_ok=True)

    downloaded = set()
    for country, (model_name, lang, url) in COUNTRIES.items():
        if model_name in downloaded:
            continue

        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            size = dir_size_mb(model_path)
            print(f"  [OK] {model_name} ({size:.1f} MB)")
            downloaded.add(model_name)
            continue

        print(f"  [DL] {model_name} ...")
        tar_path = os.path.join(model_dir, f'{model_name}.tar')

        # 여러 URL 패턴 시도
        urls_to_try = [url]
        # PP-OCRv4 multilingual 경로 폴백
        if 'korean' in model_name:
            urls_to_try.append(url.replace('/korean/', '/multilingual/'))
            # PP-OCRv3 폴백
            urls_to_try.append(url.replace('PP-OCRv4', 'PP-OCRv3').replace('/korean/', '/multilingual/'))

        success = False
        for try_url in urls_to_try:
            print(f"       URL: {try_url}")
            if _download_file(try_url, tar_path):
                # tar 압축 해제
                r = subprocess.run(['tar', '-xf', tar_path, '-C', model_dir])
                if r.returncode == 0:
                    if os.path.exists(tar_path):
                        os.remove(tar_path)
                    # tar 안의 폴더명이 다를 수 있으므로 확인
                    if not os.path.exists(model_path):
                        # 압축 해제된 폴더 찾기
                        for item in os.listdir(model_dir):
                            item_path = os.path.join(model_dir, item)
                            if os.path.isdir(item_path) and item not in downloaded and item != model_name:
                                if 'rec' in item.lower():
                                    os.rename(item_path, model_path)
                                    print(f"       renamed: {item} -> {model_name}")
                                    break
                    success = True
                    break
                else:
                    print(f"       tar 해제 실패")
            if os.path.exists(tar_path):
                os.remove(tar_path)

        if success and os.path.exists(model_path):
            size = dir_size_mb(model_path)
            print(f"  [OK] {model_name} ({size:.1f} MB)")
            downloaded.add(model_name)
        else:
            print(f"  [FAIL] {model_name} 다운로드 실패!")
            print(f"         수동 다운로드 후 {model_path}/ 에 넣어주세요")
            print(f"         또는: wget {url} && tar -xf {model_name}.tar -C {model_dir}")


def _find_pretrained_path(model_name):
    """pretrained 모델의 .pdparams 파일 경로 자동 탐색 (PADDLE_OCR_DIR 기준 상대경로 반환)"""
    base_dir = os.path.join(PADDLE_OCR_DIR, 'pretrained_models', model_name)
    if not os.path.exists(base_dir):
        return None

    # 가능한 경로 후보 (우선순위)
    candidates = [
        os.path.join(base_dir, 'best_model', 'model'),
        os.path.join(base_dir, 'best_accuracy', 'model'),
        os.path.join(base_dir, 'best_accuracy'),         # korean, en PP-OCRv4
        os.path.join(base_dir, 'student'),                # ch PP-OCRv4
        os.path.join(base_dir, 'model'),
        os.path.join(base_dir, 'train', 'model'),
    ]
    for path in candidates:
        if os.path.exists(path + '.pdparams'):
            return os.path.relpath(path, PADDLE_OCR_DIR)

    # 후보에 없으면 .pdparams 파일 직접 탐색 (._ macOS 메타파일 제외)
    for root, dirs, files in os.walk(base_dir):
        for f in sorted(files):
            if f.endswith('.pdparams') and not f.startswith('._'):
                full = os.path.join(root, f.replace('.pdparams', ''))
                return os.path.relpath(full, PADDLE_OCR_DIR)

    return None


def _detect_gpu_count():
    """사용 가능한 GPU 수 확인"""
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return len([line for line in r.stdout.strip().split('\n') if line.strip()])
    except Exception:
        pass
    return 0


# ════════════════════════════════════════════════════════
#  STEP 3. Config YAML 생성 + 백업
# ════════════════════════════════════════════════════════
def generate_configs():
    print("\n" + "=" * 60)
    print("  STEP 3. Config 생성 + 백업")
    print("=" * 60)

    config_dir = os.path.join(PADDLE_OCR_DIR, 'configs', 'rec', 'finetune')
    backup_dir = ensure_dir(os.path.join(REPORT_DIR, 'configs'))
    os.makedirs(config_dir, exist_ok=True)

    for country, (model_name, lang, url) in COUNTRIES.items():
        dict_path  = os.path.join('..', 'ocr_train_data_v1', country, 'ocr_dataset', f'{country}_dict.txt')

        # pretrained 모델 경로 자동 탐색
        pretrained = _find_pretrained_path(model_name)
        if pretrained:
            print(f"  {country}: pretrained -> {pretrained}")
        else:
            pretrained = os.path.join('pretrained_models', model_name, 'best_model', 'model')
            print(f"  [!] {country}: .pdparams 못 찾음, 기본값 사용 -> {pretrained}")

        config = {
            'Global': {
                'debug': False,
                'use_gpu': True,
                'epoch_num': EPOCH_NUM,
                'log_smooth_window': 20,
                'print_batch_step': 10,
                'save_model_dir': f'./output/rec_{country}',
                'save_epoch_step': 10,
                'eval_batch_step': [0, EVAL_STEP],
                'cal_metric_during_train': True,
                'pretrained_model': pretrained,
                'checkpoints': None,
                'save_inference_dir': f'./inference/rec_{country}',
                'use_visualdl': False,
                'infer_img': None,
                'character_dict_path': dict_path,
                'max_text_length': 25,
                'infer_mode': False,
                'use_space_char': False,
            },
            'Optimizer': {
                'name': 'Adam',
                'beta1': 0.9,
                'beta2': 0.999,
                'lr': {
                    'name': 'Cosine',
                    'learning_rate': LEARNING_RATE,
                    'warmup_epoch': WARMUP_EPOCH,
                },
                'regularizer': {
                    'name': 'L2',
                    'factor': 3.0e-05,
                },
            },
            'Architecture': {
                'model_type': 'rec',
                'algorithm': 'SVTR_LCNet',
                'Transform': None,
                'Backbone': {
                    'name': 'PPLCNetV3',
                    'scale': 0.95,
                },
                'Head': {
                    'name': 'MultiHead',
                    'head_list': [
                        {'CTCHead': {
                            'Neck': {
                                'name': 'svtr',
                                'dims': 120,
                                'depth': 2,
                                'hidden_dims': 120,
                                'kernel_size': [1, 3],
                                'use_guide': True,
                            },
                            'Head': {'fc_decay': 0.00001},
                        }},
                        {'NRTRHead': {
                            'nrtr_dim': 384,
                            'max_text_length': 25,
                        }},
                    ],
                },
            },
            'Loss': {
                'name': 'MultiLoss',
                'loss_config_list': [
                    {'CTCLoss': None},
                    {'NRTRLoss': None},
                ],
            },
            'PostProcess': {'name': 'CTCLabelDecode'},
            'Metric': {'name': 'RecMetric', 'main_indicator': 'acc'},
            'Train': {
                'dataset': {
                    'name': 'SimpleDataSet',
                    'data_dir': f'../ocr_train_data_v1/{country}',
                    'ext_op_transform_idx': 1,  # 🔥 수정: ext_op_transform_idx 추가
                    'label_file_list': [
                        f'../ocr_train_data_v1/{country}/ocr_dataset/rec_gt_train.txt',
                    ],
                    'transforms': [
                        {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                        {'RecConAug': {
                            'prob': 0.5,
                            'ext_data_num': 2,
                            'image_shape': [48, 320, 3],
                            'max_text_length': 25,
                        }},
                        {'MultiLabelEncode': {'gtc_encode': 'NRTRLabelEncode'}},
                        {'RecResizeImg': {'image_shape': [3, 48, 320]}},
                        {'KeepKeys': {
                            'keep_keys': ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio'], # 🔥 수정: label_nrtr -> label_gtc
                        }},
                    ],
                },
                'loader': {
                    'shuffle': True,
                    'batch_size_per_card': BATCH_SIZE,
                    'drop_last': True,
                    'num_workers': NUM_WORKERS,
                },
            },
            'Eval': {
                'dataset': {
                    'name': 'SimpleDataSet',
                    'data_dir': f'../ocr_train_data_v1/{country}',
                    'label_file_list': [
                        f'../ocr_train_data_v1/{country}/ocr_dataset/rec_gt_val.txt',
                    ],
                    'transforms': [
                        {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                        {'MultiLabelEncode': {'gtc_encode': 'NRTRLabelEncode'}},
                        {'RecResizeImg': {'image_shape': [3, 48, 320]}},
                        {'KeepKeys': {
                            'keep_keys': ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio'], # 🔥 수정: label_nrtr -> label_gtc
                        }},
                    ],
                },
                'loader': {
                    'shuffle': False,
                    'batch_size_per_card': BATCH_SIZE,
                    'drop_last': False,
                    'num_workers': NUM_WORKERS,
                },
            },
        }

        config_path = os.path.join(config_dir, f'rec_{country}.yml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        # 백업
        shutil.copy2(config_path, os.path.join(backup_dir, f'rec_{country}.yml'))
        print(f"  {country} -> {config_path}  (백업 완료)")


# ════════════════════════════════════════════════════════
#  STEP 4. 학습 (Early Stopping + GPU 모니터링 + 메트릭 기록)
# ════════════════════════════════════════════════════════
def _gpu_monitor_loop(gpu_id, log_path, stop_event):
    """GPU 사용량 주기적 기록 (백그라운드 스레드)"""
    with open(log_path, 'w') as f:
        f.write('timestamp,gpu_util_%,mem_used_mb,mem_total_mb,temp_c,power_w\n')
        while not stop_event.is_set():
            try:
                r = subprocess.run(
                    ['nvidia-smi', f'--id={gpu_id}',
                     '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    f.write(f"{now_str()},{r.stdout.strip()}\n")
                    f.flush()
            except Exception:
                pass
            stop_event.wait(GPU_LOG_INTERVAL)


def train_country(gpu_id, country):
    """단일 국가 학습 (Early Stopping + 실시간 메트릭 기록 + GPU 모니터링)"""
    config_path  = os.path.join('configs', 'rec', 'finetune', f'rec_{country}.yml')
    train_script = os.path.join('tools', 'train.py')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    env['PYTHONUNBUFFERED'] = '1'

    cmd = [sys.executable, train_script, '-c', config_path]

    # 디렉토리
    log_dir     = os.path.join(PADDLE_OCR_DIR, 'output', f'rec_{country}')
    metrics_dir = ensure_dir(os.path.join(REPORT_DIR, 'metrics'))
    gpu_log_dir = ensure_dir(os.path.join(REPORT_DIR, 'gpu_logs'))
    os.makedirs(log_dir, exist_ok=True)

    log_path     = os.path.join(log_dir, 'train.log')
    gpu_log_path = os.path.join(gpu_log_dir, f'{country}_gpu.csv')

    # 메트릭 초기화
    metrics = {
        'country': country,
        'gpu_id': gpu_id,
        'start_time': now_str(),
        'end_time': None,
        'elapsed_minutes': 0,
        'total_epochs_config': EPOCH_NUM,
        'actual_epochs': 0,
        'early_stopped': False,
        'early_stop_eval_count': 0,
        'best_acc': 0.0,
        'best_ned': 0.0,
        'best_epoch': 0,
        'best_step': 0,
        'final_acc': 0.0,
        'train_steps': [],      # [{step, epoch, loss, loss_ctc, loss_nrtr, lr, ips}, ...]
        'eval_steps': [],       # [{step, epoch, acc, ned, is_best}, ...]
        'pretrained_model_size_mb': 0,
        'finetuned_model_size_mb': 0,
    }

    # pretrained 모델 크기 기록
    model_name = COUNTRIES[country][0]
    pretrained_dir = os.path.join(PADDLE_OCR_DIR, 'pretrained_models', model_name)
    metrics['pretrained_model_size_mb'] = round(dir_size_mb(pretrained_dir), 2)

    # GPU 모니터링 시작
    gpu_stop = threading.Event()
    gpu_thread = threading.Thread(target=_gpu_monitor_loop,
                                  args=(gpu_id, gpu_log_path, gpu_stop), daemon=True)
    gpu_thread.start()

    print(f"\n  >>> [{country}] GPU {gpu_id} 학습 시작 ({now_str()})")
    print(f"      cmd: CUDA_VISIBLE_DEVICES={gpu_id} python {' '.join(cmd[1:])}")

    start_time    = time.time()
    no_improve    = 0
    current_epoch = 0
    is_eval_line  = False   # 다음 줄이 best 판정 줄인지

    proc = subprocess.Popen(
        cmd, env=env, cwd=PADDLE_OCR_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True
    )

    with open(log_path, 'w', encoding='utf-8') as log_file:
        try:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()

                # ── 학습 스텝 파싱 ──
                m_epoch = RE_EPOCH.search(line)
                m_step  = RE_STEP.search(line)
                m_loss  = RE_LOSS.search(line)

                if m_epoch and m_step and m_loss:
                    current_epoch = int(m_epoch.group(1))
                    step_data = {
                        'step':  int(m_step.group(1)),
                        'epoch': current_epoch,
                        'loss':  float(m_loss.group(1)),
                    }
                    m_lr = RE_LR.search(line)
                    if m_lr:
                        step_data['lr'] = float(m_lr.group(1))
                    m_ctc = RE_LOSS_CTC.search(line)
                    if m_ctc:
                        step_data['loss_ctc'] = float(m_ctc.group(1))
                    m_nrtr = RE_LOSS_NRTR.search(line)
                    if m_nrtr:
                        step_data['loss_nrtr'] = float(m_nrtr.group(1))
                    m_ips = RE_IPS.search(line)
                    if m_ips:
                        step_data['ips'] = float(m_ips.group(1))
                    metrics['train_steps'].append(step_data)

                # ── 평가 결과 파싱 ──
                if ('metric' in line.lower() or 'acc' in line) and RE_ACC.search(line):
                    # 'cur metric' 또는 'metric eval' 줄
                    if any(kw in line.lower() for kw in ['cur metric', 'metric eval', 'cur_metric']):
                        m_acc = RE_ACC.search(line)
                        m_ned = RE_NED.search(line)
                        if m_acc:
                            acc = float(m_acc.group(1))
                            ned = float(m_ned.group(1)) if m_ned else 0.0
                            last_step = metrics['train_steps'][-1]['step'] if metrics['train_steps'] else 0
                            eval_data = {
                                'step': last_step,
                                'epoch': current_epoch,
                                'acc': acc,
                                'ned': ned,
                                'is_best': False,
                            }
                            metrics['eval_steps'].append(eval_data)
                            is_eval_line = True

                    # 'best metric' 줄 - best 여부 판정
                    if 'best' in line.lower() and is_eval_line:
                        is_best = False
                        if re.search(r'(?:update_best|is_best).*?True', line, re.IGNORECASE):
                            is_best = True
                        elif re.search(r'(?:update_best|is_best).*?False', line, re.IGNORECASE):
                            is_best = False
                        else:
                            # 직접 비교 (로그 포맷이 다를 때 폴백)
                            m_bacc = RE_ACC.search(line)
                            if m_bacc and metrics['eval_steps']:
                                is_best = (float(m_bacc.group(1))
                                           >= metrics['eval_steps'][-1]['acc'] - 1e-6)

                        if metrics['eval_steps']:
                            metrics['eval_steps'][-1]['is_best'] = is_best

                        if is_best:
                            ea = metrics['eval_steps'][-1]
                            metrics['best_acc']   = ea['acc']
                            metrics['best_ned']   = ea['ned']
                            metrics['best_epoch'] = current_epoch
                            metrics['best_step']  = ea['step']
                            no_improve = 0
                            print(f"      [{country}] NEW BEST acc={ea['acc']:.4f} "
                                  f"ned={ea['ned']:.4f} (epoch {current_epoch})")
                        else:
                            no_improve += 1

                        is_eval_line = False

                        # Early Stopping 체크
                        if EARLY_STOP_PATIENCE > 0 and no_improve >= EARLY_STOP_PATIENCE:
                            print(f"\n  <<< [{country}] EARLY STOPPING "
                                  f"({EARLY_STOP_PATIENCE}번 연속 개선 없음, "
                                  f"best acc={metrics['best_acc']:.4f})")
                            metrics['early_stopped'] = True
                            metrics['early_stop_eval_count'] = no_improve
                            proc.terminate()
                            try:
                                proc.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            break

        except Exception as e:
            print(f"  [!] [{country}] 로그 파싱 중 예외: {e}")

    proc.wait()

    # GPU 모니터링 종료
    gpu_stop.set()
    gpu_thread.join(timeout=5)

    # 결과 기록
    elapsed = time.time() - start_time
    metrics['end_time']        = now_str()
    metrics['elapsed_minutes'] = round(elapsed / 60, 1)
    metrics['actual_epochs']   = current_epoch
    metrics['final_acc']       = metrics['eval_steps'][-1]['acc'] if metrics['eval_steps'] else 0.0

    # finetuned 모델 크기
    best_model_dir = os.path.join(PADDLE_OCR_DIR, 'output', f'rec_{country}', 'best_model')
    metrics['finetuned_model_size_mb'] = round(dir_size_mb(best_model_dir), 2)

    # 체크포인트 목록
    output_dir = os.path.join(PADDLE_OCR_DIR, 'output', f'rec_{country}')
    checkpoints = []
    if os.path.exists(output_dir):
        for item in sorted(os.listdir(output_dir)):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and item.startswith(('iter_', 'epoch_', 'best_model', 'latest')):
                checkpoints.append({'name': item, 'size_mb': round(dir_size_mb(item_path), 2)})
    metrics['checkpoints'] = checkpoints

    # 메트릭 JSON 저장
    metrics_path = os.path.join(metrics_dir, f'{country}_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    ok = proc.returncode == 0 or metrics['early_stopped']
    tag = "DONE" if ok else "FAIL"
    es  = " (early stopped)" if metrics['early_stopped'] else ""
    print(f"  [{tag}] {country} | GPU {gpu_id} | "
          f"{metrics['elapsed_minutes']:.1f}min | "
          f"best_acc={metrics['best_acc']:.4f} | "
          f"epoch {current_epoch}/{EPOCH_NUM}{es}")

    # 실패 시 로그 마지막 20줄 출력 (원인 파악용)
    if not ok and os.path.exists(log_path):
        print(f"\n  ---- [{country}] 에러 로그 (마지막 20줄) ----")
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                for line in lines[-20:]:
                    print(f"  | {line.rstrip()}")
        except Exception:
            pass
        print(f"  ---- 전체 로그: {log_path} ----\n")

    return {
        'country': country,
        'returncode': proc.returncode,
        'early_stopped': metrics['early_stopped'],
        'best_acc': metrics['best_acc'],
        'elapsed_minutes': metrics['elapsed_minutes'],
    }


def run_training():
    print("\n" + "=" * 60)
    print("  STEP 4. 학습 시작 (Early Stopping)")
    print("=" * 60)

    os.chdir(PADDLE_OCR_DIR)

    # GPU 수 자동 감지 → 스케줄 조정
    gpu_count = _detect_gpu_count()
    print(f"  GPU 감지: {gpu_count}개")

    if gpu_count >= 2:
        schedule = GPU_SCHEDULE
        print(f"  -> 2 GPU 병렬 스케줄 사용")
    else:
        # GPU 1개면 순차 실행 (모두 GPU 0)
        all_countries = [ctry for pairs in GPU_SCHEDULE for _, ctry in pairs]
        schedule = [[('0', ctry)] for ctry in all_countries]
        print(f"  -> 1 GPU 순차 스케줄로 변경: {', '.join(all_countries)}")

    total_start = time.time()

    for round_num, pairs in enumerate(schedule, 1):
        print(f"\n  ====== Round {round_num}/{len(schedule)} ======")
        if len(pairs) == 1:
            gpu_id, country = pairs[0]
            train_country(gpu_id, country)
        else:
            with ProcessPoolExecutor(max_workers=len(pairs)) as executor:
                futures = [executor.submit(train_country, gid, ctry)
                           for gid, ctry in pairs]
                for f in futures:
                    f.result()

    total_min = (time.time() - total_start) / 60
    print(f"\n  === 전체 학습 완료: {total_min:.1f}분 ===")


# ════════════════════════════════════════════════════════
#  STEP 5. 분석 & 시각화
# ════════════════════════════════════════════════════════
def _load_metrics(country):
    """메트릭 JSON 로드 (없으면 로그 파싱 폴백)"""
    # JSON 먼저
    json_path = os.path.join(REPORT_DIR, 'metrics', f'{country}_metrics.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 로그 파싱 폴백
    log_path = os.path.join(PADDLE_OCR_DIR, 'output', f'rec_{country}', 'train.log')
    if not os.path.exists(log_path):
        return None

    print(f"  [{country}] JSON 없음, 로그 파싱 중...")
    metrics = {
        'country': country,
        'train_steps': [],
        'eval_steps': [],
        'best_acc': 0.0,
    }
    current_epoch = 0

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m_epoch = RE_EPOCH.search(line)
            m_step  = RE_STEP.search(line)
            m_loss  = RE_LOSS.search(line)

            if m_epoch and m_step and m_loss:
                current_epoch = int(m_epoch.group(1))
                sd = {'step': int(m_step.group(1)), 'epoch': current_epoch,
                      'loss': float(m_loss.group(1))}
                m_lr = RE_LR.search(line)
                if m_lr:
                    sd['lr'] = float(m_lr.group(1))
                m_ctc = RE_LOSS_CTC.search(line)
                if m_ctc:
                    sd['loss_ctc'] = float(m_ctc.group(1))
                metrics['train_steps'].append(sd)

            if any(kw in line.lower() for kw in ['cur metric', 'metric eval']) and RE_ACC.search(line):
                m_acc = RE_ACC.search(line)
                m_ned = RE_NED.search(line)
                if m_acc:
                    acc = float(m_acc.group(1))
                    ned = float(m_ned.group(1)) if m_ned else 0.0
                    step = metrics['train_steps'][-1]['step'] if metrics['train_steps'] else 0
                    metrics['eval_steps'].append({
                        'step': step, 'epoch': current_epoch, 'acc': acc, 'ned': ned
                    })
                    if acc > metrics['best_acc']:
                        metrics['best_acc'] = acc

    return metrics


def analyze_and_visualize():
    print("\n" + "=" * 60)
    print("  STEP 5. 분석 & 시각화")
    print("=" * 60)

    if not HAS_MPL:
        print("  [!] matplotlib 없음 -> 그래프 생성 불가")
        return

    setup_matplotlib()
    chart_dir = ensure_dir(os.path.join(REPORT_DIR, 'charts'))

    all_metrics = {}
    colors = {'korea': '#E53935', 'china': '#FDD835', 'brazil': '#43A047',
              'europe': '#1E88E5', 'india': '#FB8C00'}

    for country in COUNTRIES:
        m = _load_metrics(country)
        if not m or (not m.get('train_steps') and not m.get('eval_steps')):
            print(f"  [{country}] 메트릭 없음 (건너뜀)")
            continue
        all_metrics[country] = m
        print(f"  [{country}] train_steps={len(m.get('train_steps',[]))}  "
              f"eval_steps={len(m.get('eval_steps',[]))}")

    if not all_metrics:
        print("  [!] 분석할 메트릭 없음")
        return

    # ── 국가별 개별 차트 ──
    for country, m in all_metrics.items():
        c = colors.get(country, '#666666')

        # Loss 곡선
        if m.get('train_steps'):
            steps = [s['step'] for s in m['train_steps']]
            losses = [s['loss'] for s in m['train_steps']]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, losses, color=c, linewidth=0.8, alpha=0.6, label='Total Loss')
            # CTC / NRTR 서브 loss
            if 'loss_ctc' in m['train_steps'][0]:
                ax.plot(steps, [s.get('loss_ctc', 0) for s in m['train_steps']],
                        color='#42A5F5', linewidth=0.6, alpha=0.5, label='CTC Loss')
            if 'loss_nrtr' in m['train_steps'][0]:
                ax.plot(steps, [s.get('loss_nrtr', 0) for s in m['train_steps']],
                        color='#AB47BC', linewidth=0.6, alpha=0.5, label='NRTR Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title(f'{country.upper()} - Training Loss')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, f'{country}_loss.png'))
            plt.close()

        # Accuracy 곡선
        if m.get('eval_steps'):
            eval_steps = [e['step'] for e in m['eval_steps']]
            accs = [e['acc'] for e in m['eval_steps']]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(eval_steps, accs, color=c, marker='o', markersize=3, linewidth=1.2)
            # best 포인트 표시
            if m.get('best_acc'):
                best_idx = max(range(len(accs)), key=lambda i: accs[i])
                ax.axhline(y=accs[best_idx], color='gray', linestyle='--', alpha=0.5)
                ax.annotate(f'Best: {accs[best_idx]:.4f}',
                            xy=(eval_steps[best_idx], accs[best_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=9, fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='gray'))
            ax.set_xlabel('Step')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{country.upper()} - Validation Accuracy')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, f'{country}_accuracy.png'))
            plt.close()

        # Learning Rate 곡선
        if m.get('train_steps') and 'lr' in m['train_steps'][0]:
            steps = [s['step'] for s in m['train_steps']]
            lrs = [s.get('lr', 0) for s in m['train_steps']]
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(steps, lrs, color='#26A69A', linewidth=1)
            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title(f'{country.upper()} - Learning Rate Schedule')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(chart_dir, f'{country}_lr.png'))
            plt.close()

    # ── 전체 비교 차트 ──
    countries_with_eval = [c for c in all_metrics if all_metrics[c].get('eval_steps')]

    # 1) 모든 국가 Accuracy 곡선 (하나의 그래프에)
    if countries_with_eval:
        fig, ax = plt.subplots(figsize=(12, 5))
        for country in countries_with_eval:
            m = all_metrics[country]
            eval_steps = [e['step'] for e in m['eval_steps']]
            accs = [e['acc'] for e in m['eval_steps']]
            ax.plot(eval_steps, accs, color=colors.get(country, '#666'),
                    label=f"{country.upper()} (best={m.get('best_acc',0):.4f})",
                    linewidth=1.2, marker='o', markersize=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.set_title('All Countries - Accuracy Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, 'all_accuracy_curves.png'))
        plt.close()

    # 2) 모든 국가 Loss 곡선
    countries_with_train = [c for c in all_metrics if all_metrics[c].get('train_steps')]
    if countries_with_train:
        fig, ax = plt.subplots(figsize=(12, 5))
        for country in countries_with_train:
            m = all_metrics[country]
            steps = [s['step'] for s in m['train_steps']]
            losses = [s['loss'] for s in m['train_steps']]
            ax.plot(steps, losses, color=colors.get(country, '#666'),
                    label=country.upper(), linewidth=0.8, alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('All Countries - Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, 'all_loss_curves.png'))
        plt.close()

    # 3) Best Accuracy 비교 (bar chart)
    if countries_with_eval:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [c.upper() for c in countries_with_eval]
        accs  = [all_metrics[c].get('best_acc', 0) * 100 for c in countries_with_eval]
        clrs  = [colors.get(c, '#666') for c in countries_with_eval]
        bars  = ax.bar(names, accs, color=clrs)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Best Accuracy Comparison')
        ax.set_ylim(min(accs) - 5 if accs else 0, 102)
        ax.bar_label(bars, fmt='%.2f%%', fontsize=9, padding=3)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, 'accuracy_comparison.png'))
        plt.close()

    # 4) 학습 시간 비교 (bar chart)
    countries_with_time = [c for c in all_metrics if all_metrics[c].get('elapsed_minutes')]
    if countries_with_time:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [c.upper() for c in countries_with_time]
        times = [all_metrics[c]['elapsed_minutes'] for c in countries_with_time]
        clrs  = [colors.get(c, '#666') for c in countries_with_time]
        bars  = ax.bar(names, times, color=clrs)
        ax.set_ylabel('Minutes')
        ax.set_title('Training Time Comparison')
        ax.bar_label(bars, fmt='%.1f min', fontsize=9, padding=3)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(chart_dir, 'training_time_comparison.png'))
        plt.close()

    # 5) GPU 사용량 차트
    for country in COUNTRIES:
        gpu_csv = os.path.join(REPORT_DIR, 'gpu_logs', f'{country}_gpu.csv')
        if not os.path.exists(gpu_csv):
            continue
        try:
            timestamps, utils, mems = [], [], []
            with open(gpu_csv, 'r') as f:
                next(f)  # header
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        timestamps.append(len(timestamps))  # index as time
                        utils.append(float(parts[1].strip()))
                        mems.append(float(parts[2].strip()))

            if timestamps:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                t_min = [t * GPU_LOG_INTERVAL / 60 for t in timestamps]

                ax1.plot(t_min, utils, color='#E53935', linewidth=1)
                ax1.set_ylabel('GPU Util (%)')
                ax1.set_title(f'{country.upper()} - GPU Usage')
                ax1.set_ylim(0, 105)
                ax1.grid(True, alpha=0.3)

                ax2.plot(t_min, mems, color='#1E88E5', linewidth=1)
                ax2.set_ylabel('Memory Used (MB)')
                ax2.set_xlabel('Time (minutes)')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(chart_dir, f'{country}_gpu_usage.png'))
                plt.close()
        except Exception as e:
            print(f"  [{country}] GPU 차트 생성 실패: {e}")

    print(f"  -> 차트 저장 완료: {chart_dir}")


# ════════════════════════════════════════════════════════
#  STEP 6. 모델 내보내기
# ════════════════════════════════════════════════════════
def export_models():
    print("\n" + "=" * 60)
    print("  STEP 6. 모델 내보내기 (inference)")
    print("=" * 60)

    os.chdir(PADDLE_OCR_DIR)
    export_script = os.path.join('tools', 'export_model.py')
    model_sizes = {}

    for country in COUNTRIES:
        config_path = os.path.join('configs', 'rec', 'finetune', f'rec_{country}.yml')
        best_model  = os.path.join('output', f'rec_{country}', 'best_model', 'model')
        save_dir    = os.path.join('inference', f'rec_{country}')

        if not os.path.exists(best_model + '.pdparams'):
            print(f"  [!] {country}: best_model 없음 (건너뜀)")
            continue

        cmd = [
            sys.executable, export_script,
            '-c', config_path,
            '-o', f'Global.pretrained_model={best_model}',
            '-o', f'Global.save_inference_dir={save_dir}',
        ]

        proc = subprocess.run(cmd, cwd=PADDLE_OCR_DIR,
                              capture_output=True, text=True)
        ok = proc.returncode == 0
        size = dir_size_mb(save_dir)
        model_sizes[country] = size
        tag = "OK" if ok else "FAIL"
        print(f"  [{tag}] {country} -> {save_dir} ({size:.1f} MB)")

    # 모델 크기 기록
    if model_sizes:
        sizes_path = os.path.join(REPORT_DIR, 'model_sizes.json')
        with open(sizes_path, 'w') as f:
            json.dump(model_sizes, f, indent=2)


# ════════════════════════════════════════════════════════
#  STEP 7. 샘플 예측
# ════════════════════════════════════════════════════════
def run_sample_predictions():
    print("\n" + "=" * 60)
    print("  STEP 7. 검증 데이터 샘플 예측")
    print("=" * 60)

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("  [!] paddleocr 미설치 -> pip install paddleocr (건너뜀)")
        return

    pred_dir = ensure_dir(os.path.join(REPORT_DIR, 'sample_predictions'))
    sample_n = 30

    for country in COUNTRIES:
        model_dir = os.path.join(PADDLE_OCR_DIR, 'inference', f'rec_{country}')
        dict_path = os.path.join(DATA_DIR, country, 'ocr_dataset', f'{country}_dict.txt')
        val_path  = os.path.join(DATA_DIR, country, 'ocr_dataset', 'rec_gt_val.txt')

        if not os.path.exists(model_dir):
            print(f"  [!] {country}: inference 모델 없음 (건너뜀)")
            continue
        if not os.path.exists(val_path):
            continue

        # 검증 샘플 읽기
        samples = []
        with open(val_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_path = os.path.join(DATA_DIR, country, parts[0])
                    if os.path.exists(img_path):
                        samples.append((img_path, parts[1]))
                if len(samples) >= sample_n:
                    break

        if not samples:
            continue

        # 추론
        try:
            ocr = PaddleOCR(rec_model_dir=model_dir,
                            rec_char_dict_path=dict_path,
                            use_angle_cls=False, use_det=False, show_log=False)
        except Exception as e:
            print(f"  [!] {country}: PaddleOCR 로드 실패 ({e})")
            continue

        correct = 0
        results = []
        for img_path, label in samples:
            try:
                result = ocr.ocr(img_path, det=False, cls=False)
                pred = result[0][0][0] if result and result[0] else ""
                conf = result[0][0][1] if result and result[0] else 0.0
            except Exception:
                pred, conf = "", 0.0

            is_ok = (pred == label)
            if is_ok:
                correct += 1
            results.append({
                'image': os.path.basename(img_path),
                'label': label,
                'prediction': pred,
                'confidence': round(conf, 4),
                'correct': is_ok,
            })

        acc = correct / len(samples) * 100 if samples else 0

        # 결과 저장 (txt)
        txt_path = os.path.join(pred_dir, f'{country}_samples.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"  {country.upper()} Sample Predictions  "
                    f"({correct}/{len(samples)} = {acc:.1f}%)\n")
            f.write(f"{'='*60}\n\n")
            for r in results:
                mark = "O" if r['correct'] else "X"
                f.write(f"  [{mark}] {r['image']}\n")
                f.write(f"         Label : {r['label']}\n")
                f.write(f"         Pred  : {r['prediction']}  "
                        f"(conf: {r['confidence']:.4f})\n\n")

        # 결과 저장 (json)
        json_path = os.path.join(pred_dir, f'{country}_samples.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'country': country, 'accuracy': acc,
                       'correct': correct, 'total': len(samples),
                       'results': results}, f, ensure_ascii=False, indent=2)

        print(f"  {country}: {correct}/{len(samples)} ({acc:.1f}%)")


# ════════════════════════════════════════════════════════
#  STEP 8. 최종 리포트
# ════════════════════════════════════════════════════════
def generate_final_report():
    print("\n" + "=" * 60)
    print("  STEP 8. 최종 리포트 생성")
    print("=" * 60)

    ensure_dir(REPORT_DIR)
    lines = []
    summary_data = {}

    lines.append("=" * 62)
    lines.append("  PaddleOCR 5개국 번호판 인식 파인튜닝 결과 리포트")
    lines.append(f"  생성 시각: {now_str()}")
    lines.append("=" * 62)

    # 환경 요약
    lines.append("\n[ 학습 환경 ]")
    env_path = os.path.join(REPORT_DIR, 'environment.txt')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if any(kw in line for kw in ['GPU 정보:', 'PaddlePaddle:', 'CUDA']):
                    lines.append(f"  {line.strip()}")
                if '  0,' in line or '  1,' in line:  # GPU lines
                    lines.append(f"    {line.strip()}")

    # 하이퍼파라미터
    lines.append(f"\n[ 하이퍼파라미터 ]")
    lines.append(f"  Epoch: {EPOCH_NUM}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    lines.append(f"  Warmup: {WARMUP_EPOCH} epoch  |  Early Stop Patience: {EARLY_STOP_PATIENCE}")
    lines.append(f"  Architecture: SVTR_LCNet (PPLCNetV3 + MultiHead CTC+NRTR)")

    # 국가별 결과 테이블
    lines.append(f"\n[ 국가별 결과 ]")
    lines.append(f"  {'Country':<10} {'Best Acc':>10} {'NED':>8} {'Time':>10} "
                 f"{'Epochs':>10} {'Early Stop':>12} {'Model MB':>10}")
    lines.append(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    total_time = 0

    for country in COUNTRIES:
        m = _load_metrics(country)
        if not m:
            lines.append(f"  {country.upper():<10} {'(no data)':>10}")
            continue

        best_acc  = m.get('best_acc', 0) * 100
        best_ned  = m.get('best_ned', 0)
        elapsed   = m.get('elapsed_minutes', 0)
        actual_ep = m.get('actual_epochs', '?')
        total_ep  = m.get('total_epochs_config', EPOCH_NUM)
        es        = "Yes" if m.get('early_stopped') else "No"
        es_epoch  = f" (ep.{m.get('best_epoch', '?')})" if m.get('early_stopped') else ""
        model_mb  = m.get('finetuned_model_size_mb', 0)

        total_time += elapsed

        lines.append(f"  {country.upper():<10} {best_acc:>9.2f}% {best_ned:>8.4f} "
                     f"{elapsed:>8.1f}m {actual_ep:>4}/{total_ep:<5} "
                     f"{es + es_epoch:>12} {model_mb:>8.1f}MB")

        summary_data[country] = {
            'best_acc': round(best_acc, 2),
            'best_ned': round(best_ned, 4),
            'elapsed_minutes': elapsed,
            'actual_epochs': actual_ep,
            'early_stopped': m.get('early_stopped', False),
            'best_epoch': m.get('best_epoch', 0),
            'model_size_mb': model_mb,
        }

    lines.append(f"\n  Total Training Time: {total_time:.1f} min ({total_time/60:.1f} hours)")

    # 데이터셋 요약
    lines.append(f"\n[ 데이터셋 ]")
    ds_path = os.path.join(REPORT_DIR, 'dataset_stats.json')
    if os.path.exists(ds_path):
        with open(ds_path, 'r', encoding='utf-8') as f:
            ds = json.load(f)
        for country, s in ds.items():
            lines.append(f"  {country.upper():<10} Train: {s['train_count']:>8,}  "
                         f"Val: {s['val_count']:>7,}  Chars: {s['total_chars']:>5}")

    # 샘플 예측 요약
    lines.append(f"\n[ 샘플 예측 결과 ]")
    for country in COUNTRIES:
        pred_json = os.path.join(REPORT_DIR, 'sample_predictions', f'{country}_samples.json')
        if os.path.exists(pred_json):
            with open(pred_json, 'r', encoding='utf-8') as f:
                p = json.load(f)
            lines.append(f"  {country.upper():<10} {p['correct']}/{p['total']} "
                         f"({p['accuracy']:.1f}%)")

    # 체크포인트 목록
    lines.append(f"\n[ 저장된 체크포인트 ]")
    for country in COUNTRIES:
        m = _load_metrics(country)
        if m and m.get('checkpoints'):
            cp_list = ', '.join(f"{cp['name']}({cp['size_mb']}MB)"
                                for cp in m['checkpoints'])
            lines.append(f"  {country.upper()}: {cp_list}")

    # 생성된 파일 목록
    lines.append(f"\n[ 생성된 파일 ]")
    lines.append(f"  training_report/environment.txt      - 환경 정보")
    lines.append(f"  training_report/dataset_stats.* - 데이터셋 통계")
    lines.append(f"  training_report/configs/              - Config YAML 백업")
    lines.append(f"  training_report/metrics/              - 학습 메트릭 JSON")
    lines.append(f"  training_report/gpu_logs/             - GPU 사용량 CSV")
    lines.append(f"  training_report/charts/               - 시각화 이미지")
    lines.append(f"  training_report/sample_predictions/   - 샘플 예측 결과")
    lines.append(f"  PaddleOCR/inference/                  - 내보낸 모델")

    # 저장
    report_txt = os.path.join(REPORT_DIR, 'training_summary.txt')
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    report_json = os.path.join(REPORT_DIR, 'training_summary.json')
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': now_str(),
            'hyperparams': {
                'epoch': EPOCH_NUM, 'batch_size': BATCH_SIZE,
                'lr': LEARNING_RATE, 'warmup': WARMUP_EPOCH,
                'early_stop_patience': EARLY_STOP_PATIENCE,
            },
            'total_training_minutes': round(total_time, 1),
            'results': summary_data,
        }, f, ensure_ascii=False, indent=2)

    print(f"  -> {report_txt}")
    print(f"  -> {report_json}")

    # 콘솔에도 출력
    print("\n" + "\n".join(lines))


# ════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("""
+----------------------------------------------------------+
|  PaddleOCR 5개국 번호판 인식 파인튜닝                      |
|  GPU: 2x RTX 4090  |  병렬 학습  |  Early Stopping        |
+----------------------------------------------------------+
    """)

    import argparse
    parser = argparse.ArgumentParser(
        description='PaddleOCR 5개국 번호판 인식 파인튜닝')
    parser.add_argument('--step', type=str, default='all',
                        help='실행할 단계 (쉼표로 복수 지정 가능)')
    args = parser.parse_args()

    steps = {
        'env':      record_environment,
        'stats':    record_dataset_stats,
        'dict':     generate_dicts,
        'download': download_pretrained,
        'config':   generate_configs,
        'train':    run_training,
        'analyze':  analyze_and_visualize,
        'export':   export_models,
        'predict':  run_sample_predictions,
        'report':   generate_final_report,
    }

    available = ', '.join(steps.keys()) + ', all'

    if args.step == 'all':
        for name, func in steps.items():
            func()
    else:
        # 쉼표로 복수 단계 지정 가능: --step train,analyze,report
        for s in args.step.split(','):
            s = s.strip()
            if s in steps:
                steps[s]()
            else:
                print(f"  [!] 알 수 없는 단계: {s}")
                print(f"  사용 가능: {available}")

    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)