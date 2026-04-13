"""
PaddleOCR 유럽/인도 전용 데이터 증강 파인튜닝 스크립트
=================================================================
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
REPORT_DIR     = os.path.join(PARENT_DIR, 'training_report_aug')

COUNTRIES = {
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

EPOCH_NUM      = 200
BATCH_SIZE     = 64
LEARNING_RATE  = 0.0005
WARMUP_EPOCH   = 5
NUM_WORKERS    = 0
EVAL_STEP      = 500

EARLY_STOP_PATIENCE = 15
GPU_LOG_INTERVAL = 30       
GPU_SCHEDULE = [
    [('0', 'europe'), ('1', 'india')],
]

RE_EPOCH    = re.compile(r'epoch:\s*\[(\d+)/(\d+)\]')
RE_STEP     = re.compile(r'(?:global_step|iter):\s*(\d+)')
RE_LR       = re.compile(r'\blr:\s*([\d.eE+-]+)')
RE_LOSS     = re.compile(r'(?<!_)loss:\s*([\d.]+)')            
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

def dir_size_mb(path):
    total = 0
    if os.path.exists(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)

# ════════════════════════════════════════════════════════
#  STEP 3. Config YAML 생성 + 백업
# ════════════════════════════════════════════════════════
def _find_pretrained_path(model_name):
    base_dir = os.path.join(PADDLE_OCR_DIR, 'pretrained_models', model_name)
    if not os.path.exists(base_dir): return None
    candidates = [
        os.path.join(base_dir, 'best_model', 'model'),
        os.path.join(base_dir, 'best_accuracy', 'model'),
        os.path.join(base_dir, 'best_accuracy'),
        os.path.join(base_dir, 'model'),
    ]
    for path in candidates:
        if os.path.exists(path + '.pdparams'):
            return os.path.relpath(path, PADDLE_OCR_DIR)
    return None

def generate_configs():
    print("\n" + "=" * 60)
    print("  STEP 3. 증강 전용 Config 생성 + 백업")
    print("=" * 60)

    config_dir = os.path.join(PADDLE_OCR_DIR, 'configs', 'rec', 'finetune')
    backup_dir = ensure_dir(os.path.join(REPORT_DIR, 'configs'))
    os.makedirs(config_dir, exist_ok=True)

    for country, (model_name, lang, url) in COUNTRIES.items():
        dict_path  = os.path.join('..', 'ocr_train_data_v1', country, 'ocr_dataset', f'{country}_dict.txt')
        pretrained = _find_pretrained_path(model_name)
        if not pretrained:
            pretrained = os.path.join('pretrained_models', model_name, 'best_model', 'model')

        # 🚨 오류 완벽 수정: Head 구조 정확히 맞춤
        config = {
            'Global': {
                'debug': False, 
                'use_gpu': True, 
                'epoch_num': EPOCH_NUM,
                'log_smooth_window': 20,              # 🚨 복구된 필수 파라미터 1
                'print_batch_step': 10,               # 🚨 복구된 필수 파라미터 2
                'save_model_dir': f'./output/rec_{country}_aug',
                'save_epoch_step': 10, 
                'eval_batch_step': [0, EVAL_STEP],
                'cal_metric_during_train': True,      # 🚨 복구된 필수 파라미터 3 (평가용)
                'pretrained_model': pretrained,
                'checkpoints': None,
                'save_inference_dir': f'./inference/rec_{country}_aug',
                'use_visualdl': False,
                'infer_img': None,
                'character_dict_path': dict_path,
                'max_text_length': 25, 
                'infer_mode': False,
                'use_space_char': False,
            },
            'Optimizer': {
                'name': 'Adam', 'beta1': 0.9, 'beta2': 0.999,
                'lr': {'name': 'Cosine', 'learning_rate': LEARNING_RATE, 'warmup_epoch': WARMUP_EPOCH},
                'regularizer': {'name': 'L2', 'factor': 3.0e-05},
            },
            'Architecture': {
                'model_type': 'rec', 'algorithm': 'SVTR_LCNet',
                'Backbone': {'name': 'PPLCNetV3', 'scale': 0.95},
                'Head': {
                    'name': 'MultiHead',
                    'head_list': [
                        {'CTCHead': {
                            'Neck': {'name': 'svtr', 'dims': 120, 'depth': 2, 'hidden_dims': 120, 'kernel_size': [1, 3], 'use_guide': True},
                            'Head': {'fc_decay': 0.00001}
                        }},
                        {'NRTRHead': {'nrtr_dim': 384, 'max_text_length': 25}},
                    ],
                },
            },
            'Loss': {
                'name': 'MultiLoss',
                'loss_config_list': [{'CTCLoss': None}, {'NRTRLoss': None}],
            },
            'PostProcess': {'name': 'CTCLabelDecode'},
            'Metric': {'name': 'RecMetric', 'main_indicator': 'acc'},
            'Train': {
                'dataset': {
                    'name': 'SimpleDataSet',
                    'data_dir': f'../ocr_train_data_v1/{country}',
                    'ext_op_transform_idx': 1,
                    'label_file_list': [f'../ocr_train_data_v1/{country}/ocr_dataset/rec_gt_train.txt'],
                    'transforms': [
                        {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                        {'RecConAug': {'prob': 0.8, 'ext_data_num': 2, 'image_shape': [48, 320, 3], 'max_text_length': 25}},
                        {'MultiLabelEncode': {'gtc_encode': 'NRTRLabelEncode'}},
                        {'RecResizeImg': {'image_shape': [3, 48, 320]}},
                        {'KeepKeys': {'keep_keys': ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio']}},
                    ],
                },
                'loader': {
                    'shuffle': True, 'batch_size_per_card': BATCH_SIZE, 'drop_last': True, 'num_workers': NUM_WORKERS,
                },
            },
            'Eval': {
                'dataset': {
                    'name': 'SimpleDataSet',
                    'data_dir': f'../ocr_train_data_v1/{country}',
                    'label_file_list': [f'../ocr_train_data_v1/{country}/ocr_dataset/rec_gt_val.txt'],
                    'transforms': [
                        {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                        {'MultiLabelEncode': {'gtc_encode': 'NRTRLabelEncode'}},
                        {'RecResizeImg': {'image_shape': [3, 48, 320]}},
                        {'KeepKeys': {'keep_keys': ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio']}},
                    ],
                },
                'loader': {
                    'shuffle': False, 'batch_size_per_card': BATCH_SIZE, 'drop_last': False, 'num_workers': NUM_WORKERS,
                },
            },
        }

        config_path = os.path.join(config_dir, f'rec_{country}_aug.yml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        shutil.copy2(config_path, os.path.join(backup_dir, f'rec_{country}_aug.yml'))
        print(f"  {country} -> {config_path}  (강력한 증강 모드 세팅 완료!)")

# ════════════════════════════════════════════════════════
#  실행 파트
# ════════════════════════════════════════════════════════
def _gpu_monitor_loop(gpu_id, log_path, stop_event):
    with open(log_path, 'w') as f:
        f.write('timestamp,gpu_util_%,mem_used_mb,mem_total_mb,temp_c,power_w\n')
        while not stop_event.is_set():
            try:
                r = subprocess.run(['nvidia-smi', f'--id={gpu_id}', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    f.write(f"{now_str()},{r.stdout.strip()}\n")
                    f.flush()
            except: pass
            stop_event.wait(GPU_LOG_INTERVAL)

def _detect_gpu_count():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        if r.returncode == 0: return len([line for line in r.stdout.strip().split('\n') if line.strip()])
    except: pass
    return 0

def train_country(gpu_id, country):
    config_path  = os.path.join('configs', 'rec', 'finetune', f'rec_{country}_aug.yml')
    train_script = os.path.join('tools', 'train.py')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    env['PYTHONUNBUFFERED'] = '1'
    cmd = [sys.executable, train_script, '-c', config_path]

    log_dir     = os.path.join(PADDLE_OCR_DIR, 'output', f'rec_{country}_aug')
    metrics_dir = ensure_dir(os.path.join(REPORT_DIR, 'metrics'))
    gpu_log_dir = ensure_dir(os.path.join(REPORT_DIR, 'gpu_logs'))
    os.makedirs(log_dir, exist_ok=True)
    log_path     = os.path.join(log_dir, 'train.log')
    gpu_log_path = os.path.join(gpu_log_dir, f'{country}_gpu.csv')

    metrics = {'country': country, 'gpu_id': gpu_id, 'start_time': now_str(), 'end_time': None, 'elapsed_minutes': 0, 'total_epochs_config': EPOCH_NUM, 'actual_epochs': 0, 'early_stopped': False, 'early_stop_eval_count': 0, 'best_acc': 0.0, 'best_ned': 0.0, 'best_epoch': 0, 'best_step': 0, 'final_acc': 0.0, 'train_steps': [], 'eval_steps': [] }

    gpu_stop = threading.Event()
    gpu_thread = threading.Thread(target=_gpu_monitor_loop, args=(gpu_id, gpu_log_path, gpu_stop), daemon=True)
    gpu_thread.start()

    print(f"\n  >>> [{country}] 증강 훈련 시작! GPU {gpu_id}")
    start_time    = time.time()
    no_improve    = 0
    current_epoch = 0
    is_eval_line  = False

    proc = subprocess.Popen(cmd, env=env, cwd=PADDLE_OCR_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
    with open(log_path, 'w', encoding='utf-8') as log_file:
        try:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                m_epoch = RE_EPOCH.search(line)
                m_step  = RE_STEP.search(line)
                m_loss  = RE_LOSS.search(line)

                if m_epoch and m_step and m_loss:
                    current_epoch = int(m_epoch.group(1))
                    step_data = {'step': int(m_step.group(1)), 'epoch': current_epoch, 'loss': float(m_loss.group(1))}
                    metrics['train_steps'].append(step_data)

                if ('metric' in line.lower() or 'acc' in line) and RE_ACC.search(line):
                    if any(kw in line.lower() for kw in ['cur metric', 'metric eval', 'cur_metric']):
                        m_acc, m_ned = RE_ACC.search(line), RE_NED.search(line)
                        if m_acc:
                            acc = float(m_acc.group(1))
                            ned = float(m_ned.group(1)) if m_ned else 0.0
                            metrics['eval_steps'].append({'step': metrics['train_steps'][-1]['step'] if metrics['train_steps'] else 0, 'epoch': current_epoch, 'acc': acc, 'ned': ned, 'is_best': False})
                            is_eval_line = True

                    if 'best' in line.lower() and is_eval_line:
                        is_best = False
                        if re.search(r'(?:update_best|is_best).*?True', line, re.IGNORECASE): is_best = True
                        elif re.search(r'(?:update_best|is_best).*?False', line, re.IGNORECASE): is_best = False
                        else:
                            m_bacc = RE_ACC.search(line)
                            if m_bacc and metrics['eval_steps']:
                                is_best = (float(m_bacc.group(1)) >= metrics['eval_steps'][-1]['acc'] - 1e-6)

                        if metrics['eval_steps']: metrics['eval_steps'][-1]['is_best'] = is_best
                        if is_best:
                            ea = metrics['eval_steps'][-1]
                            metrics['best_acc'], metrics['best_ned'], metrics['best_epoch'] = ea['acc'], ea['ned'], current_epoch
                            no_improve = 0
                            print(f"      [{country}] NEW BEST acc={ea['acc']:.4f} (epoch {current_epoch})")
                        else:
                            no_improve += 1
                        is_eval_line = False

                        if EARLY_STOP_PATIENCE > 0 and no_improve >= EARLY_STOP_PATIENCE:
                            print(f"\n  <<< [{country}] EARLY STOPPING (best acc={metrics['best_acc']:.4f})")
                            metrics['early_stopped'], metrics['early_stop_eval_count'] = True, no_improve
                            proc.terminate()
                            break
        except Exception as e: print(f"  [!] [{country}] 파싱 예외: {e}")
    proc.wait()
    gpu_stop.set()
    gpu_thread.join(timeout=5)

    metrics['end_time'] = now_str()
    metrics['elapsed_minutes'] = round((time.time() - start_time) / 60, 1)
    metrics['actual_epochs'] = current_epoch
    metrics['final_acc'] = metrics['eval_steps'][-1]['acc'] if metrics['eval_steps'] else 0.0

    metrics_path = os.path.join(metrics_dir, f'{country}_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  [DONE] {country} | {metrics['elapsed_minutes']:.1f}min | best_acc={metrics['best_acc']:.4f}")

def run_training():
    print("\n" + "=" * 60)
    print("  STEP 4. 증강 학습 시작 (인도, 유럽)")
    print("=" * 60)
    os.chdir(PADDLE_OCR_DIR)
    gpu_count = _detect_gpu_count()
    if gpu_count >= 2: schedule = GPU_SCHEDULE
    else: schedule = [[('0', ctry)] for pairs in GPU_SCHEDULE for _, ctry in pairs]

    total_start = time.time()
    for round_num, pairs in enumerate(schedule, 1):
        if len(pairs) == 1:
            train_country(pairs[0][0], pairs[0][1])
        else:
            with ProcessPoolExecutor(max_workers=len(pairs)) as executor:
                futures = [executor.submit(train_country, gid, ctry) for gid, ctry in pairs]
                for f in futures: f.result()
    print(f"\n  === 증강 학습 완료: {(time.time() - total_start) / 60:.1f}분 ===")

def export_models():
    print("\n" + "=" * 60)
    print("  STEP 5. 증강 모델 내보내기")
    print("=" * 60)
    os.chdir(PADDLE_OCR_DIR)
    for country in COUNTRIES:
        config_path = os.path.join('configs', 'rec', 'finetune', f'rec_{country}_aug.yml')
        best_model  = os.path.join('output', f'rec_{country}_aug', 'best_model', 'model')
        save_dir    = os.path.join('inference', f'rec_{country}_aug')
        if not os.path.exists(best_model + '.pdparams'): continue
        subprocess.run([sys.executable, os.path.join('tools', 'export_model.py'), '-c', config_path, '-o', f'Global.pretrained_model={best_model}', '-o', f'Global.save_inference_dir={save_dir}'], capture_output=True)
        print(f"  [OK] {country} -> {save_dir}")

if __name__ == '__main__':
    generate_configs()
    run_training()
    export_models()
    print("\n  [🌟] 인도, 유럽 데이터 증강 훈련이 모두 종료되었습니다!")