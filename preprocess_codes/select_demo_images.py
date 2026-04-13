"""
시연용 이미지 선별 스크립트
============================
5개국 train+val 이미지 중 실제 서버 파이프라인(한글감지→한자감지→EfficientNet)
기준으로 국가 분류 + OCR 텍스트 인식이 모두 정확한 이미지만 골라서 demo_images/ 폴더에 복사.

사용법:
  python preprocess_codes/select_demo_images.py

출력:
  demo_images/
  ├── KOR/  (국가 정확 + OCR 정확한 이미지)
  ├── CHN/
  ├── BRA/
  ├── EUR/
  └── IND/
"""

import os
import re
import sys
import math
import glob
import shutil
import yaml
import numpy as np
import cv2
import onnxruntime as ort
import paddle.inference as pdi

os.environ['FLAGS_enable_pir_api'] = '0'

# ─── 경로 설정 ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCR_DATA_ROOT = os.path.join(PROJECT_ROOT, 'ocr_train_data_v1')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'demo_images')

EFF_ONNX_PATH = os.path.join(PROJECT_ROOT, 'finetuned_models', 'efficient_net', 'efficient_net_b0', 'finetuned_efficientnetb0.onnx')

COUNTRY_MAP = {0: 'BRA', 1: 'CHN', 2: 'EUR', 3: 'IND', 4: 'KOR'}
COUNTRY_MAP_REV = {v: k for k, v in COUNTRY_MAP.items()}

COUNTRY_MODEL_MAP = {
    'KOR': os.path.join(PROJECT_ROOT, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_korea'),
    'CHN': os.path.join(PROJECT_ROOT, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_china'),
    'BRA': os.path.join(PROJECT_ROOT, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_brazil'),
    'EUR': os.path.join(PROJECT_ROOT, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_europe_aug'),
    'IND': os.path.join(PROJECT_ROOT, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_india_aug'),
}

COUNTRY_FOLDER_MAP = {
    'KOR': 'korea', 'CHN': 'china', 'BRA': 'brazil', 'EUR': 'europe', 'IND': 'india'
}

# 제한 없음 — 전체 이미지에서 100% 정확한 것 모두 선별
TARGET_COUNT = float('inf')

# ─── EfficientNet 로드 ──────────────────────────────────────────
print("  EfficientNet 로드...")
eff_session = ort.InferenceSession(EFF_ONNX_PATH, providers=['CPUExecutionProvider'])
eff_input_name = eff_session.get_inputs()[0].name

_EFF_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_EFF_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def classify_country(img_bgr):
    """EfficientNet 국가 분류 → (country_code, confidence)"""
    img = cv2.resize(img_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _EFF_MEAN) / _EFF_STD
    tensor = img.transpose(2, 0, 1)[np.newaxis, :]  # HWC → NCHW
    logits = eff_session.run(None, {eff_input_name: tensor})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    idx = int(np.argmax(probs))
    return COUNTRY_MAP[idx], float(probs[idx])


# ─── PaddleOCR 로드 ─────────────────────────────────────────────
print("  PaddleOCR 로드...")
rec_predictors = {}
char_dicts = {}

def resize_norm_img(img, image_shape=(3, 48, 320)):
    imgC, imgH, imgW = image_shape
    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = min(imgW, int(math.ceil(imgH * ratio)))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32').transpose((2, 0, 1)) / 255.0
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, :resized_w] = resized_image
    return padding_im

def ctc_decode(preds, char_list):
    pred_indices = preds.argmax(axis=1)
    chars = []
    scores = []
    prev = -1
    for i, idx in enumerate(pred_indices):
        if idx != prev and idx != 0 and idx < len(char_list):
            chars.append(char_list[idx])
            row = preds[i]
            if abs(row.sum() - 1.0) < 0.01:
                scores.append(float(row[idx]))
            else:
                e = np.exp(row - row.max())
                scores.append(float(e[idx] / e.sum()))
        prev = idx
    text = ''.join(chars)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return text, avg_score

def run_ocr(country_code, img_bgr):
    predictor = rec_predictors[country_code]
    char_list = char_dicts[country_code]
    norm_img = resize_norm_img(img_bgr)[np.newaxis, :]
    input_handle = predictor.get_input_handle('x')
    input_handle.reshape(norm_img.shape)
    input_handle.copy_from_cpu(norm_img)
    predictor.run()
    output = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()
    return ctc_decode(output[0], char_list)

def server_pipeline(img_bgr):
    """실제 서버와 동일한 파이프라인 → (pred_country, pred_text, ocr_score)
    1) KOR 모델 실행 → 한글 감지 시 KOR 확정
    2) CHN 모델 실행 → 한자 감지 시 CHN 확정
    3) EfficientNet → BRA/EUR/IND 분류 후 해당 모델 실행
    """
    kor_text, kor_score = run_ocr('KOR', img_bgr)
    if re.search(r'[가-힣]', kor_text) and kor_score > 0.3:
        return 'KOR', kor_text, kor_score

    chn_text, chn_score = run_ocr('CHN', img_bgr)
    if re.search(r'[\u4e00-\u9fff]', chn_text) and chn_score > 0.3:
        return 'CHN', chn_text, chn_score

    pred_country, _ = classify_country(img_bgr)
    pred_text, ocr_score = run_ocr(pred_country, img_bgr)
    return pred_country, pred_text, ocr_score

for code, model_dir in COUNTRY_MODEL_MAP.items():
    config = pdi.Config(
        os.path.join(model_dir, 'inference.pdmodel'),
        os.path.join(model_dir, 'inference.pdiparams')
    )
    config.disable_gpu()
    config.disable_glog_info()
    config.enable_mkldnn()
    config.set_cpu_math_library_num_threads(4)
    config.switch_ir_optim(True)
    rec_predictors[code] = pdi.create_predictor(config)

    yml_path = os.path.join(model_dir, 'inference.yml')
    with open(yml_path, 'r', encoding='utf-8') as f:
        yml = yaml.safe_load(f)
    char_dicts[code] = ['blank'] + yml['PostProcess']['character_dict']
    print(f"    {code}: {len(char_dicts[code])} chars")


# ─── 워밍업 ─────────────────────────────────────────────────────
print("  워밍업...")
dummy = np.random.randint(0, 255, (120, 400, 3), dtype=np.uint8)
classify_country(dummy)
for code in COUNTRY_MODEL_MAP:
    run_ocr(code, dummy)
    run_ocr(code, dummy)
print("  워밍업 완료\n")


# ─── 메인: 국가별 선별 ──────────────────────────────────────────
def load_all_labels(country_code):
    """train + val 라벨 전체 로드 → [(img_path, label), ...]"""
    folder = COUNTRY_FOLDER_MAP[country_code]
    samples = []
    for txt_name in ['rec_gt_train.txt', 'rec_gt_val.txt']:
        txt_path = os.path.join(OCR_DATA_ROOT, folder, 'ocr_dataset', txt_name)
        if not os.path.exists(txt_path):
            continue
        country_dir = os.path.join(OCR_DATA_ROOT, folder)
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' not in line:
                    continue
                img_rel, label = line.split('\t', 1)
                img_path = os.path.join(country_dir, img_rel)
                samples.append((img_path, label))
    return samples


print("=" * 60)
print("  시연용 이미지 선별 시작")
print("=" * 60)

total_selected = 0

for country_code in ['KOR', 'CHN', 'BRA', 'EUR', 'IND']:
    print(f"\n--- {country_code} ---")

    out_dir = os.path.join(OUTPUT_DIR, country_code)
    os.makedirs(out_dir, exist_ok=True)

    samples = load_all_labels(country_code)
    if not samples:
        print(f"  라벨 파일 없음, 건너뜀")
        continue

    print(f"  전체: {len(samples)}장 검사 시작")

    selected = 0
    tested = 0
    country_fail = 0
    ocr_fail = 0

    for img_path, label in samples:
        if selected >= TARGET_COUNT:
            break

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        tested += 1

        # 1) 서버 파이프라인으로 국가 분류 + OCR
        pred_country, pred_text, ocr_score = server_pipeline(img)
        if pred_country != country_code:
            country_fail += 1
            continue

        # 2) OCR 결과 검증
        if pred_text != label:
            ocr_fail += 1
            continue

        # 3) 둘 다 정확 → 복사
        ext = os.path.splitext(img_path)[1]
        save_name = f"{label}{ext}"
        save_path = os.path.join(out_dir, save_name)
        counter = 1
        while os.path.exists(save_path):
            save_name = f"{label}_{counter}{ext}"
            save_path = os.path.join(out_dir, save_name)
            counter += 1

        shutil.copy2(img_path, save_path)
        selected += 1

        if tested % 1000 == 0:
            print(f"    검사 {tested}/{len(samples)} | 선별 {selected}장 (국가X {country_fail}, OCRX {ocr_fail})")

    total_selected += selected
    print(f"  결과: {selected}장 선별 (검사 {tested}장, 국가 실패 {country_fail}, OCR 실패 {ocr_fail})")

print(f"\n{'=' * 60}")
print(f"  완료! 총 {total_selected}장 → {OUTPUT_DIR}")
print(f"{'=' * 60}")

# 최종 요약
for country_code in ['KOR', 'CHN', 'BRA', 'EUR', 'IND']:
    d = os.path.join(OUTPUT_DIR, country_code)
    count = len(glob.glob(os.path.join(d, '*'))) if os.path.exists(d) else 0
    print(f"  {country_code}: {count}장")
