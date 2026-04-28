"""
다국적 자동차 번호판 인식 AI 서버 (port 8001)

파이프라인: YOLO(번호판 검출) → EfficientNet(국가 분류) → PaddleOCR(문자 인식) → 정규식(검증)
"""
import os
import re
import cv2
import math
import threading
import traceback
import time
import base64

import numpy as np
import yaml
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from ultralytics import YOLO
import paddle.inference as pdi
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

os.environ['FLAGS_enable_pir_api'] = '0'

# ═══════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# YOLO 모델 경로
YOLO_ONNX = os.path.join(BASE_DIR, 'finetuned_models', 'yolo', 'yolov11small640_acc8400', 'best.onnx')
YOLO_PT   = os.path.join(BASE_DIR, 'finetuned_models', 'yolo', 'yolov11small640_acc8400', 'best.pt')
YOLO_PATH = YOLO_ONNX if os.path.exists(YOLO_ONNX) else YOLO_PT
EFF_PATH  = os.path.join(BASE_DIR, 'finetuned_models', 'country_classification', 'efficientnet_b0', 'country_efficientnetb0_v3.onnx')

# EV 분류기 (한국 번호판 전용, 파인튜닝된 EfficientNet-B0, ONNX)
EV_MODEL_PATH = os.path.join(BASE_DIR, 'finetuned_models', 'ev_classification', 'mobilenet_v3', 'ev_mobilenetv3_v2.onnx')

# EfficientNet 출력 인덱스 → 국가코드
COUNTRY_MAP = {0: 'BRA', 1: 'CHN', 2: 'EUR', 3: 'IND', 4: 'KOR'}

# 국가별 PaddleOCR 모델 디렉토리
OCR_MODELS = {
    'KOR': os.path.join(BASE_DIR, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_korea'),
    'CHN': os.path.join(BASE_DIR, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_china'),
    'BRA': os.path.join(BASE_DIR, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_brazil'),
    'EUR': os.path.join(BASE_DIR, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_europe_aug'),
    'IND': os.path.join(BASE_DIR, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_india_aug'),
}

# 국가별 번호판 정규식
# KOR: 12가3456, 123가4567, 서울12가3456, 충북88아1511
# CHN: 皖AJ9T46  BRA: ABC1234, ABC1D23  EUR: 영숫자 4~10자  IND: MH12AB1234
PLATE_REGEX = {
    'KOR': re.compile(r'^(([가-힣]{2,4}\d{0,3})|(\d{2,3}))[가-힣]\d{4}$'),
    'CHN': re.compile(r'^[\u4e00-\u9fff][A-Z][A-Z0-9]{5}$'),
    'BRA': re.compile(r'^[A-Z]{3}\d{4}$|^[A-Z]{3}\d[A-Z]\d{2}$'),
    'EUR': re.compile(r'^[A-Z0-9]{4,10}$'),
    'IND': re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$'),
}

# EfficientNet ImageNet 정규화 상수
_EFF_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_EFF_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════
# 전처리 / 디코딩 함수
# ═══════════════════════════════════════════════════════════════════
def validate_plate(text, country_code):
    """번호판 텍스트가 해당 국가 정규식에 맞는지 검증"""
    pattern = PLATE_REGEX.get(country_code)
    if not pattern or not text:
        return False
    return bool(pattern.match(text.replace(' ', '').replace('-', '').upper()))


def eff_preprocess(img_bgr):
    """EfficientNet 입력 전처리: 224x224 리사이즈 → RGB → ImageNet 정규화 → NCHW"""
    img = cv2.resize(img_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _EFF_MEAN) / _EFF_STD
    return img.transpose(2, 0, 1)[np.newaxis, :]


def ocr_preprocess(img, target_shape=(3, 48, 320)):
    """PaddleOCR 입력 전처리: 종횡비 유지 리사이즈 + 우측 제로패딩"""
    C, H, W = target_shape
    h, w = img.shape[:2]
    resized_w = min(W, int(math.ceil(H * w / float(h))))
    resized = cv2.resize(img, (resized_w, H)).astype('float32')
    resized = resized.transpose((2, 0, 1)) / 255.0
    resized = (resized - 0.5) / 0.5
    padded = np.zeros((C, H, W), dtype=np.float32)
    padded[:, :, :resized_w] = resized
    return padded


def _to_prob(preds):
    """모델 출력을 softmax 확률로 변환 (이미 softmax면 그대로 반환)"""
    if abs(preds[0].sum() - 1.0) < 0.01:
        return preds
    e = np.exp(preds - preds.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def ctc_beam_decode(preds, char_list, beam_width=5, top_n=3):
    """CTC beam search 디코딩: 모델 출력에서 상위 N개 텍스트 후보 반환

    greedy decode(1개)와 달리, 여러 후보를 뽑아 정규식 검증 기회를 넓힘.
    예) 1순위 "12가345"(정규식 실패) → 2순위 "12가3456"(정규식 통과)
    """
    probs = _to_prob(preds)
    T, C = probs.shape
    BLANK = 0
    beams = {((), BLANK): 0.0}

    for t in range(T):
        new_beams = {}
        top_k = min(beam_width, C)
        top_indices = np.argpartition(probs[t], -top_k)[-top_k:]

        for (prefix, last), log_p in beams.items():
            for idx in top_indices:
                idx = int(idx)
                if idx >= C or probs[t][idx] < 1e-6:
                    continue
                lp = log_p + float(np.log(probs[t][idx] + 1e-12))

                if idx == BLANK:
                    key = (prefix, BLANK)
                elif idx == last:
                    key = (prefix, idx)       # CTC 규칙: 연속 동일 문자는 병합
                else:
                    key = (prefix + (idx,), idx)

                if key not in new_beams or lp > new_beams[key]:
                    new_beams[key] = lp

        # 빔 가지치기: 상위 beam_width*2 개만 유지
        if len(new_beams) > beam_width * 2:
            new_beams = dict(sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width * 2])
        beams = new_beams

    # 동일 prefix 합산 후 상위 top_n 추출
    merged = {}
    for (prefix, _), lp in beams.items():
        if prefix not in merged or lp > merged[prefix]:
            merged[prefix] = lp

    results = []
    for prefix, lp in sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        text = ''.join(char_list[i] for i in prefix if 0 < i < len(char_list))
        avg_score = float(np.exp(lp / max(len(prefix), 1)))
        results.append((text, avg_score))

    return results if results else [("", 0.0)]

# ═══════════════════════════════════════════════════════════════════
# 모델 로드 (서버 기동 시 1회)
# ═══════════════════════════════════════════════════════════════════
yolo_model = YOLO(YOLO_PATH)

eff_session    = ort.InferenceSession(EFF_PATH, providers=['CPUExecutionProvider'])
eff_input_name = eff_session.get_inputs()[0].name

# EV 분류기 (EfficientNet-B0, ONNX, 2-class: ev_false=0, ev_true=1)
ev_session    = ort.InferenceSession(EV_MODEL_PATH, providers=['CPUExecutionProvider'])
ev_input_name = ev_session.get_inputs()[0].name
print(f"  EV Classifier (EfficientNet-B0, ONNX): {EV_MODEL_PATH}")

_EV_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_ev(plate_img_bgr):
    """번호판 크롭 이미지에서 EV 여부 판별 (한국 번호판 전용, ONNX 추론)
    Returns: (is_ev: bool, ev_confidence: float 0~1)
    """
    rgb = cv2.cvtColor(plate_img_bgr, cv2.COLOR_BGR2RGB)
    tensor = _EV_TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).numpy()  # (1,3,224,224)
    logits = ev_session.run(None, {ev_input_name: tensor})[0]          # (1, 2)
    # softmax (수치 안정성 위해 max를 빼서 계산)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    ev_prob = float(probs[0, 1])  # class 1 = ev_true
    return ev_prob > 0.5, ev_prob

# PaddleOCR: 5개국 모델 + Lock (동시 요청 시 predictor 보호)
rec_predictors = {}
char_dicts     = {}
_ocr_locks     = {}

for code, model_dir in OCR_MODELS.items():
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

    with open(os.path.join(model_dir, 'inference.yml'), 'r', encoding='utf-8') as f:
        char_dicts[code] = ['blank'] + yaml.safe_load(f)['PostProcess']['character_dict']

    _ocr_locks[code] = threading.Lock()
    print(f"  PaddleOCR ({code}): {len(char_dicts[code])} chars")


def run_ocr(country_code, plate_img):
    """단일 국가 OCR 실행 → 상위 3개 후보 [(text, score), ...] 반환"""
    norm_img = ocr_preprocess(plate_img)[np.newaxis, :]

    with _ocr_locks[country_code]:
        predictor = rec_predictors[country_code]
        handle = predictor.get_input_handle('x')
        handle.reshape(norm_img.shape)
        handle.copy_from_cpu(norm_img)
        predictor.run()
        output = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()

    return ctc_beam_decode(output[0], char_dicts[country_code])

# ═══════════════════════════════════════════════════════════════════
# 서버 시작 (워밍업 → API 등록)
# ═══════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """MKL-DNN 그래프 컴파일을 위해 더미 추론 2회 실행 (첫 요청 지연 방지)"""
    print("  워밍업 시작...")
    dummy_img   = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_plate = np.random.randint(0, 255, (120, 400, 3), dtype=np.uint8)

    t = time.time()
    yolo_model(dummy_img, verbose=False)
    print(f"    YOLO 워밍업 완료 ({time.time()-t:.2f}s)")

    t = time.time()
    eff_session.run(None, {eff_input_name: eff_preprocess(dummy_plate)})
    print(f"    EfficientNet 워밍업 완료 ({time.time()-t:.2f}s)")

    t = time.time()
    predict_ev(dummy_plate)
    print(f"    MobileNet-V3 워밍업 완료 ({time.time()-t:.2f}s)")

    for code in rec_predictors:
        t = time.time()
        run_ocr(code, dummy_plate)
        run_ocr(code, dummy_plate)
        print(f"    PaddleOCR ({code}) 워밍업 완료 ({time.time()-t:.2f}s)")

    print("  서버 준비 완료")
    yield

app = FastAPI(lifespan=lifespan)


@app.post("/license-plates-recognition")
async def recognize(file: UploadFile = File(...)):
    try:
        # 이미지 디코딩
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"status": "error", "message": "이미지 읽기 실패"}

        t1 = time.time()

        # ── 1단계: YOLO 번호판 검출 → 크롭 ──
        is_detected = False
        plate_crop = img

        boxes = yolo_model(img)[0].boxes
        if len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
            h_img, w_img = img.shape[:2]
            pad = int((x2 - x1) * 0.03)
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
            plate_crop = img[y1:y2, x1:x2]
            is_detected = True

        if plate_crop.size == 0:
            plate_crop = img

        t2 = time.time()

        # ── 2단계: EfficientNet 국가 분류 → 확률순 정렬 ──
        logits    = eff_session.run(None, {eff_input_name: eff_preprocess(plate_crop)})[0][0]
        e         = np.exp(logits - logits.max())
        all_probs = e / e.sum()

        country_code     = COUNTRY_MAP[int(np.argmax(all_probs))]
        country_accuracy = round(float(all_probs.max()) * 100, 2)
        ranked_countries = [COUNTRY_MAP[int(i)] for i in np.argsort(all_probs)[::-1]]

        t3 = time.time()

        # ── 3단계: 확률 높은 국가부터 순차 OCR → 정규식 통과 시 확정 ──
        v_num    = "인식불가"
        acc      = 0.0
        is_valid = False
        attempts = []

        for rank, code in enumerate(ranked_countries, 1):
            try:
                candidates = run_ocr(code, plate_crop)
                for ci, (text, score) in enumerate(candidates, 1):
                    valid = validate_plate(text, code)
                    attempts.append((code, text, score, valid, rank, ci))

                    if valid and not is_valid:
                        country_code     = code
                        v_num            = text
                        acc              = round(score * 100, 2)
                        country_accuracy = round(float(all_probs[list(COUNTRY_MAP.values()).index(code)]) * 100, 2)
                        is_valid         = True
            except Exception as ocr_err:
                print(f"    {rank}순위 {code}: OCR 오류 → {ocr_err}")

            if is_valid:
                break

        t4 = time.time()

        # ── 4단계: EV 판별 (한국 번호판만) ──
        is_ev = False
        ev_conf = 0.0
        if country_code == 'KOR':
            is_ev, ev_conf = predict_ev(plate_crop)

        t5 = time.time()

        # ── 콘솔 로그 ──
        # 5개국 분류 확률을 확률순으로 정렬 (국가명 + %)
        sorted_idx = np.argsort(all_probs)[::-1]
        eff_ranking = " > ".join(
            f"{COUNTRY_MAP[int(i)]}({float(all_probs[i]) * 100:.2f}%)" for i in sorted_idx
        )

        print("=" * 75)
        print(f"결과  : {country_code} ({country_accuracy}%) | {v_num} ({acc}%)")
        print(f"YOLO  : {'적용' if is_detected else '미적용'}")
        print(f"Eff   : {eff_ranking}")
        for c, t, s, v, r, ci in attempts:
            tag = " [정규식 통과]" if v else ""
            print(f"Eff-{r} : {c} | YOLO-{ci} : {t or '(빈 결과)'} ({round(s*100, 2)}%){tag}")
        if not is_valid:
            print(f"!! 5개국 정규식 모두 실패 (인식불가)")
        if country_code == 'KOR':
            print(f"EV    : {'전기차 ✓' if is_ev else '일반 차량'} (EV 확률 {ev_conf*100:.2f}%)")
        print(f"속도  : YOLO {t2-t1:.3f}s | Eff {t3-t2:.3f}s | OCR {t4-t3:.3f}s | EV {t5-t4:.3f}s | 합계 {t5-t1:.3f}s")
        print("=" * 75)

        # ── 응답 ──
        _, encoded = cv2.imencode('.jpg', plate_crop)
        return {
            "status": "success",
            "is_yolo_detected": is_detected,
            "license_plate_country": country_code,
            "country_accuracy": country_accuracy,
            "vehicle_num": v_num,
            "ocr_accuracy": acc,
            "is_ev_license_plate": is_ev,
            "ev_confidence": round(ev_conf * 100, 2),
            "plate_img_base64": base64.b64encode(encoded).decode('utf-8')
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": f"AI 서버 오류: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
 