import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import yaml
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image
import paddle.inference as pdi
import onnxruntime as ort
import traceback
import time
import base64

os.environ['FLAGS_enable_pir_api'] = '0'

# ─── 모델 경로 설정 ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ONNX_PATH     = os.path.join(BASE_DIR, 'finetuned_models', 'yolo', 'yolo_11n', 'finetuned_yolo11n_acc822.onnx')
PT_PATH       = os.path.join(BASE_DIR, 'finetuned_models', 'yolo', 'yolo_11n', 'finetuned_yolo11n_acc822.pt')
YOLO_PATH     = ONNX_PATH if os.path.exists(ONNX_PATH) else PT_PATH
EFF_ONNX_PATH = os.path.join(BASE_DIR, 'finetuned_models', 'efficient_net', 'efficient_net_b0', 'finetuned_efficientnetb0.onnx')
EFF_PT_PATH   = os.path.join(BASE_DIR, 'finetuned_models', 'efficient_net', 'efficient_net_b0', 'finetuned_efficientnetb0.pt')

COUNTRY_MAP = {0: 'BRA', 1: 'CHN', 2: 'EUR', 3: 'IND', 4: 'KOR'}

COUNTRY_MODEL_MAP = {
    'KOR': os.path.join(BASE_DIR, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_korea'),
    'CHN': os.path.join(BASE_DIR, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_china'),
    'BRA': os.path.join(BASE_DIR, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_brazil'),
    'EUR': os.path.join(BASE_DIR, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_europe_aug'),
    'IND': os.path.join(BASE_DIR, 'finetuned_models', 'paddle_ocr', 'inference_v2', 'rec_india_aug'),
}

# character dict도 같은 폴더의 inference.yml에서 읽기

# ─── YOLO 로드 ──────────────────────────────────────────────────
yolo_model = YOLO(YOLO_PATH)

# ─── EfficientNet 로드 ──────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(EFF_ONNX_PATH):
    eff_session    = ort.InferenceSession(EFF_ONNX_PATH, providers=['CPUExecutionProvider'])
    eff_input_name = eff_session.get_inputs()[0].name
    use_onnx_eff   = True
    print(f"  EfficientNet ONNX: {EFF_ONNX_PATH}")
else:
    classify_model = models.efficientnet_b0()
    classify_model.classifier[1] = nn.Linear(classify_model.classifier[1].in_features, 5)
    classify_model.load_state_dict(torch.load(EFF_PT_PATH, map_location=device))
    classify_model.to(device).eval()
    use_onnx_eff = False
    print(f"  EfficientNet PyTorch: {EFF_PT_PATH}")

eff_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─── PaddleOCR Recognition: Paddle Inference 직접 사용 ───────────
rec_predictors = {}  # country_code -> predictor
char_dicts = {}      # country_code -> ['blank', '-', '0', '1', ...]

def resize_norm_img(img, image_shape=(3, 48, 320)):
    """PaddleOCR 학습과 동일한 전처리: 종횡비 유지 + 우측 패딩"""
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
    """CTC greedy decode: 연속 중복 제거 + blank(0) 제거"""
    pred_indices = preds.argmax(axis=1)
    chars = []
    scores = []
    prev = -1
    for i, idx in enumerate(pred_indices):
        if idx != prev and idx != 0 and idx < len(char_list):
            chars.append(char_list[idx])
            # softmax 확률 계산
            row = preds[i]
            if abs(row.sum() - 1.0) < 0.01:  # 이미 softmax된 경우
                scores.append(float(row[idx]))
            else:  # logits인 경우
                e = np.exp(row - row.max())
                scores.append(float(e[idx] / e.sum()))
        prev = idx
    text = ''.join(chars)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return text, avg_score

def load_rec_predictor(country_code):
    """Paddle Inference로 rec 모델 로드"""
    model_dir = COUNTRY_MODEL_MAP[country_code]

    config = pdi.Config(
        os.path.join(model_dir, 'inference.pdmodel'),
        os.path.join(model_dir, 'inference.pdiparams')
    )
    config.disable_gpu()
    config.disable_glog_info()
    config.enable_mkldnn()
    config.set_cpu_math_library_num_threads(4)
    config.switch_ir_optim(True)
    rec_predictors[country_code] = pdi.create_predictor(config)

    # character dict 로드 (같은 폴더의 inference.yml)
    yml_path = os.path.join(model_dir, 'inference.yml')
    with open(yml_path, 'r', encoding='utf-8') as f:
        yml = yaml.safe_load(f)
    char_dicts[country_code] = ['blank'] + yml['PostProcess']['character_dict']

    n_chars = len(char_dicts[country_code])
    print(f"  PaddleOCR rec ({country_code}): {model_dir} ({n_chars} chars)")

def run_ocr(country_code, plate_img):
    """단일 국가 OCR 실행 → (text, score)"""
    predictor = rec_predictors[country_code]
    char_list = char_dicts[country_code]

    norm_img = resize_norm_img(plate_img)[np.newaxis, :]

    input_handle = predictor.get_input_handle('x')
    input_handle.reshape(norm_img.shape)
    input_handle.copy_from_cpu(norm_img)
    predictor.run()

    output = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()
    return ctc_decode(output[0], char_list)

# 5개국 모델 전부 미리 로드
for code in COUNTRY_MODEL_MAP:
    load_rec_predictor(code)

# ─── 워밍업 & 서버 시작 ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("  모델 워밍업 시작...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_plate = np.random.randint(0, 255, (120, 400, 3), dtype=np.uint8)

    try:
        yolo_model(dummy, verbose=False)
        print("    YOLO 워밍업 완료")
    except Exception as e:
        print(f"    YOLO 워밍업 실패: {e}")

    try:
        pil_dummy = Image.fromarray(dummy)
        tensor = eff_preprocess(pil_dummy).unsqueeze(0)
        if use_onnx_eff:
            eff_session.run(None, {eff_input_name: tensor.numpy()})
        else:
            with torch.no_grad():
                classify_model(tensor.to(device))
        print("    EfficientNet 워밍업 완료")
    except Exception as e:
        print(f"    EfficientNet 워밍업 실패: {e}")

    try:
        # 실제 추론과 동일한 크기로 2회씩 워밍업 (MKL-DNN 그래프 컴파일 완료)
        for code in rec_predictors:
            run_ocr(code, dummy_plate)
            run_ocr(code, dummy_plate)
        print("    PaddleOCR rec 워밍업 완료")
    except Exception as e:
        print(f"    PaddleOCR rec 워밍업 실패: {e}")

    print("  서버 준비 완료")
    yield

app = FastAPI(lifespan=lifespan)

# ─── API 엔드포인트 ──────────────────────────────────────────────
@app.post("/license-plates-recognition")
async def predict_all(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "이미지 읽기 실패"}

        t1 = time.time()

        # ── 1단계: YOLO 번호판 검출 ──
        is_detected = False
        plate_crop = img

        results = yolo_model(img)
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img[y1:y2, x1:x2]
            is_detected = True

        if plate_crop.size == 0:
            plate_crop = img

        t2 = time.time()

        # ── 2단계: EfficientNet 국가 분류 ──
        pil_plate    = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
        input_tensor = eff_preprocess(pil_plate).unsqueeze(0)

        if use_onnx_eff:
            logits    = eff_session.run(None, {eff_input_name: input_tensor.numpy()})[0][0]
            e         = np.exp(logits - logits.max())
            all_probs = e / e.sum()
            max_idx   = int(np.argmax(all_probs))
            max_prob  = float(all_probs[max_idx])
        else:
            with torch.no_grad():
                output    = classify_model(input_tensor.to(device))
                probs_t   = torch.softmax(output, dim=1)
                all_probs = probs_t[0].cpu().numpy()
                max_idx   = int(np.argmax(all_probs))
                max_prob  = float(all_probs[max_idx])

        country_code     = COUNTRY_MAP.get(max_idx, 'KOR')
        country_accuracy = round(max_prob * 100, 2)

        country_ranking = sorted(
            [(COUNTRY_MAP[i], round(float(all_probs[i]) * 100, 2)) for i in range(5)],
            key=lambda x: x[1], reverse=True
        )

        t3 = time.time()

        # ── 3단계: PaddleOCR Recognition ──
        # 1) KOR 모델 → 한글 있으면 KOR 확정
        # 2) CHN 모델 → 한자 있으면 CHN 확정
        # 3) 둘 다 아니면 → EfficientNet 1위 국가(BRA/EUR/IND) 결과 사용
        import re
        v_num = "Unknown"
        acc = 0.0
        detect_method = "EfficientNet"

        try:
            # 1) 한국어 모델
            kor_text, kor_score = run_ocr('KOR', plate_crop)
            if re.search(r'[가-힣]', kor_text) and kor_score > 0.3:
                v_num = kor_text
                acc = round(kor_score * 100, 2)
                country_code = 'KOR'
                country_accuracy = acc
                detect_method = "한글 감지"
            else:
                # 2) 중국어 모델
                chn_text, chn_score = run_ocr('CHN', plate_crop)
                if re.search(r'[\u4e00-\u9fff]', chn_text) and chn_score > 0.3:
                    v_num = chn_text
                    acc = round(chn_score * 100, 2)
                    country_code = 'CHN'
                    country_accuracy = acc
                    detect_method = "한자 감지"
                else:
                    # 3) BRA/EUR/IND → EfficientNet 1위 국가 사용
                    for rank_country, rank_prob in country_ranking:
                        if rank_country not in ('KOR', 'CHN'):
                            text, score = run_ocr(rank_country, plate_crop)
                            v_num = text
                            acc = round(score * 100, 2)
                            country_code = rank_country
                            country_accuracy = rank_prob
                            break

        except Exception as ocr_err:
            print(f"  OCR 오류: {ocr_err}")

        t4 = time.time()

        # ── 응답 생성 ──
        _, plate_encoded = cv2.imencode('.jpg', plate_crop)
        plate_base64 = base64.b64encode(plate_encoded).decode('utf-8')

        print("===========================================================================")
        print(f"국가 : {country_code} [{detect_method}]")
        print(f"번호 : {v_num} ({acc}%)")
        print(f"YOLO : {is_detected} | YOLO: {t2-t1:.3f}s | EfficientNet: {t3-t2:.3f}s | OCR: {t4-t3:.3f}s | 합계: {t4-t1:.3f}s")
        print("===========================================================================")

        return {
            "status": "success",
            "is_yolo_detected": is_detected,
            "license_plate_country": country_code,
            "country_accuracy": country_accuracy,
            "vehicle_num": v_num,
            "ocr_accuracy": acc,
            "is_ev_license_plate": False,
            "plate_img_base64": plate_base64
        }

    except Exception as e:
        print(f"  전체 프로세스 에러!")
        traceback.print_exc()
        return {"status": "error", "message": f"AI 서버 오류: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
