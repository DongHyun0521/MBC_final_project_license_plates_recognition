import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from torchvision import models, transforms
from paddleocr import PaddleOCR
from PIL import Image
import traceback
import time
import base64
import re

os.environ['FLAGS_enable_pir_api'] = '0'

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, 'finetuned_yolo11n_acc822.pt')
EFF_PATH = os.path.join(BASE_DIR, 'finetuned_efficientnetb0.pt')

yolo_model = YOLO(YOLO_PATH)
classify_model = models.efficientnet_b0()
classify_model.classifier[1] = nn.Linear(classify_model.classifier[1].in_features, 5)
classify_model.load_state_dict(torch.load(EFF_PATH, map_location=device))
classify_model.to(device).eval()

COUNTRY_MAP = {0: 'BRA', 1: 'CHN', 2: 'EUR', 3: 'IND', 4: 'KOR'}

# 국가별 OCR 언어 매핑
COUNTRY_LANG_MAP = {
    'KOR': 'korean',
    'CHN': 'ch',
    'BRA': 'en',
    'EUR': 'en',
    'IND': 'en',
}

# OCR 엔진 캐시 (최초 사용 시 로드)
ocr_engines = {}

def get_ocr_engine(lang):
    if lang not in ocr_engines:
        print(f"🔄 PaddleOCR ({lang}) 초기화 중...")
        ocr_engines[lang] = PaddleOCR(use_textline_orientation=True, lang=lang, enable_mkldnn=False)
    return ocr_engines[lang]

# 한국어, 중국어 엔진 미리 로드
get_ocr_engine('korean')
get_ocr_engine('ch')

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/license-plates-recognition")
async def predict_all(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None: return {"status": "error", "message": "이미지 읽기 실패"}

        t1 = time.time()

        # 1. YOLO 검출
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

        # 2. 국가 판별 (EfficientNet)
        pil_plate = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(pil_plate).unsqueeze(0).to(device)
        with torch.no_grad():
            output = classify_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            max_prob, max_idx = torch.max(probs, 1)
            country_code = COUNTRY_MAP.get(max_idx.item(), 'KOR')
            country_accuracy = round(float(max_prob.item()) * 100, 2)

            t3 = time.time()

            all_probs = probs[0].cpu().numpy()
            country_ranking = sorted(
                [(COUNTRY_MAP[i], round(float(all_probs[i]) * 100, 2)) for i in range(5)],
                key=lambda x: x[1], reverse=True
            )

        # 3. 국가별 PaddleOCR
                # 3. OCR로 국가 확정 (한글→KOR, 한자→CHN, 그 외→EfficientNet)
        v_num, acc = ("Unknown", 0.0)
        ocr_details = []
        detect_method = 'EfficientNet'

        try:
            ph, pw = plate_crop.shape[:2]
            target_w = 480
            scale = target_w / pw
            plate_for_ocr = cv2.resize(plate_crop, (target_w, int(ph * scale)))

            # 1단계: 한국어 OCR 우선 실행
            ocr_engine = get_ocr_engine('korean')
            ocr_res = ocr_engine.predict(plate_for_ocr)

            kor_text = ''
            if ocr_res and len(ocr_res) > 0:
                r = ocr_res[0]
                if 'rec_texts' in r and len(r['rec_texts']) > 0:
                    kor_text = ''.join(r['rec_texts']).replace(' ', '')

            if re.search(r'[가-힣]', kor_text):
                # 한글 발견 → KOR 확정
                country_code = 'KOR'
                country_accuracy = 100.0
                detect_method = '한글 감지'
            else:
                # 2단계: 중국어 OCR 실행
                ocr_engine = get_ocr_engine('ch')
                ocr_res = ocr_engine.predict(plate_for_ocr)

                chn_text = ''
                if ocr_res and len(ocr_res) > 0:
                    r = ocr_res[0]
                    if 'rec_texts' in r and len(r['rec_texts']) > 0:
                        chn_text = ''.join(r['rec_texts']).replace(' ', '')

                if re.search(r'[\u4e00-\u9fff]', chn_text):
                    # 한자 발견 → CHN 확정
                    country_code = 'CHN'
                    country_accuracy = 100.0
                    detect_method = '한자 감지'
                else:
                    # 3단계: 한글도 한자도 없음 → EfficientNet 신뢰 (BRA/EUR/IND)
                    ocr_engine = get_ocr_engine('en')
                    ocr_res = ocr_engine.predict(plate_for_ocr)

            # OCR 결과 처리 (최종 ocr_res 사용)
            if ocr_res and len(ocr_res) > 0:
                result = ocr_res[0]
                if 'rec_texts' in result and len(result['rec_texts']) > 0:
                    texts = list(result['rec_texts'])
                    scores = list(result['rec_scores'])

                    if len(texts) > 1 and 'dt_polys' in result:
                        polys = result['dt_polys']
                        centers = []
                        for poly in polys:
                            pts = np.array(poly)
                            cx = float(pts[:, 0].mean())
                            cy = float(pts[:, 1].mean())
                            centers.append((cy, cx))

                        order = sorted(range(len(texts)), key=lambda i: centers[i])
                        texts = [texts[i] for i in order]
                        scores = [scores[i] for i in order]

                    v_num = ''.join(texts).replace(' ', '')
                    if not v_num:
                        v_num = "Unknown"
                        acc = 0.0
                    else:
                        acc = float(sum(scores) / len(scores)) * 100
                        ocr_details = [(texts[i], round(scores[i] * 100, 2)) for i in range(min(3, len(texts)))]

        except Exception as ocr_err:
            print(f"⚠️ OCR 내부 오류 (무시하고 진행): {ocr_err}")

        t4 = time.time()

        _, plate_encoded = cv2.imencode('.jpg', plate_crop)
        plate_base64 = base64.b64encode(plate_encoded).decode('utf-8')

        print("===========================================================================")
        print(f"국가 : {country_code} ({country_accuracy}%) [{detect_method}]")
        for i, (c, p) in enumerate(country_ranking[:3]):
            print(f"  {i+1}. {c} ({p}%)")
        print(f"번호 : {v_num} ({round(acc, 2)}%)")
        print(f"YOLO 감지 여부 : {is_detected}")
        print(f"⏱️ YOLO: {t2-t1:.3f}초 | EfficientNet: {t3-t2:.3f}초 | PaddleOCR: {t4-t3:.3f}초 | 합계: {t4-t1:.3f}초")
        print("===========================================================================")
    
        return {
            "status": "success",
            "is_yolo_detected": is_detected,
            "license_plate_country": country_code,
            "country_accuracy": country_accuracy,
            "vehicle_num": v_num,
            "ocr_accuracy": round(acc, 2),
            "is_ev_license_plate": False,
            "plate_img_base64": plate_base64
        }

    except Exception as e:
        print(f"❌ 전체 프로세스 에러 발생!")
        traceback.print_exc()
        return {"status": "error", "message": f"AI 서버 상세 오류: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
