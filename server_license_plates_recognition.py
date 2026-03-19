# ai_server.py

from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
import cv2
import numpy as np
import uvicorn

# 💡 6개국 언어 판별/라우팅 전문가 호출!
from plate_router import detect_country_and_route

app = FastAPI(title="🚗 글로벌 6개국 차량 번호판 인식 AI 서버 (PaddleOCR 단독)")


print("======================================")
print("🤖 AI 뇌(PaddleOCR) 로딩 중...")
# 참고: 중국어 테스트 시 lang='ch'로 변경해야 한자를 잘 읽습니다. (기본은 한국어/영어 최적화)
ocr = PaddleOCR(lang='korean', enable_mkldnn=False)
print("✅ AI 서버 세팅 완료! (프론트엔드 연동 대기 중)")
print("======================================")

# 💡 전기차 판별 함수 (파란색 픽셀 비율 분석)
def detect_ev_plate(img, box):
    if box is None: return False
    pts = np.array(box, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    padding = 10
    
    # 이미지 밖으로 넘어가지 않게 좌표 보정
    x, y = max(0, x - padding), max(0, y - padding)
    w = min(img.shape[1] - x, w + padding * 2)
    h = min(img.shape[0] - y, h + padding * 2)
    
    plate_crop = img[y:y + h, x:x + w]
    if plate_crop.size == 0: return False
    
    hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
    lower_blue, upper_blue = np.array([85, 50, 50]), np.array([115, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return bool((cv2.countNonZero(mask) / (w * h)) > 0.15)


@app.post("/api/v1/ocr")
async def read_license_plate(file: UploadFile = File(...)):
    print(f"\n======================================")
    print(f"📥 [요청 수신] 사진이 도착했습니다! (파일명: {file.filename})")

    try:
        # 1. 프론트엔드에서 받은 사진을 OpenCV 배열로 변환
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "이미지를 읽을 수 없습니다."}

        # 2. PaddleOCR로 글자 추출
        result = ocr.ocr(img)

        raw_text = ""
        is_ev = False
        box = None
        
        # 기본 응답값 세팅
        final_plate = "인식실패"
        country_code = "UNKNOWN"
        language_code = "ko"
        welcome_msg = "번호판 인식에 실패했습니다."

        try:
            # 3. 텍스트 및 좌표 추출 로직
            if result and isinstance(result, list) and len(result) > 0 and result[0] is not None:
                first_item = result[0]

                if isinstance(first_item, dict) and 'rec_texts' in first_item:
                    rec_texts = first_item.get('rec_texts', [])
                    if len(rec_texts) > 0:
                        raw_text = "".join(rec_texts)

                    polys = first_item.get('rec_polys') or first_item.get('dt_polys')
                    if polys is not None and len(polys) > 0:
                        box = polys[0].tolist() if hasattr(polys[0], 'tolist') else polys[0]

                elif isinstance(first_item, list):
                    text_parts = []
                    for line in first_item:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            if box is None: box = line[0]  
                            t_data = line[1]
                            if isinstance(t_data, (list, tuple)) and len(t_data) >= 1:
                                text_parts.append(t_data[0])  
                    raw_text = "".join(text_parts) 

            # 4. 추출된 글자가 있다면 6개국 라우터(Router)로 넘김!
            if raw_text:
                raw_text = raw_text.replace(" ", "")
                print(f"🔍 [AI 텍스트 추출 성공] : '{raw_text}'")
                
                # 전기차 파란색 비율 계산
                if box is not None: is_ev = detect_ev_plate(img, box)
                
                # 라우터 호출하여 국가, 번호판, 언어 응답 받아오기
                route_result = detect_country_and_route(raw_text)
                country_code = route_result["country"]
                final_plate = route_result["plate_number"]
                language_code = route_result["language"]
                welcome_msg = route_result["welcome_message"]
                
            else:
                print("🚨 텍스트 추출에 실패했습니다. (사진에 글자가 없거나 흐림)")

        except Exception as parse_err:
            print(f"⚠️ [파싱 에러]: {parse_err}")

        # 서버 콘솔에 최종 결과 출력
        print(f"✨ [판별 완료] 번호판: '{final_plate}' | 국가: {country_code} | 전기차: {is_ev}")
        print(f"======================================\n")

        # 5. 프론트엔드/자바로 쏴줄 6개국 지원 JSON
        return {
            "status": "success",
            "filename": file.filename,
            "plate_number": final_plate,
            "country": country_code,
            "is_ev": is_ev,
            "language": language_code,
            "welcome_message": welcome_msg
        }

    except Exception as e:
        print(f"🚨 [파이썬 서버 에러]: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8001, reload=True)