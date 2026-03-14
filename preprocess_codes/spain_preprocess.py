import os
import glob
import json
import cv2
import shutil
from pathlib import Path

def unified_spain_preprocess(base_dir):
    print("🇪🇸 [스페인(유럽 V2.0) 데이터셋] YOLO 전용 전처리를 시작합니다!")
    print("   🚨 [긴급 패치] 개인정보 블러 처리로 인해 OCR(글자 읽기) 데이터는 생성하지 않습니다.")
    print("   👉 오직 YOLO(번호판 위치 탐지) 학습용 고화질 사진만 추출합니다!\n")

    # 1. 출력 폴더 구조 (YOLO만 생성)
    yolo_img_dir = os.path.join(base_dir, "yolo_dataset", "images")
    yolo_lbl_dir = os.path.join(base_dir, "yolo_dataset", "labels")
    
    for d in [yolo_img_dir, yolo_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    success_count = 0

    # 2. train과 test 폴더를 순회하며 전처리
    for split in ["train", "test"]:
        folder_path = os.path.join(base_dir, split)
        if not os.path.exists(folder_path): continue
        
        json_files = glob.glob(os.path.join(folder_path, "*.json"))
        print(f"\n🔍 [{split}] 폴더에서 {len(json_files)}개의 JSON 파일을 찾았습니다. 변환 시작!")

        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 껍데기 파일명 무시! JSON 내부의 진짜 이미지 이름
                real_img_name = data.get("imagePath")
                if not real_img_name: continue
                
                # 진짜 이미지 경로
                img_path = os.path.join(folder_path, real_img_name)
                if not os.path.exists(img_path): continue

                if 'lps' not in data or len(data['lps']) == 0: continue
                plate_obj = data['lps'][0]

                # 좌표 추출
                points = plate_obj.get("poly_coord", [])
                if len(points) != 4: continue

                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)

                # --- [STEP A: YOLO 데이터셋 생성만 수행!] ---
                img = cv2.imread(img_path)
                h, w, _ = img.shape
                
                x_center = ((xmin + xmax) / 2.0) / w
                y_center = ((ymin + ymax) / 2.0) / h
                box_width = (xmax - xmin) / w
                box_height = (ymax - ymin) / h
                
                # 💡 [수정됨] 확장자(.jpg)를 가장 확실하게 제거하고 파일명만 가져옵니다!
                file_stem = os.path.splitext(real_img_name)[0]
                
                with open(os.path.join(yolo_lbl_dir, f"{file_stem}_{split}.txt"), 'w') as lf:
                    lf.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                
                # YOLO 원본 이미지 복사
                shutil.copy(img_path, os.path.join(yolo_img_dir, f"{file_stem}_{split}.jpg"))

                success_count += 1
                if success_count % 100 == 0:
                    print(f"⏳ YOLO 박스 생성 중... {success_count}장 완료 ({real_img_name})")

            except Exception as e:
                pass 

    print(f"\n🎉 [스페인 데이터셋] YOLO 학습용 고화질 추출 완료! (총 {success_count}장)")
    print("   👉 OCR 주소록(rec_gt.txt)은 안전을 위해 생성하지 않았습니다.")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    SPAIN_DIR = os.path.join(PROJECT_ROOT, "spain")
    
    print(f"📂 인식된 스페인 데이터 경로: {SPAIN_DIR}\n")
    unified_spain_preprocess(SPAIN_DIR)