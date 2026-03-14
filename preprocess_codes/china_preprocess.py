import os
import glob
import cv2
import shutil
import random
import numpy as np
from pathlib import Path

# 💡 CCPD 암호 해독용 매핑표 (절대 수정 금지!)
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def decode_plate_text(lp_str):
    """'0_0_14_28_24_26_29' 같은 문자열을 '皖AQ4025'로 번역합니다."""
    indices = lp_str.split('_')
    if len(indices) != 7: return ""
    
    province = PROVINCES[int(indices[0])]
    alphanums = "".join([ALPHABETS[int(i)] for i in indices[1:]])
    return province + alphanums

def unified_china_preprocess(base_dir):
    print("🇨🇳 [중국 CCPD 데이터셋] 암호 해독 및 통합 전처리(YOLO+OCR)를 시작합니다!\n")

    # 1. 출력 폴더 구조 싹 다 만들기
    yolo_img_dir = os.path.join(base_dir, "yolo_dataset", "images")
    yolo_lbl_dir = os.path.join(base_dir, "yolo_dataset", "labels")
    ocr_train_dir = os.path.join(base_dir, "ocr_train")
    ocr_val_dir = os.path.join(base_dir, "ocr_val")
    ocr_dataset_dir = os.path.join(base_dir, "ocr_dataset")
    
    for d in [yolo_img_dir, yolo_lbl_dir, ocr_train_dir, ocr_val_dir, ocr_dataset_dir]:
        os.makedirs(d, exist_ok=True)

    # 2. ccpd_base 폴더 안의 이미지 찾기
    img_files = glob.glob(os.path.join(base_dir, "ccpd_base", "*.jpg"))
    print(f"✅ 원본 사진 {len(img_files)}장 발견! 풀패키지 변환 시작...\n")

    ocr_train_lines, ocr_val_lines = [], []
    success_count = 0
    random.seed(42)

    for img_path in img_files:
        try:
            filename = Path(img_path).name
            parts = filename.replace(".jpg", "").split("-")
            
            # 🚨 CCPD 파일명은 '-'를 기준으로 나뉩니다.
            bb_str = parts[-5]       # 바운딩 박스 (예: 298&341_449&414)
            vertices_str = parts[-4] # 4모서리 좌표 (예: 458&394_... )
            lp_str = parts[-3]       # 글자 코드 (예: 0_0_14_...)

            # --- [1. 글자 번역] ---
            plate_text = decode_plate_text(lp_str)
            if not plate_text: continue

            # --- [2. YOLO 데이터셋 생성 (Bounding Box)] ---
            img = cv2.imread(img_path)
            if img is None: continue
            h, w, _ = img.shape
            
            # 좌상단(xmin, ymin) _ 우하단(xmax, ymax) 추출
            pt1, pt2 = bb_str.split('_')
            xmin, ymin = map(int, pt1.split('&'))
            xmax, ymax = map(int, pt2.split('&'))

            x_center, y_center = ((xmin + xmax) / 2.0) / w, ((ymin + ymax) / 2.0) / h
            box_width, box_height = (xmax - xmin) / w, (ymax - ymin) / h
            
            file_stem = Path(img_path).stem
            with open(os.path.join(yolo_lbl_dir, f"{file_stem}.txt"), 'w') as lf:
                lf.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            
            # 원본 이미지 복사 (YOLO 학습용)
            shutil.copy(img_path, os.path.join(yolo_img_dir, f"{filename}"))

            # --- [3. OCR 데이터셋 생성 (Crop & 4 Corners)] ---
            # CCPD의 4모서리 순서: 우하(RB) _ 좌하(LB) _ 좌상(LT) _ 우상(RT)
            corners = vertices_str.split('_')
            rb = list(map(int, corners[0].split('&')))
            lb = list(map(int, corners[1].split('&')))
            lt = list(map(int, corners[2].split('&')))
            rt = list(map(int, corners[3].split('&')))

            # 투시 변환 (반듯하게 펴기)
            pts1 = np.float32([lt, rt, rb, lb])
            crop_w, crop_h = 300, 100
            pts2 = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])
            
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            cropped_img = cv2.warpPerspective(img, matrix, (crop_w, crop_h))

            # 💡 [핵심: 한자 깨짐 방지] 파일명에 한자(plate_text)를 넣지 않고 영어/숫자만 사용
            crop_filename = f"crop_{file_stem}.jpg"

            # 8:2 스플릿 및 저장
            is_train = random.random() < 0.8
            target_dir = ocr_train_dir if is_train else ocr_val_dir
            folder_name = "ocr_train" if is_train else "ocr_val"
            
            # 한자가 없으므로 윈도우(CP949) 환경에서도 100% 안전하게 저장됨
            cv2.imwrite(os.path.join(target_dir, crop_filename), cropped_img)

            # 글로벌 상대경로 주소록 작성 (예: china/ocr_train/crop_...jpg)
            # 주소록 내용물(정답)은 여전히 예쁜 중국어 한자가 기록됨
            relative_path = f"china/{folder_name}/{crop_filename}"
            if is_train:
                ocr_train_lines.append(f"{relative_path}\t{plate_text}\n")
            else:
                ocr_val_lines.append(f"{relative_path}\t{plate_text}\n")

            success_count += 1
            if success_count % 100 == 0: 
                print(f"⏳ 처리 중... {success_count} / {len(img_files)} 장 완료 (최근 해독: {plate_text})")

        except Exception as e:
            pass # 가끔 규격에 안 맞는 깨진 파일은 무시

    # --- [4. OCR 주소록 최종 저장] ---
    with open(os.path.join(ocr_dataset_dir, "rec_gt_train.txt"), 'w', encoding='utf-8') as f:
        f.writelines(ocr_train_lines)
    with open(os.path.join(ocr_dataset_dir, "rec_gt_val.txt"), 'w', encoding='utf-8') as f:
        f.writelines(ocr_val_lines)

    print(f"\n🎉 [중국 데이터셋] YOLO 라벨링 및 OCR 암호 해독 세팅 완벽 종료! (총 {success_count}장)")

if __name__ == "__main__":
    # 현재 코드 파일이 있는 곳: .../preprocess_codes
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 프로젝트 최상위 폴더로 한 칸 올라가기: .../MBC_final_project_license_plates_recognition
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    
    # 진짜 china 폴더 경로
    CHINA_DIR = os.path.join(PROJECT_ROOT, "china")
    
    print(f"📂 인식된 중국 데이터 경로: {CHINA_DIR}")
    unified_china_preprocess(CHINA_DIR)