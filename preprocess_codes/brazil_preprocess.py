# 브라질 데이터셋을 YOLO 학습용 데이터(Bounding Box 포맷 + 이미지)로 변환하는 코드
# pip install opencv-python numpy

import os
import glob
import cv2
import shutil
import random
import numpy as np
from pathlib import Path

def unified_brazil_preprocess(base_dir):
    print("🇧🇷 [브라질 ALPR 데이터셋] 통합 전처리를 시작합니다!")
    print("   - 목표: YOLO(위치 탐지) + OCR(글자 인식) 데이터 일괄 생성\n")

    # 1. 모든 출력 폴더 구조 한 번에 만들기
    yolo_img_dir = os.path.join(base_dir, "yolo_dataset", "images")
    yolo_lbl_dir = os.path.join(base_dir, "yolo_dataset", "labels")
    ocr_train_dir = os.path.join(base_dir, "ocr_train")
    ocr_val_dir = os.path.join(base_dir, "ocr_val")
    ocr_dataset_dir = os.path.join(base_dir, "ocr_dataset") # txt 저장용
    
    for d in [yolo_img_dir, yolo_lbl_dir, ocr_train_dir, ocr_val_dir, ocr_dataset_dir]:
        os.makedirs(d, exist_ok=True)
    
    print(f"📁 폴더 구조 생성 완료: {base_dir}")

    # 2. 원본 txt 파일 전부 찾기
    txt_files = glob.glob(os.path.join(base_dir, "**", "*.txt"), recursive=True)
    # yolo_dataset이나 ocr_dataset 안에 있는 txt는 제외 (원본만 필터링)
    raw_txt_files = [f for f in txt_files if "yolo_dataset" not in f and "ocr_dataset" not in f]
    
    total_files = len(raw_txt_files)
    print(f"✅ 원본 라벨 파일 총 {total_files}개 발견! 전처리 루프를 시작합니다...\n")

    ocr_train_lines, ocr_val_lines = [], []
    success_count = 0

    # 일관된 분할을 위해 시드 고정
    random.seed(42)

    for txt_path in raw_txt_files:
        img_path = txt_path.replace(".txt", ".png")
        if not os.path.exists(img_path): continue

        # 파싱
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        plate_text, corners = "", []
        for line in lines:
            line = line.strip()
            if line.startswith("plate:"): plate_text = line.split(":")[1].strip()
            elif line.startswith("corners:"):
                coords_str = line.split(":")[1].strip().replace(',', ' ')
                corners = [int(x) for x in coords_str.split()]

        if len(corners) != 8 or not plate_text: continue

        # --- [STEP A: YOLO 데이터셋 생성] ---
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        x_coords = [corners[0], corners[2], corners[4], corners[6]]
        y_coords = [corners[1], corners[3], corners[5], corners[7]]
        xmin, xmax, ymin, ymax = min(x_coords), max(x_coords), min(y_coords), max(y_coords)

        # YOLO 라벨 작성
        x_center, y_center = ((xmin + xmax) / 2.0) / w, ((ymin + ymax) / 2.0) / h
        box_width, box_height = (xmax - xmin) / w, (ymax - ymin) / h
        file_stem = Path(txt_path).stem
        
        with open(os.path.join(yolo_lbl_dir, f"{file_stem}.txt"), 'w') as lf:
            lf.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
        
        # YOLO 원본 이미지 복사
        shutil.copy(img_path, os.path.join(yolo_img_dir, f"{file_stem}.png"))

        # --- [STEP B: OCR 크롭 및 분할 (Train/Val)] ---
        # 반듯하게 펴기 위한 Perspective Transform
        pts1 = np.float32([[corners[6], corners[7]], [corners[4], corners[5]], 
                           [corners[0], corners[1]], [corners[2], corners[3]]])
        width, height = 300, 100
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        cropped_img = cv2.warpPerspective(img, matrix, (width, height))

        # 8:2 랜덤 스플릿
        is_train = random.random() < 0.8
        target_dir = ocr_train_dir if is_train else ocr_val_dir
        folder_name = "ocr_train" if is_train else "ocr_val"
        
        # 크롭된 이미지 저장 (jpg로 변환하여 저장)
        crop_filename = f"{plate_text}_{file_stem}.jpg"
        cv2.imwrite(os.path.join(target_dir, crop_filename), cropped_img)

        # 글로벌 상대경로 주소록 한 줄 기록 (예: brazil/ocr_train/파일명.jpg)
        relative_path = f"brazil/{folder_name}/{crop_filename}"
        if is_train:
            ocr_train_lines.append(f"{relative_path}\t{plate_text}\n")
        else:
            ocr_val_lines.append(f"{relative_path}\t{plate_text}\n")

        success_count += 1
        
        # 💡 [진행 상황 출력 로직 수정: 퍼센트 제거, N장 완료 스타일]
        if success_count % 100 == 0: 
            print(f"⏳ 처리 중... {success_count} / {total_files} 장 완료 (최근 번호: {plate_text})")

    # --- [STEP C: OCR 주소록(TXT) 최종 저장] ---
    print("\n📝 최종 OCR 정답지(rec_gt.txt)를 작성 중입니다...")
    with open(os.path.join(ocr_dataset_dir, "rec_gt_train.txt"), 'w', encoding='utf-8') as f:
        f.writelines(ocr_train_lines)
    with open(os.path.join(ocr_dataset_dir, "rec_gt_val.txt"), 'w', encoding='utf-8') as f:
        f.writelines(ocr_val_lines)

    print(f"\n🎉 [브라질 데이터셋] 모든 전처리가 완벽하게 종료되었습니다! (최종 성공: {success_count}장)")

if __name__ == "__main__":
    # 💡 폴더 이사 반영: 한 칸 위로 올라가서 최상위 폴더 찾기!
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    
    # 진짜 brazil 폴더 경로 연결
    BRAZIL_DIR = os.path.join(PROJECT_ROOT, "brazil")
    
    print(f"📂 인식된 브라질 데이터 경로: {BRAZIL_DIR}")
    unified_brazil_preprocess(BRAZIL_DIR)