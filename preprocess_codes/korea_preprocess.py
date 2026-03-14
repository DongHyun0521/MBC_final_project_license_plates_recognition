# 한국 번호판 데이터셋을 OCR 학습용 데이터(Crop 이미지 + 라벨 텍스트)로 변환하는 코드

import os
import glob
import json
import csv

def make_ocr_label_csv(json_dir, output_csv_path):
    """(사람용) 데이터 분석을 위한 CSV 장부 생성"""
    print(f"📊 [{os.path.basename(json_dir)}] CSV 장부 생성을 시작합니다...")
    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image_filename', 'license_plate_text'])
        count = 0
        for json_name in os.listdir(json_dir):
            if not json_name.endswith('.json'): continue
            json_path = os.path.join(json_dir, json_name)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    img_name, plate_text = data.get('imagePath', ''), data.get('value', '')
                    if img_name and plate_text:
                        writer.writerow([img_name, plate_text])
                        count += 1
                        
                        # 💡 [진행 상황 출력 로직 추가]
                        if count % 100 == 0:
                            print(f"⏳ CSV 기록 중... {count}장 완료 (최근 번호: {plate_text})")
            except Exception: pass
    print(f"✅ CSV 장부 생성 완벽 종료: {output_csv_path} (총 {count}장)\n")

def process_split(split_name, label_dir, img_dir, folder_name, output_file_path):
    """(AI용) PaddleOCR 글로벌 상대경로 주소록 생성"""
    print(f"📝 [{split_name}] AI 학습용 TXT 주소록 생성을 시작합니다...")
    json_files = glob.glob(os.path.join(label_dir, "*.json"))
    if not json_files: return 0

    success_count = 0
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f_in:
                    data = json.load(f_in)
                img_name, plate_text = data.get("imagePath"), data.get("value")
                if not img_name or not plate_text: continue
                
                # 글로벌 통합 상대경로 (예: korea/ocr_train/파일명.jpg)
                relative_img_path = f"korea/{folder_name}/{img_name}"
                actual_img_path = os.path.join(img_dir, img_name)
                
                if os.path.exists(actual_img_path):
                    f_out.write(f"{relative_img_path}\t{plate_text}\n")
                    success_count += 1
                    
                    # 💡 [진행 상황 출력 로직 추가]
                    if success_count % 100 == 0:
                        print(f"⏳ [{split_name}] 주소록 작성 중... {success_count}장 완료 (최근 번호: {plate_text})")
            except Exception: pass
    print(f"✅ [{split_name}] TXT 주소록 생성 완벽 종료 (총 {success_count}장)\n")
    return success_count

if __name__ == "__main__":
    print("🇰🇷 [한국 데이터셋] 통합 전처리를 시작합니다!")
    print("   - 목표: 사람용 CSV 장부 + AI용 OCR 주소록(TXT) 일괄 생성\n")
    
    # 💡 폴더 이사 반영: 한 칸 위로 올라가서 최상위 폴더 찾기!
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    KOREA_DIR = os.path.join(PROJECT_ROOT, "korea")
    
    print(f"📂 인식된 한국 데이터 경로: {KOREA_DIR}\n")
    
    # 1. 사람용 CSV 만들기
    make_ocr_label_csv(os.path.join(KOREA_DIR, "label_train"), os.path.join(KOREA_DIR, "korea_train_labels.csv"))
    make_ocr_label_csv(os.path.join(KOREA_DIR, "label_val"), os.path.join(KOREA_DIR, "korea_val_labels.csv"))
    
    # 2. AI 학습용 주소록 폴더 생성
    ocr_dataset_dir = os.path.join(KOREA_DIR, "ocr_dataset")
    os.makedirs(ocr_dataset_dir, exist_ok=True)
    
    # 3. AI 학습용 글로벌 상대경로 TXT 만들기
    process_split("Train", os.path.join(KOREA_DIR, "label_train"), os.path.join(KOREA_DIR, "ocr_train"), "ocr_train", os.path.join(ocr_dataset_dir, "rec_gt_train.txt"))
    process_split("Val", os.path.join(KOREA_DIR, "label_val"), os.path.join(KOREA_DIR, "ocr_val"), "ocr_val", os.path.join(ocr_dataset_dir, "rec_gt_val.txt"))
    
    print("🎉 [한국 데이터셋] 모든 전처리가 완벽하게 종료되었습니다!")