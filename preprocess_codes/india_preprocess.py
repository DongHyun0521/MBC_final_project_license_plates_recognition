import os
import glob
import cv2
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

def unified_india_preprocess(base_dir):
    print("🇮🇳 [인도(야생 실전) 데이터셋] 영혼까지 끌어모으는 통합 전처리를 시작합니다!")
    print("   - 목표: 깊은 폴더 속 모든 XML을 싹쓸이하여 YOLO + OCR 데이터 일괄 생성\n")

    # 1. 모든 출력 폴더 구조 한 번에 만들기
    yolo_img_dir = os.path.join(base_dir, "yolo_dataset", "images")
    yolo_lbl_dir = os.path.join(base_dir, "yolo_dataset", "labels")
    ocr_train_dir = os.path.join(base_dir, "ocr_train")
    ocr_val_dir = os.path.join(base_dir, "ocr_val")
    ocr_dataset_dir = os.path.join(base_dir, "ocr_dataset")
    
    for d in [yolo_img_dir, yolo_lbl_dir, ocr_train_dir, ocr_val_dir, ocr_dataset_dir]:
        os.makedirs(d, exist_ok=True)

    # 2. 하위 폴더 상관없이 모든 .xml 파일 영끌하기 (recursive=True의 마법)
    # yolo_dataset이나 ocr_dataset 안의 파일은 무시하도록 필터링
    all_xml_files = glob.glob(os.path.join(base_dir, "**", "*.xml"), recursive=True)
    raw_xml_files = [f for f in all_xml_files if "yolo_dataset" not in f and "ocr_dataset" not in f]
    
    total_files = len(raw_xml_files)
    print(f"✅ 원본 라벨(XML) 파일 총 {total_files}개 발견! 전처리 루프를 시작합니다...\n")

    ocr_train_lines, ocr_val_lines = [], []
    success_count = 0
    
    # 데이터 스플릿을 위한 시드 고정
    random.seed(42)

    for xml_path in raw_xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # XML 내부의 진짜 이미지 이름 찾기
            filename_node = root.find("filename")
            if filename_node is None or not filename_node.text: continue
            real_img_name = filename_node.text

            # 이미지 경로는 XML 파일과 같은 폴더에 있다고 가정
            xml_dir = os.path.dirname(xml_path)
            img_path = os.path.join(xml_dir, real_img_name)
            
            # 혹시 확장자가 .jpeg 등으로 달라서 못 찾을 경우 방어 코드
            if not os.path.exists(img_path):
                file_stem = Path(xml_path).stem
                # 같은 폴더 안에서 이름이 같은 이미지 파일 찾기
                possible_imgs = glob.glob(os.path.join(xml_dir, f"{file_stem}.*"))
                valid_imgs = [img for img in possible_imgs if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if not valid_imgs: continue
                img_path = valid_imgs[0]

            # 번호판 정보 찾기 (object 태그)
            obj = root.find("object")
            if obj is None: continue

            # 번호판 글자 정답 (name 태그)
            name_node = obj.find("name")
            if name_node is None or not name_node.text: continue
            plate_text = name_node.text.replace(" ", "").replace("-", "") # 띄어쓰기, 특수문자 제거
            
            # 좌표 찾기 (bndbox 태그)
            bndbox = obj.find("bndbox")
            if bndbox is None: continue
            
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # --- [STEP A: YOLO 데이터셋 생성] ---
            img = cv2.imread(img_path)
            if img is None: continue
            h, w, _ = img.shape
            
            x_center = ((xmin + xmax) / 2.0) / w
            y_center = ((ymin + ymax) / 2.0) / h
            box_width = (xmax - xmin) / w
            box_height = (ymax - ymin) / h
            
            file_stem = Path(img_path).stem
            
            # YOLO 라벨 저장
            with open(os.path.join(yolo_lbl_dir, f"{file_stem}.txt"), 'w') as lf:
                lf.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            
            # YOLO 이미지 복사 (jpg로 통일해서 저장)
            yolo_img_save_path = os.path.join(yolo_img_dir, f"{file_stem}.jpg")
            cv2.imwrite(yolo_img_save_path, img)

            # --- [STEP B: OCR 크롭 이미지 생성] ---
            cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            if cropped_img.shape[0] < 10 or cropped_img.shape[1] < 10: continue

            crop_filename = f"{plate_text}_{file_stem}.jpg"
            
            # 8:2 랜덤 스플릿
            is_train = random.random() < 0.8
            target_ocr_dir = ocr_train_dir if is_train else ocr_val_dir
            folder_name = "ocr_train" if is_train else "ocr_val"
            
            cv2.imwrite(os.path.join(target_ocr_dir, crop_filename), cropped_img)

            # 글로벌 상대경로 주소록 한 줄 기록 (예: india/ocr_train/파일명.jpg)
            relative_path = f"india/{folder_name}/{crop_filename}"
            
            if is_train:
                ocr_train_lines.append(f"{relative_path}\t{plate_text}\n")
            else:
                ocr_val_lines.append(f"{relative_path}\t{plate_text}\n")

            success_count += 1
            if success_count % 100 == 0:
                print(f"⏳ 처리 중... {success_count}장 완료 (최근 번호: {plate_text})")

        except Exception as e:
            pass # 에러난 파일은 쿨하게 무시

    # --- [STEP C: OCR 주소록(TXT) 최종 저장] ---
    print("\n📝 최종 OCR 정답지(rec_gt.txt)를 작성 중입니다...")
    with open(os.path.join(ocr_dataset_dir, "rec_gt_train.txt"), 'w', encoding='utf-8') as f:
        f.writelines(ocr_train_lines)
    with open(os.path.join(ocr_dataset_dir, "rec_gt_val.txt"), 'w', encoding='utf-8') as f:
        f.writelines(ocr_val_lines)

    print(f"\n🎉 [인도 데이터셋] 폴더 대통합 및 전처리 완벽 종료! (최종 성공: {success_count}장)")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    
    # 🚨 india 폴더 경로 연결
    INDIA_DIR = os.path.join(PROJECT_ROOT, "india")
    
    print(f"📂 인식된 인도 데이터 경로: {INDIA_DIR}\n")
    unified_india_preprocess(INDIA_DIR)