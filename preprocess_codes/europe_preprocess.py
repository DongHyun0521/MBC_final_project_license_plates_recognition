import os
import glob
import shutil
import random
from pathlib import Path

def setup_europe_ocr(raw_europe_dir, project_root):
    print("🇪🇺 [유럽 데이터셋] 영혼까지 끌어모으는 OCR 전처리를 시작합니다!")
    print("   - 목표: Train/Val/Test 폴더 통합 및 글로벌 8:2 스플릿\n")

    # 1. 우리 프로젝트 공식 폴더 구조 만들기
    europe_dir = os.path.join(project_root, "europe")
    ocr_train_dir = os.path.join(europe_dir, "ocr_train")
    ocr_val_dir = os.path.join(europe_dir, "ocr_val")
    ocr_dataset_dir = os.path.join(europe_dir, "ocr_dataset")
    
    for d in [ocr_train_dir, ocr_val_dir, ocr_dataset_dir]:
        os.makedirs(d, exist_ok=True)
        
    print(f"📁 폴더 구조 생성 완료: {europe_dir}")

    # 2. train, val, test 폴더의 모든 이미지 영끌하기
    all_images = []
    for folder in ["train", "val", "test"]:
        folder_path = os.path.join(raw_europe_dir, folder)
        if os.path.exists(folder_path):
            all_images.extend(glob.glob(os.path.join(folder_path, "*.png")))
            all_images.extend(glob.glob(os.path.join(folder_path, "*.jpg")))

    total_files = len(all_images)
    print(f"✅ 원본 이미지 총 {total_files}장 발견! 8:2 황금비율 전처리 루프를 시작합니다...\n")

    # 3. 데이터 골고루 섞기 (편향 방지)
    random.seed(42)
    random.shuffle(all_images)

    train_lines = []
    val_lines = []
    
    # 4. 이미지 복사 및 주소록 작성
    for i, img_path in enumerate(all_images):
        filename = Path(img_path).name
        plate_text = Path(img_path).stem  # 확장자 뺀 파일명 (예: AB123CD)
        
        # 특수문자나 띄어쓰기가 있다면 제거
        plate_text = plate_text.replace(" ", "").replace("-", "")
        
        # 전체의 80%는 train, 20%는 val로 분배
        if i < total_files * 0.8:
            target_dir = ocr_train_dir
            folder_name = "ocr_train"
            train_lines.append(f"europe/{folder_name}/{filename}\t{plate_text}\n")
        else:
            target_dir = ocr_val_dir
            folder_name = "ocr_val"
            val_lines.append(f"europe/{folder_name}/{filename}\t{plate_text}\n")
            
        # 이미지 복사
        shutil.copy(img_path, os.path.join(target_dir, filename))
        
        success_count = i + 1
        
        # 💡 [진행 상황 출력 로직 추가: 100장 단위]
        if success_count % 100 == 0:
            print(f"⏳ 처리 중... {success_count}장 완료 (최근 번호: {plate_text})")

    # 5. 주소록 텍스트 파일 저장
    print("\n📝 최종 OCR 정답지(rec_gt.txt)를 작성 중입니다...")
    with open(os.path.join(ocr_dataset_dir, "rec_gt_train.txt"), 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(os.path.join(ocr_dataset_dir, "rec_gt_val.txt"), 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    print(f"\n🎉 [유럽 데이터셋] 모든 전처리가 완벽하게 종료되었습니다! (Train: {len(train_lines)}장 / Val: {len(val_lines)}장)")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    
    # 🚨 여기에 원본 유럽 데이터 폴더 경로를 넣어주세요!
    RAW_EUROPE_DIR = r"C:\Users\dhlim\PycharmProjects\MBC_final_project_license_plates_recognition\europe"
    
    print(f"📂 인식된 유럽 원본 데이터 경로: {RAW_EUROPE_DIR}\n")
    setup_europe_ocr(RAW_EUROPE_DIR, PROJECT_ROOT)