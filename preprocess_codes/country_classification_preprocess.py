import os
import shutil

def build_classification_dataset(base_dir):
    print("🌍 [국가 판별 AI용] Classification 데이터셋 구축을 시작합니다!")

    countries = ["korea", "china", "brazil", "europe", "india", "malaysia"]
    
    class_dir = os.path.join(base_dir, "country_classification_data_v1")
    train_dir = os.path.join(class_dir, "train")
    val_dir = os.path.join(class_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    total_copied = 0
    total_skipped = 0  # 💡 건너뛴 파일 개수도 세어봅시다!

    for country in countries:
        os.makedirs(os.path.join(train_dir, country), exist_ok=True)
        os.makedirs(os.path.join(val_dir, country), exist_ok=True)

        print(f"\n🔍 [{country}] 데이터 확인 중...")

        # --- Train 데이터 처리 ---
        src_train = os.path.join(base_dir, country, "ocr_train")
        if os.path.exists(src_train):
            files = os.listdir(src_train)
            for f in files:
                dest_path = os.path.join(train_dir, country, f)
                
                # 💡 핵심 로직: 목적지에 파일이 없으면 복사하고, 있으면 건너뜀!
                if not os.path.exists(dest_path):
                    shutil.copy(os.path.join(src_train, f), dest_path)
                    total_copied += 1
                else:
                    total_skipped += 1
            print(f"  - Train: {len(files)}장 완료")

        # --- Val 데이터 처리 ---
        src_val = os.path.join(base_dir, country, "ocr_val")
        if os.path.exists(src_val):
            files = os.listdir(src_val)
            for f in files:
                dest_path = os.path.join(val_dir, country, f)
                
                if not os.path.exists(dest_path):
                    shutil.copy(os.path.join(src_val, f), dest_path)
                    total_copied += 1
                else:
                    total_skipped += 1
            print(f"  - Val: {len(files)}장 완료")

    print(f"\n🎉 작전 완료! (새로 복사됨: {total_copied}장 / 이미 있어서 건너뜀: {total_skipped}장)")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    
    build_classification_dataset(PROJECT_ROOT)