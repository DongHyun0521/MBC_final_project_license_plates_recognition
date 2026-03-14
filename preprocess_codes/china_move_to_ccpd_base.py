import os
import glob
import shutil

def move_ccpd_data(ccpd_root):
    print("🚀 [CCPD 파일 순간이동] 윈도우 탐색기를 무시하고 파일 이동을 시작합니다...")

    # 모일 장소 (기존 ccpd_base 폴더)
    target_dir = os.path.join(ccpd_root, "ccpd_base")
    if not os.path.exists(target_dir):
        print(f"🚨 에러: 타겟 폴더가 없습니다! 경로를 확인하세요: {target_dir}")
        return

    # 합칠 악조건 폴더들 이름 (여기에 있는 사진들을 base로 싹 다 옮깁니다)
    # 다운받으신 폴더명에 맞게 추가/삭제 가능합니다.
    folders_to_move = [
        "ccpd_challenge", "ccpd_db", "ccpd_fn", 
        "ccpd_rotate", "ccpd_tilt", "ccpd_weather"
    ]

    total_moved = 0

    for folder in folders_to_move:
        source_dir = os.path.join(ccpd_root, folder)
        
        # 해당 폴더가 있는지 확인
        if not os.path.exists(source_dir):
            print(f"⚠️ {folder} 폴더를 찾을 수 없어 건너뜁니다.")
            continue

        print(f"📦 [{folder}] 폴더에서 짐 싸는 중...")
        
        # 폴더 안의 모든 jpg 파일 찾기
        jpg_files = glob.glob(os.path.join(source_dir, "*.jpg"))
        
        for file_path in jpg_files:
            file_name = os.path.basename(file_path)
            target_path = os.path.join(target_dir, file_name)
            
            # 파일 순간이동 (shutil.move는 덮어쓰기 위험 없이 아주 빠르게 이동시킵니다)
            # 만약 같은 드라이브 내 이동이면 1초도 안 걸립니다.
            try:
                shutil.move(file_path, target_path)
                total_moved += 1
            except Exception as e:
                pass # 이미 이동했거나 에러 난 파일은 무시

        print(f"✅ [{folder}] 이동 완료! (현재 누적: {total_moved}장)")

    print(f"🎉 대성공! 총 {total_moved}장의 사진이 ccpd_base로 안전하게 이사했습니다.")

if __name__ == "__main__":
    # 🚨 여기에 CCPD 전체 폴더(ccpd_base, ccpd_weather 등이 모여있는 상위 폴더) 경로를 넣어주세요!
    CCPD_ROOT_DIR = r"C:\Users\dhlim\PycharmProjects\MBC_final_project_license_plates_recognition\china" # 본인 경로로 수정
    
    move_ccpd_data(CCPD_ROOT_DIR)