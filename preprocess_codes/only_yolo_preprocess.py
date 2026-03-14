import os
import csv
import shutil
from pathlib import Path

def unified_csv_yolo_preprocess(base_dir):
    print("🚀 [Only_YOLO 데이터셋] CSV -> YOLO 포맷 대변환 작전을 시작합니다!")

    # 1. 우리의 메인 파이프라인 폴더 (기존 폴더에 그대로 이어 붙입니다)
    # base_dir은 '.../only_yolo' 이므로, 여기서 한 칸 더 올라가면 프로젝트 루트입니다.
    project_root = os.path.dirname(base_dir) 
    yolo_img_dir = os.path.join(project_root, "only_yolo/yolo_dataset", "images")
    yolo_lbl_dir = os.path.join(project_root, "only_yolo/yolo_dataset", "labels")
    
    os.makedirs(yolo_img_dir, exist_ok=True)
    os.makedirs(yolo_lbl_dir, exist_ok=True)

    success_count = 0

    # 2. train, valid, test 3개 폴더를 돌면서 작업
    for split in ["train", "valid", "test"]:
        folder_path = os.path.join(base_dir, split)
        if not os.path.exists(folder_path): continue
        
        csv_file = os.path.join(folder_path, "_annotations.csv")
        if not os.path.exists(csv_file): 
            print(f"⚠️ [{split}] 폴더에 _annotations.csv 파일이 없습니다. 패스!")
            continue

        print(f"\n🔍 [{split}] 폴더 처리 시작...")
        
        # 3. CSV 파일 읽기
        with open(csv_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    filename = row['filename']
                    width = float(row['width'])
                    height = float(row['height'])
                    
                    # 좌표 추출 (픽셀 단위 절대 좌표)
                    xmin = float(row['xmin'])
                    ymin = float(row['ymin'])
                    xmax = float(row['xmax'])
                    ymax = float(row['ymax'])
                    
                    # 💡 핵심 수학: YOLO용 정규화(0~1) 좌표로 변환
                    x_center = ((xmin + xmax) / 2.0) / width
                    y_center = ((ymin + ymax) / 2.0) / height
                    box_width = (xmax - xmin) / width
                    box_height = (ymax - ymin) / height
                    
                    # 소수점 6자리까지 깔끔하게 포맷팅
                    yolo_line = f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
                    
                    file_stem = Path(filename).stem
                    
                    # 원본 이미지 파일 경로 확인
                    src_img_path = os.path.join(folder_path, filename)
                    if not os.path.exists(src_img_path): continue
                    
                    # 라벨 파일(.txt) 쓰기
                    txt_path = os.path.join(yolo_lbl_dir, f"{file_stem}.txt")
                    
                    # 만약 이미 파일이 있으면 내용 덧붙이기 (한 사진에 번호판 여러 개)
                    mode = 'a' if os.path.exists(txt_path) else 'w'
                    with open(txt_path, mode) as lf:
                        lf.write(yolo_line)
                        
                    # 이미지는 한 번만 복사하면 됨
                    dest_img_path = os.path.join(yolo_img_dir, filename)
                    if not os.path.exists(dest_img_path):
                        shutil.copy(src_img_path, dest_img_path)

                    success_count += 1
                    
                    if success_count % 100 == 0:
                        print(f"⏳ 열일 중... {success_count}장 변환 완료!")

                except Exception as e:
                    pass # 헤더 에러나 찌꺼기 파일 무시

    print(f"\n🎉 [Only_YOLO 데이터셋] 1만 장 털어넣기 작전 완료! (총 {success_count}개 박스 처리)")

if __name__ == "__main__":
    # 💡 폴더 이사 완벽 반영: 현재 코드 파일이 'preprocess_codes' 안에 있으므로 한 칸 올라감
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    
    # 진짜 only_yolo 폴더 경로 연결
    ONLY_YOLO_DIR = os.path.join(PROJECT_ROOT, "only_yolo")
    
    print(f"📂 인식된 타겟 데이터 경로: {ONLY_YOLO_DIR}\n")
    unified_csv_yolo_preprocess(ONLY_YOLO_DIR)