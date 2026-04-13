import time
import datetime
from ultralytics import YOLO

def run_overnight_experiments():
    # 💡 8시간 밤샘용 4가지 실험 세팅
    experiments = [
        # 실험 1: [기준점] 가장 기본 세팅
        #{'exp_name': '01_baseline_11n_640', 'model': 'yolo11n.pt', 'imgsz': 640, 'batch': 16, 'optimizer': 'auto'},
        
        # 실험 2: [해상도 UP] 번호판은 작으니까 이미지를 더 크게 줘보자 (추천!)
        {'exp_name': '02_highres_11n_800', 'model': 'yolo11n.pt', 'imgsz': 800, 'batch': 8, 'optimizer': 'auto'},
        
        # 실험 3: [모델 UP] 더 무겁고 똑똑한 Small 모델을 써보자
        {'exp_name': '03_model_11s_640', 'model': 'yolo11s.pt', 'imgsz': 640, 'batch': 8, 'optimizer': 'auto'},
        
        # 실험 4: [옵티마이저 변경] 파인튜닝에 좋다는 AdamW를 써보자
        {'exp_name': '04_adamw_11n_640', 'model': 'yolo11n.pt', 'imgsz': 640, 'batch': 16, 'optimizer': 'AdamW'}
    ]

    print("🌙 8시간 오버나이트 YOLO 파인튜닝 자동화를 시작합니다...")
    total_start_time = time.time()

    for i, exp in enumerate(experiments):
        print(f"\n==================================================")
        print(f"🧪 [실험 {i+1}/{len(experiments)}] 시작: {exp['exp_name']}")
        print(f"==================================================")

        model = YOLO(exp['model'])

        model.train(
            data='data.yaml',
            epochs=300,            # 넉넉하게 300 (얼리스탑핑이 잡아줌)
            patience=30,           # 30번 동안 개선 없으면 다음 실험으로 넘어감
            batch=exp['batch'],    
            imgsz=exp['imgsz'],    # 변경된 이미지 사이즈 적용
            optimizer=exp['optimizer'], # 변경된 옵티마이저 적용
            project='Overnight_Experiments', 
            name=exp['exp_name'],  
            device=1,
            workers=0,
            exist_ok=True,
            plots=True
        )
        print(f"✅ {exp['exp_name']} 실험 완료!\n")

    total_end_time = time.time()
    elapsed_time = total_end_time - total_start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"🎉 기상 시간입니다! 총 소요 시간: {int(hours)}시간 {int(minutes)}분")
    print(f"📁 'Overnight_Experiments' 폴더에서 각 실험의 results.png와 mAP 점수를 비교해보세요.")

if __name__ == '__main__':
    run_overnight_experiments()