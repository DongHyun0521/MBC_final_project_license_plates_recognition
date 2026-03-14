# plate_router.py
import re

def detect_country_and_route(plate_text):
    """
    OCR로 인식된 번호판 텍스트를 분석하여 6개 타겟 국가를 판별하고,
    프론트엔드에 전달할 응답(JSON 형태의 딕셔너리)을 반환합니다.
    """
    # 공백 및 특수문자 제거 (알파벳, 숫자, 한글, 한자만 남김)
    clean_text = re.sub(r'[^a-zA-Z0-9가-힣一-龥]', '', plate_text).upper()
    
    # 1. 대한민국 (한국어)
    if re.search(r'[가-힣]', clean_text):
        return {
            "country": "대한민국",
            "plate_number": clean_text,
            "language": "ko",
            "welcome_message": "전기차 충전소에 오신 것을 환영합니다. 결제 수단을 확인합니다."
        }
        
    # 2. 중국 (중국어 간체)
    elif re.search(r'[一-龥]', clean_text):
        return {
            "country": "中国", # 중국 (Zhongguo)
            "plate_number": clean_text,
            "language": "zh",
            "welcome_message": "欢迎来到电动汽车充电站。正在确认支付方式。"
        }
        
    # 3. 인도 (힌디어)
    # 패턴: 알파벳2 + 숫자1~2 + 알파벳1~3 + 숫자4 (예: MH12AB1234)
    elif re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$', clean_text):
        return {
            "country": "भारत", # 인도 (Bharat)
            "plate_number": clean_text,
            "language": "hi",
            "welcome_message": "इलेक्ट्रिक वाहन चार्जिंग स्टेशन में आपका स्वागत है।"
        }
        
    # 4. 브라질 (포르투갈어)
    # 패턴: 알파벳3 + 숫자1 + 알파벳/숫자1 + 숫자2 (구형 ABC1234, 신형 ABC1D23)
    elif re.match(r'^[A-Z]{3}\d[A-Z0-9]\d{2}$', clean_text):
        return {
            "country": "Brasil", # 브라질
            "plate_number": clean_text,
            "language": "pt",
            "welcome_message": "Bem-vindo à estação de carregamento de veículos elétricos."
        }
        
    # 5. 유럽연합 (영어 공용)
    # 대표 패턴: 프랑스/이탈리아(알파벳2+숫자3+알파벳2) 또는 영국(알파벳2+숫자2+알파벳3)
    elif re.match(r'^[A-Z]{2}\d{2,3}[A-Z]{2,3}$', clean_text):
        return {
            "country": "Europe", # 유럽연합
            "plate_number": clean_text,
            "language": "en",
            "welcome_message": "Welcome to the European EV charging station."
        }
        
    # 6. 말레이시아 (말레이어)
    # 패턴: 알파벳1~3 + 숫자1~4 + (선택)알파벳1 (예: W1234C, BGY333)
    elif re.match(r'^[A-Z]{1,3}\d{1,4}[A-Z]?$', clean_text):
        return {
            "country": "Malaysia", # 말레이시아
            "plate_number": clean_text,
            "language": "ms",
            "welcome_message": "Selamat datang ke stesen pengecasan kenderaan elektrik."
        }
        
    # 7. 예외 처리 (알 수 없는 형태)
    else:
        return {
            "country": "Unknown", # 미상
            "plate_number": clean_text,
            "language": "en",
            "welcome_message": "Welcome to the EV charging station. Checking payment method."
        }