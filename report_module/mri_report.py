from datetime import datetime

def generate_mri_report(prediction, heatmap_path):
    label_map = {0: '정상', 1: '이상 소견 있음'}
    label = label_map.get(prediction, '미확인')
    
    report = f"""
    [MRI AI 분석 리포트]
    - 분석 일자: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - 판독 결과: {label}
    - 주의가 필요한 부위는 Grad-CAM 이미지(첨부파일: {heatmap_path})를 참고하십시오.
    """
    return report