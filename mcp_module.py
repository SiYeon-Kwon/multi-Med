from analyze_module.ecg_module import analyze_ecg
from analyze_module.eeg_module import analyze_eeg
from gradcam_module.gradcam import generate_gradcam
from report_module.report_generator import generate_report

# 예시: X-ray, CT, MRI 등도 추가 가능
from analyze_module.xray_module import analyze_xray
from analyze_module.ct_module import analyze_ct
from analyze_module.mri_module import analyze_mri

def run_analysis(data_type, data_input):
    analyzer = {
        "xray": analyze_xray,
        "ct": analyze_ct,
        "mri": analyze_mri,
        "ecg": analyze_ecg,
        "eeg": analyze_eeg,
    }.get(data_type)

    if analyzer is None:
        raise ValueError("지원하지 않는 데이터 타입입니다.")

    prediction = analyzer(data_input)
    gradcam = generate_gradcam(None, data_input, data_type)
    return generate_report(data_type, prediction, gradcam)