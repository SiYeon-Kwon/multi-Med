from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(temperature=0.4)

def generate_report(data_type, prediction, gradcam_path):
    content = f"""
    환자 {data_type.upper()} 데이터에 대한 분석 결과는 다음과 같습니다.\n
    예측 값: {prediction}\n
    중요한 시각화(GRAD-CAM): {gradcam_path}\n
    위 내용을 기반으로 의료 리포트를 생성해주세요.
    """
    response = llm([HumanMessage(content=content)])
    print("=== AI 기반 리포트 ===\n", response.content)
    return response.content
