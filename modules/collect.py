from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import json

class ModelCollect:
    def __init__(self, tools):        
        self.tools = tools   

    @staticmethod
    def is_data_enough(collected_data):
           null_count = sum(1 for v in collected_data.values() if v is None)
           if len(collected_data.values()) - null_count >= len(collected_data.values()) // 2 :
               return True
           else:
               return False

    def get_response(self, messages, collected_data):
        
        prompt = f"""
            다음과 같은 상황에 대해 반드시 json 형식으로 응답하세요.
            당신은 식물 인테리어 추천 또는 식물 선물 추천을 위한 상담 챗봇입니다.  
            당신의 목표는 아래 JSON 형식을 사용자와의 대화를 통해 하나씩 채워가는 것입니다.

            현재까지 수집된 정보는 다음과 같습니다:
            {collected_data}

            ### 지침
            1. JSON에서 아직 값이 null인 항목을 확인합니다.
            2. 한 번의 턴에서는 정보를 채울수 있는 자연스러운 질문을 **한 가지** 합니다.
            3. 모든 질문은 자연스러운 대화를 통해 이루어져야 하고, **최대한 여러개의 정보를 한번에 얻을 수 있는 질문**을 해야합니다.
            4. 사용자의 답변을 기반으로 **한개 또는 여러개**의 해당 항목을 업데이트합니다.
            5. **응답은 아래 출력 예시 형식 구조를 반드시 따라야 합니다**:

            - "assistant_message":  
            사용자의 답변을 반영하여 다음 질문을 하거나 대화를 이어가는 문장입니다.  
            두 문장 이상으로 이루어져야 하며, 자연스러운 이야기의 흐름을 연결하며 마지막에 질문하도록 합니다.
            (아직 채워야 할 항목이 있다면 질문을 계속하며, 모든 항목이 채워졌다면 전체 데이터를 요약하고 질문을 멈춥니다.)

            - "updated_json":  
            지금까지 수집된 모든 데이터를 포함하는 JSON 객체입니다.  
            null이 아닌 값은 이전 사용자 답변을 기반으로 유지합니다.

            ### 규칙
            - JSON 값은 절대 임의로 추측하지 말고, 반드시 사용자 답변으로만 채웁니다.
            - 질문 순서는 반드시 랜덤한 순서대로 진행해야 합니다.
            - yes/no 형태 질문은 true 또는 false로 변환하여 저장합니다.
            - 사용자 경험 수준이나 감정 등 주관적 답변도 사용자가 말한 표현을 그대로 저장합니다.
            - **항상 "assistant_message"와 "updated_json"이 포함된 유효한 JSON 형태**로 반환해야 합니다.

            ### 출력 예시 형식
            {{
            "assistant_message": "먼저, 어떤 용도에 맞는 식물을 추천받고 싶으신가요?",
            "updated_json": {{
                "purpose": null, # 목적 (사용자의 입력 그대로)
                "preferred_style": null, # 인테리어 스타일, 선호하는 스타일
                "preferred_color": null, # 선호하는 색상
                "plant_type": null, # 인테리어 시 선호하는 식물
                "season": null, # 현재 계절
                "humidity": null, # 습도
                "has_dog": null, # 강아지를 키우는지
                "has_cat": null, # 고양이를 키우는지
                "isAirCond": null, # 공기 정화가 필요한지
                "watering_frequency": null, # 물을 줄 수 있는 주기
                "user_experience": null, # 기타 정보
                "emotion": null # 담고 싶은 분위기나 감정
            }}
            }}
        """
        
        system_msg = SystemMessage(prompt)
        input_msg = [system_msg] + messages

        model = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.3,
        )

        error_count = 0
        while True:
            response = model.invoke(input_msg)
            
            try:
                res_json = json.loads(response.content)
                response_message = res_json['assistant_message']
                collected_data = res_json['updated_json']
                break
            except:
                error_count +=1
                if error_count > 1 :
                    raise ValueError("모델이 올바른 JSON을 반환하지 않습니다.")
                continue                    
        
        return response, response_message, collected_data