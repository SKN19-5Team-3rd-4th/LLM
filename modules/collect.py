from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import json
import streamlit as st

collected_data = {
                "room": "거실 / 거실의 빈 벽이 너무 심심해서 놓고 싶어",
                "humidity": "건조",
                "preferred_style": "미니멀",
                "preferred_color": "따뜻한 색",
                "has_dog": True,
                "has_cat": False,
                "isAirCond": True,
                "watering_frequency": "주 1회",
                "user_experience": "쉬움"
            }

class ModelCollect:
    question_list = {
        "room" : {"question" : "식물을 놓을 공간은 어디인가요?",
                  "choice" : ["거실", "베란다", "침실", "사무실", "서재", "현관", "상업적 공간"]},
        "purpose" : {"question" : "식물을 구매하려는 주요 목적이 무엇인가요?",
                     "choice" : []},
        "humidity" : {"question" : "공간의 환경은 어떤 편인가요?",
                      "choice": ["습한 편", "건조한 편"]},
        "preferred_style" : {"question" : "인테리어 스타일은 어떤 느낌인가요?",
                             "choice": ["모던", "미니멀", "스칸디나비아", "전통", "빈티지", "내추럴", "레트로", "클래식"]},
        "preferred_color" : {"question" : "선호하는 색상은 어떤 색상인가요?",
                             "choice": ["무채색 계열", "따뜻한 색", "차가운 색", "선명한 색"]},
        "has_dog_or_cat" : {"question" : "강아지나 고양이를 키우시나요?",
                     "choice" : ["강아지", "고양이", "키우지 않음"]},
        "isAirCond" : {"question" : "공기 정화가 가능한 식물을 원하시나요?",
                       "choice" : ["예", "아니오"]},
        "watering_frequency": {
                         "question": "원하는 물 주기는?",
                         "choice": ["주 1회", "2주에 1회", "3주에 1회", "4주에 1회", "상관 없음"]},
        "uesr_experience": {
                            "question": "원하는 식물 난이도?",
                            "choice": ["쉬움", "중간", "어려움", "상관 없음"]}
        
    }

    def __init__(self, tools):        
        self.tools = tools   

    @staticmethod
    def is_data_enough(collected_data):
           null_count = sum(1 for v in collected_data.values() if v is None)
           if null_count == 0 :
               return True
           else:
               return False

    def get_response(self, messages, collected_data):
        if len(messages) >= 2 :
            key = None
            for k, v in self.question_list.items():
                if v["question"] == messages[-2].content:
                    key = k
                    break

            data = [word for word in self.question_list[key]["choice"] if word in messages[-1].content.lower()]

            updated_data = collected_data

            if data:
                if key == "has_dog_or_cat":
                    if "강아지" in data :
                        updated_data["has_dog"] = True
                    if "고양이" in data :
                        updated_data["has_cat"] = True

                updated_data[key] = ', '.join(data)
            else:
                updated_data[key] = messages[-1].content

            keys = [k for k, v in updated_data.items() if v is None]
            if len(keys) > 0 :
                message = self.question_list[keys[0]]["question"]
            else :
                message = "정보 수집이 끝났습니다. 잠시만 기다려 주세요."
        else : 
            updated_data = collected_data
            message = self.question_list["purpose"]["question"]

        response = AIMessage(content=message)
        
        return response, updated_data