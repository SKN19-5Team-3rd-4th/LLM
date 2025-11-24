from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import json
import streamlit as st

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

    def get_response(self, collected_data):
        
        options = {
            "purpose": ["공기 정화", "인테리어", "선물", "학습/관찰", "반려용"],
            "style": ["모던/심플", "빈티지", "내추럴/우드", "화려함"],
            "color": ["초록색(기본)", "알록달록", "흰색 꽃", "분홍/빨강 계열"],
            "type": ["관엽식물", "다육/선인장", "꽃이 피는 식물", "행잉 플랜트"],
            "season": ["봄", "여름", "가을", "겨울", "사계절 무관"],
            "humidity": ["건조한 편", "보통", "습한 편"],
            "watering": ["자주 (주 2회 이상)", "보통 (주 1회)", "가끔 (월 2-3회)", "거의 안 함 (월 1회)"],
            "experience": ["식집사 입문 (초보)", "경험 있음 (중수)", "전문가 (고수)"],
            "emotion": ["행복/기쁨", "차분함/힐링", "우울/위로", "피곤/활력필요"],
            "yes_no": ["예", "아니오"] 
        }

        # 1. 폼(Form) 시작: 이 블록 안의 위젯들은 즉시 반응하지 않습니다.
        with st.form(key="plant_preference_form"):
            st.caption("모든 항목을 선택한 후 하단의 버튼을 눌러주세요.")

            # 3. 화면 레이아웃 구성
            col1, col2 = st.columns(2)

            # 헬퍼 함수들 (폼 내부에서 작동)
            def get_selection(label, options_list):
                selection = st.selectbox(label, ["선택하세요"] + options_list)
                return selection if selection != "선택하세요" else None

            def get_bool_selection(label):
                selection = st.selectbox(label, ["선택하세요"] + options["yes_no"])
                if selection == "예": return True
                elif selection == "아니오": return False
                else: return None

            # --- 컬럼 1 입력 ---
            with col1:
                st.subheader("🏠 환경 및 목적")
                collected_data["purpose"] = get_selection("구매 목적", options["purpose"])
                collected_data["season"] = get_selection("현재 계절", options["season"])
                collected_data["humidity"] = get_selection("설치 공간 습도", options["humidity"])
                collected_data["isAirCond"] = get_bool_selection("에어컨/히터 바람이 직접 닿나요?")
                collected_data["has_dog"] = get_bool_selection("강아지를 키우시나요?")
                collected_data["has_cat"] = get_bool_selection("고양이를 키우시나요?")

            # --- 컬럼 2 입력 ---
            with col2:
                st.subheader("🎨 취향 및 경험")
                collected_data["preferred_style"] = get_selection("선호하는 스타일", options["style"])
                collected_data["preferred_color"] = get_selection("선호하는 색상", options["color"])
                collected_data["plant_type"] = get_selection("원하는 식물 종류", options["type"])
                collected_data["watering_frequency"] = get_selection("선호하는 물주기 빈도", options["watering"])
                collected_data["user_experience"] = get_selection("식물 키우기 경험", options["experience"])
                collected_data["emotion"] = get_selection("현재 기분/얻고 싶은 감정", options["emotion"])

            st.divider()

            # 2. 폼 제출 버튼 (Form Submit Button)
            # 이 버튼을 눌러야만 위의 선택값들이 확정되고 스크립트가 Rerun 됩니다.
            submitted = st.form_submit_button("식물 추천 받기 🪴")

        # 4. 제출 후 로직 처리 (폼 블록 바깥에서 처리)
        if submitted:
            return collected_data