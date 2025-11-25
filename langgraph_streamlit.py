from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Optional, Literal, List

from functools import partial
import operator
from dotenv import load_dotenv
import warnings
import json
import streamlit as st
import requests
from PIL import Image
import io

from modules.collect import ModelCollect
from modules.recommend import ModelRecommend, tool_rag_recommend
from modules.qna import ModelQna, tool_rag_qna

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 정보 저장 state 선언 --------------------

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]                         # 모든 메시지를 저장하는 리스트

    current_stage: Literal["collect", "recommend", "qna", "exit"]   # 현재 어떤 작업을 하고 있는지 저장

    collected_data: Optional[dict]                                  # 사용자에게서 모은 데이터(정보)를 저장하는 딕셔너리

    recommend_result: Annotated[Optional[List[str]], operator.add]  # 사용자에게 추천한 결과(해당 추천 결과는 재추천할때에 고려하지 않게 하기 위함)

    # None: 아무 행동도 하지 않음, Skip: 다음 단계로, Continue: 추천 만족, Retry: 추천 다시 받기, Restart: 처음부터 재시작, QnA: QnA로 이동
    user_action: Literal["None", "Skip", "Continue", "Retry", "Restart", "QnA", "Exit"]


initial_state = {
    "messages": [AIMessage(content="안녕하세요. AI입니다.")],
    "current_stage": "collect",
    "user_action": "None",
    "collected_data": {
                "purpose": None,            
                "preferred_style": None,    
                "preferred_color": None,
                "plant_type": None,
                "season": None,
                "humidity": None,
                "has_dog": None,
                "has_cat": None,
                "isAirCond": None,
                "watering_frequency": None,
                "user_experience": None,
                "emotion": None
            },
    "recommend_result": " "
}
### tools 선언 ---------------------------
# tool 함수 선언

# tools 에는, 각각 RAG를 수행하는 두가지 함수가 들어가야 함
tools = [tool_rag_recommend, tool_rag_qna]

### 노드 선언 -----------------------------

def node_collect(state: GraphState, collector: ModelCollect):
    collected_data = collector.get_response(state["collected_data"])  # 어떤 정보를 전달했는지 알아야 하니까 collected_data도 같이 전달
    
    return {
        "current_stage" : "recommend",
        "collected_data": collected_data,
    }

def node_recommend(state: GraphState, recommender: ModelRecommend):

    response, recommend_result = recommender.get_response(state["messages"], state["collected_data"], state["recommend_result"])  

    return {
        "current_stage" : "recommend",
        "messages": [response],
        "recommend_result": recommend_result,
    }

def node_qna(state: GraphState, chatbot: ModelQna):
    response = chatbot.get_response(state["messages"])

    return {
        "current_stage": "qna",
        "messages": [response],
    }

def node_end_state(state:GraphState):
    return {
        "current_stage": "exit"
    }


### router 선언 -----------------------

# 해당 router의 결과에 따라, 어떤 노드로 향할지 컨트롤
def main_router(state: GraphState):
    stage = state["current_stage"]
    action = state["user_action"]

    if action == "Restart":
        return "restart"
    
    if action == "Exit":
        return "exit"
    
    if action == "QnA":
        return "qna"
    
    
    if stage == "collect":
        if action == "Continue":
            return "recommend"
        
        if ModelCollect.is_data_enough(state["collected_data"]):
            return "recommend"
        else:
            return "collect"
    
    elif stage == "recommend":
        if action == "Continue":
            return "exit"
        
        elif action == "QnA":
            return "qna"
        else:   # action == "Retry"
            return "recommend"
    
    elif stage == "qna":
        return "qna"
    
    elif stage == "exit":
        return "exit"
    
def is_tool_calls(state: GraphState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool_call"
    else:
        return "done"
    
def tool_back_to_caller(state: GraphState) -> str:
    current_state = state.get("current_stage")

    if current_state == "recommend":
        print(f"[ToolMessages] [RAG] [Pinecone Index name is plant-rec]")
    elif current_state == "qna":
        print(f"[ToolMessages] [RAG] [Pinecone Index name is plant-qna]")
    print(state["messages"][-1])

    if current_state and current_state in ["collect", "recommend", "qna"]:
        return current_state
    
    return "exit"


model_collect = ModelCollect(tools)
model_recommend = ModelRecommend(tools)
model_qna = ModelQna(tools)

workflow = StateGraph(GraphState)

workflow.add_node("collect", partial(node_collect, collector=model_collect))
workflow.add_node("recommend", partial(node_recommend, recommender=model_recommend))
workflow.add_node("qna", partial(node_qna, chatbot=model_qna))
workflow.add_node("exit", node_end_state)
workflow.add_node("rag_tool", ToolNode(tools))

workflow.add_edge("exit", END)
workflow.add_edge("collect", END)

workflow.add_conditional_edges(
    START,
    main_router,
    {
        "collect": "collect",
        "recommend": "recommend",
        "qna": "qna",
        "exit": "exit"
    }
)

workflow.add_conditional_edges(
    "recommend",
    is_tool_calls,
    {
        "tool_call": "rag_tool",
        "done": END,
    }
)

workflow.add_conditional_edges(
    "qna",
    is_tool_calls,
    {
        "tool_call": "rag_tool",
        "done": END,
    }
)

workflow.add_conditional_edges(
    "rag_tool",
    tool_back_to_caller,
    {
        "collect": "collect",
        "recommend": "recommend",
        "qna": "qna",
        "exit": "exit",
    }
)

### streamlit -----------------------

# 메시지 파싱 함수
def parse_ai_content(content):
    if isinstance(content, str) and content.startswith('{'):    # 메시지가 json 형태라면 딕셔너리로 변환
        try:
            data = json.loads(content)
            if "assistant_message" in data: return data["assistant_message"], None
            if "response" in data: return data["response"], data["flowNm"]
        except: pass
    return content, None

if "is_collected" not in st.session_state:
    st.session_state.is_collected = False

if "collected_data" not in st.session_state:
    st.session_state.collected_data = {
                "purpose": None,            
                "preferred_style": None,    
                "preferred_color": None,
                "plant_type": None,
                "season": None,
                "humidity": None,
                "has_dog": None,
                "has_cat": None,
                "isAirCond": None,
                "watering_frequency": None,
                "user_experience": None,
                "emotion": None
            }

if st.session_state.is_collected is False:
    options = {
        "purpose": ["공기 정화", "인테리어", "선물", "학습/관찰", "반려용"],
        "style": ["모던/심플", "빈티지", "내추럴/우드", "화려함"],
        "color": ["초록색(기본)", "알록달록", "흰색 꽃", "분홍/빨강 계열"],
        "type": ["관엽식물", "다육/선인장", "꽃이 피는 식물", "행잉 플랜트"],
        "season": ["봄", "여름", "가을", "겨울", "사계절 무관"],
        "humidity": ["건조한 편", "보통", "습한 편"],
        "experience": ["식집사 입문 (초보)", "경험 있음 (중수)", "전문가 (고수)"],
        "emotion": ["행복/기쁨", "차분함/힐링", "우울/위로", "피곤/활력필요"],
    }

    with st.form(key="plant_preference_form"):
        collected_data = {
                "purpose": None,            
                "preferred_style": None,    
                "preferred_color": None,
                "plant_type": None,
                "season": None,
                "humidity": None,
                "has_dog": None,
                "has_cat": None,
                "isAirCond": None,
                "watering_frequency": None,
                "user_experience": None,
                "emotion": None
        }
        st.caption("일부 항목을 선택한 후 하단의 버튼을 눌러주세요.")

        col1, col2 = st.columns(2)

        def get_selection(label, options_list):
            selection = st.selectbox(label, ["선택하세요"] + options_list)
            return selection if selection != "선택하세요" else None

        def get_bool_selection(label):
            selection = st.selectbox(label, ["선택하세요"] + options["yes_no"])
            if selection == "예": return True
            elif selection == "아니오": return False
            else: return None

        with col1:
            st.subheader("환경 및 목적")
            collected_data["purpose"] = get_selection("구매 목적", options["purpose"])
            collected_data["season"] = get_selection("현재 계절", options["season"])
            collected_data["humidity"] = get_selection("설치 공간 습도", options["humidity"])
            collected_data["user_experience"] = get_selection("식물 키우기 경험", options["experience"])

        with col2:
            st.subheader("취향 및 경험")
            collected_data["preferred_style"] = get_selection("선호하는 스타일", options["style"])
            collected_data["preferred_color"] = get_selection("선호하는 색상", options["color"])
            collected_data["plant_type"] = get_selection("원하는 식물 종류", options["type"])
            collected_data["emotion"] = get_selection("현재 기분/얻고 싶은 감정", options["emotion"])

        st.divider()

        submitted = st.form_submit_button("식물 추천 받기")
    if submitted:
        st.session_state.collected_data = collected_data
        st.session_state.is_collected = True
else:
    initial_state = {
        "messages": [AIMessage(content="안녕하세요. AI입니다.")],
        "current_stage": "recommend",
        "user_action": "None",
        "collected_data": st.session_state.collected_data,
        "recommend_result": " "
    }

    # "compile()" 은 rerun마다 재사용되도록 session_state에 저장
    if "app" not in st.session_state:
        memory = MemorySaver()
        st.session_state.app = workflow.compile(checkpointer=memory)

    st.set_page_config(page_title="PLANT AI", page_icon="🌿")

    st.title("A.P.T(AI Plant Teller)")


    app = st.session_state.app

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "user_1234" # 고유 ID

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # 초기 메시지/상태가 없으면 초기화
    current_state_snapshot = app.get_state(config)
    if not current_state_snapshot.values:
        app.invoke(initial_state, config=config)
        
        st.rerun()

    # 현재 상태 가져오기
    state_values = app.get_state(config).values
    messages = state_values.get("messages", [])
    current_stage = state_values.get("current_stage", "collect")
    collected_data = state_values.get("collected_data", {})


    with st.sidebar:
        st.header("진행 상황")
        stage_map = {"collect": "정보 수집", "recommend": "추천", "qna": "상담", "exit": "종료"}
        st.info(f"현재 단계: **{stage_map.get(current_stage, current_stage)}**")

        if st.button("처음부터 다시 시작"):
            # 상태 리셋 로직 (새 thread_id 발급 등)
            st.session_state.thread_id = f"user_{int(st.session_state.thread_id.split('_')[1]) + 1}"
            st.rerun()


    # 히스토리 출력
    for msg in messages[1:]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            if msg.content:
                text, flowNm = parse_ai_content(msg.content)
                with st.chat_message("assistant", avatar="🌿"):
                    if flowNm is not None:
                        with open("datas/flower_preprocessed_data.json", "r", encoding="utf-8") as f:
                            flower_list = json.load(f)

                        target = next((item for item in flower_list if item.get("flowNm") == flowNm), None)

                        if target is None:
                            st.warning(f"'{flowNm}' 데이터가 없습니다.")
                        else:
                            image_url = target.get("imgUrl1")

                            if not image_url:
                                st.warning(f"'{flowNm}' 데이터에 이미지 URL이 없습니다.")
                            else:
                                # 3. 이미지 다운로드
                                response_img = requests.get(image_url)
                                image_data = response_img.content

                                # 4. 이미지 객체 변환
                                pil_img = Image.open(io.BytesIO(image_data))

                                # 5. Streamlit에 출력
                                st.image(pil_img, caption=flowNm)
                    st.write(text)



    if user_input := st.chat_input("메시지를 입력하세요..."):
        # 사용자 입력 즉시 표시
        with st.chat_message("user"):
            st.write(user_input)
        
        # Action 결정 로직
        action = "None"
        actual_input = user_input

        if user_input.lower() == "종료":
            action = "Exit"
        elif user_input.lower() == "qna":
            action = "QnA"
            actual_input = "안녕? 자기소개 해줘" # 상태 전환 트리거용
        elif user_input.lower() == "next" or user_input == "추천해줘":
            action = "Continue" # 혹은 로직에 따라 Skip
            actual_input = "추천해줘"

        input_payload = {
            "messages": [HumanMessage(content=actual_input)],
            "user_action": action
        }

        with st.chat_message("assistant", avatar="🌿"):
            with st.spinner("생각 중..."):
                # Graph 실행
                result = app.invoke(input_payload, config=config)
                
                # 마지막 응답 출력
                last_msg = result["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    response, flowNm = parse_ai_content(last_msg.content)
                    if flowNm is not None:
                        with open("datas/flower_preprocessed_data.json", "r", encoding="utf-8") as f:
                            flower_list = json.load(f)

                        target = next((item for item in flower_list if item.get("flowNm") == flowNm), None)

                        if target is None:
                            st.warning(f"'{flowNm}' 데이터가 없습니다.")
                        else:
                            image_url = target.get("imgUrl1")

                            if not image_url:
                                st.warning(f"'{flowNm}' 데이터에 이미지 URL이 없습니다.")
                            else:
                                response_img = requests.get(image_url)
                                image_data = response_img.content

                                pil_img = Image.open(io.BytesIO(image_data))

                                st.image(pil_img, caption=flowNm)
                    st.write(response)
                    
