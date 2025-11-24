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
from pyfiglet import figlet_format
import json
import streamlit as st

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

# tools 에는, 각각 이미지 처리 혹은 RAG를 수행하는 세가지 함수가 들어가야 함
tools = [tool_rag_recommend, tool_rag_qna]

### 노드 선언 -----------------------------

def node_collect(state: GraphState, collector: ModelCollect):
    response, message, collected_data = collector.get_response(state["messages"], state["collected_data"])  # 어떤 정보를 전달했는지 알아야 하니까 collected_data도 같이 전달
    
    return {
        "current_stage" : "collect",
        "messages": [response],
        "collected_data": collected_data,
    }

def node_recommend(state: GraphState, recommender: ModelRecommend):

    response, recommend_result = recommender.get_response(state["messages"], state["collected_data"], state["recommend_result"])  
    # collected_data (정보를 저장한 딕셔너리) 도 같이 전달해주는 것이 낫지 않을지...
    # 사용자에게 보여줘야할 값 : response와, 추천 결과: recommend_result를 같이 반환해줘야 할듯 (추천 결과는 다시 추천 받을때 제외하기 위함)
    # collected_data: dict         # 사용자에게서 모은 데이터(정보)를 저장하는 딕셔너리
    # recommend_result: List[str]  # 사용자에게 추천한 결과(해당 추천 결과는 재추천할때에 고려하지 않게 하기 위함)

    print(type(response))

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

# "compile()" 은 rerun마다 재사용되도록 session_state에 저장
if "app" not in st.session_state:
    memory = MemorySaver()
    st.session_state.app = workflow.compile(checkpointer=memory)


# ==========================================
# [4] Streamlit UI 시작
# ==========================================

st.set_page_config(page_title="PLANT AI", page_icon="🌿")

st.title("🌿 PLANT AI")
st.caption("나만의 식물 추천 파트너 (LangGraph Powered)")



app = st.session_state.app

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_1234" # 고유 ID

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 초기 메시지/상태가 없으면 초기화
current_state_snapshot = app.get_state(config)
if not current_state_snapshot.values:
    # 초기 상태 주입
    initial_state = {
        "messages": [AIMessage(content="안녕하세요. AI입니다.")],
        "current_stage": "collect",
        "user_action": "None",
        "collected_data": {
            "purpose": None, "preferred_style": None, "preferred_color": None,
            "plant_type": None, "season": None, "humidity": None,
            "has_dog": None, "has_cat": None, "isAirCond": None,
            "watering_frequency": None, "user_experience": None, "emotion": None
        },
        "recommend_result": []
    }
    # 초기 실행으로 상태 설정
    app.invoke(initial_state, config=config)
    st.rerun()

# 현재 상태 가져오기
state_values = app.get_state(config).values
messages = state_values.get("messages", [])
current_stage = state_values.get("current_stage", "collect")
collected_data = state_values.get("collected_data", {})

# ==========================================
# [사이드바]
# ==========================================
with st.sidebar:
    st.header("📊 진행 상황")
    stage_map = {"collect": "정보 수집", "recommend": "추천", "qna": "상담", "exit": "종료"}
    st.info(f"현재 단계: **{stage_map.get(current_stage, current_stage)}**")

    if current_stage == 'collect' and collected_data:
        total = len(collected_data)
        filled = sum(1 for v in collected_data.values() if v is not None)
        if total > 0:
            pct = int((filled / total) * 100)
            st.progress(pct / 100)
            st.write(f"정보 수집률: {pct}%")

    if st.button("처음부터 다시 시작"):
        # 상태 리셋 로직 (새 thread_id 발급 등)
        st.session_state.thread_id = f"user_{int(st.session_state.thread_id.split('_')[1]) + 1}"
        st.rerun()

# ==========================================
# [메인] 채팅창
# ==========================================

# 메시지 파싱 함수
def parse_ai_content(content):
    if isinstance(content, str) and content.startswith('{'):
        try:
            data = json.loads(content)
            if "assistant_message" in data: return data["assistant_message"]
            if "response" in data: return data["response"]
        except: pass
    return content

# 히스토리 출력
for msg in messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.content:
            text = parse_ai_content(msg.content)
            with st.chat_message("assistant", avatar="🌿"):
                st.write(text)

# ==========================================
# [입력] 처리
# ==========================================
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

    # LangGraph 입력 페이로드
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
                st.write(parse_ai_content(last_msg.content))
            
            # 상태 갱신을 위해 리런 (필수는 아니지만 UI 동기화 확실함)
            # st.rerun()