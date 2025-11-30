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
                "room" : None,
                "purpose": None,            
                "preferred_style": None,    
                "preferred_color": None,
                "humidity": None,
                "has_dog": None,
                "has_cat": None,
                "isAirCond": None,
                "watering_frequency": None,
                "user_experience": None,
            },
    "recommend_result": " "
}
### tools 선언 ---------------------------
# tool 함수 선언
""" 
@tool
def tool_func(들어갈 인자들(타입 힌트 포함)) -> str:
    # RAG 수행
    return string 
"""
# tools 에는, 각각 이미지 처리 혹은 RAG를 수행하는 세가지 함수가 들어가야 함
tools = [tool_rag_recommend, tool_rag_qna]

### 노드 선언 -----------------------------

def node_collect(state: GraphState, collector: ModelCollect):
    response, collected_data = collector.get_response(state["messages"], state["collected_data"])  # 어떤 정보를 전달했는지 알아야 하니까 collected_data도 같이 전달
    
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


### workflow 구현----------------------

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

### 그래프 컴파일 ----------------------
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

### 실제 구동 코드 ---------------------
def run_chat_loop(app, memory: MemorySaver, initial_state: dict):
    thread_id_01 = "basic_user"
    config = {"configurable": {"thread_id": thread_id_01}}

    response = app.invoke(initial_state, config=config)

    print("\n"*5)
    print(figlet_format("PLANT", font="roman"))

    while True:
        current_state = response
        message = current_state["messages"][-1]
        collected_data = current_state["collected_data"]

        print("\n", "="*80, "\n")
        if current_state["current_stage"] == "exit":
            print("종료합니다...")
            break
        elif current_state['current_stage'] == 'collect':
            null_count = sum(1 for v in collected_data.values() if v is None)
            process = (len(collected_data.values()) - null_count)*100 / len(collected_data.values())

            print(f"[INFO] [Work Stage] 정보 수집 (collect)")
            print(f"[INFO] [AIMessages] 정보 수집률: {int(process)}%")
            print(collected_data)
        elif current_state['current_stage'] == 'recommend':
            print(f"[INFO] [Work Stage] 추천 (recommend)")
        else:
            print(f"[INFO] [Work Stage] 상담 (QnA)")
        print(f"AI   : {message.content}")
        print("\n", "-"*80, "\n")
        print("[대화창]")
        for message in current_state["messages"][1:]:
            if message.content == "안녕? 자기소개 해줘" or message.content == "추천해줘":
                continue
            else:
                if isinstance(message, AIMessage) and (not hasattr(message, 'tool_calls') or not message.tool_calls):

                    if message.content.startswith('{'):
                        data_dict = json.loads(message.content)
                        if "assistant_message" in data_dict:
                            response = data_dict["assistant_message"]
                            print(f"AI(플랜이) : {response}")
                        elif "response" in data_dict:
                            response = data_dict["response"]
                            print(f"AI(플랜이) : {response}")
                    else:
                        print(f"AI(플랜이) : {message.content}")

                elif isinstance(message, HumanMessage):
                    print(f"User({thread_id_01}) : {message.content}")
        print("\n", "="*80, "\n")


        user_input = input("User : ")
        action = "None"

        if user_input.lower() == "종료":    # 종료 누르면 종료
            action = "Exit"
        elif user_input.lower() == "qna":
            action = "QnA"
            user_input = "안녕? 자기소개 해줘"
        elif user_input.lower() == "next":
            print(f"[INFO] User Action is next")
            action = "Continue"
            user_input = "추천해줘"

        input_delta = {
            "messages": [HumanMessage(content=user_input)],
            "user_action": action,
        }
        
        response = app.invoke(input_delta, config=config)
### -----------------------------------

if __name__ == "__main__":
    run_chat_loop(app, memory, initial_state)

