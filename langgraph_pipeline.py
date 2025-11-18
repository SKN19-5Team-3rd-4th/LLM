from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Optional, Literal, List

from functools import partial
import operator

from models.collect.collect import ModelCollect
from models.recommend.recommend import ModelRecommend
from models.qna.qna import ModelQna

# 정보 저장 state 선언 --------------------

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]                         # 모든 메시지를 저장하는 리스트

    current_stage: Literal["collect", "recommend", "qna", "exit"]   # 현재 어떤 작업을 하고 있는지 저장

    collected_data: Optional[dict]                                  # 사용자에게서 모은 데이터(정보)를 저장하는 딕셔너리

    recommend_result: Annotated[Optional[List[str]], operator.add]  # 사용자에게 추천한 결과(해당 추천 결과는 재추천할때에 고려하지 않게 하기 위함)

    # None: 아무 행동도 하지 않음, Skip: 다음 단계로, Continue: 추천 만족, Retry: 추천 다시 받기, Restart: 처음부터 재시작, QnA: QnA로 이동
    user_action: Literal["None", "Skip", "Continue", "Retry", "Restart", "QnA",]

    picture_exist: Optional[str]

initial_state = {
    "messages": []

}

### 노드 선언 -----------------------------

def node_collect(state: GraphState, collector: ModelCollect):
    response, collected_data = collector.get_response(state["messages"], state["collected_data"])   # 어떤 정보를 전달했는지 알아야 하니까 collected_data도 같이 전달
    picture = collector.get_picture()   # 아마 get_picture에서 picture를 저장하고 해당 경로를 전달해야 할듯 (str)
    if picture is not None:
        return {
            "current_stage" : "collect",
            "messages": [AIMessage(content=response)],  # 사용자에게 보여줄 질문
            "picture_exist": picture,                   # 사진이 있다면 사진의 저장 경로
            "collected_data": collected_data,            # 업데이트된 새로운 정보 딕셔너리
            "user_action": "None",
        }
    else:
        return {
            "current_stage" : "collect",
            "messages": [AIMessage(content=response)],
            "picture_exist": "None",
            "collected_data": collected_data,
            "user_action": "None",
        }

def node_recommend(state: GraphState, recommender: ModelRecommend):
    response, recommend_result = recommender.get_response(state["messages"], state["collected_data"])  
    # collected_data (정보를 저장한 딕셔너리) 도 같이 전달해주는 것이 낫지 않을지...
    # 사용자에게 보여줘야할 값 : response와, 추천 결과: recommend_result를 같이 반환해줘야 할듯 (추천 결과는 다시 추천 받을때 제외하기 위함)

    return {
        "current_stage" : "recommend",
        "messages": [AIMessage(content=response)],
        "recommend_result": [recommend_result],
        "user_action": "None",
    }

def node_qna(state: GraphState, chatbot: ModelQna):
    response = chatbot.get_response(state["messages"])

    return {
        "current_stage": "qna",
        "messages": [AIMessage(content=response)],
        "user_action": "None",
    }

def node_end_state(state:GraphState):
    return {
        "current_stage": "exit"
    }

def node_image_process(state:GraphState):
    # 이미지 처리 구현
    collected_data = None
    return {
        "collected_data": collected_data,
        "user_action": "None",
        "current_stage": "collect",
        "picture_exist": None
    }

### router 선언 -----------------------

# 해당 router의 결과에 따라, 어떤 노드로 향할지 컨트롤
def main_router(state: GraphState):
    stage = state["current_stage"]
    last_message = state["messages"][-1]
    action = state["user_action"]

    if state["user_action"] == "Restart":
        return "Restart"
    
    if stage == "collect":
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
    
def is_picture_input(state: GraphState):
    if state["picture_exist"] is not None:
        return "image"
    else:
        return "done"


### workflow 구현----------------------

model_collect = ModelCollect()
model_recommend = ModelRecommend()
model_qna = ModelQna()

workflow = StateGraph(GraphState)

workflow.add_node("collect", partial(node_collect, collector=model_collect))
workflow.add_node("recommend", partial(node_collect, collector=model_recommend))
workflow.add_node("qna", partial(node_collect, collector=model_qna))
workflow.add_node("exit", node_end_state)
workflow.add_node("image", node_image_process)

workflow.set_entry_point("__start__")

workflow.add_conditional_edges(
    "__start__",
    main_router,
    {
        "collect": "collect",
        "recommend": "recommend",
        "qna": "qna",
        "done": END,
        "exit": "exit"
    }
)

workflow.add_conditional_edges(
    "collect",
    is_picture_input,
    {
        "image": "image",
        "done": END,
    }
)



### -----------------------------------



