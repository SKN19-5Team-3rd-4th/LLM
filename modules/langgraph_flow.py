from typing import TypedDict, Annotated, Optional, Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from functools import partial

### 각각 모델에 대한 클래스 작성 (각각 작업하여 모듈 또는 패키지로 import)
class ModelCollect():
    def __init__(self):
        # 모델 선언
        # 프롬프트 선언
        # DB 연결
        # 기타 등등
        self.pictures = None


    def connect_db(self):
        # RAG 사용 시 DB 연결하는 함수
        pass

    def get_memory(self):
        pass

    def set_memory(self, momory):
        pass

    # 기타 필요 함수 작성
    @staticmethod
    def is_data_enough(data: str) -> bool :
        # 데이터의 양이 충분한지 확인
        return True
    
    def is_pictures(self):
        return self.pictures

    def get_response(self, messages):
        print("요약 모델")
        return "집들이 선물용 식물 추천"

class ModelRecommend:
    def get_response(self, collected_data):
        print("추천 모델")
        return "몬스테라, 스투키"


class ModelQna:
    def get_response(self, messages):
        print("QA 모델")
        return "물은 2주에 한 번 주시면 됩니다!"



### 기본 상태 state
class GraphState(TypedDict):

    messages: Annotated[list, add_messages]

    ### 현재 작업 단계
    # 'collect' : 정보 수집 로직
    # 'recommend' : 추천 로직
    # 'qna' : QnA 로직
    # 'done' : 작업 마무리
    current_stage: Literal["collect", "recommend", "qna", "done"]

    # 'collect'에서 수집된 정보
    collected_data: Optional[dict]

    # 'recommend'에서 추천한 결과
    recommend_result: Optional[str]

    # 사용자가 임의의 버튼을 클릭했을때 (처음부터, 다음 단계로 이동, QnA로 이동,...). 버튼 동작에 대해 요소 삽입
    user_action: Optional[Literal["None", "Next", "Retry" "Restart", "QnA"]]

def node_collect(state: GraphState, collector: ModelCollect):
    # response = collector.get_response()
    # picture = collector.pictures()
    # if picture is not None:
    #     state["picture_input"] = "True"
    # else:
    #     state["picture_input"] = "False"

    response = collector.get_response(state["messages"])
    state["collected_data"] = response
    state["messages"].append({"role":"assistant", "content": response})
    state["current_stage"] = "recommend"
    return state



def node_recommend(state: GraphState, recommender: ModelRecommend):
    response = recommender.get_response(state["collected_data"])
    state["recommend_result"] = response
    state["messages"].append({"role":"assistant", "content": response})
    state["current_stage"] = "qna"
    return state


def node_qna(state: GraphState, answer: ModelQna):
    response = answer.get_response(state["messages"])
    state["messages"].append({"role":"assistant", "content": response})
    state["current_stage"] = "done"
    return state

# def node_image(state: GraphState, description: ModelImage):
#     response = description.get_response()

# 조건에 따라 어떤 노드로 향할지 컨트롤
def main_router(state: GraphState):
    return state["current_stage"]
# def main_router(state: GraphState):
#     stage = state["current_stage"]
#     last_message = state["messages"][-1]

#     if state["user_sction"] == "Restart":
#         # 초기화 함수 작성
#         return "collect"

#     if stage == "collect":
#         if ModelCollect.is_data_enough(state["messages"]):
#             state["current_stage"] = "recommend"
#             return "recommend"
#         else:
#             return "collect"

#     elif stage == "recommend":
#         if state["user_action"] == "Next":
#             state["current_stage"] = "done"
#             return "done"
#         else:
#             state["current_stage"] == "Retry"
#             return "Retry"

#     elif stage == "qna":
#         return "qna"

#     elif stage == "done":
#         return "done"

# def is_picture_input(state: GraphState):
#     if state["picture_input"] == "True":
#         return "image"
#     else:
#         return "done"


model_collect = ModelCollect()
model_recommend = ModelRecommend()
model_qna = ModelQna()

workflow = StateGraph(GraphState)

workflow.add_node("collect", partial(node_collect, collector=model_collect))
workflow.add_node("recommend", partial(node_recommend, recommender=model_recommend))
workflow.add_node("qna", partial(node_qna, answer=model_qna))
# workflow.add_node("image", node_image)

workflow.set_entry_point("collect")

workflow.add_conditional_edges(
    "collect",
    main_router,
    {"collect": "collect", "recommend": "recommend"}
)

workflow.add_conditional_edges(
    "recommend",
    main_router,
    {"recommend": "recommend", "qna": "qna"}
)

workflow.add_edge("qna", END)

app = workflow.compile()

# workflow.add_node("router", main_router)   # 노드처럼 등록
# workflow.set_entry_point("router")         # entry point 사용 가능

# workflow.add_conditional_edges(
#     "main_router",
#     main_router,
#     {
#         "collect": "collect",
#         "recommend": "recommend",
#         "qna":"qna",
#         "done": END
#     }
# )

# workflow.add_conditional_edges(
#     "collect",
#     is_picture_input,
#     {"image": "image", "done": END}
# )

# workflow.add_edge("collect", END)
# workflow.add_edge("recommend", END)
# workflow.add_edge("qna", END)


# 임시포맷
def format_output(final_state):
    messages = final_state["messages"]

    collect_result = final_state.get("collected_data", None)
    recommend_result = final_state.get("recommend_result", None)
    current_stage = final_state.get("current_stage", None)

    if recommend_result:
        recommend_result = recommend_result.split(", ")

    qna_result = None
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage":
            qna_result = msg.content
            break

    return {
        "input": messages[0].content,         # 사용자 입력
        "collect_result": collect_result,     # 정보 수집 결과
        "recommend_result": recommend_result, # 추천 결과 (list 형태)
        "qna_result": qna_result,             # qna 응답
        "status": current_stage               # 최종 stage
    }


if __name__ == "__main__":
    state = {
        "messages": [{"role":"user","content":"식물 추천해줘!"}],
        "current_stage": "collect",
        "collected_data": None,
        "recommend_result": None,
        "user_action": "None",
        "picture_input": "False",
    }
    final_state = app.invoke(state) 
    result = format_output(final_state)

    import json
    print("\n=== 최종 상태 ===")
    print(json.dumps(result, ensure_ascii=False, indent=4))