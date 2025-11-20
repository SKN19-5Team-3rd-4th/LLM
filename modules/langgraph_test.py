from typing import TypedDict, Annotated, Optional, Literal, List
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

    # 기타 필요 함수 작성

    def get_response(self, messages: List):
        """ 
        messages는 Humanmessage와 AIMessage로 구성된 리스트 형태
        처음부터 모든 대화의 내용이 차례대로 들어가 있음
        message[index].content를 통해서 필요한 내용의 텍스트만 가져올 수 있음
        """
        
        response = None
        ### 각 모델에 맞는 작업 수행
        return response

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
    user_action = Optional[Literal["None", "Next", "Retry" "Restart", "QnA"]]

def node_collect(state: GraphState, collector: ModelCollect):
    response = collector.get_response()
    picture = collector.pictures()
    if picture is not None:
        state["picture_input"] = "True"
    else:
        state["picture_input"] = "False"





def node_recommend(state: GraphState, recommender: ModelRecommend):
    response = recommender.get_response()

def node_qna(state: GraphState, answer: ModelQna):
    response = answer.get_response()

def node_image(state: GraphState, description: ModelImage):
    response = description.get_response()

# 조건에 따라 어떤 노드로 향할지 컨트롤
def main_router(state: GraphState):
    stage = state["current_stage"]
    last_message = state["messages"][-1]

    if state["user_sction"] == "Restart":
        # 초기화 함수 작성
        return "collect"

    if stage == "collect":
        if ModelCollect.is_data_enough(state["messages"]):
            state["current_stage"] = "recommend"
            return "recommend"
        else:
            return "collect"

    elif stage == "recommend":
        if state["user_action"] == "Next":
            state["current_stage"] = "done"
            return "done"
        else:
            state["current_stage"] == "Retry"
            return "Retry"

    elif stage == "qna":
        return "qna"

    elif stage == "done":
        return "done"

def is_picture_input(state: GraphState):
    if state["picture_input"] == "True":
        return "image"
    else:
        return "done"



workflow = StateGraph(GraphState)

workflow.add_node("collect", partial(node_collect, collector=model_collect))
workflow.add_node("recommend", node_recommend)
workflow.add_node("qna", node_qna)
workflow.add_node("image", node_image)

workflow.set_entry_point("main_router")

workflow.add_conditional_edges(
    "main_router",
    main_router,
    {
        "collect": "collect",
        "recommend": "recommend",
        "qna":"qna",
        "done": END
    }
)

workflow.add_conditional_edges(
    "collect",
    is_picture_input,
    {"image": "image", "done": END}
)

workflow.add_edge("collect", END)
workflow.add_edge("recommend", END)
workflow.add_edge("qna", END)



### state 초기값 설정
state = {
    "current_stage" : "collect",
    "collected_data" : "",
    "recommend_result" : "",
    "user_action" : "None",
    "picture_input" : "False",
}

### 실제 실행되는 코드 (UI로 구현 시 반복 구동되어야할 코드)
while state["current_stage"] != "done":

    ai_response = state["messages"][-1]









    

# (7) 그래프 컴파일 (Checkpointer 연결)
app = workflow.compile(checkpointer=memory)

# (8) 대화 세션 ID 설정
config = {"configurable": {"thread_id": "cli-session-1"}}

# ==============================================================================
# 2. 반복문 내부: 사용자 상호작용 및 그래프 실행 (사용자 입력 시마다 실행)
# ==============================================================================

print("\n--- 챗봇 시뮬레이션 시작 (종료: '종료') ---")

# (가상) AI의 첫 인사 (그래프를 한 번 실행하여 첫 메시지 받기)
try:
    initial_state = {"current_stage": "collect", "messages": []}
    response = app.invoke(initial_state, config=config)
    print(f"AI: {response['messages'][-1].content}")
except Exception as e:
    print(f"초기화 오류: {e}")
    sys.exit()

while True:
    # 1. 사용자 입력 수집
    user_input = input("You: ")

    # 2. 종료 조건 확인
    if user_input.lower() == "종료":
        print("--- 시뮬레이션 종료 ---")
        break

    # 3. 입력 처리 (버튼 클릭을 텍스트로 시뮬레이션)
    user_action = None
    if user_input.lower() in ["예", "좋아요", "ㅇㅋ"]:
        user_action = "ACCEPT"
    elif user_input.lower() in ["아니오", "별로", "다시"]:
        user_action = "REJECT"

    # 4. GraphState에 전달할 '델타(delta)' 생성
    input_delta = {
        "messages": [HumanMessage(content=user_input)],
        "user_action": user_action
    }

    try:
        # 5. LangGraph 실행 (한 턴)
        # Checkpointer가 config의 thread_id를 보고 이전 상태를 로드하여
        # input_delta를 병합한 후, 그래프를 실행합니다.
        response_state = app.invoke(input_delta, config=config)
        
        # 6. AI 응답 출력
        ai_message = response_state["messages"][-1]
        print(f"AI: {ai_message.content}")

        # 7. 최종 단계 확인
        if response_state["current_stage"] == "done":
            print("--- 모든 단계가 완료되었습니다 ---")
            break

    except Exception as e:
        print(f"오류 발생: {e}")
        break

    


