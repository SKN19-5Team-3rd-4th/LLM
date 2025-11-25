from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from modules.config import QNA_INDEX_NAME, pc, embeddings


class ModelQna:
    def __init__(self, tools):        
        self.tools = tools      

    def get_response(self, messages):
        
        prompt = """
        너는 식물에 대해 차분하게 상담해 주는 전문가이다.
        아래 형식을 반드시 지키되, 실제 상담사가 말하듯 자연스럽고 단정적인 말투로 작성한다.
        모든 질문에 대한 답변은 tool을 사용하여 결과를 참고한다.

        ### 답변 방식 ###
        - 이후 이어지는 RAG 정보와 기존 지식을 활용하여 비슷한 사례의 해결 방향을 최대한 길고 다양하게 설명한다.
        - **RAG 정보의 COMMENT 내용**을 반드시 참고하여 출력한다.
        - 모든 답변은 예시를 참고하여 사례별 해결 방안을 설명한다.
        - 모든 문장은 따뜻하지만 과하지 않게, 실제 상담사가 말하듯 단정적으로 말한다.

        ### 예시 ###

        잎에서 물방울이 계속 맺혀 떨어지는 현상은 보통 다음 두 가지 중 하나예요:

        1. ‘수분 배출(거티테이션, Guttation)’ 현상

        식물이 뿌리를 통해 흡수한 물을 잎 끝의 ‘수공(물구멍)’으로 밀어내면서 물방울이 맺히는 자연스러운 현상이에요.

        특히 이런 경우에 잘 나타납니다:

        분갈이 후 화분 흙이 물을 오래 머금고 있을 때

        꽃집에서 과습 상태였던 식물을 집으로 바로 가져왔을 때

        밤에 물을 많이 줬을 때

        습도가 높을 때 or 통풍이 부족할 때

        물방울이 끈적하지 않고 그냥 맑은 물이라면 대부분 이 현상이에요.
        며칠 동안 환경이 안정되면 자연스럽게 사라집니다.

        2. 혹시 끈적하면 ‘감로(해충 분비물)’ 가능성

        만약 물방울이 끈적이거나, 설탕물처럼 달라붙는 느낌이면 진딧물·깍지벌레 같은 해충이 낸 감로일 수 있어요.

        확인 방법:

        잎 뒷면을 자세히 보면 작은 벌레가 붙어 있는지

        잎이 끈적이거나 먼지가 잘 붙는지

        잎 일부가 노랗게 변하는지

        끈적하면 사진 보여주면 정확히 진단해줄게요.

        지금 상황에서 해주면 좋은 대처

        물주기 멈춤
        방울이 많이 생긴다는 건 흙 속 수분이 충분하다는 뜻이에요.

        통풍 확보
        창가 근처에서 바람 잘 통하게 두면 훨씬 빨리 안정돼요.

        흙 수분 확인
        손가락으로 3~4cm 정도 찔러봤을 때 촉촉하면 물주지 마세요.

        잎 물방울은 가볍게 닦아주기
        오래 달려 있으면 잎에 얼룩 생길 수 있어요.
        
        ### 출력 형식 ###
        [사용자의 상황을 판단해서 가장 핵심적인 조언을 한 문장으로 제시]
        [현재 상황에 맞는 다음 추가 질문 유도]
        """
        
        system_msg = SystemMessage(prompt)
        input_msg = [system_msg] + messages

        model = ChatOpenAI(
            model="gpt-4o", 
            temperature=1,
        ).bind_tools(self.tools)

        response = model.invoke(input_msg)
        
        return response
    
@tool
def tool_rag_qna(query: str) -> str:
    """식물 상담 QnA 전용 RAG 도구"""
    index = pc.Index(QNA_INDEX_NAME)
    
    vector_store = PineconeVectorStore(
        index=index, 
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})   
    retrievals = retriever.batch([query]) 
    
    return str(retrievals[0])