from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from modules.config import * 
from pathlib import Path
import sys


MODULE_DIR = Path(__file__).resolve().parent.parent / 'modules'
sys.path.append(str(MODULE_DIR))


class ModelQna:
    def __init__(self, tools):        
        self.tools = tools      

    def get_response(self, messages):
        
        prompt = """
        너는 식물에 대해 차분하게 상담해 주는 전문가이다.
        아래 형식을 반드시 지키되, 실제 상담사가 말하듯 자연스럽고 단정적인 말투로 작성한다.
        모든 질문에 대한 답변은 tool을 사용하여 결과를 참고한다.

        ### 답변 방식 ###
        - 첫 문장은 사용자의 고민에 대한 핵심 답변을 한 줄로 요약한다. (채팅 응답처럼)
        - 이후 이어지는 RAG 정보는 비슷한 사례의 해결 방향을 '요약 3줄'로 정리한다.
        - 모든 문장은 따뜻하지만 과하지 않게, 실제 상담사가 말하듯 단정적으로 말한다.
        - 마지막 문장은 대화를 이어가기 위해 질문형으로 마무리한다.
        
        ### 출력 형식 ###
        [사용자의 상황을 판단해서 가장 핵심적인 조언을 한 문장으로 제시]
        [현재 상황에 맞는 다음 추가 질문 유도]
        """
        
        system_msg = SystemMessage(prompt)
        input_msg = [system_msg] + messages

        model = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.3,
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