from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv 
import os


#-11/20논의후수정
#-------------------------------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "plant-qna"
index = pc.Index(index_name)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def tool_rag_qna(query: str) -> str:
    """식물 상담 QnA 전용 RAG 도구"""
    
    #----- 전역에서 한번만 바꿔야 할듯한
    vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    ##-----

    docs = retriever.invoke(query) # retriever(query)
    return "\n\n".join([d.page_content for d in docs])
#-------------------------------------------------------


class ModelQna:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.prompt_template = """
        너는 식물에 대해 차분하게 상담해 주는 전문가이다.
        아래 형식을 반드시 지키되, 실제 상담사가 말하듯 자연스럽고 단정적인 말투로 작성한다.
        RAG 검색 정보를 참고하여 상담을 진행한다.
        
        ### RAG 검색 정보 ###
        {context}

        ### 답변 방식 ###
        - 첫 문장은 사용자의 고민에 대한 핵심 답변을 한 줄로 요약한다. (채팅 응답처럼)
        - 이후 이어지는 RAG 정보는 비슷한 사례의 해결 방향을 '요약 3줄'로 정리한다.
        - 모든 문장은 따뜻하지만 과하지 않게, 실제 상담사가 말하듯 단정적으로 말한다.
        - 마지막 문장은 대화를 이어가기 위해 질문형으로 마무리한다.
        
        ### 출력 형식 ###
        [사용자의 상황을 판단해서 가장 핵심적인 조언을 한 문장으로 제시]
        [현재 상황에 맞는 다음 추가 질문 유도]

        사용자 질문: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)


    def extract_question(self, messages):
        human_msgs = [m.content for m in messages if isinstance(m, HumanMessage)]
        return human_msgs[-1] if human_msgs else ""


    def build_chain(self):
        return (
            {
                "context": RunnableLambda(lambda q: next(t.run(q) for t in self.tools if t.name == "tool_rag_qna")),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    

    def get_response(self, messages):
        q = self.extract_question(messages)
        chain = self.build_chain()
        response = chain.invoke(q)
        return response