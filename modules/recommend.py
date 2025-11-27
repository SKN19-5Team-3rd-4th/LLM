# 1. node_recommend()
# 2. (rag 전) get_response(): tool 호출 
# 3. tool recommend_rag(): 벡터DB에서 검색 후 반환
# 4. (rag 후) get_response(): 검색된 데이터를 바탕으로 LLM이 응답 생성
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from modules.config import REC_INDEX_NAME, pc, embeddings
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datetime import datetime

class ModelRecommend:
    def __init__(self, tools):
        self.tools = tools

    def get_response(self, messages, collected_data, prev_results): 
        if all(v is None for v in collected_data.values()):
            day = datetime.now().day
            month = datetime.now().month

            message = "입력하신 정보가 없어 오늘의 꽃으로 추천드릴게요."

            with open("datas/data.json", "r", encoding='utf-8') as f :
                data_list = json.load(f)

            select_data  = None
            for data in data_list :
                if data["fMonth"] == str(month) and data["fDay"] == str(day):
                    select_data = data
                    break

            if select_data is None:
                select_data = data_list[0]

            responses = [AIMessage(content=message)]

            ### ----------------- 프롬프트 작성해야함
            prompt = ""

            system_msg = SystemMessage(prompt)
            input_msg = [system_msg] + messages

            model = ChatOpenAI(
                model='gpt-4o',
                temperature=0.5
            )

            response = model.invoke(input_msg)

            responses.append(response)

            return responses, select_data["flowNm"]
            
        prompt= f"""
            ### 요구사항 ###
            {collected_data}

            ### 이미 추천한 식물 (다시 추천하지 않습니다) ###
            {prev_results}

            ### 설명 ###
            당신은 친절한 식물 전문가입니다. 한국어로 친절하게 답변하세요.
            반드시 JSON만 출력하세요. JSON 앞뒤에 설명, 문장, 코드블록, ``` 표시를 넣지 마세요.
            키우는 식물과 관련된 결과는 반드시 RAG 검색 결과를 참고해서 반드시 조건에 맞는 식물 1가지를 추천하세요.
            꽃다발과 관련된 결과는 꽃 종류만 RAG로 판단하고 스스로 생각하여 추천하세요.
            **제외 또는 부정적인 단어가 섞인 정보와 관련된 식물은 절대 추천하지 않습니다.**
            **제외해야 하는 정보가 포함된 RAG 결과에 대해서는 다시 검색합니다.**
            이미 추천한 식물은 사용자가 거부한 식물이니 추천하지 않습니다.

            ### 응답목록 ###
            - 식물 이름
            - 추천하는 이유 & 식물의 특징

            ### 출력형식 ###
            {{
                "flowNm": "식물 이름",
                "response": "추천하는 이유 & 식물의 특징"
            }}
        """
        
        system_msg = SystemMessage(prompt)
        input_msg = [system_msg] + messages
            
        model = ChatOpenAI(
            model='gpt-4o',
            temperature=1
        ).bind_tools(self.tools)
            
        recommend_result = ""
        
        while True:
            response = model.invoke(input_msg)
            
            if response.content == '':
                return response, recommend_result

            try:
                res_json = json.loads(response.content)
                recommend_result = res_json['flowNm']
                break
            except:
                continue                    
        
        return [response], recommend_result


def rerank(query, documents):
    model_path = "Dongjin-kr/ko-reranker"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    texts = [
        doc.page_content if hasattr(doc, "page_content") 
        else doc.get("text") if isinstance(doc, dict) 
        else doc
        for doc in documents
    ]

    pairs = [[query, text] for text in texts]
    
    inputs = tokenizer(
        pairs, 
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.squeeze(-1)
    
    ranked = sorted(
        list(zip(documents, scores.cpu().tolist())),
        key=lambda x: x[1],
        reverse=True
    )
    
    return ranked



@tool
def tool_rag_recommend(query: str) -> str:
    """식물 추천 전용 RAG 도구"""
    index = pc.Index(REC_INDEX_NAME)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # retrievals: [[doc1, doc2, ...]]
    retrievals = retriever.batch([query])[0]

    ranked = rerank(query, retrievals)

    best_doc, best_score = ranked[0]

    return str({
        "best": best_doc.page_content if hasattr(best_doc, "page_content") else str(best_doc),
        "score": best_score,
        "all_ranked": [
            (d.page_content if hasattr(d, "page_content") else str(d), s) 
            for d, s in ranked
        ]
    })