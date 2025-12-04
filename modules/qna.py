from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from modules.config import QNA_INDEX_NAME, pc, embeddings
import json


#---------------------------------------------------------------------
# RAG 파이프라인 설정 상수
#---------------------------------------------------------------------
K_RAG = 5                       # 기본 k 값
SCORE_THRESHOLD = 0.7           # 동문서답 방지 기준 스코어
CATEGORY_CONF_THRESHOLD = 0.6   # 카테고리 예측 신뢰도 기준
USE_CATEGORY_FILTER = True      # 카테고리 필터 T/F 플래그


#---------------------------------------------------------------------
# 카테고리 예측 프롬프트
#---------------------------------------------------------------------
CATEGORY_PREDICTION_PROMPT = """
너는 식물 상담 질문을 카테고리로 분류하는 전문가이다.

사용자의 질문을 다음 카테고리 중 하나로 분류해야 한다:

**[주요 카테고리 (category_main)]:**
- symptom_diagnosis: 증상 진단 (잎 변색, 시들음, 병충해 등)
- watering: 물주기 관련 질문
- light: 빛/햇빛 관련 질문
- fertilizer: 비료/영양제 관련
- transplant: 분갈이/화분 교체
- id_request: 식물 종류 식별 요청
- urgent_care: 긴급 처치가 필요한 상황
- general_care: 일반적인 관리 방법
- environment: 온도/습도/통풍 등 환경 조건
- propagation: 번식/삽목/씨앗
- unknown: 식물 관련이 아니거나 판단하기 어려운 경우

**[부가 카테고리 (category_sub)]:** 
위 카테고리들 중 추가로 연관된 것이 있으면 배열로 포함 (없으면 빈 배열)

**[출력 형식 (JSON만 반환)]:**
{
  "category_main": "<카테고리명>",
  "category_sub": ["<카테고리명1>", "<카테고리명2>"],
  "confidence": <0.0~1.0>
}

**[판단 기준]:**
- 명확하게 판단 가능: confidence >= 0.7
- 대략적으로 추정 가능: confidence 0.4~0.7
- 판단하기 어려움: confidence < 0.4, category_main = "unknown"

사용자 질문을 분석하고 JSON만 출력하라.
"""

#---------------------------------------------------------------------
# LLM을 사용하여 질문의 카테고리를 예측
# 신뢰도가 낮거나 파싱 실패 시 안전한 기본값 반환
#---------------------------------------------------------------------
def gen_category(query):

    try:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        messages = [
            SystemMessage(content=CATEGORY_PREDICTION_PROMPT),
            HumanMessage(content=query)
        ]
        
        response = model.invoke(messages)
        content = response.content.strip()
        
        # JSON 파싱 시도
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        category_data = json.loads(content)
        
        # 필수 필드 검증
        if "category_main" not in category_data:
            raise ValueError("category_main 필드가 없음")
            
        return {
            "category_main": category_data.get("category_main", "unknown"),
            "category_sub": category_data.get("category_sub", []),
            "confidence": float(category_data.get("confidence", 0.0))
        }
        
    except Exception as e:
        # 예외 발생 시 안전한 기본값 반환
        return {
            "category_main": "unknown",
            "category_sub": [],
            "confidence": 0.0
        }


#---------------------------------------------------------------------
# 카테고리 기반 Pinecone 필터 구성
# 카테고리 정보를 기반으로 Pinecone filter dict를 생성
# 신뢰도가 낮거나 unknown이면 None을 반환하여 전체 검색을 사용
#---------------------------------------------------------------------
def search_filter_dict(category):

    if not USE_CATEGORY_FILTER:
        return None
        
    cat_main = category.get("category_main", "unknown")
    cat_sub_list = category.get("category_sub", [])
    confidence = float(category.get("confidence", 0.0))
    
    # 신뢰도가 낮거나 unknown이면 필터 사용하지 않음
    if confidence < CATEGORY_CONF_THRESHOLD or cat_main == "unknown":
        return None
    
    # category_sub에 category_main도 포함시켜 OR 검색
    all_categories = [cat_main] + cat_sub_list
    
    # Pinecone 필터: category_main이 일치하거나 category_sub에 포함되어 있는 문서 검색
    filter_dict = {
        "$or": [
            {"category_main": cat_main},
            {"category_sub": {"$in": all_categories}},
        ]
    }
    
    return filter_dict


#---------------------------------------------------------------------
# 필터 검색 + fallback + no-answer 판단
# max_score < threshold이면 전체 검색으로 fallback
#---------------------------------------------------------------------
def search_filter(vector_store, query, filter_dict, k, score_threshold):

    docs_scores = []
    used_filter = False
    
    # 1차 검색: 필터가 있으면 필터 검색, 없으면 전체 검색
    if filter_dict is not None:
        docs_scores = vector_store.similarity_search_with_score(
            query, 
            k=k, 
            filter=filter_dict
        )
        used_filter = True
    else:
        docs_scores = vector_store.similarity_search_with_score(
            query, 
            k=k
        )
        used_filter = False
    
    # max_score 계산
    max_score = max([score for _, score in docs_scores], default=0.0)
    
    # 필터 검색을 사용했고 max_score가 threshold보다 낮으면 fallback
    if used_filter and max_score < score_threshold:
        docs_scores = vector_store.similarity_search_with_score(
            query, 
            k=k
        )
        used_filter = False
    
    return docs_scores, used_filter


#---------------------------------------------------------------------
# ModelQna 식물상담 LLM
#---------------------------------------------------------------------
class ModelQna:
    def __init__(self, tools):        
        self.tools = tools      

    def get_response(self, messages):
        
        prompt = """
        너는 식물에 대해 차분하게 상담해 주는 전문가이다.
        아래 형식을 반드시 지키되, 실제 상담사가 말하듯 자연스럽고 단정적인 말투로 작성한다.
        모든 질문에 대한 답변은 tool을 사용하여 결과를 참고한다.

        ### RAG 결과 해석 방법 ###
        - tool_rag_qna가 반환하는 JSON에서 "no_answer": true이면 관련 정보를 찾기 어려운 것이다.
        - 이 경우 "정확한 답변을 드리기 어렵습니다. 더 구체적인 정보를 알려주시겠어요?"와 같이 안전하게 응답한다.
        - no_answer가 false이면 "documents" 배열의 content, metadata, score를 활용하여 RAG 기반 답변을 생성한다.

        ### 답변 방식 ###
        - 이후 이어지는 RAG 정보와 기존 지식을 활용하여 비슷한 사례의 해결 방향을 최대한 길고 다양하게 설명한다.
        - **RAG 정보의 COMMENT 내용**을 반드시 참고하여 출력한다.
        - 모든 답변은 예시를 참고하여 사례별 해결 방안을 설명한다.
        - 모든 문장은 따뜻하지만 과하지 않게, 실제 상담사가 말하듯 단정적으로 말한다.

        ### 예시 ###

        잎에서 물방울이 계속 맺혀 떨어지는 현상은 보통 다음 두 가지 중 하나예요:

        1. '수분 배출(거티테이션, Guttation)' 현상

        식물이 뿌리를 통해 흡수한 물을 잎 끝의 '수공(물구멍)'으로 밀어내면서 물방울이 맺히는 자연스러운 현상이에요.

        특히 이런 경우에 잘 나타납니다:

        분갈이 후 화분 흙이 물을 오래 머금고 있을 때

        꽃집에서 과습 상태였던 식물을 집으로 바로 가져왔을 때

        밤에 물을 많이 줬을 때

        습도가 높을 때 or 통풍이 부족할 때

        물방울이 끈적하지 않고 그냥 맑은 물이라면 대부분 이 현상이에요.
        며칠 동안 환경이 안정되면 자연스럽게 사라집니다.

        2. 혹시 끈적하면 '감로(해충 분비물)' 가능성

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


#---------------------------------------------------------------------
# tool_rag_qna
#---------------------------------------------------------------------
@tool
def tool_rag_qna(query: str) -> str:
    """
    식물 상담 QnA 전용 RAG 도구
    
    카테고리 예측 → 필터 검색 → fallback → no-answer 판단을 수행하고
    LLM이 활용하기 쉬운 JSON 페이로드를 반환한다.
    """
    # Step 1: 카테고리 예측
    category = gen_category(query)
    
    # Step 2: 필터 구성
    filter_dict = search_filter_dict(category)
    
    # Step 3: Pinecone 인덱스 및 벡터스토어 초기화
    index = pc.Index(QNA_INDEX_NAME)
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )
    
    # Step 4: 검색 수행 (필터 + fallback)
    docs_scores, used_filter = search_filter(
        vector_store=vector_store,
        query=query,
        filter_dict=filter_dict,
        k=K_RAG,
        score_threshold=SCORE_THRESHOLD,
    )
    
    # Step 5: max_score 계산 및 no_answer 판단
    max_score = max([score for _, score in docs_scores], default=0.0)
    no_answer = (max_score < SCORE_THRESHOLD)
    
    # Step 6: JSON 페이로드 구성
    docs_payload = []
    for doc, score in docs_scores:
        docs_payload.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        })
    
    result = {
        "query": query,
        "category": category,
        "used_filter": used_filter,
        "k": K_RAG,
        "score_threshold": SCORE_THRESHOLD,
        "max_score": float(max_score),
        "no_answer": no_answer,
        "documents": docs_payload,
    }
    
    # Step 7: JSON 문자열 반환
    return json.dumps(result, ensure_ascii=False)