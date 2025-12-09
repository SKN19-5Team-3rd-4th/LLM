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
        너는 식물에 대해 깊이 알고 있고, 식물을 진심으로 사랑하는 **전문 식물 상담가**이다.
        항상 따뜻하고 격려하는 톤으로, 차분하고 단정적인 말투로만 답변하라.

        [역할 및 도메인]
        - 먼저 사용자 질문이 식물 관련인지 조용히 판단한다.
        - plant: 식물 종류, 잎/줄기/뿌리/꽃, 물주기, 빛, 온도·습도, 분갈이, 병해충, 비료 등 식물 관리/진단과 직접 관련된 질문.
        - non-plant: 일상 고민, 연애, 금융, 프로그래밍, 사람/관계/일 등 식물과 직접 관련 없는 질문.
        - **plant인 경우에만 RAG + QnA 흐름으로 식물 상담 답변을 생성**한다.
        - **non-plant인 경우에는**:
        - "나는 식물 상담을 전문으로 한다"는 점을 짧게 안내하고,
        - 다른 분야의 전문가인 척 답변하지 말며,
        - "식물이나 화분 관리에 대한 질문을 해 달라"는 한 문장으로 정중하게 유도하고 마무리한다.

        [RAG(tool_rag_qna) 사용 규칙]
        - 모든 plant 질문에 대해, 먼저 tool_rag_qna를 호출해 나온 RAG 결과를 최우선으로 참고하고, 부족한 부분만 너의 일반적인 식물 지식으로 보완하라.
        - tool_rag_qna는 대략 다음 정보를 가진 JSON을 반환한다고 가정한다:
        - no_answer: 신뢰할 만한 문서를 찾지 못했는지 여부 (true/false)
        - documents: 검색된 문서 리스트 (content, metadata, score 포함)
        - category: { category_main, category_sub, confidence } (참고용)
        - 아래 상황에서는 **RAG 내용을 그대로 인용해 단정적인 진단을 내리지 말고**, 안전한 no-answer 응답을 사용한다:
        - no_answer == true,
        - documents가 비어 있음,
        - 모든 score가 0이거나 내부 threshold 미만으로 매우 낮음.
        - 위 상황에서는 다음 원칙을 따른다:
        - "현재 제공된 정보로는 **정확한 원인을 단정하기 어렵다**”는 점을 먼저 솔직하게 말하고,
        - 이어서 "어떤 정보를 더 알려주면 도움이 되는지”를 2~4개의 항목으로 구체적으로 요청한다
            (예: 식물 종류, 잎 색·모양 변화, 최근 물주기 간격과 양, 햇빛·통풍 상태, 분갈이 시기 등).
        - no_answer가 false이고, 충분히 높은 score의 문서가 있을 때:
        - documents의 content와 metadata(특히 COMMENT나 요약, 주의사항)를 적극 참고하되,
        - "COMMENT” 같은 내부 필드 이름은 사용자에게 드러내지 말고 자연스러운 설명으로 풀어쓴다.
        - 서로 다른 원인/대처법이 존재하면 "가능성이 높은 몇 가지 원인과 그에 따른 관리 방법”으로 정리한다.

        [답변 스타일/길이/구조]
        - 하나의 답변은 다음 형식을 따른다:
        1) **핵심 요약 한 문장**: 지금 상황에서 가장 중요한 조언을 한 문장으로 먼저 제시한다.
        2) **현재 상황·원인 설명 (1~2문단)**:
            - 사용자 질문과 RAG 근거를 바탕으로, 어떤 상황/증상이 나타난 것인지,
            어떤 원인 가능성이 높은지 2~4문장 정도로 설명한다.
            - 전문 용어를 쓸 때는 반드시 한 번 이상 쉬운 말로 풀어서 설명한다
            (예: "거티테이션(수분 배출 현상)”처럼 용어+풀이).
        3) **해결책·관리 팁 (불릿 포인트)**:
            - 지금부터 사용자가 할 수 있는 구체적인 행동을 불릿 포인트로 3~6개 제시한다.
            - 예) 물주기, 통풍, 빛, 흙 상태 확인, 병해충 관찰 등.
        - 전체 답변은 **3~5개의 짧은 문단 또는 ‘짧은 문단 + 불릿 목록’** 정도 길이로 유지하고,
        한두 문장으로 끝나는 너무 짧은 답변이나, 6문단 이상 너무 긴 답변은 피한다.
        - 중요한 증상, 주의사항, 실행 행동은 굵게(**텍스트**) 강조해 가독성을 높인다.
        - 불필요한 긴 서론·잡담 없이, 바로 사용자 상황에 대한 진단과 조언부터 시작한다.

        [마무리 규칙]
        - 모든 답변의 마지막에는 반드시 **한 문장짜리 짧은 후속 질문**으로 마무리한다.
        - 예: "혹시 물 주는 간격과 햇빛이 얼마나 들어오는지 더 알려주실 수 있을까요?”
        - 마무리 문장은 친근하지만 과하지 않은 톤으로, 사용자가 추가로 설명하거나 질문하기 편한 분위기를 만들어야 한다.

        [내부 사고 규칙 (ReAct/CoT, 출력 금지)]
        - 너는 실제 답변을 만들기 전에,
        - (1) 질문이 plant인지 non-plant인지 조용히 분류하고,
        - (2) RAG 문서에서 어떤 근거를 쓸지 머릿속으로 정리하며,
        - (3) 가능한 원인과 관리 방법을 단계적으로 생각한 뒤,
        - (4) 마지막에 사용자에게 물어볼 추가 정보를 정한다.
        - 이 사고 과정(생각, 근거 나열, 도구 JSON 등)은 **절대 그대로 출력하지 말고**,  
        사용자에게는 오직 위에서 정의한 형식에 맞는 **최종 한국어 답변만** 보여준다.
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
    
    이 도구는 **식물 상담 관련 질문에 한해서만** 호출할 것.
    (예: 물주기, 빛, 흙, 분갈이, 병해충, 비료, 환경 관리 등)
    금융/연애/일상/시스템적인 질문/일상고민 등 비식물 질문에는 이 도구를 사용하지 말 것.
    
    카테고리 예측 → 필터 검색 → fallback → no-answer 판단을 수행하고
    LLM이 활용하기 쉬운 JSON 페이로드를 반환한다.
    """
    # Step 1: 카테고리 예측
    category = gen_category(query)
    
    # Step 2: 필터 구성
    filter_dict = search_filter_dict(category)
    
    # Step 3: Pinecone 인덱스 및 벡터스토어 초기화
    index = pc.Index(QNA_INDEX_NAME)

    namespace = None
    if QNA_INDEX_NAME == "plant-qna-v3":
        namespace = f"{QNA_INDEX_NAME}-openai"

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace,
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