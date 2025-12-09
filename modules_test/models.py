from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

# Endpoint: POST /api/recommend/
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="사용자 식별을 위한 고유 세션 ID (필수)")
    user_message: str = Field(..., description="사용자가 입력한 메시지 (필수)")
    
    # user_action 값 정의:
    # "None": 일반 대화 (기본값)
    # "Skip": 다음 단계로
    # "Continue": collect -> recommend, recommend -> exit
    # "Retry": 재추천
    # "Restart": 재시작
    # "QnA": QnA 모드로 전환
    # "Exit": 대화 종료
    user_action: Literal["None", "Skip", "Continue", "Retry", "Restart", "QnA", "Exit"] = Field(
        "None", description="사용자의 의도된 행동 (선택, 기본값: None)"
    )

# --- 2. 서버 응답 데이터 형식 (Response) ---
class ChatResponse(BaseModel):
    ai_message: str = Field(..., description="AI의 응답 텍스트")
    
    current_stage: Literal["collect", "recommend", "qna", "exit"] = Field(
        ..., description="현재 대화 단계"
    )
    
    # 추천 결과 (문자열 리스트 또는 None)
    recommend_result: Optional[List[str]] = Field(
        None, description="현재 단계에서 추천된 식물 목록"
    )
    
    # 수집된 사용자 데이터 (딕셔너리 또는 None)
    collected_data: Optional[Dict[str, Any]] = Field(
        None, description="현재까지 수집된 사용자 정보 딕셔너리"
    )