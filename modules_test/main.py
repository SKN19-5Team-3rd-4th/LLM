from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Any
from langchain_core.messages import HumanMessage

from base_langgraph import app as graph_app

app = FastAPI()


### 추천 (/api/recommend/) ================================================================================

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="사용자 식별을 위한 고유 세션 ID (UUID 등)")
    user_message: str = Field(..., description="사용자가 보낸 메시지")
    user_action: Literal["None", "Skip", "Continue", "Retry", "Restart", "QnA", "Exit"] = "None"

class ChatResponse(BaseModel):
    ai_message: str
    current_stage: Literal["collect", "recommend", "qna", "exit"] 
    collected_data: Optional[Dict[str, Any]] = None
    recommend_result: Optional[List[str]] = None


@app.post("/api/recommend/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        
        inputs = {
            "messages": [HumanMessage(content=request.user_message)],
            "user_action": request.user_action,
        }

        result_state = await graph_app.ainvoke(inputs, config=config)

        all_message = result_state["messages"]

        return ChatResponse(
            reply=all_message,
            current_stage=result_state.get("current_stage", "exit"),
            collected_data=result_state.get("collected_data"),
            recommend_result=result_state.get("recommend_result")
        )

    except Exception as e:
        # 에러 처리
        print(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))