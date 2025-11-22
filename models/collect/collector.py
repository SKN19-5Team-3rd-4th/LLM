from dotenv import load_dotenv
import os
import json
import sys
from typing import Optional, Dict, Any, List

from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

FIELDS = [
    ("purpose", "선물 목적이 어떻게 되시나요?"),
    ("relation", "선물할 사람과의 관계는 무엇인가요?"),
    ("gender", "선물받는 분의 성별은 무엇인가요?"),
    ("preferred_color", "원하시는 색상을 입력해주세요."),
    ("personality", "식물을 키울 사람의 성향(성격)을 간단히 알려주세요."),
    ("has_pet", "강아지나 고양이를 키우나요?"),
    ("user_experience", "선물받는 분의 원예 경험이 있나요?"),
    ("preferred_style", "원하는 스타일이 있나요?"),
    ("plant_type", "원하는 식물 유형이 있나요?"),
    ("season", "선물할 시기(계절)가 정해져 있나요?"),
    ("humidity", "특별히 신경쓸 실내 습도가 있나요?"),
    ("isAirCond", "에어컨 사용이 잦은 환경인가요?"),
    ("watering_frequency", "물을 얼마나 자주 줄 수 있을 것 같나요?"),
    ("emotion", "선물에 담고 싶은 감정이나 메시지가 있나요?"),
]

# 맨 처음 질문
INITIAL_OPEN_QUESTION = (
    "선물하실 상황을 간단히 알려주세요."
    "누구에게, 어떤 목적, 언제 주실 예정인지 한두 문장으로 설명해 주세요."
)

# 모른다고 했을 떄
def is_unknown(answer: Optional[str]) -> bool:
    if answer is None:
        return True
    a = answer.strip().lower()
    if a == "":
        return True
    unknown_keywords = ["모름", "모르겠", "모르겠어요", "몰라", "모르겠음"]
    for kw in unknown_keywords:
        if kw in a:
            return True
    return False

def parse_pets(answer: Optional[str]) -> Dict[str, Optional[bool]]:
    if answer is None:
        return {"has_dog": None, "has_cat": None}
    a = answer.strip().lower()
    if is_unknown(a):
        return {"has_dog": None, "has_cat": None}
    has_dog = None
    has_cat = None
    if any(k in a for k in ["강아지", "개"]):
        has_dog = True
    if any(k in a for k in ["고양이", "고양"]):
        has_cat = True
    if any(k in a for k in ["없음", "없어", "안키움", "아니오", "없습니다"]):
        has_dog = False if has_dog is None else has_dog
        has_cat = False if has_cat is None else has_cat
    return {"has_dog": has_dog, "has_cat": has_cat}

def extract_json_from_conversation(conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    system_prompt = (
        "당신은 친절한 식물 전문가입니다. 한국어로 친절하게 답변하세요.\n"
        "아래 형식을 반드시 지키되, 실제 상담사가 말하듯 자연스럽고 단정적인 말투로 작성하세요."
        "1) 역할과 말투\n"
    "- 사용자를 존중하는 단정하고 친절한 말투로 답변하세요. 길이는 간결하게(권장 1~4문장).\n"
    "2) 'end' 처리(엄격)\n"
    "- 사용자가 'end'를 입력하면, 지금까지의 대화 전체를 분석하여 **오직** 아래 JSON 스키마 하나만 출력하세요.\n"
    "- JSON 외의 어떤 텍스트(설명, 주석, 코드블럭 등)도 출력하면 안 됩니다.\n"
    "- 값이 없거나 사용자가 '모름'/'pass'로 넘겼으면 해당 필드는 JSON의 null로 표기하세요.\n"
    "3) JSON 스키마(정확히 이 키들을 포함)\n"
    "- purpose, relation, gender, preferred_color, personality,\n"
    "  has_dog, has_cat, user_experience, preferred_style,\n"
    "  plant_type, season, humidity, isAirCond, watering_frequency, emotion\n"
    "4) 반려동물 처리\n"
    "- 대화에서 반려동물 관련 응답이 있으면 반드시 has_dog / has_cat 을 true/false/null 로 채우세요 (문자열 X).\n"
    "5) 출력 예시(반드시 JSON만):\n"
    "{\n"
    '  \"purpose\": \"... or null\",\n'
    '  \"relation\": \"... or null\",\n'
    '  \"gender\": \"... or null\",\n'
    '  \"preferred_color\": \"... or null\",\n'
    '  \"personality\": \"... or null\",\n'
    '  \"has_dog\": true/false/null,\n'
    '  \"has_cat\": true/false/null,\n'
    '  \"user_experience\": \"... or null\",\n'
    '  \"preferred_style\": \"... or null\",\n'
    '  \"plant_type\": \"... or null\",\n'
    '  \"season\": \"... or null\",\n'
    '  \"humidity\": \"... or null\",\n'
    '  \"isAirCond\": \"... or null\",\n'
    '  \"watering_frequency\": \"... or null\",\n'
    '  \"emotion\": \"... or null\"\n'
    "}\n"
    "반드시 위 규칙을 지켜 JSON만 출력하세요."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in conversation:
        messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=800,
        temperature=0.0,
    )

    try:
        text = resp.choices[0].message.content
    except Exception:
        text = resp.choices[0].message.content if resp.choices else ""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()

    try:
        parsed = json.loads(cleaned)
        return parsed
    except Exception:
        template = {
            "purpose": None,
            "relation": None,
            "gender": None,
            "preferred_color": None,
            "personality": None,
            "has_dog": None,
            "has_cat": None,
            "user_experience": None,
            "preferred_style": None,
            "plant_type": None,
            "season": None,
            "humidity": None,
            "isAirCond": None,
            "watering_frequency": None,
            "emotion": None,
        }
        return template

def extract_partial_fields(initial_text: str) -> Dict[str, Any]:
    prompt = (
    "아래 사용자 문장을 읽고, 문장에서 명확히 드러나는 정보만 JSON으로 반환하세요.\n"
    "규칙:\n"
    "- 가능한 경우 사용자가 말한 내용을 짧게 내부적으로 '되짚어서' 이해한 형태로 해석하되, 출력은 **오직 JSON**으로만 하세요.\n"
    "- 문장에 명확히 드러나지 않은 항목은 절대 포함하지 마세요.\n"
    "- 반려동물 관련 내용이 있다면 has_dog / has_cat 을 true/false 로 표기하세요.\n"
    "출력 키(허용): purpose, relation, gender, preferred_color, personality,\n"
    "                 has_dog, has_cat, user_experience, preferred_style,\n"
    "                 plant_type, season, humidity, isAirCond, watering_frequency, emotion\n"
    "문장:\n"
    f"{initial_text}\n\n"
    "예시: '친구 졸업식에 선물하려고' -> {\"purpose\": \"친구 졸업식에 선물하려고\", \"relation\": \"친구\"}\n"
    "중요: 모델이 사용자 문장을 내부적으로 요약(예: '친구분 졸업식에 드리시려는거군요')하는 방식으로 다음 질문을 자연스럽게 이어가도록 유도하되, 이 내부 요약은 출력에 포함하지 말고 **오직 JSON**만 반환하세요."
)


    messages = [{"role": "system", "content": prompt}]
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=400,
        temperature=0.0,
    )
    try:
        text = resp.choices[0].message.content
    except Exception:
        text = resp.choices[0].message.content if resp.choices else ""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()

    try:
        parsed = json.loads(cleaned)
        if "has_dog" in parsed or "has_cat" in parsed:
            pd = {}
            pd["has_dog"] = parsed.get("has_dog", None)
            pd["has_cat"] = parsed.get("has_cat", None)
            parsed.update(pd)
        return parsed
    except Exception:
        return {}

def save_json(data: Dict[str, Any], filename: str = "collected_user_info.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {filename}")

def main():
    print("안녕하세요 — 친절한 화원 직원입니다. 먼저 선물 상황을 간단히 알려주세요.")
    print("예: '친구 졸업식에 선물하려고' 같은 한두 문장으로 설명해 주세요.")
    print("답변을 원하지 않으면 'pass'를 입력하세요. 바로 추천(종료) 받고 싶으면 'end'를 입력하세요.")
    print("모르는 경우에는 '모름' 또는 '모르겠음' 등으로 답해 주세요.")
    print()

    collected: Dict[str, Optional[Any]] = {
        "purpose": None,
        "relation": None,
        "gender": None,
        "preferred_color": None,
        "personality": None,
        "has_dog": None,
        "has_cat": None,
        "user_experience": None,
        "preferred_style": None,
        "plant_type": None,
        "season": None,
        "humidity": None,
        "isAirCond": None,
        "watering_frequency": None,
        "emotion": None,
    }

    conversation: List[Dict[str, str]] = []

    # 1) 맨 처음: 전체적인 상황 설명 받기
    print(INITIAL_OPEN_QUESTION)
    initial = input("A: ").strip()
    conversation.append({"role": "user", "content": initial})

    if initial.strip().lower() == "end":
        print("지금까지의 대화를 분석해서 JSON으로 저장합니다...")
        extracted = extract_json_from_conversation(conversation)
        save_json(extracted)
        return
    if initial.strip().lower() == "pass":
        initial_parsed = {}
    elif is_unknown(initial):
        initial_parsed = {}
    else:
        initial_parsed = extract_partial_fields(initial)

    for k, v in initial_parsed.items():
        if k in collected:
            collected[k] = v
    if "has_pet" in initial_parsed:
        pet_map = parse_pets(initial_parsed.get("has_pet"))
        collected["has_dog"] = pet_map["has_dog"]
        collected["has_cat"] = pet_map["has_cat"]

    # 2) 남은 필드들만 질문
    for key, question in FIELDS:

        if key == "has_pet":
            continue

        if collected.get(key) is not None:
            continue

        context_summary = []

        if collected.get("relation"):
            context_summary.append(f"{collected.get('relation')}분")
        if collected.get("purpose"):
            context_summary.append(f"{collected.get('purpose')}")
        if context_summary:
            lead = " ".join(context_summary)
            print(f"{question}")
        else:
            print(f"Q: {question}")

        user_input = input("A: ").strip()
        conversation.append({"role": "user", "content": user_input})

        low = user_input.strip().lower()
        if low == "end":
            print("지금까지의 대화를 분석해서 JSON으로 저장합니다...")
            extracted = extract_json_from_conversation(conversation)

            if "has_dog" not in extracted or "has_cat" not in extracted:
                pet_field = None
                for turn in reversed(conversation):
                    if any(k in turn["content"] for k in ["강아지", "고양이", "없음", "개", "고양"]):
                        pet_field = turn["content"]
                        break
                pet_parsed = parse_pets(pet_field) if pet_field else {"has_dog": None, "has_cat": None}
                extracted["has_dog"] = extracted.get("has_dog", pet_parsed["has_dog"])
                extracted["has_cat"] = extracted.get("has_cat", pet_parsed["has_cat"])
            save_json(extracted)
            return

        if low == "pass":
            print(f"'{key}' 항목은 건너뜁니다.")
            collected[key] = None
            continue

        if is_unknown(user_input):
            collected[key] = None
            continue

        if key == "gender":
            collected[key] = user_input
        elif key == "preferred_color":
            collected[key] = user_input
        elif key == "personality":
            collected[key] = user_input
        elif key == "user_experience":
            collected[key] = user_input
        elif key == "preferred_style":
            collected[key] = user_input
        elif key == "plant_type":
            collected[key] = user_input
        elif key == "season":
            collected[key] = user_input
        elif key == "humidity":
            collected[key] = user_input
        elif key == "isAirCond":
            collected[key] = user_input
        elif key == "watering_frequency":
            collected[key] = user_input
        elif key == "emotion":
            collected[key] = user_input
        else:
            collected[key] = user_input

        if any(k in user_input for k in ["강아지", "고양이", "개", "고양"]):
            pets = parse_pets(user_input)
            collected["has_dog"] = pets["has_dog"]
            collected["has_cat"] = pets["has_cat"]

    print("지금까지 수집된 내용을 바탕으로 JSON파일 저장")
    final_json = collected.copy()
    save_json(final_json)

if __name__ == "__main__":
    main()
