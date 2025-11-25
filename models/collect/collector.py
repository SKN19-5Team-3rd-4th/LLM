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
    # 필수 수집 항목
    ("purpose", "선물 목적이 어떻게 되시나요?"),
    ("relation", "선물할 사람과의 관계는 무엇인가요?"),
    ("gender", "선물받는 분의 성별은 무엇인가요?"),
    ("preferred_color", "원하시는 색상을 입력해주세요."),
    ("personality", "식물을 키울 사람의 성향(성격)을 간단히 알려주세요."),
    ("has_pet", "강아지나 고양이를 키우나요?"),
    # 추가 수집 항목
    ("user_experience", "선물받는 분의 원예 경험이 있나요?"),
    ("preferred_style", "원하는 스타일이 있나요?"),
    ("plant_type", "원하는 식물 유형이 있나요?"),
    ("season", "선물할 시기(계절)가 정해져 있나요?"),
    ("humidity", "특별히 신경쓸 실내 습도가 있나요?"),
    ("isAirCond", "에어컨 사용이 잦은 환경인가요?"),
    ("watering_frequency", "물을 얼마나 자주 줄 수 있을 것 같나요?"),
    ("emotion", "선물에 담고 싶은 감정이나 메시지가 있나요?"),
]

def is_unknown(answer: str) -> bool:
    if not answer:
        return True
    a = answer.strip().lower()
    unknown_keywords = ["모름", "모르겠", "모르겠어요", "모르겠습", "모르겠", "몰라", "모르겠음"]
    for kw in unknown_keywords:
        if kw in a:
            return True
    return False

def parse_pets(answer: str) -> Dict[str, Optional[bool]]:
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
    """
    conversation: [{"role": "user", "content": "..."} , ...]
    반환: 필드에 맞춘 dict (없는 항목은 null/None)
    """
    system_prompt = (
        "너는 친절한 화원 직원이다. 아래의 conversation(사용자와의 대화)을 읽고, "
        "다음 키들을 포함하는 JSON 하나만 출력해라. "
        "만약 사용자가 모른다고 했거나 관련 대화가 전혀 없다면 그 필드의 값은 null로 해라. "
        "사용자가 'pass'로 넘긴 질문은 무시하되 해당 값이 채워지지 않았다면 null로 둬라. "
        "사용자가 강아지/고양이 관련 대답을 했으면 has_dog/has_cat 을 True/False/null 로 정확히 채워라. "
        "출력 형식(예시):\n"
        "{\n"
        '  "purpose": "...",\n'
        '  "relation": "...",\n'
        '  "gender": "...",\n'
        '  "preferred_color": "...",\n'
        '  "personality": "...",\n'
        '  "has_dog": true/false/null,\n'
        '  "has_cat": true/false/null,\n'
        '  "user_experience": "...",\n'
        '  "preferred_style": "...",\n'
        '  "plant_type": "...",\n'
        '  "season": "...",\n'
        '  "humidity": "...",\n'
        '  "isAirCond": "...",\n'
        '  "watering_frequency": "...",\n'
        '  "emotion": "..."\n'
        "}\n"
        "반드시 JSON만 출력하고 다른 텍스트를 출력하지 마라. null 값은 JSON의 null이어야 한다."
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

    text = ""
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

def save_json(data: Dict[str, Any], filename: str = "collected_user_info.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {filename}")

def main():
    print("안녕하세요 — 친절한 화원 직원입니다. 추천을 위해서 몇 가지 질문드릴게요.")
    print("답변을 원하지 않으시는 경우 'pass'를 입력하시고, 바로 추천을 원하신다면 'end'를 입력하세요.")
    print("모르는 경우에는 '모름' 또는 '모르겠음' 등으로 답변해 주세요.")
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

    for key, question in FIELDS:
        print(question)
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
            continue

        # 모른다고 할 때
        if is_unknown(user_input):
            collected[key] = None
            continue

        if key == "has_pet":
            pets = parse_pets(user_input)
            collected["has_dog"] = pets["has_dog"]
            collected["has_cat"] = pets["has_cat"]
        else:
            collected[key] = user_input

    # json 파일로 저장
    print("모든 질문을 마쳤습니다. 지금까지 수집된 내용을 바탕으로 JSON을 저장합니다.")

    final_json = collected.copy()
    save_json(final_json)
    return

if __name__ == "__main__":
    main()
