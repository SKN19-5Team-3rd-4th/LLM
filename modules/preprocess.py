from dotenv import load_dotenv
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config import RAW_FILE_PATH, NEW_FILE_PATH, REC_FILE_PATH

load_dotenv()

def supply_plants_data():
    with open(RAW_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
    input_datas = [plant['flowNm'] for plant in data]
    
    print('데이터 로드 성공')

    prompt = PromptTemplate(
        template="""
        ### 입력 데이터 ###
        {input_data}

        ### 설명 ###
        당신은 식물 전문가입니다. 입력 데이터의 식물에 대해 정보를 생성하여 반환하세요.
        반드시 JSON만 출력하세요. JSON 앞뒤에 설명, 문장, 코드블록, ``` 표시를 넣지 마세요.
        반드시 flowNm, watering_frequency, difficulty, interior, style을 그대로 사용하세요.

        0. flowNm: 식물 이름
        1. watering_frequency: 물 주기
        2. difficulty: 키우는 난이도 ('매우 쉬움', '쉬움', '보통', '어려움', '매우 어려움' 중 하나)
        3. interior: 이 식물로 공간을 연출하는 구체적인 방법 자세히 설명
        4. style: 이 식물과 가장 잘 어울리는 인테리어 스타일 3개
        
        ### 출력 형식 ###
        {{
            "flowNm": "",
            "watering_frequency": "",
            "difficulty": "",
            "interior": "",
            "style": "",
        }}
        """,
        input_variables=["input_data"],
    )

    model = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=1,
    )

    chain = prompt | model

    print('체인 생성 성공')
    
    completed_list = []

    for idx, item in enumerate(input_datas):
        raw = chain.invoke({'input_data': item})
        
        try:
            obj = json.loads(raw.content)
        except json.JSONDecodeError:
            print(f"{idx}번째 입력에서 JSON 파싱 실패: {item}")
            print("raw:", raw)
            continue

        completed_list.append(obj)
        
    with open(NEW_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(completed_list, f, ensure_ascii=False, indent=2)
    print('데이터 저장 성공')


def merge_data():
    with open(RAW_FILE_PATH, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
        
    with open(NEW_FILE_PATH, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    merged = [{**a, **b} for a, b in zip(data1, data2)]
    
    with open(REC_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    

if __name__ == "__main__":
    supply_plants_data()
    merge_data()