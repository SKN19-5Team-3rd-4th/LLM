from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from config import RAW_FILE_PATH, NEW_FILE_PATH, REC_FILE_PATH, QNA_RAW_FILE_PATH, QNA_FILE_PATH
from dotenv import load_dotenv
import pandas as pd 
import argparse
import json
import emoji
import re
load_dotenv()


#----------------------------------------------------------------------------------------------
# 1) 식물 추천 데이터 전처리
# RAW_FILE_PATH 로드 → 정보생성 → NEW_FILE_PATH 저장 → 정보병합 → REC_FILE_PATH 저장 → RAG 활용
#----------------------------------------------------------------------------------------------
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
            temperature=1
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
    
    print(f"[INFO] 저장 경로: {REC_FILE_PATH} | {len(merged)}건 전처리 완료")
    

#----------------------------------------------------------------------------------------------
# 2) 식물 커뮤니티 게시글 데이터 전처리
# QNA_RAW_FILE_PATH 로드 → 정제 → QNA_FILE_PATH 저장 → RAG 활용
#----------------------------------------------------------------------------------------------
def clean_qna_data():

    with open(QNA_RAW_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 게시글 데이터 필터링
    df = pd.DataFrame(data)[['channel', 'author', 'content', 'comments', 'post_url', 'date', 'hashtags']]
    df = df.drop_duplicates(['content']).query("channel == '식물Q&A' and author != '그로로알림이'").reset_index(drop=True)

    # 게시글 id 생성 및 코멘츠 댓글 {} > row 분리
    df = df.explode('comments').reset_index(drop=True)   
    df['post_id'] = df['post_url'].str.split('/').str[-1]
    df['comments_id'] = df.groupby(['post_url']).cumcount()+1
    df['comments'] = df['comments'].apply(lambda x: x if not 'content' in str(x) else x['content'])
    df = df.drop(['author', 'post_url'], axis=1)

    # 해시태그 합치기 f"#태그 #태그"
    df = df.explode('hashtags').reset_index(drop=True)  
    df = df.groupby(['post_id', 'channel', 'content', 'comments', 'comments_id', 'date'], dropna=False)['hashtags']\
        .apply(lambda x: ' '.join(f"#{v}" for v in sorted(set(x.dropna())))).reset_index()

    # 텍스트 정제 
    def clean_text(text):
        if pd.isna(text) or not isinstance(text, str): return ''        
        text = str(text).lower()
        text = emoji.demojize(text, language='es') # 이모지 텍스트화
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r'\b(\d+)k\b', lambda m: str(int(m.group(1)) * 1000), text)
        text = re.sub(r'\b(\d+)m\b', lambda m: str(int(m.group(1)) * 1000000), text)
        # 문맥에 필요한 구두점/분수 패턴, demojize 콜론(:) 보존
        text = re.sub(r'[^\w\s!?,:/]', '', text)        
        # 반복자음 일부 줄이기
        text = re.sub(r'(ㅋ|ㅎ|ㅠ|ㅜ)\1{2,}', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df['content'] = df['content'].apply(clean_text)
    df['comments'] = df['comments'].apply(clean_text)

    # 댓글 순서 유지 
    df = df.sort_values(by=['post_id', 'comments_id'],key=lambda x:x.astype(int)).reset_index(drop=True)
    df['comments'] = "[COMMENT" + df['comments_id'].astype(str) + "] " + df['comments']

    # 본문/댓글 정제 내용 병합
    df_comment = df.groupby('post_id')['comments'].agg(lambda x: '\n'.join(x.astype(str))).reset_index()
    df_content = df.drop_duplicates('post_id').reset_index(drop=True)
    df_merge = df_content.merge(df_comment, on='post_id', suffixes=('_drop', ''))
    df_merge = df_merge.drop(columns=df_merge.filter(regex='_drop$').columns)
    df_merge = df_merge[['post_id', 'channel', 'content', 'comments', 'hashtags', 'date']]
    df_merge = df_merge.sort_values(by='post_id', key=lambda x:x.astype(int)).reset_index(drop=True)

    # RAG 포맷 가공
    df_merge['ids'] = 'groro_' + df_merge['post_id']
    df_merge['question'] = df_merge['content']
    df_merge['answer'] = df_merge['comments']

    df_merge['metadata'] = df_merge.apply(\
        lambda x: {'post_id': x['post_id'],
                   'channel': x['channel'],
                   'hashtags': x['hashtags'],
                   'date': x['date'],
                   'source_platform': 'groro'}
                   , axis=1
    )

    result = df_merge[['ids', 'question', 'answer', 'metadata']].to_dict('records')

    # 저장
    with open(QNA_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 저장 경로: {QNA_FILE_PATH} | {len(result)}건 전처리 완료")

def main():
    parser = argparse.ArgumentParser(description="데이터 전처리 스크립트")
    parser.add_argument(
        "--idx",
        type=int,
        required=True,
        choices=[0, 1], 
        help=(
            "데이터 종류 선택\n"
            "[0] 식물 추천용 데이터 전처리\n"
            "[1] 식물 상담 QnA 데이터 전처리"
        )
    )
    args = parser.parse_args()

    if args.idx == 0:
        print(f"[INFO] 식물 추천용 데이터 전처리 시작")       
        supply_plants_data()
        merge_data()
    
    elif args.idx == 1:
        print(f"[INFO] 식물 상담 QnA 데이터 전처리 시작") 
        clean_qna_data()


if __name__ == "__main__":
    main()