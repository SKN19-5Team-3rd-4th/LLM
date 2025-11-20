from langchain_core.documents import Document
from pinecone import ServerlessSpec
from config import *
import pandas as pd
import json


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    return raw_data


def create_db(index_name):   
    if index_name not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return pc.Index(index_name)

    
def _create_text_for_embedding(row):
    air_cond_text = '공기정화식물이다.' if row['isAirCond'] else ''
    toxic_dog_text = '강아지에게 해롭다.' if row['isToxicToDog'] else ''
    toxic_cat_text = '고양이에게 해롭다.' if row['isToxicToCat'] else ''

    return (
        f"꽃 이름: {row['flowNm']}, "
        f"꽃말: {row['flowLang']}, "
        f"특징: {row['fContent']}, "
        f"용도: {row['fUse']}, "
        f"재배법: {row['fGrow']}, "
        f"종류: {row['fType']}. "
        f"{air_cond_text} "
        f"{toxic_dog_text} "
        f"{toxic_cat_text}"
    ).strip()


def insert_rec_index(raw_data):
    
    df = pd.DataFrame(raw_data)
    df['text_to_embed'] = df.apply(_create_text_for_embedding, axis=1)
    texts = df['text_to_embed'].tolist()
    vector = embeddings.embed_documents(texts)

    for i, row in df.iterrows():
        metadatas = row.drop(['fSctNm', 'fEngNm', 'fileName1', 'fileName2', 'fileName3', 'publishOrg', 'colors', 'text_to_embed']).to_dict()
        metadatas["text"] = row["text_to_embed"]

        rec_index.upsert([
            (
                str(row['dataNo']),
                vector[i],
                metadatas
            )
        ])


def insert_qna_index(raw_data):

    documents = []
    for item in raw_data:
        text = f"Q: {item['question']}\nA: {item['answer']}"
        documents.append(
            Document(page_content=text, metadata=item.get("metadata", {}))
        )

    for idx, docs in enumerate(documents):
        id_ = "groro" + "_" + str(idx + 1) + "_" + str(docs.metadata['post_id'])
        vector = embeddings.embed_query(docs.page_content)

        qna_index.upsert([
            (
                id_, 
                vector, 
                {'text': docs.page_content, **docs.metadata}
            )
        ])


if __name__ == "__main__":

    print("데이터로드")
    rec_data = load_data(REC_FILE_PATH)
    qna_data = load_data(QNA_FILE_PATH)

    print("인덱스생성")
    rec_index = create_db(REC_INDEX_NAME)
    qna_index = create_db(QNA_INDEX_NAME)

    print("추천 데이터적재")
    insert_rec_index(rec_data)
    
    print("식물 데이터적재")
    insert_qna_index(qna_data)