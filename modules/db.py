from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime 
from config import PINECONE_API_KEY, REC_INDEX_NAME, QNA_INDEX_NAME, REC_FILE_PATH, QNA_FILE_PATH, pc, embeddings
import pandas as pd
import argparse
import json


class PineconeManager:
    def __init__(self):
        # self.pc = Pinecone(api_key=PINECONE_API_KEY)
        # self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.pc = pc 
        self.embeddings = embeddings
        self.dimension = 1536
        self.upsert_dt = datetime.now().strftime("%y%m%d")
        self.INDEX = None


    def load_index(self, index_name):   
        if index_name not in [idx["name"] for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.INDEX = self.pc.Index(index_name)
        print(f"[INFO] {index_name} | 로드 완료")
        return self.INDEX
    

    def read_index(self, query, namespace=""):        
        query_embeddings = self.embeddings.embed_query(query)        
        query_response = \
        self.INDEX.query(
            namespace = namespace,     # 인덱스 내 검색할 네임스페이스
            top_k = 3,                 # 결과 반환 갯수
            include_values=False,      # 벡터 임베딩 반환 여부
            include_metadata=True,     # 메타 데이터 반환 여부
            vector=query_embeddings    # 검색할 벡터 임베딩
        )
        return query_response
        

    def update_index(self, id, query, metadata={}, namespace=""):
        query_embeddings = self.embeddings.embed_query(query)  
        update_response = \
        self.INDEX.update(
            id=id,                     # 업데이트 할 기존 문서 ID
            values=query_embeddings,   # NEW 벡터임베딩
            set_metadata=metadata,     # NEW 메타데이터
            namespace=namespace
        )
        return update_response


    def delete_index(self, id_list, namespace=""):
        delete_response = \
        self.INDEX.delete(
            ids=id_list,               # 삭제할 문서ID (type: 리스트)
            namespace=namespace        # 인덱스 내 검색할 네임스페이스
        )
        return delete_response


class PineconeRagIngestor(PineconeManager):
    def __init__(self):
        super().__init__()

    # 적재할 데이터 불러오기
    def load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)        
        self.df = pd.DataFrame(raw_data)  
        print(f"[INFO] {len(self.df)}건 데이터 로드 완료")
        return self.df
    
    # 임베딩 텍스트 전처리 
    def _create_text_for_embedding(self, index_name, row):
        if REC_INDEX_NAME == index_name:
            air_cond_text = '공기정화식물이다.' if row['isAirCond'] else ''
            toxic_dog_text = '' if row['isToxicToDog'] else '강아지에게 안전하다.'
            toxic_cat_text = '' if row['isToxicToCat'] else '고양이에게 안전하다.'
            colors = " ".join(row['colors'])           
            return (
                f"{row['flowLang']}, "
                f"{row['fContent']}, "
                f"물주기: {row['watering_frequency']}, "
                f"난이도 {row['difficulty']}, "
                f"{row['interior']}, "
                f"{row['style']}, "
                f"{row['fUse']}, "
                f"{row['fGrow']}, "
                f"{row['fType']}. "
                f"{air_cond_text} "
                f"{toxic_dog_text} "
                f"{toxic_cat_text} "
                f"{colors} "
            ).strip()

        elif QNA_INDEX_NAME == index_name:            
            return f"Question: {row['question']}\nAnswer: {row['answer']}"
        
    # 데이터 적재
    def upsert_index(self, index_name):

        # 임베딩 텍스트
        self.df['text_to_embed'] = self.df.apply(lambda x: self._create_text_for_embedding(index_name, x), axis=1)
        print(f"[INFO] {index_name} 전처리 완료")

        # 문서ID, 벡터임베딩, 메타데이터
        if index_name == REC_INDEX_NAME:
            id_ = self.df['dataNo']
            vector_ = self.embeddings.embed_documents(self.df['text_to_embed'].tolist())
            metadata_ = self.df.rename(columns={'text_to_embed': 'text'})\
                        .drop(['fSctNm', 'fEngNm', 'fileName1', 'fileName2', 'fileName3', 'publishOrg', 'colors'], axis=1)\
                        .to_dict(orient='records')
            
        elif index_name == QNA_INDEX_NAME:
            id_ = self.df['metadata'].apply(lambda x: "groro" + "_" + str(x['post_id']))
            vector_ = self.embeddings.embed_documents(self.df['text_to_embed'].tolist())
            metadata_ = self.df.apply(lambda x: {**x['metadata'], 'text': x['text_to_embed']}, axis=1)
        
        # 적재
        insert_data = [{'id':i, 'values':v, 'metadata':m} for i, v, m in zip(id_, vector_, metadata_)]

        # 파인콘 한번에 4MB 이상 request 보내면 거부
        def batch(iterable, n=100):
            for idx in range(0, len(iterable), n):
                yield iterable[idx: idx + n]

        for idx, batch_data in enumerate(batch(insert_data, 100)):
            self.INDEX.upsert(
                vectors=batch_data,
                namespace=f'{index_name}-{self.upsert_dt}'
            )
            print(f"[INFO] Batch {idx+1}")

        print(f"[INFO] {index_name} | {len(insert_data)}건 적재 완료")


def main():

    parser = argparse.ArgumentParser(description="파인콘 임베딩 데이터 적재 스크립트")
    parser.add_argument(
        "--idx",
        type=int,
        required=True,
        choices=[0, 1], 
        help=(
            "데이터 종류 선택\n"
            "[0] 식물 추천용 데이터 (REC_FILE_PATH / REC_INDEX_NAME 사용)\n"
            "[1] 식물 상담 QnA 데이터 (QNA_FILE_PATH / QNA_INDEX_NAME 사용)"
        )
    )
    args = parser.parse_args()

    file_path = (REC_FILE_PATH, QNA_FILE_PATH)
    index_name = (REC_INDEX_NAME, QNA_INDEX_NAME)    

    PRI = PineconeRagIngestor()
    PRI.load_index(index_name[args.idx])
    PRI.load_data(file_path[args.idx])
    PRI.upsert_index(index_name[args.idx])


if __name__ == "__main__":
    main()

    # 테스트
    # PRI = PineconeRagIngestor()
    # PRI.load_index(QNA_INDEX_NAME)
    # response = PRI.read_index(query="스투키같은 식물은 어떻게 키워야할까요?", namespace="plant-qna-251122")
    # response = PRI.delete_index(id_list=['groro_12976'], namespace="plant-qna-251122")
    # print(response)