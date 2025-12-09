from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv()

# API 키 로드
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 데이터 경로 설정
DATA_PATH = Path(__file__).resolve().parent.parent / 'datas'
IMG_PATH = Path(__file__).resolve().parent.parent / 'datas/images'

# 원본 데이터
RAW_FILE_PATH = DATA_PATH / 'flower_data.json'
NEW_FILE_PATH = DATA_PATH / 'new_data.json'
QNA_RAW_FILE_PATH = DATA_PATH / 'post.json'

# 벡터 DB 적재 데이터
REC_FILE_PATH = DATA_PATH / 'flower_preprocessed_data.json'
QNA_FILE_PATH = DATA_PATH / 'post_preprocessed_data.json'
IMG_FILE_PATH =[os.path.join(IMG_PATH, f) for f in os.listdir(IMG_PATH) if f.lower().endswith('.jpg')]

# 인덱스 이름 설정
REC_INDEX_NAME = "plant-recommend"
QNA_INDEX_NAME = "plant-qna-v3"
IMG_INDEX_NAME = "plant-img"

# 파인콘DB, 임베딩 모델
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")