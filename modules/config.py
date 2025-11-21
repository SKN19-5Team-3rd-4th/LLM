from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from pathlib import Path
from dotenv import load_dotenv 
import os


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

DATA_PATH = Path(__file__).resolve().parent.parent / 'datas'
RAW_FILE_PATH = os.path.join(DATA_PATH, 'flower_data.json')
NEW_FILE_PATH = os.path.join(DATA_PATH, 'new_data.json')
REC_FILE_PATH = os.path.join(DATA_PATH, 'flower_preprocessed_data.json')
QNA_FILE_PATH = os.path.join(DATA_PATH, 'post_preprocessed_data.json')
REC_INDEX_NAME = "plant-recommend"
QNA_INDEX_NAME = "plant-qna"

pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")