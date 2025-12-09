from langchain_pinecone import PineconeVectorStore
from config import embeddings, IMG_INDEX_NAME
import base64

def _image_to_base64(image_path):
  with open(image_path, 'rb') as f:
    return base64.b64encode(f.read()).decode('utf-8')

class ImageSearch(): 
    def search_top3(self, user_img, k=3):
        query = _image_to_base64(user_img)

        vector_store = PineconeVectorStore(
            index_name=IMG_INDEX_NAME,
            embedding=embeddings
        )
        docs = vector_store.similarity_search(query, k=k)

        if not docs:
            print("검색 결과 없음")
            return
        
        reference_text = docs[0].page_content
        
        img_path = []
        img_path.append(docs[0].metadata["img_path"])
        img_path.append(docs[1].metadata["img_path"])
        img_path.append(docs[2].metadata["img_path"])
        
        return reference_text, img_path