from dotenv import load_dotenv
import os
from PIL import Image
from modules.similar_imgs import ImageSearch
import json
from modules.image_generation import ImageGenerator

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

system_prompt = "입력되는 이미지를 설명 예시와 유사하게 인테리어 스타일을 묘사하여라."
user_prompt = "다음 방 이미지의 인테리어 스타일을 묘사해줘."


# ==== front에서 받아올 것 ====
room_image_path = 'datas/images/test1.jpg'
flower_name = "산세베리아"       
flower_image_path = "datas/images/730017_565845_3235.png" 
# ============================

searcher = ImageSearch()
reference_text, img = searcher.search_top3(room_image_path)

# 이미지를 파일에서 Pillow Image로 불러옴
input_image = Image.open(room_image_path)
image_base64 = None

generator = ImageGenerator()
summarize = generator.generate_image_summarize(input_image_path=room_image_path, system_prompt=system_prompt, user_prompt=user_prompt, reference_text=reference_text)
summ_json = json.loads(summarize)

prompt = f"다음은 내 집 사진과 {flower_name} 사진이야. 이 두 사진을 다음 예시 설명에 맞춰서 적절한 위치에 화분으로 합성해줘. 예시 설명: {summarize}"
created_img = generator.generate_image_with_image_edit(room_image_path, flower_image_path, prompt)

# ==== django로 넘겨서 db에 저장 ====
summ_json
created_img
# ==================