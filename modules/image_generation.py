from openai import OpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import json
from typing import TypedDict, Annotated, Optional, Literal, List, Dict
from PIL import Image
import io
import base64
from dotenv import load_dotenv


def image_to_base64(path) -> base64 :
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# base64를 출력용 이미지로 변환 
def base64_to_image(b64_image) -> Image.Image :
    image_data = base64.b64decode(b64_image)
    return Image.open(io.BytesIO(image_data))

# Image형태를 모델에 전달하기 위한 형태의 bytes 형태로 변환
def image_to_bytes(image: Image.Image, path: str = None, format="PNG") -> bytes :
    # 경로가 존재한다면 이미지를 저장 후 bytes 형태로 오픈
    if path:
        image.save(path, format=format)
        return open(path, "rb")
    # 경로가 존재하지 않는다면 버퍼에 저장 후 오픈
    else:
        buf = io.BytesIO()
        image.save(buf, format=format)
        buf.seek(0)
        return buf
    
def bytes_to_base64(img):
    return base64.b64encode(img.read()).decode('utf-8')

class ImageGenerator() :
    def __init__(self):
        load_dotenv()
        self.client = OpenAI()

    def generate_image_summarize(self, input_image: bytes, system_prompt:str, user_prompt:str, reference_text: str = None) -> Dict :
        image_base64 = bytes_to_base64(input_image)

        response = self.client.chat.completions.create(
            model = 'gpt-5.1',
            input=[
                {
                    "role": "system",
                    "content" :[
                        {
                            "type" : "input_text",
                            "text" : system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content" :[
                        {
                            "type" : "input_text", 
                            "text" : user_prompt,
                        },
                        {
                            "type" : "input_text", 
                            "text" : f"참고 예시 : {reference_text}",
                        },

                        {
                            "type" : "input_image",
                            "image_url" : f"data:image/png;base64,{image_base64}",
                        }
                    ]
                }
            ]
        )

        return response.choices[0].message.content

    def generate_image_with_response(self, room_img_bytes: bytes, flower_img_bytes: bytes, system_prompt, user_prompt=None) :

        room_img = bytes_to_base64(room_img_bytes)
        flower_img = bytes_to_base64(flower_img_bytes)
        response = self.client.responses.create(
            model="gpt-5.1",
            input=[
                {
                    "role": "system",
                    "content" :[
                        {
                            "type" : "input_text",
                            "text" : system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content" : [
                        {
                            "type" : "input_text", 
                            "text" : user_prompt,
                        },
                        {
                            "type" : "input_image",
                            "image_url" : f"data:image/png;base64,{room_img}",
                        },
                        {
                            "type" : "input_image",
                            "image_url" : f"data:image/png;base64,{flower_img}",
                        },
                    ],
                }
            ],
            tools=[{"type": "image_generation", "input_fidelity": "high"}],
        )

        image_data = [
            output.result for output in response.output
            if output.type == "image_generation_call"
        ]

        if image_data:
            image = base64_to_image(image_data[0])

            return image
        else: 
            return None
        
    def generate_image_with_image_edit(self, room_img: bytes, flower_img: bytes, prompt=None) :
        response = self.client.images.edit(
            model="gpt-image-1",
            image=[room_img, flower_img],
            prompt=prompt,
            input_fidelity="high",
        )

        image_base64 = response.data[0].b64_json
        if image_base64:
            image = base64_to_image(image_base64)
            return image
        else:
            return None





