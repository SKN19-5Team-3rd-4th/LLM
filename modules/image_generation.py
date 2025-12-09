from openai import OpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import json
from typing import TypedDict, Annotated, Optional, Literal, List, Dict
from PIL import Image
import io
from io import BufferedReader, BytesIO
import tempfile
import base64
from dotenv import load_dotenv

def image_open_to_base64(path) -> base64 :
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# base64를 출력용 이미지로 변환 
def base64_to_image(b64_image) -> Image.Image :
    image_data = base64.b64decode(b64_image)
    return Image.open(io.BytesIO(image_data))



class ImageGenerator() :
    def __init__(self):
        load_dotenv()
        self.client = OpenAI()

    def generate_image_summarize(self, input_image_path: str, system_prompt:str, user_prompt:str, reference_text: str = None) -> Dict :
        image_base64 = image_open_to_base64(input_image_path)

        response = self.client.chat.completions.create(
            model = 'gpt-4o',
            messages=[
                {
                    "role": "system",
                    "content" :[
                        {
                            "type" : "text",
                            "text" : system_prompt
                        }
                    ]
                },
                {
                    "role": "user",
                    "content" :[
                        {
                            "type" : "text", 
                            "text" : user_prompt,
                        },
                        {
                            "type" : "text", 
                            "text" : f"참고 예시 : {reference_text}",
                        },
                    ]
                },
                {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url', 
                                'image_url': {'url': f'data:image/png;base64,{image_base64}'}
                            }
                        ]
                    }
            ]
        )

        return response.choices[0].message.content

    def generate_image_with_response(self, room_img_path: str, flower_img_path: str, system_prompt, user_prompt=None) -> Image.Image :

        room_img = image_open_to_base64(room_img_path)
        flower_img = image_open_to_base64(flower_img_path)
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
        
    def generate_image_with_image_edit(self, room_img_path: str, flower_img_path: str, prompt=None) -> Image.Image:
        response = self.client.images.edit(
            model="gpt-image-1",
            image=[open(room_img_path, "rb"), open(flower_img_path, "rb")],
            prompt=prompt,
            input_fidelity="high",
        )

        image_base64 = response.data[0].b64_json
        if image_base64:
            image = base64_to_image(image_base64)
            return image
        else:
            return None





