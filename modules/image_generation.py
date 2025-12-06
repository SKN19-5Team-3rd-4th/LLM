from openai import OpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import json
from typing import TypedDict, Annotated, Optional, Literal, List
from PIL import Image
import io
import base64
from dotenv import load_dotenv

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')
    
def base64_to_image(b64_image):
    image_data = base64.b64decode(b64_image)
    return Image.open(io.BytesIO(image_data))

class ImageGenerator() :
    def __init__(self):
        load_dotenv()
        self.client = OpenAI()

    def image_generation_with_response(self, room_img: base64, flower_img: base64, prompt=None) :
        response = self.client.responses.create(
            model="gpt-5.1",
            input=[
                {
                    "role": "user",
                    "content" : [
                        {
                            "type" : "input_text", 
                            "text" : prompt,
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
        
    def image_generation_with_image_edit(self, room_img_path: str, flower_img_path: str, prompt=None) :
        response = self.client.images.edit(
            model="gpt-image-1",
            image=[open(room_img_path, 'rb'), open(flower_img_path, 'rb')],
            prompt=prompt,
            input_fidelity="high",
        )

        image_base64 = response.data[0].b64_json
        if image_base64:
            image = base64_to_image(image_base64)
            return image
        else:
            return None





