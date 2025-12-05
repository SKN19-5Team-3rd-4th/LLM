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

    def image_generation_with_response(self, room_img, flower_img, prompt=None) :
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

                        },
                    
                    ],
                }
            ],
            tools=[{"type": "image_generation", "input_fidelity": "high"}],
        )


