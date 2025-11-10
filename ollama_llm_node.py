import requests
import time
from PIL import Image
import numpy as np
import base64
import os

def encode_image_b64(ref_image):
    i = 255. * ref_image.cpu().numpy()[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    lsize = np.max(img.size)
    factor = 1
    while lsize / factor > 2048:
        factor *= 2
    img = img.resize((img.size[0] // factor, img.size[1] // factor))

    image_path = f'{time.time()}.webp'
    img.save(image_path, 'WEBP')

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    os.remove(image_path)
    return base64_image


class Ollama_LLMAPI_Node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_baseurl": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": ""}),
                "role": ("STRING", {"multiline": True, "default": "You are a helpful assistant"}),
                "prompt": ("STRING", {"multiline": True, "default": "Hello"}),
                "temperature": ("FLOAT", {"default": 0.6}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("describe",)
    FUNCTION = "rh_run_llmapi"
    CATEGORY = "Runninghub"

    def rh_run_llmapi(self, api_baseurl, model, role, prompt, temperature, ref_image=None):

        headers = {
            "Content-Type": "application/json",
        }

        if ref_image is None:
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": role},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
            }
        else:
            base64_image = encode_image_b64(ref_image)
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64_image]
                    }
                ],
                "temperature": temperature,
            }

        response = requests.post(
            f"{api_baseurl}/v1/completions",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            result = response.json()
            text = result.get("completion", "No response")
        else:
            text = f"Error {response.status_code}: {response.text}"

        return (text,)
