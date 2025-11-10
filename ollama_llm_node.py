import requests
import base64
from PIL import Image
import numpy as np
import os
import time

def encode_image_b64(ref_image):
    i = 255. * ref_image.cpu().numpy()[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    # 限制最大尺寸
    lsize = max(img.size)
    factor = 1
    while lsize / factor > 2048:
        factor *= 2
    img = img.resize((img.size[0] // factor, img.size[1] // factor))

    path = f"{time.time()}.webp"
    img.save(path, "WEBP")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    os.remove(path)
    return b64


class Ollama_LLMAPI_Node:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_baseurl": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "Hello"}),
                "system": ("STRING", {"multiline": True, "default": "You are a helpful assistant"}),
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

    def rh_run_llmapi(self, api_baseurl, model, prompt, system, temperature, ref_image=None):

        headers = {"Content-Type": "application/json"}

        data = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "options": {"temperature": temperature},
            "stream": False  # 返回完整对象
        }

        if ref_image is not None:
            b64_image = encode_image_b64(ref_image)
            data["images"] = [b64_image]

        response = requests.post(f"{api_baseurl}/v1/generate", headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            # 新版 Ollama API 返回文本在 'response' 字段
            text = result.get("response", "No response")
        else:
            text = f"Error {response.status_code}: {response.text}"

        return (text,)
