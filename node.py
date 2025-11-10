import requests
import time
from PIL import Image
import numpy as np
import base64
import os
import torch

def encode_image_b64(ref_image):
    """将ComfyUI图像tensor编码为base64"""
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


class RH_LLMAPI_Node():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_baseurl": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "model": ("STRING", {"default": "llama3"}),  # 默认使用Ollama本地模型
                "role": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
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
        """
        调用 Ollama /api/chat 接口。
        """
        url = f"{api_baseurl}/api/chat"

        # 构造消息内容
        if ref_image is None:
            messages = [
                {"role": "system", "content": role},
                {"role": "user", "content": prompt},
            ]
        else:
            base64_image = encode_image_b64(ref_image)
            # Ollama 尚未官方支持 multimodal image base64，但可以文本提示中嵌入描述性提示
            # 这里作为示例：将图像base64附在消息中，模型如llava/llama3-vision类才能理解
            messages = [
                {"role": "system", "content": role},
                {"role": "user", "content": f"{prompt}\n[image data: data:image/webp;base64,{base64_image}]"},
            ]

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,  # 如果想逐步输出改为 True
            "options": {"temperature": temperature},
        }

        try:
            resp = requests.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                # Ollama 的结果一般在 message.content 里
                return (data["message"]["content"],)
            else:
                return (f"Error {resp.status_code}: {resp.text}",)
        except Exception as e:
            return (f"Exception: {str(e)}",)
