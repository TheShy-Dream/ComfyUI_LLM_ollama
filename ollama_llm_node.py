import requests
import time
from PIL import Image
import numpy as np
import os
import torch
import base64

class Ollama_LLMAPI_Node():
    """调用本地 Ollama /api/chat 接口的节点，支持视觉输入与随机种子"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_baseurl": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "model": ("STRING", {"default": "llama3.2-vision"}),
                "role": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image."}),
                "temperature": ("FLOAT", {"default": 0.6}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999999}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "run_ollama"
    CATEGORY = "Ollama"

    def encode_image_b64(self, ref_image):
        """把 Torch Tensor 转成 Base64，支持大图缩放和 WEBP"""
        i = 255. * ref_image.cpu().numpy()[0]
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # 缩放大图，保证最大边不超过 2048
        lsize = max(img.size)
        factor = 1
        while lsize / factor > 2048:
            factor *= 2
        img = img.resize((img.size[0] // factor, img.size[1] // factor))

        # 保存临时 WEBP 文件
        image_path = f'{time.time()}.webp'
        img.save(image_path, 'WEBP')

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        os.remove(image_path)
        return base64_image

    def run_ollama(self, api_baseurl, model, role, prompt, temperature, seed, ref_image=None):
        """调用 Ollama 聊天接口，支持视觉输入 Base64"""
        url = f"{api_baseurl}/api/chat"

        if ref_image is None:
            messages = [
                {"role": "system", "content": role},
                {"role": "user", "content": prompt},
            ]
        else:
            if isinstance(ref_image, torch.Tensor):
                img_b64 = self.encode_image_b64(ref_image)
            elif isinstance(ref_image, Image.Image):
                buffered = io.BytesIO()
                ref_image.save(buffered, format="WEBP")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            else:
                raise ValueError("ref_image must be Torch Tensor or PIL Image")

            messages = [
                {"role": "system", "content": role},
                {"role": "user", "content": prompt, "images": [img_b64]}
            ]

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        if seed != 0:
            payload["options"]["seed"] = seed

        try:
            resp = requests.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                msg_content = data.get("message", {}).get("content")
                if isinstance(msg_content, list) and len(msg_content) > 0 and "text" in msg_content[0]:
                    return (msg_content[0]["text"],)
                else:
                    return ("No content returned.",)
            else:
                return (f"Error {resp.status_code}: {resp.text}",)
        except Exception as e:
            return (f"Exception: {str(e)}",)
