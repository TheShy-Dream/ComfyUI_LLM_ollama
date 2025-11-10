import requests
import time
from PIL import Image
import numpy as np
import os
import torch
import tempfile

class Ollama_LLMAPI_Node():
    """调用本地 Ollama /api/chat 接口的节点，支持视觉输入与随机种子"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_baseurl": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "model": ("STRING", {"default": "llama3.2-vision"}),  # 推荐支持图像的模型
                "role": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the image."}),
                "temperature": ("FLOAT", {"default": 0.6}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999999}),  # 新增 seed 参数
            },
            "optional": {
                "ref_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "run_ollama"
    CATEGORY = "Ollama"

    def run_ollama(self, api_baseurl, model, role, prompt, temperature, seed, ref_image=None):
        """调用 Ollama 聊天接口"""
        url = f"{api_baseurl}/api/chat"

        # 构造基础消息
        if ref_image is None:
            messages = [
                {"role": "system", "content": role},
                {"role": "user", "content": prompt},
            ]
        else:
            # 将图像 tensor 保存为临时文件
            i = 255. * ref_image.cpu().numpy()[0]
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            tmp_dir = tempfile.gettempdir()
            img_path = os.path.join(tmp_dir, f"ollama_ref_{time.time()}.png")
            img.save(img_path)

            # 用 file:// 引用本地路径
            messages = [
                {"role": "system", "content": role},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"file://{img_path}"}
                    ]
                },
            ]

        # 构造 payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        # 如果 seed != 0，就添加进 options
        if seed != 0:
            payload["options"]["seed"] = seed

        try:
            resp = requests.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                # Ollama 响应格式通常为 {"message": {"content": "..."}}
                return (data.get("message", {}).get("content", "No content returned."),)
            else:
                return (f"Error {resp.status_code}: {resp.text}",)
        except Exception as e:
            return (f"Exception: {str(e)}",)
        finally:
            # 删除临时文件
            if ref_image is not None and os.path.exists(img_path):
                os.remove(img_path)
