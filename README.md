🧩 节点名称

Ollama_LLMAPI_Node

📘 节点简介

Ollama_LLMAPI_Node 是一个用于 ComfyUI 的自定义计算节点，
可直接调用本地或远程部署的 Ollama 大语言模型（LLM）API，
实现文本生成、对话、图像理解等任务。

节点支持传入文字提示（Prompt）、系统角色（System Prompt），
并可选地输入一张参考图像进行多模态分析（如果模型支持）。

⚙️ 输入参数
参数名	类型	说明
api_baseurl	STRING	Ollama API 地址，例如 http://127.0.0.1:11434
model	STRING	要调用的 Ollama 模型名称（如 llama3, mistral, qwen2 等）
prompt	STRING	用户输入的提示文本
system	STRING	系统角色设定（例如 "You are a helpful assistant"）
temperature	FLOAT	控制生成的随机性（默认 0.6）
ref_image (可选)	IMAGE	参考输入图像，用于图像描述或多模态模型（自动转换为 base64）
🔁 输出
名称	类型	说明
describe	STRING	LLM 返回的生成文本结果
🧠 功能说明

自动将输入图像转为 WebP 并编码为 Base64，支持多模态模型输入

兼容 Ollama /api/generate 接口

默认返回完整的文本响应（非流式）

遵循 OpenAI 类似 API 参数格式

可用于文本生成、问答、摘要、图像描述等场景
