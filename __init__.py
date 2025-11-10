from .ollama_llm_node import Ollama_LLMAPI_Node

NODE_CLASS_MAPPINGS = {
    "Ollama_LLMAPI_Node": Ollama_LLMAPI_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ollama_LLMAPI_Node": "🦙 Ollama LLM API Node",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
