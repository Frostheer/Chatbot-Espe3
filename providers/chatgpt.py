import os
from openai import OpenAI
from typing import List, Dict, Any
from .base import Provider

class ChatGPTProvider(Provider):
    """Adapter para la API de OpenAI (ChatGPT)."""
    
    def __init__(self, model: str = "openai/gpt-4.1-mini"):
        # La clave se lee automáticamente de la variable OPENAI_API_KEY
        self.client = OpenAI(api_key=os.getenv("CHATGPT_API_KEY"),
                             base_url="https://openrouter.ai/api/v1/")
        self._model = model

    @property
    def name(self) -> str:
        return f"ChatGPT ({self._model})"

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.1),
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error en ChatGPT: {e}")
            return "ERROR: Falló la comunicación con ChatGPT."