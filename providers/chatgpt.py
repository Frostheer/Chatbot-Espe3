import os
from openai import OpenAI
from typing import List, Dict, Any
from .base import Provider

class ChatGPTProvider(Provider):
    """Adapter para la API de OpenAI (ChatGPT)."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        # La clave se lee automáticamente de la variable CHATGPT_API_KEY
        self.client = OpenAI(api_key=os.getenv("CHATGPT_API_KEY"))
        self._model = model

    @property
    def name(self) -> str:
        return f"ChatGPT ({self._model})"

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.4),
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error en ChatGPT: {e}")
            return "ERROR: Falló la comunicación con ChatGPT."