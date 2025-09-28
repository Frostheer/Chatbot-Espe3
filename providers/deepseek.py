import os
from openai import OpenAI
from typing import List, Dict, Any
from .base import Provider

class DeepSeekProvider(Provider):
    """Adapter para la API de DeepSeek (compatible con OpenAI)."""
    
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, model: str = "deepseek-chat"):
        # Usamos el cliente de OpenAI pero apuntando a DeepSeek
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=self.DEEPSEEK_BASE_URL
        )
        self._model = model

    @property
    def name(self) -> str:
        return f"DeepSeek ({self._model})"

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.1),
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error en DeepSeek: {e}")
            return "ERROR: Falló la comunicación con DeepSeek."