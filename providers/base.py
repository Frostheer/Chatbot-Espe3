from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Provider(ABC):
    """Protocolo base para cualquier proveedor de LLM."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Envía un mensaje al LLM y obtiene la respuesta.

        :param messages: Historial de mensajes (incluye el system prompt).
        :param kwargs: Parámetros adicionales (e.g., temperatura, model_name).
        :return: La respuesta de texto generada.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del proveedor (para logging/comparativas)."""
        pass