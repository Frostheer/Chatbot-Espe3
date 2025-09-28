import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import Tuple, List

class Retriever:
    """Clase para manejar la carga del índice y la búsqueda vectorial."""
    
    def __init__(self, data_dir: str = "data"):
        FAISS_PATH = os.path.join(data_dir, "index.faiss")
        CHUNKS_PATH = os.path.join(data_dir, "processed", "chunks.parquet")
        
        # Cargar Modelo de Embeddings
        self.model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        
        # Cargar Índice FAISS
        try:
            self.index = faiss.read_index(FAISS_PATH)
            # Cargar Metadatos de Chunks
            self.chunks_df = pd.read_parquet(CHUNKS_PATH)
            print("✅ Retriever inicializado (FAISS y Chunks cargados).")
        except Exception as e:
            print(f"🔴 ERROR al cargar índice/chunks: {e}. ¿Ejecutaste 'ingest.py' y 'embed.py'?")
            self.index = None
            self.chunks_df = None

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[float], List[str]]:
        """Busca los k chunks más relevantes para la query."""
        if not self.index or self.chunks_df is None:
            return [], []

        # 1. Vectorizar la query
        query_embedding = self.model.encode([query]).astype('float32')
        
        # 2. Buscar en FAISS
        # D: Distancias (similitud), I: Índices
        distances, indices = self.index.search(query_embedding, k)
        
        # 3. Recuperar los chunks y formatear el contexto
        context = []
        # Los índices son los del DataFrame original
        retrieved_chunks = self.chunks_df.iloc[indices[0]]
        
        for index, chunk in retrieved_chunks.iterrows():
            # Formato de cita requerido por el prompt de sistema
            citation_format = f"[{chunk['title']}, Pág. {chunk['page_approx']}]"
            context_entry = f"{citation_format}: {chunk['text']}\n"
            context.append(context_entry)
            
        return distances[0].tolist(), context