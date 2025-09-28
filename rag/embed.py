import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def run_embedding_and_faiss(data_dir: str = "data"):
    """Genera embeddings, construye el índice FAISS y lo persiste."""
    PROCESSED_DIR = os.path.join(data_dir, "processed")
    FAISS_PATH = os.path.join(data_dir, "index.faiss")
    CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.parquet")
    
    if not os.path.exists(CHUNKS_PATH):
        print(f"ERROR: No se encontró {CHUNKS_PATH}. Ejecuta primero rag/ingest.py")
        return

    # 1. Cargar datos
    print("--- 🧠 Cargando Chunks y Modelo de Embeddings ---")
    df = pd.read_parquet(CHUNKS_PATH)
    texts = df['text'].tolist()
    
    # 2. Cargar modelo de embeddings (all-MiniLM-L6-v2 es un buen default)
    model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    
    # 3. Generar Embeddings
    print(f"Generando {len(texts)} embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # 4. Construir Índice FAISS
    dimension = embeddings.shape[1]
    # IndexFlatIP es simple y efectivo para similitud de coseno
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings)
    
    # 5. Persistir el índice
    faiss.write_index(index, FAISS_PATH)
    print(f"\n✅ Índice FAISS persistido en {FAISS_PATH}. Dimensión: {dimension}.")
    

    return model, index, df

if __name__ == '__main__':
    # Primero asegura que el .env esté cargado
    from dotenv import load_dotenv
    load_dotenv()
    run_embedding_and_faiss()