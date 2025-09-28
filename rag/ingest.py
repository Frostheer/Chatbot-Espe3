import os
import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm

def get_pdf_text(filepath: str) -> str:
    """Extrae texto de un PDF simple."""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        # Aquí puedes añadir lógica de limpieza simple (ej. quitar saltos de línea repetidos)
        text += page.extract_text() + "\n\n"
    return text

def chunk_text(text: str, doc_metadata: Dict[str, Any], chunk_size: int = 900, chunk_overlap: int = 120) -> List[Dict[str, Any]]:
    """Divide el texto en chunks con solapamiento y metadatos."""
    # Nota: Usar un tokenizer real (ej. tiktoken) para medir el tamaño en tokens es mejor,
    # pero aquí usaremos una aproximación simple basada en caracteres/palabras.
    
    # Adaptar un método de chunking basado en caracteres para simplicidad CLI:
    words = text.split()
    tokens_per_word = 4 # Aproximación
    target_len_words = chunk_size // tokens_per_word
    overlap_len_words = chunk_overlap // tokens_per_word
    
    chunks = []
    i = 0
    while i < len(words):
        end = min(i + target_len_words, len(words))
        chunk_text = " ".join(words[i:end])
        
        chunk = {
            "text": chunk_text,
            "doc_id": doc_metadata["doc_id"],
            "title": doc_metadata["title"],
            "url": doc_metadata["url"],
            # Nota: para la página exacta, necesitarías un parseo más sofisticado. 
            # Aquí solo indicaremos el inicio del chunk.
            "page_approx": doc_metadata.get("page", 1), 
            "vigencia": doc_metadata["vigencia"],
        }
        chunks.append(chunk)
        
        # Mover el índice con el solapamiento
        i += target_len_words - overlap_len_words if end < len(words) else len(words)
        if i >= len(words) and end < len(words): # caso borde
            break

    return chunks

def run_ingestion(data_dir: str = "data"):
    """Función principal para ingestar todos los documentos."""
    RAW_DIR = os.path.join(data_dir, "raw")
    PROCESSED_DIR = os.path.join(data_dir, "processed")
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # 1. Cargar metadatos de fuentes
    sources_path = os.path.join(data_dir, "sources.csv")
    if not os.path.exists(sources_path):
        print(f"ERROR: No se encontró {sources_path}. ¡Crea este archivo primero!")
        return

    sources_df = pd.read_csv(sources_path)
    all_chunks = []

    print("--- 📄 Iniciando Ingesta y Chunking ---")
    for index, row in tqdm(sources_df.iterrows(), total=sources_df.shape[0], desc="Procesando documentos"):
        doc_path = os.path.join(RAW_DIR, row['filename']) # Asumiendo columna 'filename' en sources.csv
        if not os.path.exists(doc_path):
            print(f"AVISO: No se encontró el archivo {row['filename']}. Saltando.")
            continue
            
        doc_metadata = row.to_dict()
        
        # 2. Extraer texto
        text = get_pdf_text(doc_path)
        
        # 3. Chunking
        chunks = chunk_text(text, doc_metadata)
        all_chunks.extend(chunks)

    # 4. Guardar chunks con metadatos
    if all_chunks:
        chunks_df = pd.DataFrame(all_chunks)
        output_path = os.path.join(PROCESSED_DIR, "chunks.parquet")
        chunks_df.to_parquet(output_path, index=False)
        print(f"\n✅ Ingesta completa. {len(all_chunks)} chunks guardados en {output_path}")

if __name__ == '__main__':
    run_ingestion()