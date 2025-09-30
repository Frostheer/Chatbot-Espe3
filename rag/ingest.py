import os
import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm
from typing import List, Dict, Any

def chunk_text(text: str, doc_metadata: dict, page_number: int, chunk_size: int = 900, chunk_overlap: int = 120) -> list:
    """Divide el texto en chunks con solapamiento y metadatos de página."""
    words = text.split()
    tokens_per_word = 4
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
            "page_approx": page_number,
            "vigencia": doc_metadata["effective_date"],
        }
        chunks.append(chunk)
        i += target_len_words - overlap_len_words
        if i >= len(words):
            break
    return chunks

def get_pdf_chunks(filepath: str, doc_metadata: dict) -> List[Dict[str, Any]]:
    """Extrae texto por página de un PDF y lo divide en chunks con metadatos de página."""
    reader = PdfReader(filepath)
    all_chunks_from_doc = []
    for i, page in enumerate(tqdm(reader.pages, desc=f"Chunking {doc_metadata['title']}", leave=False)):
        page_number = i + 1
        text = page.extract_text()
        if text and text.strip():
            chunks = chunk_text(text, doc_metadata, page_number=page_number)
            all_chunks_from_doc.extend(chunks)
    return all_chunks_from_doc

def run_ingestion(data_dir: str = "data"):
    """Función principal para ingestar todos los documentos."""
    RAW_DIR = os.path.join(data_dir, "raw")
    PROCESSED_DIR = os.path.join(data_dir, "processed")
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    sources_path = os.path.join(data_dir, "sources.csv")
    if not os.path.exists(sources_path):
        print(f"ERROR: No se encontró {sources_path}. Crea este archivo primero!")
        return
    sources_df = pd.read_csv(sources_path)
    all_chunks = []
    print("--- Iniciando Ingesta y Chunking ---")
    for index, row in tqdm(sources_df.iterrows(), total=sources_df.shape[0], desc="Procesando documentos"):
        doc_path = os.path.join(RAW_DIR, row['filename'])
        if not os.path.exists(doc_path):
            print(f"AVISO: No se encontró el archivo {row['filename']}. Saltando.")
            continue
        doc_metadata = row.to_dict()
        chunks = get_pdf_chunks(doc_path, doc_metadata)
        all_chunks.extend(chunks)
    if all_chunks:
        chunks_df = pd.DataFrame(all_chunks)
        output_path = os.path.join(PROCESSED_DIR, "chunks.parquet")
        chunks_df.to_parquet(output_path, index=False)
        print(f"\nIngesta completa. {len(all_chunks)} chunks guardados en {output_path}")

if __name__ == '__main__':
    run_ingestion()
