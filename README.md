# Chatbot Normativa UFRO

Este proyecto implementa un asistente conversacional (chatbot) basado en RAG (Retrieval-Augmented Generation) para responder preguntas sobre normativa y reglamentos universitarios de la Universidad de La Frontera (UFRO), utilizando documentos oficiales como fuente.

## Características

- Recuperación de contexto relevante usando embeddings y FAISS.
- Generación de respuestas con modelos LLM (ChatGPT, DeepSeek).
- Citas automáticas de documentos según la normativa UFRO.
- Abstención responsable si la información no está en los documentos.
- CLI interactivo y modo batch para evaluación.

## Estructura del Proyecto

```
.env
app.py
requirements.txt
providers/
    base.py
    chatgpt.py
    deepseek.py
rag/
    embed.py
    ingest.py
    prompts.py
    retrieve.py
data/
    sources.csv
    raw/
    processed/
    index.faiss
eval/
scripts/
```

## Instalación

1. Clona el repositorio y entra al directorio.
2. Instala las dependencias:
   ```sh
   pip install -r requirements.txt
   ```
3. Crea un archivo `.env tus claves de API:
   ```
   OPENAI_API_KEY=tu_clave_openai
   DEEPSEEK_API_KEY=tu_clave_deepseek
   ```

## Uso

1. **Ingesta de documentos**  
   Procesa los PDFs y genera los chunks:
   ```sh
   python rag/ingest.py
   ```

2. **Generación de embeddings e índice FAISS**  
   ```sh
   python rag/embed.py
   ```

3. **Ejecuta el chatbot en modo interactivo**  
   ```sh
   python app.py --provider chatgpt -k 5
   ```
   Cambia `--provider` por `deepseek` si lo prefieres.

## Evaluación

Para modo batch y evaluación automática, implementa el módulo en `eval/`.

## Personalización

- Agrega nuevos documentos en `data/raw/` y actualiza `data/sources.csv`.
- Ajusta el prompt en [`rag/prompts.py`](rag/prompts.py).

## Créditos

Desarrollado para la UFRO.  
Modelos y APIs: OpenAI, DeepSeek, Sentence Transformers, FAISS.

## Licencia

Uso académico y educativo.