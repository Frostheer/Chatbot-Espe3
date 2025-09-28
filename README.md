# Chatbot Normativa UFRO

Este proyecto implementa un asistente conversacional (chatbot) basado en RAG (Retrieval-Augmented Generation) para responder preguntas sobre normativa y reglamentos universitarios de la Universidad de La Frontera (UFRO), utilizando documentos oficiales como fuente.

## Características

- Recuperación de contexto relevante usando embeddings y FAISS.
- Generación de respuestas con modelos LLM (ChatGPT, DeepSeek).
- Citas automáticas de documentos según la normativa UFRO.
- Abstención responsable si la información no está en los documentos.
- CLI interactivo y modo batch para evaluación.
- Modo A/B para comparar respuestas y latencias entre ChatGPT y DeepSeek.

## Estructura del Proyecto

```
.env
.env.example
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
3. Copia el archivo de ejemplo y añade tus claves de API:
   ```sh
   cp .env.example .env
   ```
   Edita `.env` con tus credenciales:
   ```
   OPENAI_API_KEY=tu_clave_openai
   DEEPSEEK_API_KEY=tu_clave_deepseek
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   DEEPSEEK_BASE_URL=https://api.deepseek.com
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
   - Usando un proveedor específico:
     ```sh
     python app.py --provider chatgpt -k 5
     ```
     o
     ```sh
     python app.py --provider deepseek -k 5
     ```
   - Modo A/B (compara ChatGPT vs DeepSeek):  
     Ejecuta la misma consulta en ambos proveedores y muestra latencias y respuestas:
     ```sh
     python app.py --ab -k 5
     ```

4. **Modo batch / evaluación**  
   Ejecuta la evaluación automática sobre `eval/gold_set.jsonl`:
   ```sh
   python app.py --batch -k 5
   ```

## Evaluación

El módulo `eval/` contiene el gold set y scripts de evaluación. El modo batch calcula métricas básicas (Exact Match, cobertura de citas, Prec@k, latencia). Recomendado: ejecutar evaluación en ambos proveedores para comparar costos y calidad.

## Personalización

- Agrega nuevos documentos en `data/raw/` y actualiza `data/sources.csv`.
- Ajusta el prompt en [`rag/prompts.py`](rag/prompts.py).
- Cambia el modelo de embeddings en `.env` si usa otro modelo de sentence-transformers.

## Notas operativas

- Asegúrate de que `.env` no esté versionado (`.gitignore` debe incluirlo).
- Para trazabilidad, valida que `data/sources.csv` incluya `doc_id,title,filename,url,effective_date`.
- El modo A/B facilita medición de latencia y comparación; los resultados se imprimen en consola y pueden guardarse si se extiende `app.py`.

## Créditos

Desarrollado para la UFRO.  
Modelos y APIs: OpenAI, DeepSeek, Sentence Transformers, FAISS.

## Licencia

Uso académico y educativo.