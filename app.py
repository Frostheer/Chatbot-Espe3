import os
import json
import re 
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys 
import traceback # <-- NUEVA IMPORTACIÓN PARA TRAZAS DETALLADAS

# Aseguramos que los módulos locales se puedan importar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- IMPORTACIONES RAG REALES ---
# Las importaciones que necesita tu sistema (asumiendo que están en carpetas 'providers' y 'rag')
try:
    from providers.chatgpt import ChatGPTProvider
    from providers.deepseek import DeepSeekProvider
    from rag.retrieve import Retriever
    from rag.prompts import build_messages
    from providers.base import Provider
except ImportError as e:
    # Este bloque maneja fallos si las carpetas/archivos de tu proyecto no se encuentran.
    print(f"Error al importar módulos RAG/Providers: {e}. Verifique la estructura de carpetas.")
    exit(1)


# --- 1. CONFIGURACIÓN E INICIALIZACIÓN GLOBAL (EJECUTADO POR GUNICORN) ---

# Cargar variables de entorno (claves API)
load_dotenv()
API_KEY = os.getenv("CHATGPT_API_KEY") # <-- ¡CLAVE DE API CORREGIDA AQUÍ!

# Asegúrate de que la clave API esté disponible
if not API_KEY:
    print("FATAL ERROR: No se encontró CHATGPT_API_KEY en el archivo .env")

# Instancia de Flask (DEBE estar expuesta globalmente)
app = Flask(__name__)
# Solución al Unicode: Deshabilita el escapado de caracteres que no son ASCII
app.config['JSON_AS_ASCII'] = False 
CORS(app) # Habilitar CORS para peticiones front-end

# Estado de la aplicación
RAG_SYSTEM = None

def has_evidence(response_text: str) -> bool:
    """Heurística simple para detectar citas/evidencia en la respuesta (Tu función original)."""
    if not response_text:
        return False
    patterns = [
        r'https?://',             
        r'\[.+?\]',               
        r'\b(pág|página|pag)\b',  
        r'\bPágina\b', 
        r'Ref(erencia)?[:\-]', 
        r'Doc(?:umento)?[:\-]'
    ]
    for p in patterns:
        if re.search(p, response_text, re.IGNORECASE):
            return True
    return False

def initialize_rag():
    """Carga los modelos, el índice FAISS y el proveedor LLM una sola vez al inicio."""
    global RAG_SYSTEM
    print("--- INICIANDO SISTEMA RAG ---")
    
    try:
        # Inicializar el Retriever (carga FAISS y embeddings)
        # Esto solía estar en tu `main()`
        retriever = Retriever()
        print("Retriever (FAISS/Embeddings) cargado.")
        
        # Inicializar el Proveedor LLM por defecto
        # Usamos ChatGPTProvider como proveedor por defecto
        provider = ChatGPTProvider() 
        print(f"Proveedor LLM activo: {provider.name}.")
        
        RAG_SYSTEM = {
            "status": "ready", 
            "retriever": retriever, 
            "provider": provider,
            "k_default": 5 
        }
        print(f"Sistema RAG cargado exitosamente. Listo para consultas.")
        
    except Exception as e:
        print(f"ERROR FATAL al cargar el sistema RAG: {e}")
        # AÑADIDO: Imprimir el traceback completo para depuración
        traceback.print_exc(file=sys.stderr)
        RAG_SYSTEM = {"status": "failed", "error": str(e)}

# LLAMADA DE INICIALIZACIÓN: Gunicorn ejecuta esto al iniciar el worker.
initialize_rag()


# --- 2. RUTAS DE FLASK ---

@app.route('/')
def index():
    """Ruta principal para la interfaz HTML (requiere templates/index.html)."""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Endpoint API para procesar la consulta RAG del usuario."""
    # Verificación de estado de inicialización
    if not RAG_SYSTEM or RAG_SYSTEM.get("status") != "ready":
        # Este es el error que recibiste antes: se dispara si initialize_rag falló.
        return jsonify({"error": "Sistema RAG no inicializado o fallido. Revise los logs."}), 503

    try:
        data = request.json
        user_prompt = data.get('prompt')
        k = RAG_SYSTEM['k_default'] 

        if not user_prompt:
            return jsonify({"response": "Por favor, ingrese una pregunta."})
        
        retriever = RAG_SYSTEM['retriever']
        provider = RAG_SYSTEM['provider']

        # 1. Recuperación de Contexto (Retrieve)
        distances, context_list = retriever.retrieve(user_prompt, k=k)
        context_str = "\n".join(context_list)
        
        if not context_str:
            abst_msg = "🔴 ERROR: No se pudo recuperar contexto. No hay documentos relacionados."
            return jsonify({"response": abst_msg, "sources": [], "retrieved_chunks": []})

        # 2. Extracción de Fuentes Limpias (MOVIDA AQUÍ)
        # Extrae el título del documento (todo lo que está antes de ': ')
        sources = [ctx.strip().split(': ')[0] for ctx in context_list]
        
        # 3. Generación (Generate)
        messages = build_messages(context=context_str, query=user_prompt)
        response_text = provider.chat(messages)
        
        # 4. Verificación de Evidencia (Lógica de Abstención - Tu función original)
        if not has_evidence(response_text):
            abst_msg = ("No encontrado en normativa UFRO. No hay evidencia en los documentos recuperados. "
                        "Consulte la unidad correspondiente (Secretaría Académica / Dirección de Estudios).")
            print(f"Abstención forzada para: {user_prompt}")
            # AHORA DEVOLVEMOS LAS FUENTES REALES
            return jsonify({"response": abst_msg, "sources": sources, "retrieved_chunks": context_list})
        
        # 5. Devolver respuesta y fuentes
        # La lista 'sources' ya está definida arriba.
        return jsonify({
            "response": response_text,
            "sources": sources,
            "retrieved_chunks": context_list # <-- Muestra los fragmentos para depuración
        })

    except Exception as e:
        # AÑADIDO: Registrar errores detallados en la ruta /query también
        print(f"Error en la consulta: {e}")
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": f"Ocurrió un error interno: {str(e)}"}), 500


# --- 3. BLOQUE DE PRUEBAS LOCALES (IGNORADO POR GUNICORN) ---

if __name__ == "__main__":
    # Esta parte se usaba antes para ejecutar la CLI y ahora solo sirve para lanzar Flask localmente sin Gunicorn
    print("Modo Local/Debug de Flask. Usar gunicorn en producción.")
    app.run(debug=True, host='0.0.0.0', port=5000)
