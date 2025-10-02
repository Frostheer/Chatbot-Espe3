import os
import json
import re 
import subprocess
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import traceback
from datetime import datetime

# Configurar logging detallado SIN EMOJIS para Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chatbot.log', encoding='utf-8')  # UTF-8 para emojis
    ]
)
logger = logging.getLogger(__name__)

# Función para logging con timestamp - SIN EMOJIS EN ARCHIVO
def log_with_timestamp(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{timestamp}] [{level}]"
    
    # Mostrar con emojis en consola
    print(f"{prefix} {message}")
    
    # Guardar sin emojis en archivo de log
    clean_message = remove_emojis(message)
    if level == "ERROR":
        logger.error(clean_message)
    elif level == "WARNING":
        logger.warning(clean_message)
    else:
        logger.info(clean_message)

def remove_emojis(text):
    """Remueve emojis para el archivo de log."""
    emoji_map = {
        '🔄': '[PROCESANDO]',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '🚀': '[INICIO]',
        '🔧': '[CONFIG]',
        '📄': '[ARCHIVO]',
        '📁': '[CARPETA]',
        '📊': '[DATOS]',
        '🔍': '[BUSCAR]',
        '⚠️': '[AVISO]',
        '🎉': '[EXITO]',
        '🌐': '[SERVIDOR]',
        '👋': '[SALIDA]',
        '🤖': '[LLM]',
        '🐋': '[DOCKER]',
        '🔴': '[ERROR_CRITICO]'
    }
    
    for emoji, replacement in emoji_map.items():
        text = text.replace(emoji, replacement)
    
    return text

# Aseguramos que los módulos locales se puedan importar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
log_with_timestamp(f"Python path actualizado: {sys.path[-1]}")

# --- IMPORTACIONES RAG REALES ---
log_with_timestamp("🔄 Iniciando importaciones de módulos RAG...")
try:
    from providers.chatgpt import ChatGPTProvider
    log_with_timestamp("✅ ChatGPTProvider importado")
    from providers.deepseek import DeepSeekProvider
    log_with_timestamp("✅ DeepSeekProvider importado")
    from rag.retrieve import Retriever
    log_with_timestamp("✅ Retriever importado")
    from rag.prompts import build_messages
    log_with_timestamp("✅ build_messages importado")
    from providers.base import Provider
    log_with_timestamp("✅ Provider base importado")
    log_with_timestamp("🎉 Todas las importaciones completadas exitosamente")
except ImportError as e:
    log_with_timestamp(f"❌ Error al importar módulos RAG/Providers: {e}", "ERROR")
    log_with_timestamp("Estructura de carpetas esperada:", "ERROR")
    log_with_timestamp("  providers/chatgpt.py", "ERROR")
    log_with_timestamp("  providers/deepseek.py", "ERROR")
    log_with_timestamp("  rag/retrieve.py", "ERROR")
    log_with_timestamp("  rag/prompts.py", "ERROR")
    exit(1)

# --- FUNCIONES DE INICIALIZACIÓN AUTOMÁTICA ---

def check_data_structure():
    """Verifica si la estructura de datos necesaria existe."""
    log_with_timestamp("🔍 Verificando estructura de datos...")
    base_path = Path(__file__).parent
    log_with_timestamp(f"Directorio base: {base_path}")
    
    checks = {
        "data_raw": base_path / "data" / "raw",
        "data_processed": base_path / "data" / "processed", 
        "faiss_index": base_path / "data" / "index.faiss",
        "sources_csv": base_path / "data" / "sources.csv"
    }
    
    status = {}
    for name, path in checks.items():
        log_with_timestamp(f"Verificando {name}: {path}")
        if name == "faiss_index":
            exists = path.exists()
            if exists:
                size = path.stat().st_size
                log_with_timestamp(f"  ✅ {name} existe ({size} bytes)")
            else:
                log_with_timestamp(f"  ❌ {name} no existe")
            status[name] = exists
        else:
            exists = path.exists()
            has_content = False
            if exists:
                try:
                    has_content = any(path.iterdir())
                    if has_content:
                        file_count = len(list(path.iterdir()))
                        log_with_timestamp(f"  ✅ {name} existe con {file_count} archivos")
                    else:
                        log_with_timestamp(f"  ⚠️ {name} existe pero está vacío")
                except Exception as e:
                    log_with_timestamp(f"  ❌ Error al verificar {name}: {e}")
            else:
                log_with_timestamp(f"  ❌ {name} no existe")
            status[name] = exists and has_content
    
    return status

def run_data_preparation():
    """Ejecuta automáticamente los scripts de preparación de datos."""
    base_path = Path(__file__).parent
    
    log_with_timestamp("🔄 Preparando datos automáticamente...")
    
    try:
        # 1. Verificar PDFs en data/raw/
        raw_path = base_path / "data" / "raw"
        log_with_timestamp(f"Verificando PDFs en: {raw_path}")
        
        if not raw_path.exists():
            log_with_timestamp("❌ Directorio data/raw/ no existe", "ERROR")
            log_with_timestamp(f"Creando directorio: {raw_path}")
            raw_path.mkdir(parents=True, exist_ok=True)
            return False
            
        pdf_files = list(raw_path.glob("*.pdf"))
        if not pdf_files:
            log_with_timestamp("❌ No se encontraron PDFs en data/raw/", "ERROR")
            log_with_timestamp(f"   Por favor, coloque archivos PDF en: {raw_path}")
            return False
        
        pdf_count = len(pdf_files)
        log_with_timestamp(f"✅ Encontrados {pdf_count} archivos PDF:")
        for pdf in pdf_files[:5]:  # Mostrar primeros 5
            log_with_timestamp(f"   📄 {pdf.name}")
        if pdf_count > 5:
            log_with_timestamp(f"   ... y {pdf_count - 5} más")
        
        # 2. Ejecutar ingesta (procesa PDFs)
        log_with_timestamp("🔄 Ejecutando ingesta de documentos...")
        ingest_script = base_path / "rag" / "ingest.py"
        log_with_timestamp(f"Script de ingesta: {ingest_script}")
        
        if ingest_script.exists():
            log_with_timestamp(f"Ejecutando: python {ingest_script}")
            result = subprocess.run([sys.executable, str(ingest_script)], 
                                  capture_output=True, text=True, cwd=base_path)
            
            log_with_timestamp(f"Return code: {result.returncode}")
            if result.stdout:
                log_with_timestamp(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                log_with_timestamp(f"STDERR:\n{result.stderr}")
                
            if result.returncode != 0:
                log_with_timestamp(f"❌ Error en ingesta (código {result.returncode})", "ERROR")
                return False
            log_with_timestamp("✅ Ingesta completada exitosamente")
        else:
            log_with_timestamp(f"❌ No se encontró el script: {ingest_script}", "ERROR")
            return False
        
        # 3. Generar embeddings e índice FAISS
        log_with_timestamp("🔄 Generando embeddings e índice FAISS...")
        embed_script = base_path / "rag" / "embed.py"
        log_with_timestamp(f"Script de embeddings: {embed_script}")
        
        if embed_script.exists():
            log_with_timestamp(f"Ejecutando: python {embed_script}")
            result = subprocess.run([sys.executable, str(embed_script)], 
                                  capture_output=True, text=True, cwd=base_path)
            
            log_with_timestamp(f"Return code: {result.returncode}")
            if result.stdout:
                log_with_timestamp(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                log_with_timestamp(f"STDERR:\n{result.stderr}")
                
            if result.returncode != 0:
                log_with_timestamp(f"❌ Error en embeddings (código {result.returncode})", "ERROR")
                return False
            log_with_timestamp("✅ Embeddings e índice FAISS generados exitosamente")
        else:
            log_with_timestamp(f"❌ No se encontró el script: {embed_script}", "ERROR")
            return False
        
        # 4. Verificar que se crearon los archivos
        log_with_timestamp("🔍 Verificando archivos generados...")
        status = check_data_structure()
        if not all([status["data_processed"], status["faiss_index"]]):
            log_with_timestamp("❌ No se generaron todos los archivos necesarios", "ERROR")
            log_with_timestamp(f"Estado: {status}")
            return False
        
        log_with_timestamp("✅ Preparación de datos completada exitosamente")
        return True
        
    except Exception as e:
        log_with_timestamp(f"❌ Error durante la preparación de datos: {e}", "ERROR")
        traceback.print_exc()
        return False

def auto_setup_data():
    """Configuración automática de datos con verificaciones inteligentes."""
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("🚀 INICIALIZACIÓN AUTOMÁTICA DEL SISTEMA RAG")
    log_with_timestamp("="*60)
    
    # Verificar estructura actual
    status = check_data_structure()
    
    log_with_timestamp("\n📋 Estado actual de los datos:")
    log_with_timestamp(f"   📁 data/raw/: {'✅' if status['data_raw'] else '❌'}")
    log_with_timestamp(f"   📁 data/processed/: {'✅' if status['data_processed'] else '❌'}")
    log_with_timestamp(f"   📊 index.faiss: {'✅' if status['faiss_index'] else '❌'}")
    log_with_timestamp(f"   📄 sources.csv: {'✅' if status['sources_csv'] else '❌'}")
    
    # Si todo está listo, no hacer nada
    if all([status["data_processed"], status["faiss_index"], status["sources_csv"]]):
        log_with_timestamp("\n✅ Todos los datos están listos. No es necesario procesar.")
        return True
    
    # Si faltan datos, intentar prepararlos automáticamente
    log_with_timestamp("\n⚠️  Faltan datos procesados. Iniciando preparación automática...")
    
    return run_data_preparation()

# --- CONFIGURACIÓN E INICIALIZACIÓN GLOBAL ---

# Cargar variables de entorno (claves API)
log_with_timestamp("🔄 Cargando variables de entorno...")
load_dotenv()
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY") 
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Verificar que al menos una clave API esté disponible
if not CHATGPT_API_KEY and not DEEPSEEK_API_KEY:
    log_with_timestamp("❌ FATAL ERROR: No se encontraron API keys en el archivo .env", "ERROR")
    log_with_timestamp("Se necesita al menos CHATGPT_API_KEY o DEEPSEEK_API_KEY")
    log_with_timestamp("Variables de entorno disponibles:")
    for key in os.environ:
        if 'API' in key.upper():
            log_with_timestamp(f"  {key}: {'[CONFIGURADA]' if os.environ[key] else '[VACÍA]'}")
    sys.exit(1)
else:
    if CHATGPT_API_KEY:
        log_with_timestamp(f"✅ CHATGPT_API_KEY encontrada (longitud: {len(CHATGPT_API_KEY)})")
    if DEEPSEEK_API_KEY:
        log_with_timestamp(f"✅ DEEPSEEK_API_KEY encontrada (longitud: {len(DEEPSEEK_API_KEY)})")

# Instancia de Flask (DEBE estar expuesta globalmente)
log_with_timestamp("🔄 Inicializando Flask...")
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False 
CORS(app)
log_with_timestamp("✅ Flask inicializado con CORS")

# Estado de la aplicación
RAG_SYSTEM = None

def has_evidence(response_text: str) -> bool:
    """Heurística simple para detectar citas/evidencia en la respuesta."""
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

def initialize_providers():
    """Inicializa ambos proveedores de LLM si están disponibles."""
    providers = {}
    
    # Intentar inicializar ChatGPT
    if CHATGPT_API_KEY:
        try:
            log_with_timestamp("🔄 Inicializando proveedor ChatGPT...")
            chatgpt_provider = ChatGPTProvider()
            providers['chatgpt'] = chatgpt_provider
            log_with_timestamp(f"✅ ChatGPT inicializado: {chatgpt_provider.name}")
        except Exception as e:
            log_with_timestamp(f"❌ Error al inicializar ChatGPT: {e}", "ERROR")
    
    # Intentar inicializar DeepSeek
    if DEEPSEEK_API_KEY:
        try:
            log_with_timestamp("🔄 Inicializando proveedor DeepSeek...")
            deepseek_provider = DeepSeekProvider()
            providers['deepseek'] = deepseek_provider
            log_with_timestamp(f"✅ DeepSeek inicializado: {deepseek_provider.name}")
        except Exception as e:
            log_with_timestamp(f"❌ Error al inicializar DeepSeek: {e}", "ERROR")
    
    if not providers:
        raise Exception("No se pudo inicializar ningún proveedor LLM")
    
    log_with_timestamp(f"🎉 Proveedores disponibles: {list(providers.keys())}")
    return providers

def chat_with_fallback(providers, messages, user_prompt):
    """Intenta usar ChatGPT primero, si falla usa DeepSeek."""
    
    # Orden de preferencia: ChatGPT primero, luego DeepSeek
    provider_order = ['chatgpt', 'deepseek']
    last_error = None
    
    for provider_name in provider_order:
        if provider_name not in providers:
            log_with_timestamp(f"⚠️ Proveedor {provider_name} no disponible, saltando...")
            continue
            
        provider = providers[provider_name]
        
        try:
            log_with_timestamp(f"🤖 Intentando generar respuesta con {provider.name}...")
            response_text = provider.chat(messages)
            
            # Verificar que la respuesta sea válida y no sea un mensaje de error
            if response_text and len(response_text.strip()) > 0 and not response_text.startswith("ERROR:"):
                log_with_timestamp(f"✅ Respuesta generada exitosamente con {provider.name} ({len(response_text)} caracteres)")
                return response_text, provider.name
            else:
                raise Exception(f"Respuesta inválida o vacía del proveedor {provider.name}")
            
        except Exception as e:
            last_error = e
            log_with_timestamp(f"❌ Error con {provider.name}: {e}", "ERROR")
            log_with_timestamp(f"Detalles del error: {str(e)}")
            
            # Continuar con el siguiente proveedor si no es el último
            remaining_providers = [p for p in provider_order if p in providers]
            if provider_name != remaining_providers[-1]:
                log_with_timestamp(f"🔄 Intentando con el siguiente proveedor...")
                continue
    
    # Si llegamos aquí, todos los proveedores fallaron
    log_with_timestamp("❌ Todos los proveedores LLM fallaron", "ERROR")
    raise Exception(f"Todos los proveedores LLM fallaron. Último error: {last_error}")

def initialize_rag():
    """Carga los modelos, el índice FAISS y los proveedores LLM una sola vez al inicio."""
    global RAG_SYSTEM
    log_with_timestamp("\n" + "="*60)
    log_with_timestamp("🔧 INICIANDO SISTEMA RAG")
    log_with_timestamp("="*60)
    
    try:
        # Inicializar el Retriever (carga FAISS y embeddings)
        log_with_timestamp("🔄 Cargando Retriever (FAISS/Embeddings)...")
        log_with_timestamp("Verificando archivos necesarios para Retriever...")
        
        base_path = Path(__file__).parent
        faiss_path = base_path / "data" / "index.faiss"
        sources_path = base_path / "data" / "sources.csv"
        
        log_with_timestamp(f"FAISS index: {faiss_path} ({'✅' if faiss_path.exists() else '❌'})")
        log_with_timestamp(f"Sources CSV: {sources_path} ({'✅' if sources_path.exists() else '❌'})")
        
        retriever = Retriever()
        log_with_timestamp("✅ Retriever cargado exitosamente.")
        
        # Inicializar TODOS los proveedores LLM disponibles
        providers = initialize_providers()
        
        RAG_SYSTEM = {
            "status": "ready", 
            "retriever": retriever, 
            "providers": providers,  # Cambiado: ahora es un diccionario de proveedores
            "k_default": 5 
        }
        log_with_timestamp("🎉 Sistema RAG cargado exitosamente. Listo para consultas.")
        
    except Exception as e:
        log_with_timestamp(f"❌ ERROR FATAL al cargar el sistema RAG: {e}", "ERROR")
        traceback.print_exc(file=sys.stderr)
        RAG_SYSTEM = {"status": "failed", "error": str(e)}

# --- RUTAS DE FLASK ---
@app.route('/')
def index():
    """Ruta principal para la interfaz HTML."""
    log_with_timestamp("📄 Acceso a ruta principal (/)")
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Endpoint API para procesar la consulta RAG del usuario."""
    log_with_timestamp("🔄 Nueva consulta recibida en /query")
    
    # Verificación de estado de inicialización
    if not RAG_SYSTEM or RAG_SYSTEM.get("status") != "ready":
        error_msg = "Sistema RAG no inicializado o fallido. Revise los logs."
        log_with_timestamp(f"❌ {error_msg}", "ERROR")
        log_with_timestamp(f"Estado RAG_SYSTEM: {RAG_SYSTEM}")
        return jsonify({"error": error_msg}), 503

    try:
        data = request.json
        log_with_timestamp(f"Datos recibidos: {data}")
        
        user_prompt = data.get('prompt')
        k = RAG_SYSTEM['k_default'] 

        if not user_prompt:
            log_with_timestamp("❌ Consulta vacía recibida", "WARNING")
            return jsonify({"response": "Por favor, ingrese una pregunta."})
        
        log_with_timestamp(f"Procesando consulta: '{user_prompt[:100]}...'")
        
        retriever = RAG_SYSTEM['retriever']
        providers = RAG_SYSTEM['providers']  

        # 1. Recuperación de Contexto (Retrieve)
        log_with_timestamp(f"🔍 Recuperando contexto (k={k})...")
        distances, context_list = retriever.retrieve(user_prompt, k=k)
        log_with_timestamp(f"Contextos recuperados: {len(context_list)}")
        
        context_str = "\n".join(context_list)
        
        if not context_str:
            abst_msg = "🔴 ERROR: No se pudo recuperar contexto. No hay documentos relacionados."
            log_with_timestamp(f"❌ {abst_msg}", "WARNING")
            return jsonify({"response": abst_msg, "sources": [], "retrieved_chunks": []})

        log_with_timestamp(f"Contexto total: {len(context_str)} caracteres")

        # 2. Extracción de Fuentes Limpias
        log_with_timestamp("🔄 Extrayendo fuentes...")
        sources = [ctx.strip().split(': ')[0] for ctx in context_list]
        log_with_timestamp(f"Fuentes extraídas: {sources}")
        
        # 3. Generación (Generate) - CON FALLBACK
        log_with_timestamp("🤖 Generando respuesta con sistema de fallback...")
        messages = build_messages(context=context_str, query=user_prompt)
        log_with_timestamp(f"Mensajes para LLM: {len(messages)} mensajes")
        
        # NUEVO: Usar el sistema de fallback
        response_text, provider_used = chat_with_fallback(providers, messages, user_prompt)
        log_with_timestamp(f"✅ Respuesta generada con {provider_used}")
        
        # 4. Verificación de Evidencia
        log_with_timestamp("🔍 Verificando evidencia en respuesta...")
        if not has_evidence(response_text):
            abst_msg = ("No encontrado en normativa UFRO. No hay evidencia en los documentos recuperados. "
                        "Consulte la unidad correspondiente (Secretaría Académica / Dirección de Estudios).")
            log_with_timestamp(f"⚠️ Abstención forzada para: {user_prompt}", "WARNING")
            return jsonify({
                "response": abst_msg, 
                "sources": sources, 
                "retrieved_chunks": context_list,
                "provider_used": provider_used
            })
        
        # 5. Devolver respuesta y fuentes
        log_with_timestamp("✅ Consulta procesada exitosamente")
        return jsonify({
            "response": response_text,
            "sources": sources,
            "retrieved_chunks": context_list,
            "provider_used": provider_used  # NUEVO: Incluir qué proveedor se usó
        })

    except Exception as e:
        log_with_timestamp(f"❌ Error en la consulta: {e}", "ERROR")
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": f"Ocurrió un error interno: {str(e)}"}), 500

# --- BLOQUE PRINCIPAL MEJORADO ---

if __name__ == "__main__":
    log_with_timestamp("\n🚀 INICIANDO CHATBOT RAG - MODO LOCAL/DEBUG")
    log_with_timestamp("="*70)
    
    # 1. Configuración automática de datos
    if not auto_setup_data():
        log_with_timestamp("\n❌ FALLO EN LA PREPARACIÓN DE DATOS", "ERROR")
        log_with_timestamp("   No se puede continuar sin los datos procesados.")
        log_with_timestamp("   Verifique que hay PDFs en data/raw/ y que los scripts funcionan correctamente.")
        sys.exit(1)
    
    # 2. Inicializar sistema RAG
    initialize_rag()
    
    # 3. Verificar que todo esté listo
    if not RAG_SYSTEM or RAG_SYSTEM.get("status") != "ready":
        log_with_timestamp("\n❌ FALLO EN LA INICIALIZACIÓN DEL SISTEMA RAG", "ERROR")
        error_detail = RAG_SYSTEM.get('error', 'Desconocido') if RAG_SYSTEM else 'Sistema no inicializado'
        log_with_timestamp(f"   Error: {error_detail}")
        sys.exit(1)
    
    # 4. Iniciar servidor Flask
    log_with_timestamp("\n🌐 INICIANDO SERVIDOR FLASK")
    log_with_timestamp("="*70)
    log_with_timestamp("   URL: http://localhost:5000")
    log_with_timestamp("   Endpoint API: http://localhost:5000/query")
    log_with_timestamp("   Presiona Ctrl+C para detener")
    log_with_timestamp("="*70)
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000)
    except KeyboardInterrupt:
        log_with_timestamp("\n\n👋 Servidor detenido por el usuario.")
    except Exception as e:
        log_with_timestamp(f"\n❌ Error al iniciar el servidor: {e}", "ERROR")
        sys.exit(1)
else:
    # Cuando se ejecuta con Gunicorn (producción)
    log_with_timestamp("🐋 MODO PRODUCCIÓN - GUNICORN")
    if not auto_setup_data():
        log_with_timestamp("❌ FALLO EN LA PREPARACIÓN DE DATOS EN PRODUCCIÓN", "ERROR")
        sys.exit(1)
    initialize_rag()