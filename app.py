import argparse
from dotenv import load_dotenv
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.retrieve import Retriever
from rag.prompts import build_messages
from providers.base import Provider

# Cargar variables de entorno
load_dotenv()

def run_rag_query(provider: Provider, retriever: Retriever, query: str, k: int):
    """Ejecuta el flujo completo RAG para un proveedor y query."""
    print(f"\n--- 🤖 {provider.name} - Procesando Consulta ---")
    
    # 1. Recuperación de Contexto (Retrieve)
    distances, context_list = retriever.retrieve(query, k=k)
    context_str = "\n".join(context_list)
    
    if not context_str:
        print("🔴 ERROR: No se pudo recuperar contexto. Verifique la ingesta y el índice.")
        return

    print(f"🔎 Contexto Recuperado ({len(context_list)} chunks):")
    for i, ctx in enumerate(context_list):
         # Muestra solo el título y el inicio del texto
         print(f"   [{i+1}] {ctx.strip().split(': ')[0]}...")

    # 2. Generación (Generate)
    messages = build_messages(context=context_str, query=query)
    
    # Usar el adapter del proveedor
    response = provider.chat(messages)

    # 3. Mostrar Resultado
    print("\n" + "="*50)
    print(f"💬 Respuesta del Asistente ({provider.name}):")
    print(response)
    print("="*50 + "\n")


def cli_interactive(retriever: Retriever, provider_name: str, k: int):
    """Modo interactivo CLI."""
    if provider_name.lower() == 'chatgpt':
        provider = ChatGPTProvider()
    elif provider_name.lower() == 'deepseek':
        provider = DeepSeekProvider()
    else:
        print("Proveedor no válido. Usando ChatGPT por defecto.")
        provider = ChatGPTProvider()
    
    print("=============================================")
    print("     Asistente CHATBOT Normativa UFRO        ")
    print("=============================================")
    print(f"Proveedor Activo: {provider.name}. K (chunks): {k}")
    print("Escribe 'salir' para terminar.")

    while True:
        query = input("\nPregunta sobre normativa UFRO > ")
        if query.lower() == 'salir':
            break
        if not query.strip():
            continue
            
        run_rag_query(provider, retriever, query, k)


def main():
    parser = argparse.ArgumentParser(description="Asistente CHATBOT RAG sobre normativa UFRO.")
    parser.add_argument(
        "--provider",
        type=str,
        default="chatgpt",
        choices=["chatgpt", "deepseek"],
        help="Proveedor LLM a utilizar (chatgpt o deepseek)."
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Número de documentos (chunks) a recuperar (top-k)."
    )
    # Tarea H6/H9: Modo Batch para evaluación
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Ejecutar en modo batch usando eval/gold_set.jsonl (requiere evaluate.py)."
    )
    
    args = parser.parse_args()
    
    # Inicializar el Retriever (carga FAISS y embeddings)
    retriever = Retriever()

    if args.batch:
        # Aquí se integrará el módulo de evaluación (H6)
        print("Modo Batch: Ejecutando evaluación del Gold Set (necesitas implementar eval/evaluate.py)")
        # from eval.evaluate import run_evaluation
        # run_evaluation(retriever, args.k)
        pass # Placeholder
    else:
        cli_interactive(retriever, args.provider, args.k)


if __name__ == '__main__':
    main()