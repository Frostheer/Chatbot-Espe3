import os
import json
import time
import csv
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Intentar importar matplotlib solo si está disponible
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    from providers.chatgpt import ChatGPTProvider
    from providers.deepseek import DeepSeekProvider
except Exception:
    ChatGPTProvider = None
    DeepSeekProvider = None

try:
    from rag.retrieve import Retriever
except Exception:
    Retriever = None

ROOT = Path(".")
GOLD_PATH = ROOT / "eval" / "gold_set.jsonl"
OUT_DIR = ROOT / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results_detailed.csv"
SUMMARY_CSV = OUT_DIR / "results_summary.csv"

EMBED_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENAI_RATE_PER_1K = float(os.getenv("OPENAI_RATE_PER_1K", "0.002"))
DEEPSEEK_RATE_PER_1K = float(os.getenv("DEEPSEEK_RATE_PER_1K", "0.0015"))

ABSTENTION_PHRASES = ["no encontrado en normativa ufro", "no puedo encontrar", "no hay evidencia", "no se encontró"]
CITATION_RE = re.compile(r"\[.+?\]|\b(pág|página|pag)\b", re.IGNORECASE)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

def load_gold(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def detect_citation(text: str) -> bool:
    if not text:
        return False
    if CITATION_RE.search(text):
        return True
    if "http" in text.lower():
        return True
    return False

def has_expected_ref_in_text(text: str, expected_refs: List[str]) -> bool:
    if not expected_refs:
        return False
    text_l = (text or "").lower()
    for r in expected_refs:
        if r.lower() in text_l:
            return True
    return False

def compute_precision_at_k(retrieved_items: List[Any], expected_refs: List[str], k: int) -> float:
    if not retrieved_items:
        return 0.0
    hits = 0
    for item in retrieved_items[:k]:
        content = ""
        if isinstance(item, dict):
            for key in ("metadata", "source", "doc_id", "title", "text"):
                if key in item and item[key]:
                    content += " " + str(item[key])
        else:
            content = str(item)
        if has_expected_ref_in_text(content, expected_refs):
            hits += 1
    return hits / k if k > 0 else 0.0

def estimate_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    if not text:
        return 0
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    words = len(text.split())
    return int(max(1, words / 0.75))

def cost_from_tokens(tokens: int, provider: str) -> float:
    rate = OPENAI_RATE_PER_1K if provider.lower() == "chatgpt" else DEEPSEEK_RATE_PER_1K
    return tokens / 1000.0 * rate

def call_provider(provider_instance, messages: List[Dict[str, str]]) -> Tuple[str, float, int]:
    start = time.time()
    try:
        resp = provider_instance.chat(messages)
    except TypeError:
        resp = provider_instance.chat(messages=messages)
    latency = time.time() - start

    if isinstance(resp, dict):
        if "content" in resp:
            text = resp["content"]
        elif "choices" in resp and resp["choices"]:
            ch = resp["choices"][0]
            if isinstance(ch, dict) and "message" in ch and "content" in ch["message"]:
                text = ch["message"]["content"]
            else:
                text = str(ch)
        else:
            text = json.dumps(resp)
    else:
        text = str(resp)

    prompt_text = " ".join([m.get("content", "") for m in messages])
    tokens = estimate_tokens(prompt_text + "\n" + text)
    return text, latency, tokens

def build_messages_simple(context: str, question: str) -> List[Dict[str,str]]:
    system = {"role": "system", "content": "Eres un asistente que responde preguntas citando la normativa UFRO. Si no hay soporte en las fuentes recuperadas, abstente indicando 'No encontrado en normativa UFRO' y sugiere la oficina correspondiente."}
    user = {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}\nResponde citando las fuentes (ej: [Título del documento, pág. X])."}
    return [system, user]

def run_evaluation(retriever, k: int = 5, out_dir: str = "eval"):
    gold_path = Path(out_dir) / "gold_set.jsonl"
    if not gold_path.exists():
        print(f"Gold set no encontrado en {gold_path}")
        return
    gold = load_gold(gold_path)
    model = SentenceTransformer(EMBED_MODEL)

    providers = []
    if ChatGPTProvider:
        providers.append(("ChatGPT", ChatGPTProvider()))
    if DeepSeekProvider:
        providers.append(("DeepSeek", DeepSeekProvider()))
    if not providers:
        print("No se encontraron providers importables. Asegura providers/chatgpt.py y providers/deepseek.py")
        return

    # Inicializar estructuras de resultados
    rows = []
    summary = {}

    # OOD questions para robustez
    ood_questions = [
        "¿Cuál es el clima habitual en Temuco en marzo?",
        "¿Cómo reiniciar un servidor Linux?",
        "¿Cuál es el capital de Francia?"
    ]
    ood_results = {pname: {"total": 0, "abstained": 0} for pname, _ in providers}

    # Evaluación principal
    for item in gold:
        qid = item.get("id")
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_refs = item.get("expected_refs", [])

        t0 = time.time()
        try:
            retrieved = retriever.retrieve(question, k=k)
        except TypeError:
            retrieved = retriever.retrieve(question, k)
        retrieve_latency = time.time() - t0

        retrieved_items = []
        if isinstance(retrieved, tuple) and len(retrieved) == 2:
            scores, texts = retrieved
            for s, t in zip(scores, texts):
                retrieved_items.append({"score": float(s), "text": t})
        elif isinstance(retrieved, list):
            retrieved_items = retrieved
        else:
            retrieved_items = [retrieved]

        context_parts = []
        for it in retrieved_items[:k]:
            if isinstance(it, dict) and "text" in it:
                context_parts.append(it["text"])
            else:
                context_parts.append(str(it))
        context = "\n---\n".join(context_parts)

        for pname, pinst in providers:
            messages = build_messages_simple(context, question)
            resp_text, llm_latency, tokens = call_provider(pinst, messages)
            total_latency = retrieve_latency + llm_latency
            em = int(normalize_text(resp_text) == normalize_text(expected_answer))
            try:
                vecs = model.encode([expected_answer or "", resp_text])
                sim = float(cosine_similarity([vecs[0]], [vecs[1]])[0][0]) if len(vecs) >= 2 else 0.0
            except Exception:
                sim = 0.0
            coverage = int(has_expected_ref_in_text(resp_text, expected_refs) or detect_citation(resp_text))
            p_at_k = compute_precision_at_k(retrieved_items, expected_refs, k)
            cost = cost_from_tokens(tokens, pname)
            possible_hallucination = 1 if (coverage == 0 and len(resp_text.split()) > 15) else 0
            abstained = any(phrase in resp_text.lower() for phrase in ABSTENTION_PHRASES)

            rows.append({
                "question_id": qid,
                "provider": pname,
                "question": question,
                "expected_refs": ";".join(expected_refs) if expected_refs else "",
                "em": em,
                "similarity": f"{sim:.4f}",
                "coverage": coverage,
                "precision_at_k": f"{p_at_k:.4f}",
                "retrieve_latency_s": f"{retrieve_latency:.3f}",
                "llm_latency_s": f"{llm_latency:.3f}",
                "total_latency_s": f"{total_latency:.3f}",
                "tokens_est": tokens,
                "cost_usd": f"{cost:.6f}",
                "possible_hallucination": possible_hallucination,
                "abstained": int(abstained),
                "response_excerpt": resp_text[:300].replace("\n", " ")
            })

            if pname not in summary:
                summary[pname] = {
                    "queries": 0, "em_sum": 0, "sim_sum": 0.0, "cov_sum": 0, "p_at_k_sum": 0.0,
                    "retrieve_latency_sum": 0.0, "llm_latency_sum": 0.0, "cost_sum": 0.0,
                    "hallucinations": 0, "abstentions": 0
                }
            s = summary[pname]
            s["queries"] += 1
            s["em_sum"] += em
            s["sim_sum"] += sim
            s["cov_sum"] += coverage
            s["p_at_k_sum"] += p_at_k
            s["retrieve_latency_sum"] += retrieve_latency
            s["llm_latency_sum"] += llm_latency
            s["cost_sum"] += cost
            s["hallucinations"] += possible_hallucination
            s["abstentions"] += int(abstained)

    # corrida OOD simple (H8)
    for q in ood_questions:
        try:
            retrieved = retriever.retrieve(q, k=k)
        except Exception:
            retrieved = []
        for pname, pinst in providers:
            messages = build_messages_simple("", q)
            resp_text, llm_latency, tokens = call_provider(pinst, messages)
            abstained = any(phrase in resp_text.lower() for phrase in ABSTENTION_PHRASES)
            ood_results[pname]["total"] += 1
            if abstained:
                ood_results[pname]["abstained"] += 1

    # Guardar filas detalladas
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["question_id","provider","question","expected_refs","em","similarity","coverage","precision_at_k","retrieve_latency_s","llm_latency_s","total_latency_s","tokens_est","cost_usd","possible_hallucination","abstained","response_excerpt"]

    with open(OUT_DIR / "results_detailed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Guardar resumen agregado
    summary_rows = []
    for pname, stats in summary.items():
        q = stats["queries"] or 1
        summary_rows.append({
            "provider": pname,
            "queries": q,
            "EM_rate": f"{stats['em_sum']/q:.4f}",
            "avg_similarity": f"{stats['sim_sum']/q:.4f}",
            "coverage_rate": f"{stats['cov_sum']/q:.4f}",
            "avg_precision_at_k": f"{stats['p_at_k_sum']/q:.4f}",
            "avg_retrieve_latency_s": f"{stats['retrieve_latency_sum']/q:.3f}",
            "avg_llm_latency_s": f"{stats['llm_latency_sum']/q:.3f}",
            "avg_cost_usd": f"{stats['cost_sum']/q:.6f}",
            "hallucination_rate": f"{stats['hallucinations']/q:.4f}",
            "abstention_rate": f"{stats['abstentions']/q:.4f}"
        })

    with open(OUT_DIR / "results_summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)

    # Guardar OOD resultados
    with open(OUT_DIR / "ood_results.json", "w", encoding="utf-8") as f:
        json.dump(ood_results, f, indent=2, ensure_ascii=False)

    # Generar gráfico comparativo si hay datos y matplotlib está disponible
    if not rows:
        print("No hay resultados para graficar. Se omite generación de gráficos.")
    elif not MATPLOTLIB_AVAILABLE:
        print("matplotlib no encontrado: se omite generación de gráficos (instalar 'matplotlib' para habilitar).")
    else:
        try:
            providers_list = [r["provider"] for r in summary_rows]
            avg_latency = [float(r["avg_retrieve_latency_s"])+float(r["avg_llm_latency_s"]) for r in summary_rows]
            cite_rate = [float(r["coverage_rate"]) for r in summary_rows]

            fig, ax1 = plt.subplots(figsize=(6,4))
            ax1.bar(providers_list, avg_latency, color='C0', alpha=0.7)
            ax1.set_ylabel('Latencia media (s)', color='C0')
            ax2 = ax1.twinx()
            ax2.plot(providers_list, cite_rate, color='C1', marker='o')
            ax2.set_ylabel('Tasa de respuestas con cita', color='C1')
            plt.title('Comparativa ChatGPT vs DeepSeek')
            plt.tight_layout()
            plot_path = Path(out_dir) / "comparison_plot.png"
            fig.savefig(plot_path)
            plt.close(fig)
            print(f"Gráfico comparativo guardado en: {plot_path}")
        except Exception as e:
            print("No fue posible generar el gráfico comparativo:", e)

    print(f"Evaluación completada. Detalles en {OUT_DIR.resolve()}")