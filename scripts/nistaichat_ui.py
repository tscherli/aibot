import os
import re
import sys
import json
import time
import glob
import logging
import argparse
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Tuple

import gradio as gr
import ollama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# --- CONSTANTS ---
LOG_DIR           = r"C:\aibot\logs\chathistory"
EMBEDDING_MODEL   = "bge-m3:567m"
READER_MODEL      = "qwen2.5:3b"
VECTOR_DB_PATH    = r"C:\aibot\data\04_vector_storage"
PARENT_STORE_PATH = r"C:\aibot\data\04_vector_storage\parent_store.jsonl"
VERIFIER_MODEL    = "bespoke-minicheck"
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUESTIONS_CONFIG_PATH = r"C:\aibot\data\06_questions_ui\questions_config.json"

_LATENCY_STATS: dict = {}  # populated once at UI startup by _load_latency_stats()

def _load_latency_stats() -> dict:
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "nistchat_*.log")))
    lines: list = []
    for lf in reversed(log_files):
        try:
            with open(lf, encoding="utf-8") as fh:
                lines = fh.readlines() + lines
        except OSError:
            continue
        if len(lines) >= 600:
            break

    low_totals: list = []
    high_totals: list = []
    current_risk = None

    for line in lines:
        rm = re.search(r"_classify_query_risk → (HIGH|LOW)", line)
        if rm:
            current_risk = rm.group(1)
            continue
        cm = re.search(r"Total Latency: ([\d.]+)s", line)
        if cm and current_risk:
            total = float(cm.group(1))
            if current_risk == "LOW":
                low_totals.append(total)
            else:
                high_totals.append(total)
            current_risk = None

    def _p80(lst):
        if not lst:
            return None
        s = sorted(lst)
        return round(s[int(len(s) * 0.8)], 1)

    stats = {
        "low_total":  _p80(low_totals),
        "high_total": _p80(high_totals),
    }
    logging.getLogger(__name__).info(f"Latency stats loaded (P80): {stats}")
    return stats

PROMPTS = {
    "generator": (
        "You are a NIST AI policy advisor. If the answer isn't in the context, say you don't know.\n\n"
        "- Be concise. Answer in plain text, no headers or bold.\n"
        "- If the answer is a list of items, output every item from the context using bullet points (-).\n"
        "- Use a numbered list only for ordered steps or requirements.\n"
        "- Answer as close as possible to the original policy language.\n"
        "- If the context only partially covers the topic, refer to the source documents listed below.\n\n"
        "CONTEXT:\n{context}\n\n"
        "CONVERSATION:\n{chat_history}\n\n"
        "QUESTION: {question}"
    ),
    "verifier": "Document: {chunk}\nClaim: {claim}",
    "retry": (
        "You are a NIST AI policy advisor. Answer ONLY using the context below.\n\n"
        "- Find the exact sentence in the context that addresses the question.\n"
        "- Answer using only that sentence, in plain text.\n"
        "- If no sentence directly addresses it, say you don't know.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}"
    ),
}

HIGH_RISK_PATTERNS = [
    r"which\s+\w+\s+answers",
    r"which\s+\w+\s+describes",
    r"which\s+\w+\s+property",
    r"what\s+are\s+the\s+(six|five|four|three|two|\d+)\s+\w+",
    r"what\s+is\s+\w+\s+defined",
    r"how\s+does\s+nist\s+\w+\s+define",
    r"when\s+a\s+user\s+asks",
    r"what\s+are\s+(examples|some\s+examples)",
    r"(give|provide|list)\s+(me\s+)?(some\s+)?examples",
    r"examples\s+of\s+\w+",
]

LIST_PATTERNS = [
    r"what\s+are\s+the\s+(six|five|four|three|two|\d+)\s+\w+",
    r"list\s+the\s+\w+",
    r"name\s+the\s+(six|five|four|three|two|\d+)",
    r"what\s+are\s+(examples|some\s+examples)",
    r"(give|provide|list)\s+(me\s+)?(some\s+)?examples",
]


# --- QUESTION CONFIG LOADER ---

def load_questions() -> list:
    """
    Load topic categories from questions_config.json.
    Each question entry may be a plain string or a dict {"label": str, "prompt": str}.
    Raises SystemExit if the file is missing or malformed.
    """
    try:
        with open(QUESTIONS_CONFIG_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        cats = data["categories"]
        for c in cats:
            if not isinstance(c.get("questions"), list):
                raise ValueError(f"Missing questions list in category '{c.get('name')}'")
        logger.info(f"Loaded {len(cats)} topic categories from questions_config.json")
        return cats
    except FileNotFoundError:
        logger.error(f"questions_config.json not found at {QUESTIONS_CONFIG_PATH}")
        raise SystemExit(1)
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse questions_config.json: {e}")
        raise SystemExit(1)


def _resolve_question(q) -> Tuple[str, str, str]:
    """
    Normalize a question entry to (label, prompt, answer).
    Plain string          → label==prompt, answer=="". Goes through RAG pipeline.
    Dict with 'answer'    → answer returned verbatim, no LLM call.
    Dict with 'prompt'    → enriched hidden prompt sent to LLM.
    If both keys present, 'answer' takes precedence.
    """
    if isinstance(q, str):
        return q, q, ""
    label = q.get("label", "")
    answer = q.get("answer", "")
    if answer:
        return label, "", answer
    return label, q.get("prompt") or label, ""


# --- DAILY-ROTATING FILE HANDLER ---

class DailyRotatingFileHandler(logging.FileHandler):
    """
    Writes to nistchat_YYYYMMDD.log in log_dir.
    At the first log call after midnight, closes the old file and opens a new one
    named for the new date — no external scheduler needed.
    """

    def __init__(self, log_dir: str, prefix: str = "nistchat", encoding: str = "utf-8"):
        self.log_dir = log_dir
        self.prefix = prefix
        os.makedirs(log_dir, exist_ok=True)
        self._current_date = datetime.now().date()
        super().__init__(self._log_path(), mode="a", encoding=encoding)

    def _log_path(self) -> str:
        return os.path.join(
            self.log_dir,
            f"{self.prefix}_{datetime.now().strftime('%Y%m%d')}.log",
        )

    def emit(self, record: logging.LogRecord) -> None:
        today = datetime.now().date()
        if today != self._current_date:
            self.close()
            self.baseFilename = self._log_path()
            self.stream = self._open()
            self._current_date = today
        super().emit(record)


# --- LOGGING SETUP ---

def _setup_logging() -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = DailyRotatingFileHandler(LOG_DIR)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    log = logging.getLogger("nistaichat")
    log.setLevel(logging.INFO)
    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    return log


logger = _setup_logging()


# --- ENGINE ---

class NistChatEngine:
    """
    RAG Application Engine utilizing a multi-pass Hybrid Retriever Layer
    (Dense FAISS + Sparse BM25 via RRF) and a sliding window character-bound memory buffer.
    """

    def __init__(self):
        self._validate_paths()

        logger.info("Initializing Dense Retrieval Layer (FAISS)...")
        start_load = time.time()
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vector_db = self._load_vector_db()

        logger.info("Extracting index fragments to construct Sparse Retrieval Layer (BM25)...")
        self.docstore_items = list(self.vector_db.docstore._dict.values())
        tokenized_corpus = [self._clean_tokenize(doc.page_content) for doc in self.docstore_items]
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info("Loading parent store...")
        self.parent_store: Dict[str, dict] = {}
        with open(PARENT_STORE_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    self.parent_store[record["id"]] = record
        logger.info(f"Parent store loaded: {len(self.parent_store)} records.")

        logger.info("Loading cross-encoder reranker...")
        self._cross_encoder = CrossEncoder(RERANKER_MODEL)
        logger.info(f"Cross-encoder ready: {RERANKER_MODEL}")

        self.chat_history: List[Dict[str, str]] = []

        load_time = time.time() - start_load
        logger.info(
            f"Engine ready — {len(self.docstore_items)} child chunks, "
            f"{len(self.parent_store)} parent records in {load_time:.2f}s."
        )
        logger.info(f"Reader model: {READER_MODEL}")

    def _validate_paths(self) -> None:
        if not os.path.exists(VECTOR_DB_PATH):
            raise FileNotFoundError(f"Vector Database directory not found at: {VECTOR_DB_PATH}")
        if not os.path.exists(PARENT_STORE_PATH):
            raise FileNotFoundError(f"Parent store not found at: {PARENT_STORE_PATH}")

    def _load_vector_db(self) -> FAISS:
        return FAISS.load_local(
            VECTOR_DB_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _clean_tokenize(self, text: str) -> List[str]:
        clean_text = re.sub(r'[\[\]\(\)\.,:;!?\-*]', ' ', text.lower())
        return [token for token in clean_text.split(" ") if token.strip()]

    def format_chat_history(self, max_char_limit: int = 4000) -> str:
        if not self.chat_history:
            return "No previous conversation."

        formatted_turns = []
        current_length = 0

        for turn in reversed(self.chat_history):
            turn_string = f"Human: {turn['human']}\nAssistant: {turn['assistant']}"
            turn_length = len(turn_string)
            if current_length + turn_length > max_char_limit:
                logger.warning("Memory ceiling met. Gracefully truncating history.")
                break
            formatted_turns.insert(0, turn_string)
            current_length += turn_length

        return "\n\n".join(formatted_turns)

    def add_to_history(self, human_message: str, assistant_message: str) -> None:
        self.chat_history.append({"human": human_message, "assistant": assistant_message})

    def clear_history(self) -> None:
        self.chat_history = []
        logger.info("Chat history cleared.")

    def _call_llm(self, system_message: str, user_message: str, model: str = READER_MODEL) -> str:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message},
            ],
            options={
                "temperature":    0.1,
                "num_predict":    600,
                "num_ctx":        4096,
                "num_thread":     6,
                "num_batch":      1024,
                "repeat_penalty": 1.15,
                "stop":           ["\n\nHuman:", "User:"],
            },
        )
        raw = response["message"]["content"]
        raw = re.sub(r'\*\*(.*?)\*\*', r'\1', raw)
        raw = re.sub(r'^#{1,6}\s+', '', raw, flags=re.MULTILINE)
        return raw

    def _is_list_query(self, question: str) -> bool:
        lowered = question.lower()
        return any(re.search(p, lowered) for p in LIST_PATTERNS)

    def _classify_query_risk(self, question: str) -> str:
        lowered = question.lower()
        for pattern in HIGH_RISK_PATTERNS:
            if re.search(pattern, lowered):
                logger.info(f"_classify_query_risk → HIGH  (matched: {pattern!r})")
                return "HIGH"
        logger.info("_classify_query_risk → LOW")
        return "LOW"

    def _extract_top_chunk(self, context: str) -> str:
        parts = context.split("\n--- Source:")
        if len(parts) >= 2:
            lines = parts[1].split("\n")
            return "\n".join(lines[1:]).strip()
        return context[:800]

    def _verify_against_context(self, top_chunk: str, answer: str) -> bool:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
        if not sentences:
            return True

        supported = 0
        not_supported = 0

        for sentence in sentences:
            prompt = PROMPTS["verifier"].format(chunk=top_chunk, claim=sentence)
            try:
                response = ollama.chat(
                    model=VERIFIER_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.0, "num_predict": 5, "num_ctx": 2048, "num_thread": 6},
                )
                verdict_raw = response["message"]["content"].strip().lower()
                verdict = "supported" if verdict_raw.startswith("yes") else "not supported"
                logger.info(f"MiniCheck verdict: [{verdict}]  claim={sentence[:80]!r}")
                if verdict == "supported":
                    supported += 1
                else:
                    not_supported += 1
            except Exception as e:
                logger.warning(f"MiniCheck call failed for sentence — skipping: {e}")
                supported += 1

        faithful = supported >= not_supported
        logger.info(f"MiniCheck overall: supported={supported}, not_supported={not_supported}, faithful={faithful}")
        return faithful

    def _assemble_hybrid_context(self, question: str, top_k: int = 6) -> Tuple[str, List[Dict[str, str]]]:
        cleaned_query = question.lower().strip()

        dense_results = self.vector_db.similarity_search_with_score(question, k=top_k * 2)

        tokenized_query = self._clean_tokenize(cleaned_query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_ranked_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:top_k * 2]

        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, any] = {}
        constant_k = 60

        def get_unique_id(doc):
            return f"{doc.metadata.get('parent_link', 'unknown')}_{doc.page_content[:50]}"

        for rank, (doc, _) in enumerate(dense_results, start=1):
            uid = get_unique_id(doc)
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + (1.0 / (constant_k + rank))
            doc_lookup[uid] = doc

        for rank, index in enumerate(sparse_ranked_indices, start=1):
            if bm25_scores[index] <= 0:
                continue
            doc = self.docstore_items[index]
            uid = get_unique_id(doc)
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + (1.0 / (constant_k + rank))
            doc_lookup[uid] = doc

        rrf_ranked  = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        rerank_pool = rrf_ranked[:top_k + 4]

        pairs     = [(question, doc_lookup[uid].page_content) for uid, _ in rerank_pool]
        ce_scores = self._cross_encoder.predict(pairs)

        reranked    = sorted(zip(ce_scores, rerank_pool), key=lambda x: x[0], reverse=True)
        sorted_docs = [(uid, rrf_score) for _, (uid, rrf_score) in reranked[:top_k]]
        logger.info(f"Cross-encoder reranked {len(rerank_pool)} candidates → top {top_k}")

        seen_ids:        set        = set()
        context_parts:   List[str]  = []
        source_metadata: List[Dict] = []

        for uid, _ in sorted_docs:
            doc         = doc_lookup[uid]
            parent_link = doc.metadata.get("parent_link", uid)
            if parent_link in seen_ids:
                continue
            seen_ids.add(parent_link)

            parent_record = self.parent_store.get(parent_link, {})
            parent_text   = parent_record.get("text", doc.page_content)
            parent_meta   = parent_record.get("metadata", {})

            source_name  = parent_meta.get("source",       doc.metadata.get("source", "NIST Document"))
            chapter      = parent_meta.get("chapter",      "")
            section      = parent_meta.get("section",      "")
            content_type = parent_meta.get("content_type", doc.metadata.get("content_type", "prose"))
            table_id     = parent_meta.get("table_id",     "")
            table_row    = parent_meta.get("table_row",    0)
            sibling_ids  = parent_meta.get("sibling_ids",  [])
            table_total  = parent_meta.get("table_total",  len(sibling_ids) + 1 if sibling_ids else 0)

            location_parts = [p for p in [chapter, section] if p]
            location       = " > ".join(location_parts) if location_parts else ""
            short_id_match = re.search(r'NIST\s+AI\s+[\d]+-[\d]+', source_name)
            short_source   = short_id_match.group(0) if short_id_match else source_name
            header         = f"--- Source: {short_source}{' | ' + location if location else ''} ---"

            context_parts.append(f"{header}\n{parent_text}")
            source_metadata.append({
                "source_title": source_name,
                "chapter":      chapter,
                "section":      section,
                "content_type": content_type,
                "table_id":     table_id,
                "table_total":  table_total,
            })

            if content_type == "table" and sibling_ids and table_row > 0:
                neighbor_ids = []
                if table_row >= 2:
                    neighbor_ids.append(sibling_ids[table_row - 2])
                if table_row - 1 < len(sibling_ids):
                    neighbor_ids.append(sibling_ids[table_row - 1])

                for nb_id in neighbor_ids:
                    if nb_id in seen_ids:
                        continue
                    nb_text = self.parent_store.get(nb_id, {}).get("text", "")
                    if not nb_text:
                        continue
                    seen_ids.add(nb_id)
                    context_parts.append(f"  [table context — adjacent row]\n{nb_text}")

        return "\n\n".join(context_parts), source_metadata

    def ask(self, question: str) -> Tuple[str, List[Dict[str, str]]]:
        try:
            start_total = time.time()
            formatted_history = self.format_chat_history(max_char_limit=1000)
            logger.info(f"User Query: {question}")
            retrieval_start = time.time()
            context, sources = self._assemble_hybrid_context(question)
            retrieval_time = time.time() - retrieval_start

            logger.info("Generator: Processing context + RRF rankings...")
            generation_start = time.time()

            full_prompt = PROMPTS["generator"].format(
                chat_history=formatted_history,
                context=context,
                question=question,
            )
            draft_answer   = self._call_llm("You are a precise NIST AI policy advisor.", full_prompt)
            generation_time = time.time() - generation_start

            risk = self._classify_query_risk(question)
            if risk == "HIGH":
                top_chunk = self._extract_top_chunk(context)
                faithful  = self._verify_against_context(top_chunk, draft_answer)
                if not faithful:
                    logger.warning("Verifier flagged contradiction — retrying with citation prompt.")
                    retry_msg    = PROMPTS["retry"].format(context=context, question=question)
                    draft_answer = self._call_llm("You are a precise NIST AI policy advisor.", retry_msg)

            if self._is_list_query(question):
                draft_answer += (
                    "\n\nNote: This list may be incomplete. "
                    "Refer to the original NIST source document for the full list."
                )

            self.add_to_history(question, draft_answer)

            total_time = time.time() - start_total
            logger.info(
                f"Query Complete | Total Latency: {total_time:.2f}s | "
                f"Hybrid Retrieval Match (R): {retrieval_time:.2f}s | "
                f"LLM Generation Inference (G): {generation_time:.2f}s"
            )

            return draft_answer, sources

        except Exception as e:
            logger.error(f"Error during Hybrid RAG execution loop: {e}", exc_info=True)
            return "I encountered an error while processing your request.", []


# --- CONSOLE MODE ---

def run_console(engine: NistChatEngine) -> None:
    print("\n" + "=" * 80)
    print("      NIST AI CYBERSECURITY ADVISOR READY (HYBRID MULTI-PASS RETRIEVAL)      ")
    print("=" * 80)

    while True:
        try:
            query = input("\n[User]: ").strip()
            if query.lower() == "exit":
                break
            if query.lower() == "clear":
                engine.clear_history()
                print("\n[System]: Conversation memory flushed.")
                continue
            if not query:
                continue

            draft_answer, sources = engine.ask(query)
            print(f"\n[Assistant]:\n{draft_answer}")

            print("\nSOURCES:")
            for i, source in enumerate(sources, 1):
                line = f"   [{i}] {source['source_title']}"
                if source.get("chapter"):
                    line += f" | {source['chapter']}"
                line += f"  ({source['content_type'].upper()})"
                print(line)
                if source.get("table_id") and source.get("table_total", 0) > 1:
                    table_name = (
                        source["table_id"].split("::")[-1]
                        if "::" in source["table_id"]
                        else source.get("section", "")
                    )
                    print(f"        Table: {table_name}  ({source['table_total']} rows total)")
            print("-" * 80)

        except KeyboardInterrupt:
            break


# --- GRADIO UI MODE ---

def _format_sources(sources: List[Dict]) -> str:
    if not sources:
        return ""
    lines = []
    for i, s in enumerate(sources, 1):
        line = f"[{i}] {s['source_title']}"
        if s.get("chapter"):
            line += f" | {s['chapter']}"
        line += f" ({s['content_type'].upper()})"
        lines.append(line)
        if s.get("table_id") and s.get("table_total", 0) > 1:
            table_name = (
                s["table_id"].split("::")[-1]
                if "::" in s["table_id"]
                else s.get("section", "")
            )
            lines.append(f"    Table: {table_name}  ({s['table_total']} rows total)")
    return "\n\nSources:\n" + "\n".join(lines)


def run_gradio_ui(engine: NistChatEngine, port: int = 7860, share: bool = False) -> None:
    global _LATENCY_STATS
    _LATENCY_STATS = _load_latency_stats()

    # Load categories from JSON config (exits with error if file missing or malformed).
    topic_categories = load_questions()

    def respond(
        message: str, history: list, enriched_prompt: str, direct_answer: str
    ):
        if not message.strip():
            yield history, "", "", "", ""
            return

        if direct_answer.strip():
            # Bypass RAG and LLM entirely — return the JSON answer verbatim.
            logger.info(f"Direct answer served for: {message!r}")
            engine.add_to_history(message, direct_answer.strip())
            new_history = history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": direct_answer.strip()},
            ]
            yield new_history, "", "Answer loaded from config", "", ""
            return

        # Use the enriched hidden prompt if set by a button click; otherwise use the typed message.
        backend_query = enriched_prompt.strip() if enriched_prompt.strip() else message

        is_high = any(re.search(p, message, re.IGNORECASE) for p in HIGH_RISK_PATTERNS)
        est_total = _LATENCY_STATS.get("high_total" if is_high else "low_total")

        # Run engine.ask() in a background thread so we can yield live status updates.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(engine.ask, backend_query)
            t0 = time.time()

            while not future.done():
                elapsed = time.time() - t0
                if est_total is not None:
                    status = f"Processing query... {elapsed:.0f}s / ~{est_total:.0f}s"
                else:
                    status = f"Processing query... {elapsed:.0f}s elapsed"
                # Yield original history unchanged — chatbot must not show partial state.
                yield history, "", status, "", ""
                time.sleep(0.5)

        answer, sources = future.result()
        elapsed_total = round(time.time() - t0)
        full_response = answer + _format_sources(sources)
        # Always store the visible label in chat history, never the backend query.
        new_history = history + [{"role": "user", "content": message},
                                  {"role": "assistant", "content": full_response}]
        yield new_history, "", f"Answer completed in {elapsed_total}s", "", ""

    def clear_fn() -> Tuple[list, str, str, str, str]:
        engine.clear_history()
        return [], "", "", "", ""

    with gr.Blocks(title="NIST AI Cybersecurity Advisor") as demo:
        gr.Markdown(
            "## NIST AI Cybersecurity Advisor\n"
            "*Hybrid Retrieval: FAISS + BM25 + Cross-Encoder Reranker*\n\n"
            "Ask about AI risk management, secure development practices, adversarial ML attacks, "
            "GenAI risks, or post-deployment monitoring — all grounded in NIST source documents."
        )

        # Server-side states — never transmitted to the browser; cleared after each respond() call.
        # hidden_prompt: enriched backend query sent to the LLM instead of the visible label.
        # hidden_answer: verbatim answer returned directly, bypassing RAG and LLM entirely.
        hidden_prompt = gr.State(value="")
        hidden_answer = gr.State(value="")

        # --- TWO-LEVEL TOPIC NAVIGATION ---
        # All component refs collected here; click handlers bound after msg_box is defined.
        cat_buttons: List[gr.Button] = []
        all_sub_panels: List[gr.Group] = []
        back_btns: List[gr.Button] = []
        topic_buttons: List[Tuple] = []  # 4-tuples: (button, label, prompt, answer)

        with gr.Group(visible=True) as category_panel:
            gr.Markdown("**With which topic would you like to get started?**")
            with gr.Row():
                for cat in topic_categories[:4]:
                    btn = gr.Button(f"{cat['emoji']}  {cat['name']}", variant="secondary", scale=1)
                    cat_buttons.append(btn)
            with gr.Row():
                for cat in topic_categories[4:]:
                    btn = gr.Button(f"{cat['emoji']}  {cat['name']}", variant="secondary", scale=1)
                    cat_buttons.append(btn)

        for cat in topic_categories:
            with gr.Group(visible=False) as sub_panel:
                back_btn = gr.Button("← Back to Topics", size="sm", variant="secondary")
                back_btns.append(back_btn)
                gr.Markdown(f"### {cat['emoji']}  {cat['name']}\n*{cat['intro']}*")
                questions = cat["questions"]
                mid = len(questions) // 2 + len(questions) % 2
                with gr.Row():
                    with gr.Column():
                        for q in questions[:mid]:
                            label, prompt, answer = _resolve_question(q)
                            q_btn = gr.Button(label, size="sm")
                            topic_buttons.append((q_btn, label, prompt, answer))
                    with gr.Column():
                        for q in questions[mid:]:
                            label, prompt, answer = _resolve_question(q)
                            q_btn = gr.Button(label, size="sm")
                            topic_buttons.append((q_btn, label, prompt, answer))
            all_sub_panels.append(sub_panel)

        chatbot = gr.Chatbot(
            label="Conversation",
            height=480,
        )

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Ask a NIST AI policy question…",
                label="",
                scale=8,
                autofocus=True,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        clear_btn = gr.Button("Clear Conversation History", variant="secondary")
        status_md = gr.Markdown("", elem_id="query-status")

        # Event binding — all visibility outputs share the same list.
        all_vis_outputs = [category_panel] + all_sub_panels

        def make_show_fn(idx: int):
            def fn():
                return [gr.update(visible=False)] + [
                    gr.update(visible=(i == idx)) for i in range(len(all_sub_panels))
                ]
            return fn

        def back_fn():
            return [gr.update(visible=True)] + [gr.update(visible=False)] * len(all_sub_panels)

        for i, cat_btn in enumerate(cat_buttons):
            cat_btn.click(fn=make_show_fn(i), inputs=[], outputs=all_vis_outputs)

        for back_btn in back_btns:
            back_btn.click(fn=back_fn, inputs=[], outputs=all_vis_outputs)

        # Default-argument capture prevents late-binding of loop vars.
        # msg_box gets the visible label; hidden_prompt/hidden_answer are server-only state.
        for q_btn, label, prompt, answer in topic_buttons:
            q_btn.click(
                fn=lambda _lbl=label, _prm=prompt, _ans=answer: (_lbl, _prm, _ans),
                inputs=[],
                outputs=[msg_box, hidden_prompt, hidden_answer],
            )

        send_btn.click(respond, [msg_box, chatbot, hidden_prompt, hidden_answer], [chatbot, msg_box, status_md, hidden_prompt, hidden_answer])
        msg_box.submit(respond, [msg_box, chatbot, hidden_prompt, hidden_answer], [chatbot, msg_box, status_md, hidden_prompt, hidden_answer])
        clear_btn.click(clear_fn, None, [chatbot, msg_box, status_md, hidden_prompt, hidden_answer])

    logger.info(f"Starting Gradio UI on port {port} (share={share})")
    demo.queue()
    demo.launch(server_port=port, share=share, inbrowser=True, theme=gr.themes.Soft())


# --- ENTRY POINT ---

def main() -> None:
    parser = argparse.ArgumentParser(description="NIST AI Cybersecurity Advisor")
    parser.add_argument(
        "--mode",
        choices=["console", "ui"],
        default="ui",
        help="Run mode: 'console' for CLI, 'ui' for Gradio web interface (default: ui)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio server port, only used with --mode ui (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link, only used with --mode ui",
    )
    args = parser.parse_args()

    engine = NistChatEngine()

    if args.mode == "ui":
        run_gradio_ui(engine, port=args.port, share=args.share)
    else:
        run_console(engine)


if __name__ == "__main__":
    main()
