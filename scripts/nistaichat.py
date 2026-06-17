import os
import re
import sys
import json
import time
import logging
from typing import List, Dict, Tuple, Any

import ollama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# --- LOGGING CONFIGURATION ---
LOG_FILE_PATH = r"C:\aibot\logs\chathistory\nistchat.log"
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- CONSTANTS & SYSTEM CONFIG ---
EMBEDDING_MODEL   = "bge-m3:567m"
READER_MODEL      = "qwen2.5:3b"
VECTOR_DB_PATH    = r"C:\aibot\data\04_vector_storage"
PARENT_STORE_PATH = r"C:\aibot\data\04_vector_storage\parent_store.jsonl"

PROMPTS = {
    "generator": (
        "You are a NIST AI policy advisor. If the answer isn't in the context, say you don't know.\n\n"
        "- Keep the answer as short as possible in plain text, no headers or bold.\n"
        "- Use a numbered list only for ordered steps or requirements.\n"
        "- Answer as close as possible to the original policy language.\n"
        "- If the context only partially covers the topic, refer to the source documents listed below.\n\n"
        "CONTEXT:\n{context}\n\n"
        "CONVERSATION:\n{chat_history}\n\n"
        "QUESTION: {question}"
    ),
    # MiniCheck native format: Document + Claim → "Yes" (supported) / "No" (not supported)
    "verifier": "Document: {chunk}\nClaim: {claim}",
    # Citation-forced retry prompt — no chat history, forces grounding before answering
    "retry": (
        "You are a NIST AI policy advisor. Answer ONLY using the context below.\n\n"
        "- Find the exact sentence in the context that addresses the question.\n"
        "- Answer using only that sentence, in plain text.\n"
        "- If no sentence directly addresses it, say you don't know.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}"
    ),
}

VERIFIER_MODEL  = "bespoke-minicheck"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

HIGH_RISK_PATTERNS = [
    r"which\s+\w+\s+answers",
    r"which\s+\w+\s+describes",
    r"which\s+\w+\s+property",
    r"what\s+are\s+the\s+(six|five|four|three|two|\d+)\s+\w+",
    r"what\s+is\s+\w+\s+defined",
    r"how\s+does\s+nist\s+\w+\s+define",
    r"when\s+a\s+user\s+asks",
]

LIST_PATTERNS = [
    r"what\s+are\s+the\s+(six|five|four|three|two|\d+)\s+\w+",
    r"list\s+the\s+\w+",
    r"name\s+the\s+(six|five|four|three|two|\d+)",
]


class NistChatEngine:
    """
    RAG Application Engine utilizing a multi-pass Hybrid Retriever Layer 
    (Dense FAISS + Sparse BM25 via RRF) and a sliding window character-bound memory buffer.
    """

    def __init__(self):
        self._validate_paths()
        
        # 1. Initialize Dense Retrieval Layer
        logger.info("Initializing Dense Retrieval Layer (FAISS)...")
        start_load = time.time()
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vector_db = self._load_vector_db()
        
        # 2. Extract internal nodes to construct Sparse Retrieval Layer natively in RAM
        logger.info("Extracting index fragments to construct Sparse Retrieval Layer (BM25)...")
        self.docstore_items = list(self.vector_db.docstore._dict.values())
        
        # RegEx processing to ensure punctuation doesn't ruin token extraction accuracy
        tokenized_corpus = [self._clean_tokenize(doc.page_content) for doc in self.docstore_items]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 3. Load parent store for full-text context lookup
        logger.info("Loading parent store...")
        self.parent_store: Dict[str, dict] = {}
        with open(PARENT_STORE_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    self.parent_store[record["id"]] = record
        logger.info(f"Parent store loaded: {len(self.parent_store)} records.")

        # 4. Cross-encoder reranker
        logger.info("Loading cross-encoder reranker...")
        self._cross_encoder = CrossEncoder(RERANKER_MODEL)
        logger.info(f"Cross-encoder ready: {RERANKER_MODEL}")

        # 5. Chat memory initialization
        self.chat_history: List[Dict[str, str]] = []

        load_time = time.time() - start_load
        logger.info(f"Engine ready — {len(self.docstore_items)} child chunks, {len(self.parent_store)} parent records in {load_time:.2f}s.")
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
            allow_dangerous_deserialization=True  
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
                logger.warning(f"Memory ceiling met. Gracefully truncating history.")
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
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': user_message},
            ],
            options={
                'temperature':    0.1,
                'num_predict':    600,
                'num_ctx':        4096,
                'num_thread':     6,
                'num_batch':      1024,
                'repeat_penalty': 1.15,
                'stop':           ["\n\nHuman:", "User:"],
            }
        )
        raw = response['message']['content']
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
            return "\n".join(lines[1:]).strip()  # skip the "Source: ..." header line
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
                    options={
                        "temperature": 0.0,
                        "num_predict": 5,
                        "num_ctx":     2048,
                        "num_thread":  6,
                    }
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
                supported += 1  # conservative: don't penalise on error

        faithful = supported >= not_supported
        logger.info(f"MiniCheck overall: supported={supported}, not_supported={not_supported}, faithful={faithful}")
        return faithful

    def _assemble_hybrid_context(self, question: str, top_k: int = 6) -> Tuple[str, List[Dict[str, str]]]:
        """
        Executes parallel dense/sparse search and ensembles results via RRF.
        Table rows are expanded with their immediate neighbours (prev + next row).
        """
        cleaned_query = question.lower().strip()

        # Dense Retrieval
        dense_results = self.vector_db.similarity_search_with_score(question, k=top_k * 2)

        # Sparse Retrieval
        tokenized_query = self._clean_tokenize(cleaned_query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_ranked_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:top_k * 2]

        rrf_scores = {}
        doc_lookup = {}
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

        # Expand pool before reranking — gives reranker access to candidates beyond top_k cut
        rrf_ranked  = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        rerank_pool = rrf_ranked[:top_k + 4]

        # Cross-encoder scores each (query, child_chunk_text) pair in a single batch
        pairs     = [(question, doc_lookup[uid].page_content) for uid, _ in rerank_pool]
        ce_scores = self._cross_encoder.predict(pairs)

        # Re-sort by cross-encoder score, cut back to top_k
        reranked    = sorted(zip(ce_scores, rerank_pool), key=lambda x: x[0], reverse=True)
        sorted_docs = [(uid, rrf_score) for _, (uid, rrf_score) in reranked[:top_k]]
        logger.info(f"Cross-encoder reranked {len(rerank_pool)} candidates → top {top_k}")

        seen_ids:      set              = set()
        context_parts: List[str]        = []
        source_metadata: List[Dict]     = []

        for uid, _ in sorted_docs:
            doc         = doc_lookup[uid]
            parent_link = doc.metadata.get("parent_link", uid)
            if parent_link in seen_ids:
                continue
            seen_ids.add(parent_link)

            # Look up full parent text and rich metadata from parent store
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

            # Expand to immediate neighbours — look up parent text from parent store
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
                question=question
            )

            draft_answer = self._call_llm("You are a precise NIST AI policy advisor.", full_prompt)
            generation_time = time.time() - generation_start

            risk = self._classify_query_risk(question)
            if risk == "HIGH":
                top_chunk = self._extract_top_chunk(context)
                faithful  = self._verify_against_context(top_chunk, draft_answer)
                if not faithful:
                    logger.warning("Verifier flagged contradiction — retrying with citation prompt.")
                    retry_msg = PROMPTS["retry"].format(context=context, question=question)
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

if __name__ == "__main__":
    engine = NistChatEngine()
    print("\n" + "="*80)
    print("      NIST AI CYBERSECURITY ADVISOR READY (HYBRID MULTI-PASS RETRIEVAL)      ")
    print("="*80)
    
    while True:
        try:
            query = input("\n[User]: ").strip()
            if query.lower() == 'exit': break
            if query.lower() == 'clear':
                engine.clear_history()
                print("\n[System]: Conversation memory flushed.")
                continue
            if not query: continue

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
                    table_name = source["table_id"].split("::")[-1] if "::" in source["table_id"] else source.get("section", "")
                    print(f"        Table: {table_name}  ({source['table_total']} rows total)")
            print("-" * 80)
            
        except KeyboardInterrupt:
            break