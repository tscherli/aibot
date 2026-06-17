"""
Step-by-step RAG pipeline diagnostic.

Replays one succeeded case (Q2, faithfulness=1.0) and one failed case (Q3,
faithfulness=0.0) from eval run 20260605_183036, printing every stage of the
pipeline so the failure cause is visible without guesswork.

Usage:
    python scripts/06_diagnose.py
"""

import os
import re
import sys
import json
import time
import textwrap
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nistaichat import NistChatEngine, PROMPTS, READER_MODEL

EVAL_STORE_PATH = r"C:\aibot\data\05_eval\eval_store.json"
NUM_CTX         = 4096
TOP_K           = 6


# ── print helpers ──────────────────────────────────────────────────────────────

def divider(title: str, width: int = 82) -> None:
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print('=' * width)


def subdiv(title: str, width: int = 82) -> None:
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print('-' * width)


def wrap_print(text: str, indent: int = 4, width: int = 100) -> None:
    prefix = " " * indent
    for paragraph in text.split("\n"):
        if paragraph.strip():
            print(textwrap.fill(paragraph, width=width, initial_indent=prefix,
                                subsequent_indent=prefix))
        else:
            print()


def tok_est(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars."""
    return len(text) // 4


def find_case(store: list, substr: str) -> dict:
    for item in store:
        if substr.lower() in item["question"].lower():
            return item
    raise ValueError(f"Question not found in eval store: {substr!r}")


# ── diagnostic engine ──────────────────────────────────────────────────────────

class DiagnosticEngine(NistChatEngine):
    """NistChatEngine with every retrieval + generation step printed verbosely."""

    def diagnose(self, question: str, expected_answer: str, case_label: str) -> None:
        print(f"\n\n{'#' * 82}")
        print(f"  {case_label}")
        print(f"{'#' * 82}")

        # ── Step 1: Question ───────────────────────────────────────────────────
        divider("STEP 1 — QUESTION")
        print(f"  Question : {question}")
        print()
        print("  Expected answer:")
        wrap_print(expected_answer, indent=4)

        tokenized_query = self._clean_tokenize(question.lower().strip())

        # ── Step 2: Dense retrieval ────────────────────────────────────────────
        divider("STEP 2 — DENSE RETRIEVAL  (top 12 child chunks, cosine similarity)")
        t0 = time.time()
        dense_results = self.vector_db.similarity_search_with_score(question, k=TOP_K * 2)
        print(f"  Elapsed: {time.time() - t0:.3f}s\n")

        for rank, (doc, score) in enumerate(dense_results, 1):
            pl   = doc.metadata.get("parent_link", "?")
            src  = doc.metadata.get("source", "?")[:60]
            snip = doc.page_content[:120].replace("\n", " ")
            print(f"  [{rank:2d}] score={score:.4f}  parent={pl}")
            print(f"       src  : {src}")
            print(f"       snip : {snip}")

        # ── Step 3: BM25 sparse retrieval ──────────────────────────────────────
        divider("STEP 3 — SPARSE RETRIEVAL  (BM25, top 12)")
        t0 = time.time()
        bm25_scores   = self.bm25.get_scores(tokenized_query)
        sparse_ranked = sorted(range(len(bm25_scores)),
                               key=lambda i: bm25_scores[i], reverse=True)[:TOP_K * 2]
        print(f"  Elapsed: {time.time() - t0:.3f}s\n")

        for rank, idx in enumerate(sparse_ranked, 1):
            sc   = bm25_scores[idx]
            doc  = self.docstore_items[idx]
            pl   = doc.metadata.get("parent_link", "?")
            src  = doc.metadata.get("source", "?")[:60]
            snip = doc.page_content[:120].replace("\n", " ")
            print(f"  [{rank:2d}] bm25={sc:.4f}  parent={pl}")
            print(f"       src  : {src}")
            print(f"       snip : {snip}")

        # ── Step 4: RRF fusion ─────────────────────────────────────────────────
        divider("STEP 4 — RRF FUSION  (rank combination -> top-6 cut)")

        constant_k              = 60
        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, any]   = {}

        def get_uid(d):
            return f"{d.metadata.get('parent_link', 'unknown')}_{d.page_content[:50]}"

        for rank, (doc, _) in enumerate(dense_results, 1):
            uid = get_uid(doc)
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + (1.0 / (constant_k + rank))
            doc_lookup[uid] = doc

        for rank, idx in enumerate(sparse_ranked, 1):
            if bm25_scores[idx] <= 0:
                continue
            doc = self.docstore_items[idx]
            uid = get_uid(doc)
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + (1.0 / (constant_k + rank))
            doc_lookup[uid] = doc

        sorted_uids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        print(f"  {'Rank':<6} {'RRF score':<14} {'In top-6':<10}  parent_link")
        print(f"  {'----':<6} {'----------':<14} {'--------':<10}  -----------")
        for rank, (uid, score) in enumerate(sorted_uids, 1):
            pl     = doc_lookup[uid].metadata.get("parent_link", "?")
            marker = "YES" if rank <= TOP_K else "no"
            print(f"  [{rank:2d}]   {score:.6f}       {marker:<10}  {pl}")

        final_uids = [uid for uid, _ in sorted_uids[:TOP_K]]

        # ── Step 5: Parent lookup ──────────────────────────────────────────────
        divider("STEP 5 — PARENT LOOKUP  (full text for each of the 6 retrieved docs)")

        seen_ids:      set       = set()
        context_parts: List[str] = []

        for slot, uid in enumerate(final_uids, 1):
            doc         = doc_lookup[uid]
            parent_link = doc.metadata.get("parent_link", uid)
            if parent_link in seen_ids:
                print(f"\n  [slot {slot}] {parent_link} -- SKIPPED (duplicate parent)")
                continue
            seen_ids.add(parent_link)

            parent_record = self.parent_store.get(parent_link, {})
            parent_text   = parent_record.get("text", doc.page_content)
            parent_meta   = parent_record.get("metadata", {})

            source_name  = parent_meta.get("source",       doc.metadata.get("source", "?"))
            chapter      = parent_meta.get("chapter",      "")
            section      = parent_meta.get("section",      "")
            content_type = parent_meta.get("content_type", "prose")
            sibling_ids  = parent_meta.get("sibling_ids",  [])
            table_row    = parent_meta.get("table_row",    0)

            m         = re.search(r'NIST\s+(?:AI\s+)?[\w]+-[\w]+', source_name)
            short_src = m.group(0) if m else source_name[:40]
            tokens    = tok_est(parent_text)

            subdiv(f"slot {slot} | {short_src} | {chapter} | {section} | "
                   f"{content_type} | ~{tokens} tokens")
            for line in parent_text.split("\n"):
                print(f"    {line}")

            location_parts = [p for p in [chapter, section] if p]
            location       = " > ".join(location_parts) if location_parts else ""
            header         = f"--- Source: {short_src}{' | ' + location if location else ''} ---"
            context_parts.append(f"{header}\n{parent_text}")

            # table neighbour expansion — mirrors nistaichat.py exactly
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
                    context_parts.append(f"  [table context -- adjacent row]\n{nb_text}")
                    print(f"\n  [table neighbour {nb_id}]")
                    for line in nb_text.split("\n"):
                        print(f"    {line}")

        full_context = "\n\n".join(context_parts)

        # ── Step 6: Assembled context ──────────────────────────────────────────
        divider("STEP 6 — ASSEMBLED CONTEXT  (the {context} block sent to the LLM)")
        ctx_tokens = tok_est(full_context)
        print(f"  Total chars : {len(full_context):,}")
        print(f"  Token est.  : ~{ctx_tokens:,}  (num_ctx budget: {NUM_CTX})")
        print()
        for line in full_context.split("\n"):
            print(f"  {line}")

        # ── Step 7: Full prompt ────────────────────────────────────────────────
        divider("STEP 7 — FULL PROMPT  (system + user message sent to ollama.chat)")

        system_msg = "You are a precise NIST AI policy advisor."
        user_msg   = PROMPTS["generator"].format(
            chat_history="No previous conversation.",
            context=full_context,
            question=question,
        )

        total_tokens = tok_est(system_msg + user_msg)
        headroom     = NUM_CTX - total_tokens

        print(f"  [SYSTEM]  ({len(system_msg)} chars)")
        print(f"    {system_msg}")
        print()
        print(f"  [USER]  ({len(user_msg):,} chars, ~{tok_est(user_msg):,} tokens)")
        print()
        for line in user_msg.split("\n"):
            print(f"    {line}")

        print()
        print(f"  Total prompt: ~{total_tokens:,} tokens  |  budget: {NUM_CTX}")
        if headroom < 0:
            print(f"  !! OVERFLOW: ~{abs(headroom)} tokens over budget — context will be truncated by ollama!")
        else:
            print(f"  OK fits within num_ctx  (headroom: ~{headroom} tokens)")

        # ── Step 8: LLM response ───────────────────────────────────────────────
        divider("STEP 8 — LLM RESPONSE")
        print(f"  Model: {READER_MODEL}  |  calling... ", end="", flush=True)
        t0      = time.time()
        answer  = self._call_llm(system_msg, user_msg)
        gen_sec = time.time() - t0
        print(f"done in {gen_sec:.2f}s\n")

        print("  Raw output:")
        for line in answer.split("\n"):
            print(f"    {line}")

        # ── Step 9: Comparison ─────────────────────────────────────────────────
        divider("STEP 9 — COMPARISON  (expected vs actual)")
        print("  EXPECTED:")
        wrap_print(expected_answer, indent=4)
        print()
        print("  ACTUAL:")
        wrap_print(answer, indent=4)


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    with open(EVAL_STORE_PATH, encoding="utf-8") as fh:
        store = json.load(fh)

    q12_item = find_case(store, "six monitoring categories for post-deployment AI systems")
    q16_item = find_case(store, "tracking model versioning and lineage for generative AI")

    cases = [
        (q12_item, "CASE 1 of 2 -- Q12 FAILURE  (correctness=0.0, relevance=0.25 — answered WHERE not WHAT)"),
        (q16_item, "CASE 2 of 2 -- Q16 FAILURE  (faithfulness=0.0, correctness=0.0 — model denied answer exists)"),
    ]

    print("Loading engine (FAISS + BM25 + parent store)...")
    engine = DiagnosticEngine()

    for item, label in cases:
        engine.diagnose(
            question        = item["question"],
            expected_answer = item["expected_answer"],
            case_label      = label,
        )

    print(f"\n\n{'=' * 82}")
    print("  DIAGNOSTIC COMPLETE")
    print(f"{'=' * 82}\n")


if __name__ == "__main__":
    main()
