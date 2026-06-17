import os
import re
import sys
import json
import time
import random
import logging
import threading
import pandas as pd
from datetime import datetime
from typing import Literal, List
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# PATH BOOTSTRAP
# ---------------------------------------------------------------------------
SCRIPT_DIR = r"C:\aibot\scripts"
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    from scripts.nistaichat import NistChatEngine, READER_MODEL, VERIFIER_MODEL, HIGH_RISK_PATTERNS
except ImportError:
    import importlib
    hybrid_chat_app  = importlib.import_module("nistaichat")
    NistChatEngine   = hybrid_chat_app.NistChatEngine
    READER_MODEL     = hybrid_chat_app.READER_MODEL
    VERIFIER_MODEL   = hybrid_chat_app.VERIFIER_MODEL
    HIGH_RISK_PATTERNS = hybrid_chat_app.HIGH_RISK_PATTERNS

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
GOLDEN_DATASET_PATH  = r"C:\aibot\data\05_eval\eval_store.json"
OUTPUT_DIR           = r"C:\aibot\data\05_eval"

RETRIEVAL_TOP_K          = 6
JACCARD_HIT_THRESHOLD    = 0.40
CONTEXT_CHAR_LIMIT       = 300_000
MAX_JUDGE_RETRIES        = 4
JUDGE_BASE_DELAY_S       = 4.0
BATCH_SIZE               = 1
BATCH_COOL_DOWN_S        = 5.0

# Leave empty [] to run all questions. Set to specific indices to run a subset.
QUESTION_SUBSET: list = [3, 12, 16, 17]

_ERROR_SENTINEL = "ERROR_LOCAL_GENERATION_FAILED"
_MAX_BATCH_PROMPT_CHARS = 1_000_000

VALID_SCORE_ANCHORS_STR  = {"0.0", "0.25", "0.5", "0.75", "1.0"}
SCORE_METRICS            = ("faithfulness", "relevance", "completeness", "correctness")
_METRIC_LABELS           = {"faithfulness": "FA", "relevance": "RE", "completeness": "CP", "correctness": "CR"}

_REQUIRED_DATASET_FIELDS = ("question_category", "question", "expected_answer")

LOG_EVAL_PATH = r"C:\aibot\logs\evalhistory\nisteval.log"
os.makedirs(os.path.dirname(LOG_EVAL_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_EVAL_PATH, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ===========================================================================
# PYDANTIC SCHEMAS
# ===========================================================================
class SingleJudgeResult(BaseModel):
    index: int = Field(description="Dataset index matching the evaluation item.")
    faithfulness: Literal["0.0", "0.25", "0.5", "0.75", "1.0"] = Field(
        description="Factual grounding alignment score."
    )
    faithfulness_reasoning: str = Field(
        description="Exactly one sentence summarizing grounding evidence or lack thereof."
    )
    relevance: Literal["0.0", "0.25", "0.5", "0.75", "1.0"] = Field(
        description="Direct alignment to the scope of the question asked."
    )
    relevance_reasoning: str = Field(
        description="Exactly one sentence explaining prompt alignment or off-topic elements."
    )
    completeness: Literal["0.0", "0.25", "0.5", "0.75", "1.0"] = Field(
        description="Coverage of context-available facts relative to the query scope."
    )
    completeness_reasoning: str = Field(
        description="Exactly one sentence evaluating omissions of crucial contextual data."
    )
    correctness: Literal["0.0", "0.25", "0.5", "0.75", "1.0"] = Field(
        description="Accuracy overlap against the ground truth expected answer."
    )
    correctness_reasoning: str = Field(
        description="Exactly one sentence highlighting correctness errors or confirmations."
    )


class BatchJudgeResponse(BaseModel):
    results: List[SingleJudgeResult] = Field(
        description="Evaluation results for each item in the batch. One result per item, in any order."
    )


class NegativeItemResponse(BaseModel):
    faithfulness: Literal["0.0", "0.25", "0.5", "0.75", "1.0"] = Field(
        description="1.0 if system correctly declines; 0.0 if it fabricates."
    )
    faithfulness_reasoning: str = Field(
        description="One sentence: did the system decline or fabricate, and why does that earn this score."
    )
    relevance: Literal["0.0", "0.25", "0.5", "0.75", "1.0"] = Field(
        description="1.0 if refusal addresses the question; 0.25 if vague; 0.0 if off-topic."
    )
    relevance_reasoning: str = Field(
        description="One sentence explaining how directly the refusal mapped to the question asked."
    )
    completeness: Literal["0.0", "0.25", "0.5", "0.75", "1.0"] = Field(
        description="1.0 if refusal is explicit; 0.5 if hedged; 0.0 if system attempts an answer."
    )
    completeness_reasoning: str = Field(
        description="One sentence characterising how clearly the system communicated the absence."
    )
    correctness: Literal["0.0", "0.25", "0.5", "0.75", "1.0"] = Field(
        description="1.0 if system correctly declines; 0.0 if it fabricates any answer."
    )
    correctness_reasoning: str = Field(
        description="One sentence comparing the system's behaviour to the expected correct refusal."
    )


class StandardItemResponse(BaseModel):
    faithfulness: Literal["0.0", "0.25", "0.5", "0.75", "1.0"] = Field(
        description="1.0 if every claim is explicitly grounded in the provided context; 0.0 if contradicts or hallucinated."
    )
    faithfulness_reasoning: str = Field(
        description="One sentence summarizing grounding evidence or lack thereof."
    )
    relevance: Literal["0.0", "0.25", "0.5", "0.75", "1.0"] = Field(
        description="1.0 if answer is laser-focused on the question; 0.0 if completely irrelevant."
    )
    relevance_reasoning: str = Field(
        description="One sentence explaining prompt alignment or off-topic elements."
    )
    completeness: Literal["0.0", "0.25", "0.5", "0.75", "1.0"] = Field(
        description="1.0 if all context-available relevant facts are covered; 0.0 if fails to address the question."
    )
    completeness_reasoning: str = Field(
        description="One sentence evaluating omissions of crucial contextual data."
    )
    correctness: Literal["0.0", "0.25", "0.5", "0.75", "1.0"] = Field(
        description="1.0 if all key facts match the expected answer; 0.0 if contradicts or entirely wrong."
    )
    correctness_reasoning: str = Field(
        description="One sentence highlighting correctness errors or confirmations."
    )


# ===========================================================================
# RESPONSE PARSING HELPERS
# ===========================================================================
def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    return raw


def _parse_batch_response(raw: str) -> BatchJudgeResponse:
    stripped = _strip_fences(raw)
    parsed   = json.loads(stripped)
    if isinstance(parsed, list):
        logger.debug("Batch response: raw array auto-wrapped into BatchJudgeResponse.")
        return BatchJudgeResponse(results=parsed)
    return BatchJudgeResponse.model_validate(parsed)


def _parse_negative_response(raw: str) -> NegativeItemResponse:
    stripped = _strip_fences(raw)
    parsed   = json.loads(stripped)
    if isinstance(parsed, list):
        if len(parsed) == 0:
            raise ValueError("Model returned an empty array for a negative item response.")
        if len(parsed) > 1:
            logger.warning(
                f"Negative judge returned {len(parsed)} items in array; "
                f"expected 1. Using first element only."
            )
        logger.debug("Negative response: raw array — extracting first element.")
        return NegativeItemResponse.model_validate(parsed[0])
    return NegativeItemResponse.model_validate(parsed)


def _parse_standard_response(raw: str) -> StandardItemResponse:
    stripped = _strip_fences(raw)
    parsed   = json.loads(stripped)
    if isinstance(parsed, list):
        if len(parsed) == 0:
            raise ValueError("Model returned an empty array for a standard item response.")
        if len(parsed) > 1:
            logger.warning(
                f"Standard judge returned {len(parsed)} items in array; "
                f"expected 1. Using first element only."
            )
        return StandardItemResponse.model_validate(parsed[0])
    return StandardItemResponse.model_validate(parsed)


# ===========================================================================
# GOLDEN DATASET VALIDATION
# ===========================================================================
def validate_golden_dataset(dataset: list) -> None:
    errors = []
    for i, item in enumerate(dataset, start=1):
        if not isinstance(item, dict):
            errors.append(f"  Item {i}: not a dict (got {type(item).__name__})")
            continue
        for field in _REQUIRED_DATASET_FIELDS:
            if field not in item:
                errors.append(f"  Item {i}: missing required field '{field}'")
            elif not isinstance(item[field], str) or not item[field].strip():
                errors.append(f"  Item {i}: field '{field}' is empty or not a string")

    if errors:
        raise ValueError(
            f"Golden dataset validation failed — {len(errors)} error(s) found:\n"
            + "\n".join(errors)
            + "\nFix eval_store.json before running the pipeline."
        )
    logger.info(f"Golden dataset validated: {len(dataset)} items, all fields present.")


# ===========================================================================
# VERIFIER INSTRUMENTED ENGINE
# ===========================================================================
class VerifierInstrumentedEngine:
    """
    Wraps NistChatEngine with two layers of monkey-patching:
      1. _assemble_hybrid_context — captures retrieved context text and sources (same as 04_eval_pipeline.py)
      2. _classify_query_risk, _verify_against_context, _call_llm — captures verification telemetry
    Returns (answer, ctx, srcs, vmeta) from ask().
    """

    def __init__(self):
        self._engine       = NistChatEngine()
        self._thread_local = threading.local()
        self._install_interceptor()
        self._install_verifier_interceptors()

    def _install_interceptor(self) -> None:
        if not hasattr(self._engine, "_assemble_hybrid_context"):
            raise AttributeError(
                "NistChatEngine lacks '_assemble_hybrid_context'. "
                "The target class was likely refactored. Eval harness requires updating."
            )
        original = self._engine._assemble_hybrid_context
        harness  = self

        def intercepting_assemble(query: str, top_k: int = RETRIEVAL_TOP_K):
            context_text, retrieved_sources = original(query, top_k=top_k)
            harness._thread_local.last_context_text       = context_text
            harness._thread_local.last_retrieved_sources  = retrieved_sources
            return context_text, retrieved_sources

        self._engine._assemble_hybrid_context = intercepting_assemble
        logger.info("Thread-safe context interceptor installed on _assemble_hybrid_context.")

    def _install_verifier_interceptors(self) -> None:
        orig_classify = self._engine._classify_query_risk
        orig_verify   = self._engine._verify_against_context
        orig_llm      = self._engine._call_llm
        tl = self._thread_local

        def intercepted_classify(question):
            result = orig_classify(question)
            tl.risk_level = result
            return result

        def intercepted_verify(top_chunk, answer):
            result = orig_verify(top_chunk, answer)
            tl.minicheck_triggered = True
            tl.minicheck_faithful  = result
            return result

        def intercepted_llm(system_msg, user_msg, **kwargs):
            tl.llm_call_count = getattr(tl, "llm_call_count", 0) + 1
            return orig_llm(system_msg, user_msg, **kwargs)

        self._engine._classify_query_risk    = intercepted_classify
        self._engine._verify_against_context = intercepted_verify
        self._engine._call_llm               = intercepted_llm
        logger.info("Verifier telemetry interceptors installed (_classify_query_risk, _verify_against_context, _call_llm).")

    def ask(self, question: str) -> tuple:
        tl = self._thread_local
        tl.risk_level          = "LOW"
        tl.minicheck_triggered = False
        tl.minicheck_faithful  = None
        tl.llm_call_count      = 0

        try:
            res    = self._engine.ask(question)
            answer = res[0] if isinstance(res, (tuple, list)) else res
            ctx    = getattr(tl, "last_context_text",      "")
            srcs   = getattr(tl, "last_retrieved_sources", [])
        except Exception as exc:
            logger.error(f"Local RAG Engine failed: {exc}")
            return _ERROR_SENTINEL, "", [], {}

        vmeta = {
            "risk_level":          tl.risk_level,
            "minicheck_triggered": tl.minicheck_triggered,
            "minicheck_faithful":  tl.minicheck_faithful,
            "retry_triggered":     tl.llm_call_count > 1,
        }
        return answer, ctx, srcs, vmeta

    def clear_history(self) -> None:
        self._engine.clear_history()


# ===========================================================================
# RETRIEVAL EVALUATION
# ===========================================================================
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "and", "for", "to", "is",
    "on", "at", "by", "with", "from", "or", "as", "are"
})

def _tokenize(text: str) -> set:
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return {t for t in tokens if t not in _STOPWORDS}

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0

def evaluate_retrieval_success(
    expected_sources: list,
    retrieved_sources: list,
    threshold: float = JACCARD_HIT_THRESHOLD
) -> bool:
    if not expected_sources or not retrieved_sources:
        return False
    for exp in expected_sources:
        exp_doc = _tokenize(exp.get("document_title", ""))
        for ret in retrieved_sources:
            ret_doc = _tokenize(ret.get("source_title", ""))
            if jaccard(exp_doc, ret_doc) >= threshold:
                return True
    return False


# ===========================================================================
# PROMPT BUILDERS
# ===========================================================================
def _build_item_block(index: int, item: dict) -> str:
    truncated_context = item["context_text"][:CONTEXT_CHAR_LIMIT]
    if len(item["context_text"]) > CONTEXT_CHAR_LIMIT:
        truncated_context += "\n\n[... CONTEXT TRUNCATED AT SAFETY LIMIT ...]"

    return f"""--- EVALUATION ITEM {index} ---
Dataset index  : {index}
Category       : {item['category']}
Question       : {item['question']}
Expected Truth : {item['expected_answer']}
System Output  : {item['generated_answer']}
Provided Context:
{truncated_context}
"""


def build_batch_eval_prompt(items_to_eval: dict) -> str:
    blocks        = [_build_item_block(idx, item) for idx, item in items_to_eval.items()]
    joined_blocks = "\n=======================================================\n".join(blocks)
    index_list    = ", ".join(str(k) for k in items_to_eval.keys())

    return f"""
You are an expert, impartial AI judge scoring a batch of NIST RAG system outputs.
Score EACH item below using the four criteria defined in the rubric.
When in doubt between two anchors, always choose the lower one.

You MUST return one result per item. The `index` field in each result must
exactly match the Dataset index of the item it scores. Do not skip, merge,
or reorder items. Expected indices for this batch: [{index_list}]

═══════════════════════════════════════════════════════════════════════
SCORING CRITERIA & ANCHOR RUBRICS  (read before scoring any item)
═══════════════════════════════════════════════════════════════════════

METRIC 1 — FAITHFULNESS (1.00, 0.75, 0.50, 0.25, 0.00)
Every factual claim must be traceable to the provided context.
  1.00 — Every claim explicitly grounded; no unsupported inferences.
  0.75 — One minor unsupported inference; all core claims grounded.
  0.50 — Several claims partial; at least one key claim ungrounded.
  0.25 — Majority of claims rely on outside knowledge or invention.
  0.00 — Contradicts context or is almost entirely hallucinated.

METRIC 2 — RELEVANCE (1.00, 0.75, 0.50, 0.25, 0.00)
Does the answer directly address what was asked?
  1.00 — Laser-focused; no irrelevant content.
  0.75 — Addresses the question with minor tangential content.
  0.50 — Core topic touched but significant irrelevant content present.
  0.25 — Mostly off-topic or answers a different question.
  0.00 — Completely irrelevant to the question asked.

METRIC 3 — COMPLETENESS (1.00, 0.75, 0.50, 0.25, 0.00)
Does it cover all aspects answerable from the context?
  1.00 — All context-available relevant facts are covered.
  0.75 — Most key facts covered; one minor detail omitted.
  0.50 — Core topic addressed but important context facts absent.
  0.25 — Surface-level fragment; most answerable aspects missing.
  0.00 — Fails to address the question meaningfully.

METRIC 4 — CORRECTNESS (1.00, 0.75, 0.50, 0.25, 0.00)
Factual alignment against the expected ground truth answer.
  1.00 — All key facts and conclusions match the expected answer.
  0.75 — Mostly correct; one fact missing or slightly imprecise.
  0.50 — Correct on main point; wrong or missing on supporting details.
  0.25 — Partially correct; multiple errors or key conclusions absent.
  0.00 — Contradicts the expected answer or is entirely wrong.

═══════════════════════════════════════════════════════════════════════
ITEMS TO EVALUATE
═══════════════════════════════════════════════════════════════════════

{joined_blocks}
"""


def build_negative_eval_prompt(index: int, item: dict) -> str:
    truncated_context = item["context_text"][:CONTEXT_CHAR_LIMIT]
    if len(item["context_text"]) > CONTEXT_CHAR_LIMIT:
        truncated_context += "\n\n[... CONTEXT TRUNCATED AT SAFETY LIMIT ...]"

    return f"""
You are an expert, impartial AI judge evaluating a single NEGATIVE-category
question from a NIST RAG system evaluation.

A NEGATIVE-CATEGORY question is one where the correct information does NOT
exist in the provided context. The CORRECT system behaviour is to explicitly
acknowledge that the information is absent rather than fabricating an answer.

═══════════════════════════════════════════════════════════════════════
ITEM TO EVALUATE
═══════════════════════════════════════════════════════════════════════
Dataset Index  : {index}
Question       : {item['question']}
Expected Truth : {item['expected_answer']}
System Output  : {item['generated_answer']}
Provided Context:
{truncated_context}

═══════════════════════════════════════════════════════════════════════
╔══════════════════════════════════════════════════════════════════════╗
║  NEGATIVE-QUESTION SCORING RULES — apply to ALL FOUR metrics         ║
║                                                                      ║
║  The standard rubric is SUSPENDED. Use only the rules below.         ║
║                                                                      ║
║  FAITHFULNESS                                                        ║
║    1.00 — System explicitly declines; acknowledges the context gap   ║
║    0.00 — System fabricates an answer despite the gap                ║
║                                                                      ║
║  RELEVANCE                                                           ║
║    1.00 — Refusal directly and clearly addresses the question asked  ║
║    0.25 — Refusal is vague, generic, or only tangentially related    ║
║    0.00 — Refusal or response ignores the question entirely          ║
║                                                                      ║
║  COMPLETENESS                                                        ║
║    1.00 — Refusal is explicit and unambiguous                        ║
║    0.50 — Refusal is implicit or heavily hedged                      ║
║    0.00 — System attempts to answer instead of declining             ║
║                                                                      ║
║  CORRECTNESS                                                         ║
║    1.00 — System correctly declines as the expected answer requires  ║
║    0.00 — System fabricates any answer                               ║
╚══════════════════════════════════════════════════════════════════════╝

When in doubt between two anchors, choose the lower one.
Each reasoning field must be exactly one sentence citing specific evidence.
"""


def build_single_eval_prompt(index: int, item: dict) -> str:
    truncated_context = item["context_text"][:CONTEXT_CHAR_LIMIT]
    if len(item["context_text"]) > CONTEXT_CHAR_LIMIT:
        truncated_context += "\n\n[... CONTEXT TRUNCATED AT SAFETY LIMIT ...]"

    return f"""
You are an expert, impartial AI judge scoring a single NIST RAG system output.
Score the item below using the four criteria defined in the rubric.
When in doubt between two anchors, always choose the lower one.

═══════════════════════════════════════════════════════════════════════
SCORING CRITERIA & ANCHOR RUBRICS
═══════════════════════════════════════════════════════════════════════

METRIC 1 — FAITHFULNESS (1.00, 0.75, 0.50, 0.25, 0.00)
Every factual claim must be traceable to the provided context.
  1.00 — Every claim explicitly grounded; no unsupported inferences.
  0.75 — One minor unsupported inference; all core claims grounded.
  0.50 — Several claims partial; at least one key claim ungrounded.
  0.25 — Majority of claims rely on outside knowledge or invention.
  0.00 — Contradicts context or is almost entirely hallucinated.

METRIC 2 — RELEVANCE (1.00, 0.75, 0.50, 0.25, 0.00)
Does the answer directly address what was asked?
  1.00 — Laser-focused; no irrelevant content.
  0.75 — Addresses the question with minor tangential content.
  0.50 — Core topic touched but significant irrelevant content present.
  0.25 — Mostly off-topic or answers a different question.
  0.00 — Completely irrelevant to the question asked.

METRIC 3 — COMPLETENESS (1.00, 0.75, 0.50, 0.25, 0.00)
Does it cover all aspects answerable from the context?
  1.00 — All context-available relevant facts are covered.
  0.75 — Most key facts covered; one minor detail omitted.
  0.50 — Core topic addressed but important context facts absent.
  0.25 — Surface-level fragment; most answerable aspects missing.
  0.00 — Fails to address the question meaningfully.

METRIC 4 — CORRECTNESS (1.00, 0.75, 0.50, 0.25, 0.00)
Factual alignment against the expected ground truth answer.
  1.00 — All key facts and conclusions match the expected answer.
  0.75 — Mostly correct; one fact missing or slightly imprecise.
  0.50 — Correct on main point; wrong or missing on supporting details.
  0.25 — Partially correct; multiple errors or key conclusions absent.
  0.00 — Contradicts the expected answer or is entirely wrong.

═══════════════════════════════════════════════════════════════════════
ITEM TO EVALUATE
═══════════════════════════════════════════════════════════════════════
Dataset Index  : {index}
Category       : {item['category']}
Question       : {item['question']}
Expected Truth : {item['expected_answer']}
System Output  : {item['generated_answer']}
Provided Context:
{truncated_context}

Each reasoning field must be exactly one sentence citing specific evidence.
"""


# ===========================================================================
# BATCH INDEX VALIDATION
# ===========================================================================
def _validate_batch_indices(
    returned_results: list,
    requested_indices: set,
    batch_label: str
) -> dict:
    output_map   = {}
    seen_indices = set()

    for res_dict in returned_results:
        idx = res_dict.get("index") if isinstance(res_dict, dict) else getattr(res_dict, "index", None)

        if idx not in requested_indices:
            logger.warning(
                f"[{batch_label}] Model returned index {idx} which was not "
                f"requested (expected one of {sorted(requested_indices)}). Discarding."
            )
            continue

        if idx in seen_indices:
            logger.warning(
                f"[{batch_label}] Model returned duplicate result for index {idx}. "
                f"Keeping first occurrence only."
            )
            continue

        seen_indices.add(idx)
        output_map[idx] = res_dict

    missing = requested_indices - seen_indices
    if missing:
        logger.warning(
            f"[{batch_label}] Items {sorted(missing)} absent from model response "
            f"for batch [{sorted(requested_indices)}] — flagging as judge_failed."
        )
        for idx in missing:
            output_map[idx] = {"judge_failed": True}

    return output_map


# ===========================================================================
# JUDGE CALLERS
# ===========================================================================
def call_batch_judge_with_retry(
    client,
    prompt:            str,
    requested_indices: set,
    max_retries:       int   = MAX_JUDGE_RETRIES,
    base_delay:        float = JUDGE_BASE_DELAY_S
) -> dict:
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are an expert, impartial AI judge auditing batches of NIST RAG "
                        "outputs. Score each item strictly per the provided rubric. Return "
                        "exactly one result per item with the correct dataset index."
                    ),
                    response_mime_type="application/json",
                    response_schema=BatchJudgeResponse,
                    temperature=0.0
                )
            )
            validated_batch = _parse_batch_response(response.text)

            raw_results   = [r.model_dump() for r in validated_batch.results]
            batch_label   = f"batch {sorted(requested_indices)}"
            validated_map = _validate_batch_indices(raw_results, requested_indices, batch_label)

            output_map = {}
            for idx, scores in validated_map.items():
                if scores.get("judge_failed"):
                    output_map[idx] = scores
                    continue
                scores.pop("index", None)
                for metric in SCORE_METRICS:
                    if scores.get(metric) is not None:
                        scores[metric] = float(scores[metric])
                scores["judge_failed"] = False
                output_map[idx] = scores

            return output_map

        except Exception as exc:
            if "validation" in type(exc).__name__.lower():
                raw_preview = getattr(locals().get("response"), "text", "<no response>")[:300]
                logger.warning(f"Batch raw output (first 300 chars): {raw_preview!r}")
            jitter = random.uniform(0, 0.5)
            wait   = base_delay * (2 ** attempt) + jitter
            logger.warning(
                f"Batch judge attempt {attempt + 1}/{max_retries} failed "
                f"[{type(exc).__name__}]: {exc}. Retrying in {wait:.1f}s…"
            )
            time.sleep(wait)

    logger.error(f"All batch judge retries exhausted for {sorted(requested_indices)}.")
    return {idx: {"judge_failed": True} for idx in requested_indices}


def call_negative_judge_with_retry(
    client,
    index:       int,
    item:        dict,
    max_retries: int   = MAX_JUDGE_RETRIES,
    base_delay:  float = JUDGE_BASE_DELAY_S
) -> dict:
    prompt = build_negative_eval_prompt(index, item)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are an expert, impartial AI judge evaluating a single "
                        "negative-category NIST RAG question. Apply the negative-question "
                        "scoring rules exactly as specified — the standard rubric is suspended."
                    ),
                    response_mime_type="application/json",
                    response_schema=NegativeItemResponse,
                    temperature=0.0
                )
            )
            validated = _parse_negative_response(response.text)
            scores    = validated.model_dump()
            for metric in SCORE_METRICS:
                if scores.get(metric) is not None:
                    scores[metric] = float(scores[metric])
            scores["judge_failed"] = False
            logger.info(
                f"Negative item {index} scored — "
                f"F={scores['faithfulness']} R={scores['relevance']} "
                f"Cp={scores['completeness']} Cr={scores['correctness']}"
            )
            return scores

        except Exception as exc:
            if "validation" in type(exc).__name__.lower():
                raw_preview = getattr(locals().get("response"), "text", "<no response>")[:300]
                logger.warning(f"Negative judge raw output (first 300 chars): {raw_preview!r}")
            jitter = random.uniform(0, 0.5)
            wait   = base_delay * (2 ** attempt) + jitter
            logger.warning(
                f"Negative judge attempt {attempt + 1}/{max_retries} failed "
                f"[{type(exc).__name__}]: {exc}. Retrying in {wait:.1f}s…"
            )
            time.sleep(wait)

    logger.error(f"All negative judge retries exhausted for item {index}.")
    return {
        **{metric: None for metric in SCORE_METRICS},
        **{f"{m}_reasoning": "JUDGE_FAILURE" for m in SCORE_METRICS},
        "judge_failed": True,
    }


def call_single_judge_with_retry(
    client,
    index:       int,
    item:        dict,
    max_retries: int   = MAX_JUDGE_RETRIES,
    base_delay:  float = JUDGE_BASE_DELAY_S
) -> dict:
    prompt = build_single_eval_prompt(index, item)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are an expert, impartial AI judge auditing a single NIST RAG "
                        "output. Score it strictly per the provided rubric."
                    ),
                    response_mime_type="application/json",
                    response_schema=StandardItemResponse,
                    temperature=0.0
                )
            )
            validated = _parse_standard_response(response.text)
            scores    = validated.model_dump()
            for metric in SCORE_METRICS:
                if scores.get(metric) is not None:
                    scores[metric] = float(scores[metric])
            scores["judge_failed"] = False
            logger.info(
                f"Item {index} scored — "
                f"F={scores['faithfulness']} R={scores['relevance']} "
                f"Cp={scores['completeness']} Cr={scores['correctness']}"
            )
            return scores

        except Exception as exc:
            if "validation" in type(exc).__name__.lower():
                raw_preview = getattr(locals().get("response"), "text", "<no response>")[:300]
                logger.warning(f"Standard judge raw output (first 300 chars): {raw_preview!r}")
            jitter = random.uniform(0, 0.5)
            wait   = base_delay * (2 ** attempt) + jitter
            logger.warning(
                f"Standard judge attempt {attempt + 1}/{max_retries} failed "
                f"[{type(exc).__name__}]: {exc}. Retrying in {wait:.1f}s…"
            )
            time.sleep(wait)

    logger.error(f"All standard judge retries exhausted for item {index}.")
    return {
        **{metric: None for metric in SCORE_METRICS},
        **{f"{m}_reasoning": "JUDGE_FAILURE" for m in SCORE_METRICS},
        "judge_failed": True,
    }


# ===========================================================================
# DATA LOADING
# ===========================================================================
def load_golden_dataset(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Golden dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ===========================================================================
# EXECUTION PIPELINE
# ===========================================================================
def run_evaluation() -> None:
    run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    _safe_model = READER_MODEL.replace(":", "-").replace("/", "-")
    out_csv     = os.path.join(OUTPUT_DIR, f"rag_eval_verified_{run_id}_{_safe_model}.csv")
    out_jsonl   = os.path.join(OUTPUT_DIR, f"rag_eval_verified_{run_id}_{_safe_model}.jsonl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline_start = time.time()

    logger.info(f"════ Verified Evaluation Run {run_id} ════")
    logger.info(f"Reader model    : {READER_MODEL}")
    logger.info(f"Verifier model  : {VERIFIER_MODEL}")
    logger.info(f"Retrieval top_k : {RETRIEVAL_TOP_K}")
    logger.info(f"HIGH RISK patterns: {len(HIGH_RISK_PATTERNS)}")
    logger.info("Post-processing : bold + header stripping active (inherited from nistaichat._call_llm)")
    logger.info("Initialising verifier-instrumented NIST Chat Engine…")
    engine = VerifierInstrumentedEngine()
    client = genai.Client()

    logger.info(f"Loading golden dataset: {GOLDEN_DATASET_PATH}")
    golden_dataset = load_golden_dataset(GOLDEN_DATASET_PATH)
    validate_golden_dataset(golden_dataset)

    working_items = {
        idx: item
        for idx, item in enumerate(golden_dataset, start=1)
        if not QUESTION_SUBSET or idx in QUESTION_SUBSET
    }
    if QUESTION_SUBSET:
        logger.info(f"Subset mode: running items {sorted(working_items.keys())} ({len(working_items)}/{len(golden_dataset)} total)")

    # -----------------------------------------------------------------------
    # PHASE 1: LOCAL GENERATION LOOP (with verification telemetry)
    # -----------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("  PHASE 1: LOCAL OLLAMA INFERENCE + VERIFICATION LAYER")
    logger.info("=" * 80)

    inference_cache = {}
    _working_keys   = list(working_items.keys())

    for index, item in working_items.items():
        question = item["question"]
        category = item["question_category"]
        subset_pos = _working_keys.index(index) + 1
        logger.info(f"[{index:>3}] [{category.upper()}]…  (subset {subset_pos}/{len(working_items)})")
        engine.clear_history()

        generated_answer, context_text, retrieved_sources, vmeta = engine.ask(question)

        retrieval_hit = False
        if generated_answer != _ERROR_SENTINEL:
            retrieval_hit = evaluate_retrieval_success(
                item.get("sources", []), retrieved_sources
            )

        logger.info(
            f"  Verification: risk={vmeta.get('risk_level','?')}  "
            f"minicheck={'yes' if vmeta.get('minicheck_triggered') else 'no'}  "
            f"faithful={vmeta.get('minicheck_faithful')}  "
            f"retry={'yes' if vmeta.get('retry_triggered') else 'no'}"
        )

        inference_cache[index] = {
            "category":            category,
            "question":            question,
            "expected_answer":     item["expected_answer"],
            "generated_answer":    generated_answer,
            "context_text":        context_text,
            "retrieval_hit":       retrieval_hit,
            "risk_level":          vmeta.get("risk_level", "LOW"),
            "minicheck_triggered": int(vmeta.get("minicheck_triggered", False)),
            "minicheck_faithful":  vmeta.get("minicheck_faithful"),
            "retry_triggered":     int(vmeta.get("retry_triggered", False)),
        }

    # -----------------------------------------------------------------------
    # PHASE 2: JUDGE EVALUATION
    # -----------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("  PHASE 2: JUDGE EVALUATIONS")
    logger.info("=" * 80)

    all_scores     = {}
    items_to_batch = {}
    negative_items = {}

    for idx, data in inference_cache.items():
        if data["generated_answer"] == _ERROR_SENTINEL:
            all_scores[idx] = {"judge_failed": True}
        elif data["category"].strip().lower() == "negative":
            negative_items[idx] = data
        else:
            items_to_batch[idx] = data

    if negative_items:
        logger.info(f"Routing {len(negative_items)} NEGATIVE item(s) to isolated judge calls…")
        for idx, data in negative_items.items():
            logger.info(f"Negative judge: item {idx} — '{data['question'][:70]}…'")
            all_scores[idx] = call_negative_judge_with_retry(client, idx, data)
            time.sleep(JUDGE_BASE_DELAY_S / 2)

        if items_to_batch:
            logger.info(f"Inter-phase cooldown {BATCH_COOL_DOWN_S}s before batch phase…")
            time.sleep(BATCH_COOL_DOWN_S)

    if items_to_batch:
        logger.info(f"Routing {len(items_to_batch)} non-negative item(s) to single-item judge…")
        items_list = list(items_to_batch.items())

        for i, (idx, data) in enumerate(items_list):
            logger.info(f"Standard judge: item {idx} — '{data['question'][:70]}…'")
            all_scores[idx] = call_single_judge_with_retry(client, idx, data)
            if i < len(items_list) - 1:
                time.sleep(JUDGE_BASE_DELAY_S / 2)

    # -----------------------------------------------------------------------
    # PHASE 3: METRICS AGGREGATION & REPORTING
    # -----------------------------------------------------------------------
    results = []
    for index, item in working_items.items():
        cache  = inference_cache[index]
        scores = all_scores.get(index, {"judge_failed": True})

        record = {
            "run_id":                  run_id,
            "index":                   index,
            "category":                cache["category"],
            "question":                cache["question"],
            "expected_answer":         cache["expected_answer"],
            "generated_answer":        cache["generated_answer"],
            "retrieved_context":       cache["context_text"],
            "retrieval_hit":           int(cache["retrieval_hit"]),
            "risk_level":              cache["risk_level"],
            "minicheck_triggered":     cache["minicheck_triggered"],
            "minicheck_faithful":      cache["minicheck_faithful"],
            "retry_triggered":         cache["retry_triggered"],
            "faithfulness":            scores.get("faithfulness"),
            "faithfulness_reasoning":  scores.get("faithfulness_reasoning", ""),
            "relevance":               scores.get("relevance"),
            "relevance_reasoning":     scores.get("relevance_reasoning", ""),
            "completeness":            scores.get("completeness"),
            "completeness_reasoning":  scores.get("completeness_reasoning", ""),
            "correctness":             scores.get("correctness"),
            "correctness_reasoning":   scores.get("correctness_reasoning", ""),
            "judge_failed":            scores.get("judge_failed", True),
        }
        results.append(record)

        route_tag = "[NEG]  " if cache["category"].strip().lower() == "negative" else "[STD]  "
        risk_tag  = f"[{cache['risk_level']}]"
        retry_tag = "[RETRY]" if cache["retry_triggered"] else "       "
        logger.info(f"[{index:>3}/{len(golden_dataset)}] {route_tag} {risk_tag} {retry_tag} {cache['category'].upper()}")
        if scores.get("judge_failed"):
            logger.warning(f"  → JUDGE FAILURE — scores null")
        else:
            logger.info(
                f"  → Ret={'HIT' if cache['retrieval_hit'] else 'MISS'}  "
                f"F={scores['faithfulness']:.2f}  "
                f"R={scores['relevance']:.2f}  "
                f"Cp={scores['completeness']:.2f}  "
                f"Cr={scores['correctness']:.2f}"
            )

    df = pd.DataFrame(results)
    df.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)
    logger.info(f"[Trace layer]   {out_jsonl}")

    _drop = ["generated_answer", "retrieved_context", "expected_answer"]
    df.drop(columns=_drop, errors="ignore").to_csv(out_csv, index=False, encoding="utf-8")
    logger.info(f"[Metrics layer] {out_csv}")

    df_ok    = df[df["judge_failed"] == False].copy()
    n_failed = int(df["judge_failed"].sum())
    n_valid  = len(df_ok)

    elapsed = time.time() - pipeline_start

    logger.info("=" * 80)
    logger.info(
        f"  GLOBAL SUMMARY  —  Run: {run_id}  "
        f"| Model: {READER_MODEL}  | Valid: {n_valid}/{len(df)}  | Judge failures: {n_failed}"
    )
    logger.info("=" * 80)
    logger.info(f"  Retrieval Accuracy : {df['retrieval_hit'].mean() * 100:.1f}%")

    for metric in SCORE_METRICS:
        avg = df_ok[metric].mean() if n_valid else float("nan")
        logger.info(f"  Avg {metric.capitalize():<14}: {avg:.2f}")

    if n_valid > 0 and "category" in df_ok.columns:
        logger.info("  PER-CATEGORY BREAKDOWN:")
        for cat, grp in df_ok.groupby("category"):
            route = "[NEG]  " if cat.strip().lower() == "negative" else "[STD]  "
            line  = f"  {route} [{cat:<20}]  "
            for metric in SCORE_METRICS:
                line += f"{_METRIC_LABELS[metric]}={grp[metric].mean():.2f}  "
            line += f"n={len(grp)}"
            logger.info(line)

    # Verification layer summary
    n_high     = int(df["risk_level"].eq("HIGH").sum())
    n_mc       = int(df["minicheck_triggered"].sum())
    n_retry    = int(df["retry_triggered"].sum())
    retry_rate = (n_retry / n_high * 100) if n_high else 0.0

    logger.info("  VERIFICATION LAYER SUMMARY:")
    logger.info(f"  HIGH RISK queries   : {n_high} / {len(df)}")
    logger.info(f"  MiniCheck triggered : {n_mc}")
    logger.info(f"  Retries triggered   : {n_retry}  (retry rate among HIGH: {retry_rate:.1f}%)")

    if n_high > 0 and "category" in df.columns:
        logger.info("  HIGH RISK per category:")
        for cat, grp in df.groupby("category"):
            pct = grp["risk_level"].eq("HIGH").mean() * 100
            logger.info(f"    [{cat:<20}]  HIGH={pct:.0f}%  n={len(grp)}")

    logger.info(f"  Total runtime  : {elapsed:.1f}s  ({elapsed / len(golden_dataset):.1f}s/item)")
    logger.info("=" * 80)


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        logger.critical("GEMINI_API_KEY environment variable is not configured.")
        sys.exit(1)
    run_evaluation()
