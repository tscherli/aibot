# NIST AI Policy Advisor

A local, privacy-preserving RAG chatbot for querying NIST AI safety and risk management documents — runs entirely on CPU via Ollama, no cloud LLM at inference time.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Ollama](https://img.shields.io/badge/Ollama-local-green) ![FAISS](https://img.shields.io/badge/FAISS-vector--db-orange) ![LangChain](https://img.shields.io/badge/LangChain-retrieval-purple) ![Gradio](https://img.shields.io/badge/Gradio-UI-yellow)

---

## Overview

This project builds a Retrieval-Augmented Generation (RAG) chatbot grounded in seven NIST publications on AI safety, risk management, and secure software development. It is designed for AI governance, security, and compliance practitioners who need precise, source-cited answers to policy questions without sending sensitive queries to a cloud provider.

All inference runs locally: the embedding model and reader model are served by Ollama, the vector index lives on disk, and the Gradio UI runs in the browser. No data leaves the machine.

## Summary of Architecture Decisions

**Test hardware:** AMD Ryzen 5 5500U · 6 cores · 16 GB RAM · CPU-only · Windows 11  

Because this experiment was conducted on highly restricted hardware, the system architecture was engineered according to the following design choices::

**Why LangChain wraps FAISS but Ollama is called directly for the reader**
FAISS requires an object with a `.embed_query()` method, not a plain callable function. LangChain provides the adapter. The reader model (`qwen2.5:3b`) is called via Ollama's REST API directly, avoiding LangChain chain overhead for generation where it adds no value.

**Why HuggingFace `AutoTokenizer` was removed from the pipeline**
Loading `transformers` + `torch` at startup consumes 1–2 GB RAM and adds 10–20 seconds before the first query — on a 16 GB machine already running Ollama and FAISS this is unacceptable. Ollama handles tokenization internally for the embedding model. Removing it keeps the Python process at ~100 MB, leaving more headroom for model inference.

**Why `bge-m3:567m` for embeddings over `all-MiniLM`**
`bge-m3:567m` has an 8,192-token context window — it embeds dense NIST compliance tables as a single vector instead of splitting them mid-row. `all-MiniLM` has a 256-token window, which fragments most policy paragraphs. Switching embedding models required a full re-ingestion (~30 min on CPU) but the retrieval quality improvement was immediate.

**Why semantic anchoring (parent/child registry) over fixed-size chunking**
NIST documents mix prose, multi-row tables, and figure captions in the same section. Fixed-size markdown splitting cuts tables mid-row and severs figure captions from their figures. The state-machine parser in `01_parse_nist.py` produces a parent/child registry with deterministic IDs — small child chunks go into FAISS for precise matching; full parent text is retrieved at query time. This is the Parent Document Retriever pattern adapted to a custom JSON store instead of a vector DB docstore.

**Why hybrid retrieval (dense FAISS + BM25 + cross-encoder)**
Dense FAISS misses exact acronyms like "GOVERN", "SSDF", "MAP". BM25 alone misses semantic paraphrases. RRF fusion combines both without requiring compatible score scales — it only uses rank positions. The cross-encoder (`ms-marco-MiniLM-L-6-v2`, 22M params) then re-scores the top-10 candidates with a full cross-attention read of both the query and the chunk, adding <1ms overhead on CPU while meaningfully improving precision on the final top-6 cut.

**Why `qwen2.5:3b` (Q4_K_M) as reader model**
Selected after a 7-model benchmark on the same hardware. Key findings: models below 3B hallucinate on NIST compliance lists; `phi4-mini:3.8b` hedges answers due to heavy RLHF safety training; `qwen2.5:3b` scored correct on all 6 lifecycle stages, generated clean plain-text output, and ran 40% faster after prompt simplification. Q4_K_M quantization beats Q4_0 on CPU latency despite lower bit-width due to better AVX2 vectorisation in llama.cpp.

**Why post-generation MiniCheck verification instead of pre-filtering**
Classification before generation is unreliable — Q3 had high retrieval confidence yet the model still overrode context with parametric memory. MiniCheck runs after generation: a misclassified query at worst returns an unverified answer, but can never silently skip the check. The retry prompt is citation-forced rather than a full regeneration, reducing latency overhead.

**Why P80 for progress bar latency estimation**
Maximum is dominated by a single 293s outlier (HIGH-risk retry). Mean causes the bar to hit zero ~50% of the time while the query is still running — looks broken. P80 errs slightly high (bar still counting when answer arrives), which feels better than running out early. Implemented as `sorted(lst)[int(len(lst) * 0.8)]`.

---

## Knowledge Base — Source Documents

| Document | ID | PDF |
|---|---|---|
| Artificial Intelligence Risk Management Framework (AI RMF 1.0) | NIST AI 100-1 | [nvlpubs.nist.gov](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf) |
| Adversarial Machine Learning: A Taxonomy and Terminology | NIST AI 100-2e2023 | [nvlpubs.nist.gov](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2023.pdf) |
| A Plan for Global Engagement on AI Standards | NIST AI 100-5 | [nvlpubs.nist.gov](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-5.pdf) |
| AI RMF: Generative Artificial Intelligence Profile | NIST AI 600-1 | [nvlpubs.nist.gov](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) |
| Challenges to the Monitoring of Deployed AI Systems | NIST AI 800-4 | [nvlpubs.nist.gov](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-4.pdf) |
| Secure Software Development Practices for Generative AI (SSDF Profile) | NIST SP 800-218A | [nvlpubs.nist.gov](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218A.pdf) |
| Trustworthy AI in Critical Infrastructure Profile (Concept Note) | NIST AI RMF Profile | [nist.gov](https://www.nist.gov/system/files/documents/2026/04/08/Concept%20Note_%20Development%20of%20the%20NIST%20AI%20RMF%20Trustworthy%20Use%20of%20AI%20in%20Critical%20Infrastructure%20Profile.pdf) |

---

## Prerequisites

- Python 3.10+
- [Ollama for Windows](https://ollama.com/download)

Pull the required models:

```powershell
ollama pull bge-m3:567m        # embedding model (8,192-token context window)
ollama pull qwen2.5:3b         # reader model (Q4_K_M quantization, default)
ollama pull llama3.2:3b        # critique/parse agent used during PDF ingestion
ollama pull bespoke-minicheck  # MiniCheck faithfulness verification model
```

---

## Installation

```
# Install Python dependencies
pip install -r requirements.txt
```

---

## Running the Application

Default is UI-Mode and script can be run without params:

```powershell
python C:\aibot\scripts\nistaichat_ui_v2.py 
```

Gradio opens automatically at `http://localhost:7860`. To use a custom port:

```powershell
python C:\aibot\scripts\nistaichat_ui_v2.py --mode ui --port 8080
```

To use in console with the following param:

```powershell
python C:\aibot\scripts\nistaichat_ui_v2.py --mode console
```

Daily chat logs are written to `logs\chathistory\nistchat_YYYYMMDD.log`.

---


## Data Ingestion Pipeline

To add new NIST documents to the knowledge base:

**Step 1 — PDF to Markdown**
Convert PDFs using [Docling](https://github.com/DS4SD/docling). For RAM-constrained machines (< 32 GB), run Docling in a Google Colab notebook — local 16 GB RAM was insufficient for the larger PDFs.

**Step 2 — Semi-Manual cleanup**
Per-document cleaning scripts handle different anchor points and boilerplate patterns across NIST publications (100-1, 600-1, etc.). Remove the table of contents and strip repeating page headers (e.g. `NIST SP 800-218A | Page 42`). Image extraction: done via manual screenshots -- Docling image export was not reliable for the multi-column NIST figure layouts on.

**Step 3 — Semantic Anchoring of tables and images**
LM-generated captions and summaries added manually using Gemini. No universal standard for anchoring; approach depends on input format. For NIST figures, generate LLM captions with a summary, structured bullet points, and keyword tags embedded in the markdown. This makes figure content semantically searchable rather than just a filename reference. Example format:

```markdown
![Taxonomy of AI Harms](../04_images/nistai100-1fig1.png)

> ### Technical Description: Taxonomy of Potential AI Harms
> **Summary:** This diagram categorizes the multi-level negative impacts that AI systems
> can exert, organized into three primary pillars: Harm to People, Harm to an
> Organization, and Harm to an Ecosystem.
>
> * **Harm to People:** Individual (civil liberties, physical/psychological safety),
>   Group/Community (discrimination), Societal (democratic participation).
> * **Harm to an Organization:** Business operations, security breaches, reputational damage.
> * **Harm to an Ecosystem:** Interconnected infrastructure, global financial system,
>   natural resources.
>
> **Keywords:** AI Risk Management Framework, Taxonomy of AI Harms, Societal Impact of AI,
> Civil Liberties, Algorithmic Discrimination, Organizational Risk, Environmental AI Impact.
```

**Step 4 — Parse to JSON registry**
```powershell
python scripts\01_parse_nist.py
```
Produces `data\04_vector_storage\parent_store.jsonl` and `data\04_vector_storage\child_store.jsonl` with deterministic tracking keys.

**Step 5 — Ingest to FAISS**
```powershell
python scripts\02_ingest_to_vector_db.py
```
Embeds 1,473 chunks via `bge-m3:567m`. Takes ~30 minutes on 16 GB RAM, CPU only. Writes `index.faiss`. Why `bge-m3:567m` for embeddings over `all-MiniLM`? `bge-m3:567m` has an 8,192-token context window — it swallows dense NIST compliance tables whole rather than splitting them mid-row. `all-MiniLM` has a 256-token window, which fragments most policy paragraphs. `bge-m3` also has stronger multilingual capability for policy terminology.

**Parent Document Retriever pattern**

The pipeline implements a classic Parent Document Retriever:

- `child_store.jsonl` — used once at index build time (`02_ingest_to_vector_db.py`) to create the FAISS embeddings. Each child chunk carries a `parent_link` metadata key pointing back to its parent block ID. Not needed at runtime.
- `parent_store.jsonl` — loaded into memory at engine startup (`nistaichat.py` line 105) as a flat dict keyed by ID.

At query time, FAISS retrieves small child chunks (precise embedding matches). The engine then reads each child's `parent_link`, looks up the full parent block in `parent_store`, and sends that to `qwen2.5:3b` instead of the child snippet:

```
Retrieval:  small child chunks  → precise semantic matching
Generation: full parent blocks  → rich context for the LLM
```

`parent_store.jsonl` must exist on disk at runtime; `child_store.jsonl` only matters during ingestion. `parent_store.jsonl` is worth committing — it contains manually enriched semantic anchoring that cannot be automatically reproduced.

---

## Reader Model Experimentation
**Benchmark query:** "How does the AI system lifecycle look from development through deployment and monitoring?"  
**Inference settings:** `num_thread: 6`, `num_batch: 1024`, `num_ctx: 3072`, `repeat_penalty: 1.15`

| Model | Generation Time | All 6 Lifecycle Stages Correct | Notes |
|---|---|---|---|
| `llama3.2:3b` | ~128s | Yes | Baseline. Correct but slow and verbose with original prompt. |
| `llama3.2:1b` | ~90s | No | Hallucinated stakeholder sections not present in NIST documents. |
| `phi4-mini:3.8b` | ~169s | No | Over-cautious, hedges answers, ignores retrieved context. Heavy RLHF safety training is counterproductive for RAG tasks. |
| `gemma3n:e2b` | ~151s | Yes | Good inline citations. ARM/NPU-optimised — no speed benefit on x86 CPU. |
| `qwen2.5:1.5b` | ~56s | No | Fast but consistently missed "Verify and validate" and "Deploy and Use" stages. |
| `qwen2.5:3b` | ~115s | **Yes** | **Winner.** All 6 stages correct, clean output, 40% faster after prompt simplification. |
| `qwen2.5:3b-instruct-q4_0` | ~147s | Yes | Slower than Q4_K_M despite lower bit-width; generated duplicate sections; weaker instruction-following. |

**Key findings:**

- **Prompt length dominates latency more than model size.** `qwen2.5:3b` ran at 190s with a 350-token verbose 6-rule prompt and at 115s after simplifying to ~50 tokens — a 40% reduction with no quality loss.
- **Models below 3B fail on faithfulness for policy extraction.** `llama3.2:1b` hallucinated generic organisational structures; `qwen2.5:1.5b` consistently dropped lifecycle stages. For compliance use where completeness is critical, 3B is the minimum viable size on this hardware.
- **Instruction-following matters more than raw parameter count.** `phi4-mini:3.8b` at 3.8B underperformed `qwen2.5:3b` at 3B because Phi's heavy RLHF safety training causes hedging that undermines RAG faithfulness.
- **Q4_K_M beats Q4_0 on CPU.** Despite lower bit-width, Q4_0 ran at 147s vs 115s for Q4_K_M. Q4_K_M uses K-quants with better AVX2 vectorisation in llama.cpp. Q4_0 also generated more output tokens due to weaker instruction-following, adding further latency.
- **CPU latency floor:** ~115s to generate ~150 words on this hardware. Further reduction requires GPU offloading or streaming output (which makes latency invisible to the user).

**Quantization reference:**

| Quantization | RAM (~3B model) | Quality loss | CPU speed |
|---|---|---|---|
| Q8_0 | ~3.1 GB | Minimal | Slowest (~2× slower than Q4_K_M) |
| Q4_K_M | ~1.9 GB | Small | **Fastest — Recommended** |
| Q4_0 | ~1.7 GB | Moderate | Slower than Q4_K_M (counterintuitive) |
| Q2_K | ~1.1 GB | Significant | Fast but noticeable faithfulness degradation |

**Production system prompt:**

```
You are a NIST AI policy advisor. If the answer isn't in the context, say you don't know.

- Be concise. Answer in plain text, no headers or bold.
- If the answer is a list of items, output every item from the context using bullet points (-).
- Use a numbered list only for ordered steps or requirements.
- Answer as close as possible to the original policy language.
- If the context only partially covers the topic, refer to the source documents listed below.

CONTEXT: {context}
CONVERSATION: {chat_history}
QUESTION: {question}
```

---

## Evaluation

### Golden Dataset Construction

18 evaluation questions were generated across 7 NIST source documents (2–3 per document) using a semi-automated approach: a structured prompt template was submitted to an LLM (Gemini / GPT) with merged NIST markdown as input, requesting diverse question types to avoid the shallow synthetic datasets that arise when all questions look alike.

All 7 NIST source documents were merged into a single file for prompting:
```powershell
# In data\02_clean_markdown\
Get-Content *.md | Set-Content merged_file.md
```
Evaluation Prompt was running on Model Sonnet 4.6 Adaptive:

```text
You are an expert AI evaluation engineer creating a high-quality golden evaluation dataset for a Retrieval-Augmented Generation (RAG) system.

Your task is to generate diverse and realistic evaluation questions based ONLY on the provided document section.

The goal is to evaluate:
- retrieval quality
- semantic search robustness
- hallucination resistance
- multi-hop reasoning
- ambiguity handling
- answer grounding

IMPORTANT RULES:
- Use ONLY information present in the provided text.
- Do not generate questions that can be answered correctly without retrieving information from the provided text.
- Do NOT invent facts.
- Questions should sound realistic and natural.
- Avoid repetitive phrasing patterns.
- Include paraphrased wording.
- Include difficult retrieval formulations.
- Negative questions should test whether the RAG system correctly admits absence instead of hallucinating.
- Sources must directly support the expected answer.
- Keep answers concise but precise.
- Try to distribute questions and question types across multiple document sections where meaningful and relevant.
- Create 2-3 questions per source (7 sources in total)
- Questions should resemble realistic user queries that might occur in production usage, including:
  - common FAQ-style questions
  - paraphrased user wording
  - vague or underspecified requests
  - expert-oriented technical questions
  - adversarial or misleading formulations

NEGATIVE QUESTION RULES:
- Include at least one negative question where the answer is NOT contained in the text.
- Negative questions should test whether the RAG system correctly admits absence instead of hallucinating.
- Negative questions should still be plausible and domain-related.

Generate the following question categories:

1. factual
   - direct retrieval
   - explicit information in text

2. reasoning
   - require combining multiple statements
   - require inference across sections or paragraphs
   - requires information from multiple parts of the text

3. edge-case
   - unusually phrased
   - indirect wording
   - retrieval robustness test

4. ambiguity
   - vague or underspecified wording
   - tests semantic retrieval quality
   - intentionally mixes related concepts
   - tests retrieval precision and conceptual distinction

6. negative
   - asks about a concept NOT present in the text
   - expected answer should indicate absence of information

OUTPUT FORMAT:

Return STRICTLY valid JSON using this schema:

[
  {
    "question_category": "",
    "question": "",
    "expected_answer": "",
    "sources": [
      {
        "document_title": "",
        "section_title": "",
        "raw_text": ""
      }
    ]
  }
]


Notes on the output format: 

- Source files are marked in the document in the following (example) format. Use this for the document_title in the source json. There are 7 sources in total:

# NIST AI 100-1 Artificial Intelligence Risk Management Framework (AI RMF 1.0)

**Published:** January 2023  
**Document Type:** NIST Pubs Report 

---

- question_category values: factual, reasoning, edge-case, ambiguity, negative

```
The prompt explicitly required multiple question categories:

| Type | Count | Purpose |
|---|---|---|
| Factual | 5 | Direct retrieval of explicit information |
| Reasoning | 4 | Multi-hop inference across sections |
| Edge-case | 4 | Unusual phrasing, indirect wording, retrieval robustness |
| Ambiguity | 3 | Vague or underspecified queries, conceptual distinctions |
| Negative | 2 | Absence detection — hallucination resistance |

**Negative questions are especially important:** a well-functioning RAG must admit when information is not present rather than confabulate a plausible-sounding answer. Example: *"What does the document say about blockchain-based authentication?"* when blockchain is never mentioned.

The golden dataset is stored at `data\05_eval\eval_store.json`.

### Evaluation Pipeline

`scripts/04_eval_pipeline.py` runs three sequential phases:

1. **Inference** — calls `NistChatEngine.ask()` for each of 18 items; captures generated answer, retrieved context, source metadata, and a `retrieval_hit` flag (Jaccard similarity ≥ 0.40 between expected and retrieved document titles)
2. **Judge evaluation** — sends each item to a Gemini 2.5 Flash judge in isolated single-item calls (`BATCH_SIZE = 1`) to prevent cross-item score contamination; negative-category items go to a dedicated negative judge
3. **Aggregation** — merges results, writes timestamped CSV + JSONL trace to `data/05_eval/`

**Scoring — four metrics on a `{0.0, 0.25, 0.5, 0.75, 1.0}` scale:**

| Label | Metric | Meaning |
|---|---|---|
| FA | Faithfulness | Every claim in the answer is grounded in the retrieved context |
| RE | Relevance | The answer addresses the question that was asked |
| CP | Completeness | The answer covers all key points from the ground truth |
| CR | Correctness | The answer matches the ground truth answer |

Two-letter fixed labels are used (`FA`, `RE`, `CP`, `CR`) rather than prefix-abbreviation because `metric[:2].upper()` produces `CO` for both Completeness and Correctness, making them indistinguishable in reports.

### Results — `qwen2.5:3b` (run `20260605_183036`)

All 18 items had `retrieval_hit = 1` — every failure was **generation-side**, not retrieval. Three distinct failure modes were identified:

**1. Parametric memory overriding context (correctness = 0.0)**

The model's training knowledge contradicted what NIST actually wrote, and the model trusted its weights over the provided text.

| Question | Failure |
|---|---|
| Q3 (edge-case) | Model mapped transparency/explainability/interpretability to wrong questions despite the correct NIST text being in slot 1 of retrieved context |
| Q16 (reasoning) | Model claimed the document doesn't address versioning challenges for generative AI — the context explicitly does |

**2. Chunk boundary truncation (correctness = 0.0, different cause)**

The answer exists in the corpus but sits at or across a chunk boundary not included in the retrieved window.

| Question | Failure |
|---|---|
| Q12 (factual) | Retrieved chunk references where the six monitoring categories are defined but does not contain their names; model answered about location instead of content |
| Q17 (ambiguity) | Context fragment cuts off before the list of AI system examples |

**3. Completeness gaps (correctness = 0.5–0.75)**

Questions 1, 5, 6, 7, 9, 10, 13 all showed the same pattern: model identified the core concept correctly but stopped at the surface level, missing supporting details, qualifications, and specific examples present in the retrieved text.

### Per-question diagnosis — `scripts/06_diagnose.py`

When the full evaluation reveals failures, `06_diagnose.py` replays specific questions through all 9 pipeline stages and prints every intermediate result to the console:

| Step | What is printed |
|---|---|
| 1 | Question and expected answer |
| 2 | Dense retrieval — top-12 child chunks with cosine similarity scores |
| 3 | BM25 sparse retrieval — top-12 by BM25 score |
| 4 | RRF fusion table — combined ranks → top-6 cut with RRF score per candidate |
| 5 | Parent lookup — full parent text for each of the 6 retrieved blocks, including table neighbour expansion |
| 6 | Assembled context — the full `{context}` block with character count and token estimate |
| 7 | Full prompt — system + user message with token budget check (overflow warning if > 4,096 tokens) |
| 8 | LLM response — raw output with generation time |
| 9 | Comparison — expected vs actual side-by-side |

By default replays Q2 (success, faithfulness=1.0) and Q3 (failure, faithfulness=0.0) from run `20260605_183036`. Edit the `find_case()` call at the bottom of the script to replay any other question.

```powershell
python scripts\06_diagnose.py
```
### First confirmed failure cases (from diagnostic run `06_diagnose.py`)

| Question | Root cause | Fix |
|---|---|---|
| Q3 (edge-case) | Parametric memory override — correct chunk in slot 1, model ignored it; three near-duplicate Figure 4 caption chunks flooded slots 2, 4, 6 | Max-per-chapter cap in `_assemble_hybrid_context` (~10 lines) |
| Q12 (factual) | Six monitoring categories live in a table; the table's chunk ranked RRF #7, just outside the top-6 cut | Cross-encoder already integrated; needs semantic chunking or two-level summary index |
| Q16 (reasoning) | 98-token boilerplate chunk scored BM25 rank #1 (false positive), pushing real answer to rank #7 | Minimum chunk length filter before BM25 indexing |
| Q17 (ambiguity) | Context fragment cuts off before the list of AI system examples (chunk boundary) | Increase chunk overlap to 200+ tokens or 25% of chunk size |

### Verification telemetry — `scripts/07_eval_verified.py`

Re-runs a configurable subset of questions with MiniCheck instrumentation active and writes extended telemetry to the output CSV and JSONL. `QUESTION_SUBSET = [3, 12, 16, 17]` targets the four known failing questions; set to `[]` to run all 18.

Adds four extra columns compared to `04_eval_pipeline.py` output:

| Column | Meaning |
|---|---|
| `risk_level` | LOW / HIGH — from `HIGH_RISK_PATTERNS` classifier |
| `minicheck_triggered` | Whether MiniCheck ran for this item |
| `minicheck_faithful` | MiniCheck verdict (True = supported, False = contradiction) |
| `retry_triggered` | Whether the citation-forced retry prompt fired |

Output: `data/05_eval/rag_eval_verified_<timestamp>_qwen2.5-3b.csv` and `.jsonl`

The script header logs the full model configuration:
```
Reader model    : qwen2.5:3b
Verifier model  : bespoke-minicheck
Retrieval top_k : 6
HIGH RISK patterns: 7
```

```powershell
python scripts\07_eval_verified.py
```

---

## Answer Verification Approaches

**Problem:** Even with correct retrieval (all 18 items had `retrieval_hit = 1`), small models sometimes override retrieved context with parametric memory. Q3 had the correct answer chunk ranked #1 in retrieval — yet the model answered from its weights, not the text.

Three verification approaches were evaluated:

| Approach | How it works | Verdict |
|---|---|---|
| Conditional pipeline | Classify query confidence upfront; route risky queries through extra verification | Unreliable — Q3 had high retrieval confidence yet still failed. Misclassification lets wrong answers through silently. |
| Universal NLI check | Every answer passes through a small NLI model (~180 MB) checking for contradiction with retrieved chunks | Catches parametric override regardless of upfront prediction. Blind to missing-context failures (returns neutral, not contradiction). |
| **Combined adaptive** | Generate first, then classify risk, then run NLI only on HIGH_RISK queries | **Chosen.** Low-risk queries pay zero overhead; NLI only runs where it matters; misclassification only affects performance, never correctness. |

**Why the combined approach is better than either alone:**
- Low-risk queries (reasoning, negative) scored well in eval and don't need verification — zero overhead
- NLI check runs *after* generation, not before — a misclassified query delivers an unverified answer at worst; it can never silently skip the check
- Retry uses a targeted citation-forced prompt rather than full regeneration: *"find the exact sentence in the context that answers this, then answer using only that sentence"*
- Neither approach alone solves Q12 — when the answer is not in the retrieved chunks, NLI returns neutral. Q12 requires retrieval-side fixes regardless of verification layer

**HIGH_RISK_PATTERNS** (trigger MiniCheck verification):
- `which \w+ answers` / `which \w+ describes` / `which \w+ property` — selection/mapping questions
- `what are the (N) \w+` — enumeration questions
- `what is \w+ defined` / `how does nist \w+ define` — definition questions
- `when a user asks` — scenario/role questions

**LIST_PATTERNS** (trigger incompleteness disclaimer, subset of above):
- `what are the (six|five|four|...) \w+`
- `list the \w+`
- `name the (N)`

### Failed experiment — model routing for list queries

**Hypothesis:** Routing list-enumeration queries (`what are the N X`) to `llama3.2:3b` would improve completeness on Q12.

**Implementation:** `_is_list_query()` regex detection in `nistaichat.py`, calling `llama3.2:3b` when matched.

**Result (eval run `20260605_234738`):** No improvement. `llama3.2:3b` generated 8 plausible-sounding category names for Q12; MiniCheck flagged all 8 as unsupported (0/8); retry also failed (0/N again). `qwen2.5:3b` had previously returned fewer claims with 3/4 supported — the larger model was *more confidently wrong*. Additionally, switching models caused a regression on Q16 (Cp/Cr dropped from 0.75 to 0.50) due to Ollama evicting and reloading models under RAM pressure. Q12 latency doubled from 118s to 245s with no accuracy gain.

**Conclusion:** Root cause confirmed as retrieval, not the reader model. The six monitoring categories are in a table in NIST AI 800-4 that is not present in the top-6 retrieved chunks. Any reader model fails when the answer is not in context. Model routing was reverted.

### Final Verification Logic implemented

**Step 1 — Dual Search (Hybrid Retrieval)**
When you ask a question, the system searches the NIST document database in two ways simultaneously:
- **Dense search (FAISS):** Converts your question into a math vector and finds chunks that are semantically similar — meaning they have the same meaning even if they use different words.
- **Sparse search (BM25):** Tokenizes your question and finds chunks that contain the same exact keywords.

Why: Neither method alone is reliable. Dense search misses exact technical terms like "GOVERN" or "SSDF". Sparse search misses paraphrasing. Running both together catches what either one would miss alone.

**Step 2 — Merge Results (Reciprocal Rank Fusion)**
Both result lists are merged into one ranked list using RRF — a formula that gives credit to chunks that ranked highly in both searches.

Why: You can't just average the raw scores because FAISS and BM25 use incompatible score scales. RRF only looks at rank position, so it's scale-agnostic and unbiased.

**Step 3 — Precision Re-ranking (Cross-Encoder)**
The top ~10 merged candidates are re-scored by a cross-encoder model (`ms-marco-MiniLM-L-6-v2`). Unlike the embedding search, the cross-encoder reads both the question and the chunk together at once to give a more precise relevance score.

Why: Embedding similarity is fast but approximate. The cross-encoder is much more accurate but too slow to run on the whole database — so it runs only on the already-filtered shortlist.

**Step 4 — Parent Document Expansion**
The FAISS index stores small child chunks for precise matching. Once the best chunks are found, the system looks up their parent passages (larger text blocks) from `parent_store.jsonl`. If a result is a table row, neighboring rows are also pulled in.

Why: Small chunks are better for retrieval precision, but the LLM needs more surrounding text to give a complete answer. The parent store gives you both: small for finding, large for answering.

**Step 5 — Generate Draft Answer (`qwen2.5:3b`)**
The top parent passages, your conversation history (last ~1,000 characters), and your question are assembled into a prompt. The reader model `qwen2.5:3b` generates an answer. Markdown formatting like bold and headers is stripped from the output.

Why: The prompt instructs the model to stick to the context and use original NIST policy language. Stripping markdown keeps the output clean for the UI.

**Step 6 — Risk Classification**
The question is pattern-matched against a list of 7 HIGH_RISK phrases (e.g. `which term describes`, `how does NIST define`). If it matches, the query is flagged as HIGH risk.

Why: Some questions have a single precise correct answer from the source document. For those, a hallucinated paraphrase is worse than no answer at all — so they get extra verification.

**Step 7 — Faithfulness Verification (MiniCheck, HIGH risk only)**
For HIGH-risk queries, every sentence in the draft answer is checked against the top retrieved chunk using `bespoke-minicheck` — a dedicated fact-checking model. It returns "Yes" (supported) or "No" (not supported) per sentence. If more sentences fail than pass, the answer is flagged as unfaithful.

Why: The reader LLM can plausibly paraphrase something that is technically wrong. MiniCheck is a small, fast model specialized for grounding claims against documents — cheaper and more focused than asking the reader to self-check.

**Step 8 — Retry with Stricter Prompt (if unfaithful)**
If MiniCheck flags the draft, the system runs the LLM again with a much tighter prompt: "find the exact sentence that answers this, copy it directly, say you don't know otherwise."

Why: Giving the LLM a second chance with a narrower instruction often recovers the correct grounded answer instead of just discarding the response entirely.

**Step 9 — List Query Disclaimer**
If the question matches patterns like `what are the five...` or `list the...`, a note is appended: "This list may be incomplete. Refer to the original NIST source document."

Why: NIST documents contain long enumerations that are split across multiple chunks during indexing. A retrieved context may only contain part of the list, so the user is warned rather than misled.

The overall philosophy is **defense in depth**: each step compensates for the weakness of the one before it, trading off latency for accuracy at each layer.

```
                    ┌─────────────────────┐
                    │    USER QUESTION     │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
     ┌──────────────────┐         ┌──────────────────┐
     │  Dense Search    │         │  Sparse Search   │
     │    (FAISS)       │         │     (BM25)       │
     │ Semantic meaning │         │ Exact keywords   │
     └────────┬─────────┘         └────────┬─────────┘
                │                             │
                └──────────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Merge via RRF     │
                    │  (Rank Fusion)      │
                    │ Combines both lists │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Cross-Encoder     │
                    │    Re-ranking       │
                    │ Precise top-6 pick  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Parent Expansion   │
                    │  + Table Neighbors  │
                    │ Small→Large context │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Build Prompt      │
                    │ Context + History   │
                    │    + Question       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Generate Answer    │
                    │   (qwen2.5:3b)      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Risk Classifier    │
                    └────────┬────────────┘
                             │
                ┌────────────┴────────────┐
              HIGH                       LOW
                │                         │
                ▼                         │
     ┌──────────────────┐                 │
     │   MiniCheck      │                 │
     │  Verify Answer   │                 │
     └────────┬─────────┘                 │
              │                           │
      ┌───────┴───────┐                   │
    PASS            FAIL                  │
      │               │                   │
      │               ▼                   │
      │    ┌─────────────────────┐        │
      │    │   Retry — Strict    │        │
      │    │   Citation Prompt   │        │
      │    └──────────┬──────────┘        │
      │               │                   │
      └───────┬────────┘                  │
              └──────────────┬────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  List Disclaimer    │  ← only if list query
                  │  (if applicable)    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Final Answer       │
                  │  + Sources          │
                  └─────────────────────┘
```

---

## UI Integration

### UI Question Configuration

`data\06_questions_ui\questions_config.json` drives all topic categories and predefined question buttons in the Gradio UI. Questions can be added, edited, or enriched without touching the Python script. Three modes are supported per question entry:

**Mode 1 — Plain string (standard RAG)**
```json
"What are open challenges of defending AI attacks"
```
Button label, chat display, and RAG query are identical.

**Mode 2 — Enriched prompt**
```json
{
  "label": "Provide me an overview of attacks grouped by predictive and generative AI.",
  "prompt": "Using NIST AI 100-2e2023, provide a structured overview of attacks..."
}
```
The UI button and chat history show only the short `"label"`. The RAG engine receives the full `"prompt"` invisibly. Use when the visible question is too vague for the LLM but you don't want to clutter the UI.

**Mode 3 — Direct answer (RAG bypassed)**
```json
{
  "label": "Provide me an overview of attacks grouped by predictive and generative AI.",
  "answer": "PREDICTIVE AI — three objective dimensions:\n- Availability: ..."
}
```
The RAG pipeline and LLM are bypassed entirely. The pre-written answer is returned verbatim. Use for deterministic taxonomy questions where LLM generation adds no value and risks paraphrasing or dropping items.

Priority rule: if both `"prompt"` and `"answer"` are present, `"answer"` takes precedence.

To add a question: edit `questions_config.json`, save, and restart the application (file is read once at startup).

## Progress Bar — Why P80 Percentile

The Gradio UI shows a countdown progress bar while the LLM generates. This requires an upfront duration estimate bucketed by query risk class (LOW vs HIGH), derived from past query logs.

Three estimation strategies were evaluated:

| Estimate | Pros | Cons |
|---|---|---|
| Maximum | Never underestimates | One 293s HIGH-risk retry outlier permanently inflates all future estimates — typical 110–165s query would show ~double the real wait time |
| Mean | Most accurate on average | ~50% of queries finish before the estimate, making the bar hit zero while still running — looks broken |
| **P80** | Works correctly 80% of the time; outlier-resistant | Slight implementation complexity |

**P80 implementation:** `sorted(lst)[int(len(lst) * 0.8)]`

The bar erring slightly high (query finishes while countdown still shows) is preferable to hitting zero early. A bar that "finishes fast" feels good; a bar that "runs out" feels broken. P80 achieves this without being dominated by single extreme outliers.

---

## Limitations and Outlook

1. **Automation of manual steps**
   - Image extraction: Docling image export was unreliable for multi-column NIST figures; currently done via manual screenshots. Integrating a reliable image extractor would remove this bottleneck.
   - Semantic anchoring: LLM-generated captions and table summaries are currently added manually using Gemini. Integrating automatic captioning into the ingestion pipeline would make it reproducible.

2. **Bigger hardware and larger reader model**
   The entire verification layer (MiniCheck, retry, disclaimer) exists because `qwen2.5:3b` is too small to reliably enumerate structured lists or self-correct. Expecially NIST-documents contain large tables that need more reasoning capabilities. On a machine with a GPU or ≥32 GB RAM, a 7B+ reader model (e.g. `qwen2.5:7b`, `llama3.1:8b`) would likely handle enumeration queries correctly without needing the verification overhead — simplifying the pipeline significantly.

3. **Q12 retrieval gap — table chunks below top-6 cut**
   The six post-deployment monitoring categories in NIST AI 800-4 live in a table whose chunk ranks RRF #7 — just outside the top-6 window. Neither larger models nor model routing fixes this; the answer is not in the retrieved context. Fixes: semantic chunking to keep full tables as single chunks, or a two-level summary index so the table is reachable via a document-level summary hit.

4. **Streaming output**
   The ~115s generation time is a CPU floor on this hardware. Streaming tokens as they generate would make the latency invisible to the user — the answer appears progressively rather than after a 115s wait. The Ollama REST API supports streaming; the Gradio UI supports it via `gr.ChatInterface` with a generator function. Currently not implemented.

5. **Retrieval improvements**
   Several retrieval enhancements are documented in the backlog:
   - Max-per-chapter cap: prevents near-duplicate figure caption chunks from flooding the top-6 slots (Q3 failure mode)
   - Minimum BM25 chunk length filter: removes 98-token boilerplate chunks that score BM25 rank #1 as false positives (Q16 failure mode)
   - Multi-query retrieval: generate 3 query variants, retrieve for each, merge — wider coverage at the cost of +1 LLM call (~40s)
   - HyDE (Hypothetical Document Embeddings): generate a hypothetical answer, embed it, retrieve — improves recall on abstract questions

6. **Evaluation coverage**
   The golden dataset has 18 questions across 7 documents. This is sufficient for identifying failure modes but too small to track regressions reliably across architecture changes. Expanding to 50–100 questions with automated regression tracking would make the eval pipeline more useful as a development tool.

7. **Knowledge base currency**
   NIST publishes new AI guidance documents regularly. The ingestion pipeline is reproducible (run `01_parse_nist.py` + `02_ingest_to_vector_db.py` on new cleaned markdown), but adding new documents requires manual cleaning and semantic anchoring (image captions, table summaries). Automating this would allow the knowledge base to stay current.

8. **Windows-only compatibility**
   All file paths in the scripts are hardcoded as Windows paths (e.g. `C:\aibot\data\04_vector_storage`). The application currently only runs on Windows. Adapting it for Linux or macOS would require replacing all hardcoded `C:\aibot\` absolute paths with relative paths or environment-variable-based configuration across `nistaichat.py`, `nistaichat_ui.py`, and the ingestion and eval scripts. Using `pathlib.Path` instead of raw string literals would make the codebase cross-platform without per-machine code changes.
