import os
import sys
import json
import logging
from collections import Counter
from typing import List

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
MODEL_ID          = "bge-m3:567m"
STORAGE_DIR       = r"C:\aibot\data\04_vector_storage"
CHILD_STORE_PATH  = r"C:\aibot\data\04_vector_storage\child_store.jsonl"
PARENT_STORE_PATH = r"C:\aibot\data\04_vector_storage\parent_store.jsonl"
EMBED_BATCH_SIZE  = 50

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class ChildStoreProcessor:
    """
    Reads child_store.jsonl and produces one LangChain Document per record.
    Child chunks are short focused texts used for retrieval precision.
    Full parent text is looked up at query time in nistaichat via parent_store.jsonl.
    """

    def extract_chunks_and_ids(self) -> tuple[List[Document], List[str]]:
        documents: List[Document] = []
        ids:       List[str]      = []

        with open(CHILD_STORE_PATH, encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSONL line {line_num}: {e}")
                    continue

                chunk_id = record.get("id", f"unknown_{line_num}")
                text     = record.get("page_content", "")
                meta     = record.get("metadata", {})

                if not text:
                    continue

                doc = Document(
                    page_content=text,
                    metadata={
                        "parent_link":  meta.get("parent_link", chunk_id),
                        "source":       meta.get("source",       "Unknown"),
                        "content_type": meta.get("content_type", "prose"),
                    }
                )
                documents.append(doc)
                ids.append(chunk_id)

        return documents, ids


class VectorStoreManager:
    """Manages local FAISS vector database operations using Ollama embeddings."""

    def __init__(self, model_id: str):
        self.embeddings = OllamaEmbeddings(model=model_id)

    def build_and_save_index(
        self,
        documents:    List[Document],
        explicit_ids: List[str],
        storage_path: str,
    ) -> None:
        total = len(documents)
        logger.info(f"Generating embeddings for {total} chunks via {MODEL_ID} (batch size {EMBED_BATCH_SIZE})...")

        first_docs = documents[:EMBED_BATCH_SIZE]
        first_ids  = explicit_ids[:EMBED_BATCH_SIZE]
        vector_db  = FAISS.from_documents(
            documents=first_docs,
            embedding=self.embeddings,
            ids=first_ids,
        )
        logger.info(f"  Embedded {len(first_docs)}/{total}")

        for start in range(EMBED_BATCH_SIZE, total, EMBED_BATCH_SIZE):
            batch      = documents[start : start + EMBED_BATCH_SIZE]
            batch_ids  = explicit_ids[start : start + EMBED_BATCH_SIZE]
            vector_db.add_documents(batch, ids=batch_ids)
            logger.info(f"  Embedded {min(start + EMBED_BATCH_SIZE, total)}/{total}")

        os.makedirs(storage_path, exist_ok=True)
        vector_db.save_local(storage_path)
        logger.info(f"FAISS index committed to: {storage_path}")


def _print_stats(chunks: List[Document]) -> None:
    type_counts = Counter(d.metadata.get("content_type", "unknown") for d in chunks)
    dirty       = [d for d in chunks
                   if "Chapter:" in d.metadata.get("source", "")
                   or "Content:" in d.metadata.get("source", "")]
    print("\n" + "=" * 50)
    print("  CHILD INDEX QUALITY REPORT")
    print("=" * 50)
    for ctype, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ctype:<14}: {n:>5}")
    print(f"  {'TOTAL':<14}: {len(chunks):>5}")
    print(f"  Dirty sources  : {len(dirty)}  (expect 0)")
    print("=" * 50)


def main():
    if not os.path.exists(CHILD_STORE_PATH):
        logger.error(f"child_store.jsonl not found: {CHILD_STORE_PATH}  — run 01_parse_nist.py first.")
        sys.exit(1)

    logger.info(f"Reading child chunks from: {CHILD_STORE_PATH}")
    processor   = ChildStoreProcessor()
    chunks, ids = processor.extract_chunks_and_ids()

    if not chunks:
        logger.warning("No chunks extracted from JSONL — aborting.")
        return

    logger.info(f"Loaded {len(chunks)} chunks. Building FAISS index...")

    manager = VectorStoreManager(MODEL_ID)
    manager.build_and_save_index(chunks, ids, STORAGE_DIR)

    _print_stats(chunks)

    logger.info("Indexing done.")
    logger.info(f"  VECTOR_DB_PATH = r\"{STORAGE_DIR}\"")


if __name__ == "__main__":
    main()
