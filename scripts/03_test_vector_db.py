import os
import sys
import json
import logging
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# --- CONFIGURATION (Must match your ingestion setup) ---
MODEL_ID          = "bge-m3:567m"
STORAGE_DIR       = r"C:\aibot\data\04_vector_storage"
PARENT_STORE_PATH = r"C:\aibot\data\04_vector_storage\parent_store.jsonl"

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def test_vector_database():
    try:
        # 1. Initialize the embedding engine connection
        logger.info(f"Connecting to embedding engine: {MODEL_ID}")
        embeddings = OllamaEmbeddings(model=MODEL_ID)

        # 2. Load the FAISS database index from local disk
        logger.info(f"Loading FAISS index from: {STORAGE_DIR}")
        if not os.path.exists(os.path.join(STORAGE_DIR, "index.faiss")):
            logger.error("Could not find index.faiss! Run your ingestion/patch scripts first.")
            return

        vector_db = FAISS.load_local(
            STORAGE_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True  # Trusted local DB generation loop
        )

        def run_query(label: str, query: str, k: int = 3) -> None:
            logger.info(f"Query [{label}]: '{query}'")
            results = vector_db.similarity_search_with_score(query, k=k)
            print("\n" + "=" * 80)
            print(f"  {label}")
            print("=" * 80)
            for rank, (doc, score) in enumerate(results, start=1):
                print(f"\n[RANK {rank}] (Distance Score: {score:.4f})")
                print("-" * 50)
                print(f"TEXT CONTENT:\n{doc.page_content.strip()}")
                print("-" * 50)
                print("METADATA ENTRIES:")
                for key, val in doc.metadata.items():
                    print(f"  • {key}: {val}")
                print("-" * 50)
                citation = (
                    f"{doc.metadata.get('source', 'Unknown Document')} "
                    f"[{doc.metadata.get('parent_link', 'N/A')}]"
                )
                print(f"CITATION: {citation}")
                print("=" * 80)

        # 3. General prose query
        run_query(
            "QUERY 1 — PROSE: secure software development",
            "secure software development framework practice and tasks",
        )

        # 4. Table-targeted query — verifies table_id field is populated in results
        run_query(
            "QUERY 2 — TABLE: AI risk governance recommendations",
            "AI risk governance recommended actions and considerations",
        )

        # 5. Parent-child wiring check
        # Load parent store and verify retrieved child chunks resolve correctly
        if not os.path.exists(PARENT_STORE_PATH):
            print("\n--- PARENT-CHILD WIRING CHECK ---")
            print(f"  ! parent_store.jsonl not found at {PARENT_STORE_PATH}")
        else:
            parent_store = {}
            with open(PARENT_STORE_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        parent_store[record["id"]] = record

            wiring_results = vector_db.similarity_search(
                "risk management table recommendations", k=10
            )
            total          = len(wiring_results)
            resolved       = 0
            missing_parent = []
            missing_table_id = []

            for d in wiring_results:
                pl     = d.metadata.get("parent_link")
                record = parent_store.get(pl)
                if not record:
                    missing_parent.append(pl)
                    continue
                resolved += 1
                if d.metadata.get("content_type") == "table":
                    if not record.get("metadata", {}).get("table_id"):
                        missing_table_id.append(pl)

            print("\n--- PARENT-CHILD WIRING CHECK ---")
            print(f"  Chunks retrieved             : {total}")
            print(f"  Parent records resolved      : {resolved} / {total}  (expect {total})")
            print(f"  Missing parent records       : {len(missing_parent)}  (expect 0)")
            print(f"  Table chunks missing table_id: {len(missing_table_id)}  (expect 0)")
            if missing_parent:
                for pl in missing_parent:
                    print(f"    ! unresolved parent_link={pl}")
            if missing_table_id:
                for pl in missing_table_id:
                    print(f"    ! missing table_id for parent_link={pl}")

    except Exception as e:
        logger.error(f"Failed to query vector store: {e}", exc_info=True)


if __name__ == "__main__":
    test_vector_database()