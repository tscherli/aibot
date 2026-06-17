import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
# Ensure these match your ingestion script exactly
DB_DIR = r"C:\aibot\data\03_vector_storage"
MODEL_ID = "all-minilm"

def inspect_tables():
    # 1. Initialize the Ollama bridge (Zero-RAM impact)
    print(f"🔄 Connecting to Ollama for model: {MODEL_ID}...")
    embeddings = OllamaEmbeddings(model=MODEL_ID)

    # 2. Load the DB
    if not os.path.exists(DB_DIR):
        print(f"❌ Error: Database directory not found at {DB_DIR}")
        return

    print("📂 Loading FAISS index...")
    vector_db = FAISS.load_local(
        DB_DIR, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # 3. Extract chunks
    # We look for the 'content_type' metadata we set during ingestion
    all_docs = vector_db.docstore._dict.values()
    
    # Logic: Prioritize metadata, fallback to "|" character search
    table_chunks = [
        doc for doc in all_docs 
        if doc.metadata.get("content_type") == "table" or "|" in doc.page_content
    ]

    print(f"\n📊 RESULTS")
    print(f"Total chunks in DB: {len(all_docs)}")
    print(f"Chunks identified as tables: {len(table_chunks)}\n")

    if not table_chunks:
        print("⚠️ No tables found. Check if your Markdown contains '|' characters.")
        return

    # 4. Print the first 2 table chunks to check their "health"
    for i, doc in enumerate(table_chunks[:2]):
        print(f"--- Table Chunk {i+1} ---")
        print(f"Source Section: {doc.metadata.get('Section', 'Unknown')}")
        print(f"Sub-Section: {doc.metadata.get('SubSection', 'N/A')}")
        print("-" * 20)
        print(doc.page_content)
        print("-" * 30 + "\n")

if __name__ == "__main__":
    inspect_tables()