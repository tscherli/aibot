from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = r"C:\aibot\data\3_vector_storage"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Load the DB
embeddings = HuggingFaceEmbeddings(model_name=MODEL_ID)
vector_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

# Extract all chunks and look for tables
all_docs = vector_db.docstore._dict.values()
table_chunks = [doc.page_content for doc in all_docs if "|" in doc.page_content]

print(f"Found {len(table_chunks)} chunks containing table data.\n")

# Print the first 2 table chunks to check their "health"
for i, content in enumerate(table_chunks[:2]):
    print(f"--- Table Chunk {i+1} ---")
    print(content)
    print("-" * 30)