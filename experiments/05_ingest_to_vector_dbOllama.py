import os
import sys
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
MODEL_ID ="all-minilm"
INPUT_FILE = r"C:\aibot\data\02_clean_markdown\NIST.SP.800-218A_CLEANED.md"
DB_EXPORT_PATH = r"C:\aibot\data\03_vector_storage"

# MiniLM-L6-v2 has a max sequence length of 256 tokens.
# 1000 characters is roughly 200-240 tokens, which is the "Goldilocks zone."
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class NISTMetadataSplitter:
    """
    Handles context-aware splitting. 
    It preserves the NIST section hierarchy in the metadata of each chunk.
    """
    
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Stage 1: Split by Headers to capture context
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Section"),
                ("##", "SubSection"),
                ("###", "SubSubSection"),
            ],
            strip_headers=False # Keep headers in the text for the LLM to read
        )
        
        # Stage 2: Split large sections into model-compatible sizes
    
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            add_start_index=True,
            strip_whitespace=True,
            # Priority: Paragraphs > Headers > Tables > Lines
            separators=["\n\n", "\n# ", "\n## ", "\n|", "\n", " ", ""]
        )

    def process_text(self, raw_text: str, file_name: str) -> List[Document]:
        """Runs the two-stage split and adds source metadata."""
        logger.info(f"✂️ Stage 1: Extracting Markdown hierarchy...")
        sections = self.header_splitter.split_text(raw_text)
        
        logger.info(f"✂️ Stage 2: Sub-splitting sections into {self.chunk_size} char chunks...")
        chunks = self.text_splitter.split_documents(sections)
        
        # Final metadata polish
        for chunk in chunks:
            chunk.metadata["source"] = file_name
            # Ensure table chunks are identifiable
            if "|" in chunk.page_content:
                chunk.metadata["content_type"] = "table"
            else:
                chunk.metadata["content_type"] = "prose"
                
        return chunks


class VectorStoreManager:
    """Handles the creation and local saving of the FAISS index."""
    
    def __init__(self, model_id: str):
        # Using langchain-huggingface for modern compatibility
        self.embeddings = self.embeddings = OllamaEmbeddings(
            model=model_id
        )

    def build_index(self, documents: List[Document], storage_path: str):
        logger.info(f"🧠 Generating embeddings for {len(documents)} chunks...")
        vector_db = FAISS.from_documents(documents, self.embeddings)
        
        os.makedirs(storage_path, exist_ok=True)
        vector_db.save_local(storage_path)
        logger.info(f"💾 FAISS index saved to: {storage_path}")


def main():
    try:
        # 1. Load the cleaned NIST file
        if not os.path.exists(INPUT_FILE):
            logger.error(f"File not found: {INPUT_FILE}")
            return

        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        # 2. Split with Metadata (Context-Aware)
        splitter = NISTMetadataSplitter(CHUNK_SIZE, CHUNK_OVERLAP)
        chunks = splitter.process_text(content, os.path.basename(INPUT_FILE))

        # 3. Embed and Save
        manager = VectorStoreManager(MODEL_ID)
        manager.build_index(chunks, DB_EXPORT_PATH)

        logger.info("✅ Professional Ingestion Complete.")

    except Exception as e:
        logger.error(f"Critical error in pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()