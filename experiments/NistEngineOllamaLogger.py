import logging
import os
import time
from typing import List, Dict, Tuple, Any

import ollama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("rag.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONSTANTS & CONFIG ---
EMBEDDING_MODEL = "all-minilm"
READER_MODEL = "smollm2:1.7b"
VECTOR_DB_PATH = r"C:\aibot\data\03_vector_storage"

# Updated Prompt to include CHAT HISTORY
PROMPTS = {
    "generator": (
        "You are a NIST advisor. Use the provided context and previous conversation to answer. "
        "Cite sections like [Section Name]. If the answer isn't in context, say you don't know.\n\n"
        "PREVIOUS CONVERSATION:\n{chat_history}\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}"
    ),
}

class NistChatEngine:
    """
    RAG Engine with integrated latency profiling and chat history management.
    """

    def __init__(self):
        self._validate_paths()
        
        logger.info("Initializing Models and Vector Store...")
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vector_db = self._load_vector_db()
        
        # New: State for chat history
        self.chat_history: List[Dict[str, str]] = []
        
        logger.info(f"Connection to Ollama verified. Using {READER_MODEL}")

    def _validate_paths(self) -> None:
        if not os.path.exists(VECTOR_DB_PATH):
            raise FileNotFoundError(f"Vector Database not found at: {VECTOR_DB_PATH}")

    def _load_vector_db(self) -> FAISS:
        return FAISS.load_local(
            VECTOR_DB_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    # --- HISTORY MANAGEMENT METHODS (Adapted from Repo) ---
    def format_chat_history(self, max_turns: int = 5) -> str:
        if not self.chat_history:
            return "No previous conversation."

        recent_history = self.chat_history[-max_turns:]
        formatted = []
        for turn in recent_history:
            formatted.append(f"Human: {turn['human']}")
            formatted.append(f"Assistant: {turn['assistant']}")
        return "\n".join(formatted)

    def add_to_history(self, human_message: str, assistant_message: str) -> None:
        self.chat_history.append({"human": human_message, "assistant": assistant_message})

    def clear_history(self) -> None:
        self.chat_history = []
        logger.info("Chat history cleared.")

    def _call_llm(self, system_message: str, user_message: str) -> str:
        # Note: Using your original 'ollama' library call
        response = ollama.chat(
            model=READER_MODEL,
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': user_message},
            ],
            options={
                'temperature': 0.1,
                'num_predict': 450 
            }
        )
        return response['message']['content']

    def _assemble_context(self, question: str) -> Tuple[str, List[Dict[str, str]]]:
        docs = self.vector_db.similarity_search(question, k=3)
        context_parts = []
        source_metadata = []
        
        for doc in docs:
            section_name = doc.metadata.get("SubSection", doc.metadata.get("Section", "General"))
            formatted_part = f"--- Section: {section_name} ---\n{doc.page_content}"
            context_parts.append(formatted_part)
            source_metadata.append({
                "section": section_name,
                "preview": doc.page_content[:150] + "..."
            })
            
        return "\n\n".join(context_parts), source_metadata

    def ask(self, question: str) -> Tuple[str, List[Dict[str, str]]]:
        try:
            start_total = time.time()

            # 1. Format the history for the prompt
            formatted_history = self.format_chat_history()

            # 2. Profile Retrieval
            logger.info(f"Question: {question}")
            retrieval_start = time.time()
            context, sources = self._assemble_context(question)
            retrieval_time = time.time() - retrieval_start

            # 3. Profile Generation
            logger.info("Generator: Drafting response with history...")
            generation_start = time.time()
            
            # Fill the template with context, history, and the question
            full_prompt = PROMPTS["generator"].format(
                chat_history=formatted_history,
                context=context,
                question=question
            )
            
            # We pass the formatted string to the user role to keep the _call_llm structure
            draft_answer = self._call_llm("You are a helpful NIST advisor.", full_prompt)
            
            generation_time = time.time() - generation_start

            # 4. Update History
            self.add_to_history(question, draft_answer)

            total_time = time.time() - start_total
            logger.info(f"Query Finished. Total Time: {total_time:.2f}s | R: {retrieval_time:.2f}s | G: {generation_time:.2f}s")

            return draft_answer, sources

        except Exception as e:
            logger.error(f"Error during RAG execution: {e}", exc_info=True)
            return "I encountered an error while processing your request.", []

# --- UPDATED TEST LOOP ---
if __name__ == "__main__":
    engine = NistChatEngine()
    
    print("NIST AI Advisor ready. Type 'exit' to quit or 'clear' to reset memory.")
    
    while True:
        query = input("\n[User]: ").strip()
        if query.lower() == 'exit':
            break
        if query.lower() == 'clear':
            engine.clear_history()
            continue
        if not query:
            continue

        draft_answer, sources = engine.ask(query)
        
        print("\nSOURCES:")
        for i, source in enumerate(sources, 1):
            print(f"   {i}. Section: {source['section']}")

        print(f"\n[Assistant]:\n{draft_answer}")