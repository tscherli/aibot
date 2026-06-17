import logging
import os
from typing import List, Dict, Tuple, Any

import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# --- CONSTANTS & CONFIG ---
# Moving these to a central place follows the "Single Source of Truth" principle
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
READER_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
VECTOR_DB_PATH = r"C:\aibot\data\03_vector_storage"

# Agent Instructions (Centralized for easy editing)
PROMPTS = {
    "generator": (
        "You are a NIST advisor. Use the provided context to answer. "
        "Cite sections like [Section Name]. If the answer isn't in context, say you don't know."
    ),
    "critic": (
        "You are a Critical Auditor. Compare the Answer to the Context. "
        "Verify grounding and citation accuracy. If incorrect or hallucinated, "
        "rewrite it to be 100% factual. If it is perfect, return the original answer."
    )
}

logger = logging.getLogger(__name__)

class NistChatEngine:
    """
    RAG Engine that implements a Generator-Critic agent loop to provide 
    validated answers based on NIST documentation.
    """

    def __init__(self):
        self._validate_paths()
        
        logger.info("📡 Initializing Models and Vector Store...")
        
        # 1. Component Initialization
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_db = self._load_vector_db()
        
        # 2. Reader Pipeline Setup
        self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            READER_MODEL, 
            torch_dtype=torch.float32,
            device_map="auto" # Dynamically uses GPU if available, else CPU
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            temperature=0.1,
            max_new_tokens=450,
            trust_remote_code=True
        )

    def _validate_paths(self) -> None:
        """Ensures required data exists before starting."""
        if not os.path.exists(VECTOR_DB_PATH):
            raise FileNotFoundError(f"Vector Database not found at: {VECTOR_DB_PATH}")

    def _load_vector_db(self) -> FAISS:
        """Safe loading of the FAISS index."""
        return FAISS.load_local(
            VECTOR_DB_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def _call_llm(self, system_message: str, user_message: str) -> str:
        """Standardized interface for LLM interaction."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        output = self.pipe(prompt)
        # Professional parsing: Splitting by the assistant marker
        raw_text = output[0]['generated_text']
        return raw_text.split("<|im_start|>assistant")[-1].strip()

    def _assemble_context(self, question: str) -> Tuple[str, List[Dict[str, str]]]:
        """Retrieves documents and formats them with metadata."""
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
        """
        Public entry point: Executes the Generator-Critic agent loop.
        """
        try:
            # 1. Retrieval
            context, sources = self._assemble_context(question)

            # 2. Generation Phase
            logger.info("🤖 Generator: Drafting response...")
            gen_input = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
            draft_answer = self._call_llm(PROMPTS["generator"], gen_input)

            # 3. Criticism/Validation Phase
            logger.info("🛡️ Critic: Validating answer quality...")
            critic_input = f"CONTEXT:\n{context}\n\nDRAFT:\n{draft_answer}"
            final_validated_answer = self._call_llm(PROMPTS["critic"], critic_input)

            return final_validated_answer, sources

        except Exception as e:
            logger.error(f"Error during RAG execution: {e}")
            return "I encountered an error while processing your request.", []