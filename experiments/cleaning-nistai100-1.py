import re
import os
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TableSanitizer:
    """Specialized logic to collapse excessive whitespace and dashes inside Markdown tables."""

    @staticmethod
    def collapse_spaces(text: str) -> str:
        """Target lines starting and ending with '|' and shrink 2+ spaces into 1."""
        def shrink_row(match):
            return re.sub(r' {2,}', ' ', match.group(0))

        table_row_pattern = r'^\|.*\|$'
        return re.compile(table_row_pattern, re.MULTILINE).sub(shrink_row, text)

    @staticmethod
    def collapse_dashes(text: str) -> str:
        """
        Finds long sequences of dashes (3+) within table structures 
        and collapses them to a standard '---'.
        """
        return re.sub(r'-{4,}', '---', text)


class NistCleaner:
    """Handles the transformation of raw NIST AI 100-1 Markdown into RAG-ready text."""
    
    def __init__(self, anchor_text: str = "## Executive Summary"):
        # Set anchor to Executive Summary to skip the title pages and TOC
        self.anchor_text = anchor_text
        self.table_sanitizer = TableSanitizer()
        
        # Updated patterns specific to NIST AI 100-1
        self.junk_patterns = [
            r"NIST AI 100-1",
            r"Artificial Intelligence Risk Management Framework \(AI RMF 1\.0\)",
            r"This publication is available free of charge from: https://doi\.org/10\.6028/NIST\.AI\.100-1",
            r"January 2023",
            r"U\.S\. Department of Commerce.*",
            r"National Institute of Standards and Technology.*",
            r"Page \d+",  # Removes injected page numbers
            r"", # Removes image placeholders if not needed for RAG
        ]

    def _remove_front_matter(self, text: str) -> str:
        """Discards administrative front matter and Table of Contents."""
        if self.anchor_text in text:
            _, content = text.split(self.anchor_text, 1)
            return self.anchor_text + content
        logger.warning(f"Anchor '{self.anchor_text}' not found. Skipping front matter removal.")
        return text

    def _remove_boilerplate(self, text: str) -> str:
        """Removes repeated document headers and footers."""
        cleaned = text
        for pattern in self.junk_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _fix_formatting_noise(self, text: str) -> str:
        """Cleans up dot leaders and excessive vertical spacing."""
        text = re.sub(r"\.{4,}", " ", text) # Remove TOC dot leaders
        text = re.sub(r"\n{3,}", "\n\n", text) # Normalize paragraph spacing
        return text.strip()

    def process(self, raw_text: str) -> str:
        """The core transformation pipeline."""
        text = self._remove_front_matter(raw_text)
        text = self._remove_boilerplate(text)
        
        # Table specific cleaning
        text = self.table_sanitizer.collapse_spaces(text)
        text = self.table_sanitizer.collapse_dashes(text)
        
        text = self._fix_formatting_noise(text)
        return text


class FileHandler:
    """Utility for safe file operations."""
    
    @staticmethod
    def read(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def write(path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def main():
    # File configuration updated for NIST AI 100-1
    INPUT_PATH = r"C:\aibot\data\02_clean_markdown\NIST.AI.100-1.md"
    OUTPUT_PATH = r"C:\aibot\data\02_clean_markdown\NIST.AI.100-1_CLEANED.md"

    try:
        cleaner = NistCleaner()
        handler = FileHandler()

        logger.info(f"Reading raw NIST markdown: {INPUT_PATH}")
        raw_data = handler.read(INPUT_PATH)
        
        logger.info("Executing Sanitization Pipeline (Tailored for AI RMF 1.0)...")
        cleaned_data = cleaner.process(raw_data)
        
        handler.write(OUTPUT_PATH, cleaned_data)
        logger.info(f"✨ Success! Cleaned file saved to {OUTPUT_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to process NIST file: {e}")

if __name__ == "__main__":
    main()