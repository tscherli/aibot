import os
import re
import json
import logging
import sys
from typing import List, Dict, Tuple, Set

from langchain_core.documents import Document

# --- CONFIGURATION ---
INPUT_DIRECTORY    = r"C:\aibot\data\02_clean_markdown"
OUTPUT_STORAGE_DIR = r"C:\aibot\data\04_vector_storage"
PARENT_REGISTRY_PATH = os.path.join(OUTPUT_STORAGE_DIR, "parent_store.jsonl")
CHILD_STORE_PATH     = os.path.join(OUTPUT_STORAGE_DIR, "child_store.jsonl")

# --- ELEMENT DETECTION PATTERNS ---
BOLD_TABLE_RE   = re.compile(r'^\*\*\s*Table\s+',  re.IGNORECASE)
BOLD_FIGURE_RE  = re.compile(r'^\*\*\s*Figure\s+', re.IGNORECASE)
FOOTNOTE_DEF_RE = re.compile(r'^\[\^([\w\d]+)\]:\s*(.+)')
REF_ENTRY_RE    = re.compile(r'^-\s+\[\d+\]\s+[A-Z]')  # - [47] Author Name...

def _is_bullet(s: str) -> bool:
    return s.startswith("- ") or s.startswith("* ") or s.startswith("  - ")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ===========================================================================
# ID EXTRACTION
# ===========================================================================
class ComplianceTokenExtractor:
    """Regex engine for extracting and normalizing framework reference IDs."""

    def __init__(self) -> None:
        self._phrase_splitter_regex = re.compile(
            r"\b(Govern|Measure|Manage|Map|GV|ME|MN|MP)\b", re.IGNORECASE
        )
        self._universal_id_regex = re.compile(r"\b([A-Z]{2,7})[-.\s](\d+(?:\.\d+)*)\b")
        self._prefix_normalization_map = {
            "GV": "GOVERN", "ME": "MEASURE", "MN": "MANAGE", "MP": "MAP"
        }
        # Extended patterns for standards body references
        self._extended_id_patterns = [
            re.compile(r'ISO(?:/IEC)?\s*\d{4,5}(?:[-:]\d+)*',           re.IGNORECASE),
            re.compile(r'(?:NIST\s+)?SP\s+\d{3}-\d{3}[A-Z]?',           re.IGNORECASE),
            re.compile(r'NIST\s+AI\s+\d{3}(?:-\d+)?(?:[A-Z]|e\d+)?',    re.IGNORECASE),
            re.compile(r'(?:E\.?O\.?|Executive\s+Order)\s+\d{5}',        re.IGNORECASE),
            re.compile(r'FIPS\s+(?:PUB\s+)?\d+(?:-\d+)*',                re.IGNORECASE),
            re.compile(r'RFC\s+\d{4,5}',                                   re.IGNORECASE),
            re.compile(r'IEEE\s+\d{3,5}(?:\.\d+)*',                       re.IGNORECASE),
        ]

    def extract_and_normalize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens: Set[str] = set()

        # NIST AI RMF function phrase extraction (GOVERN 1.2, MAP 3.1, etc.)
        string_fragments = self._phrase_splitter_regex.split(text)
        active_prefix: str = ""
        for fragment in string_fragments:
            frag_lower = fragment.lower().strip()
            if frag_lower in ["govern", "gv"]:   active_prefix = "GOVERN"
            elif frag_lower in ["measure", "me"]: active_prefix = "MEASURE"
            elif frag_lower in ["manage", "mn"]:  active_prefix = "MANAGE"
            elif frag_lower in ["map", "mp"]:     active_prefix = "MAP"
            elif active_prefix:
                for decimal in re.findall(r"\b\d+\.\d+\b", fragment):
                    tokens.add(f"{active_prefix}_{decimal}")

        # Universal alphanumeric ID pattern (PW_1.1, RV_3.1, etc.)
        for prefix, numeric_sequence in self._universal_id_regex.findall(text):
            norm = self._prefix_normalization_map.get(prefix.upper(), prefix.upper())
            tokens.add(f"{norm}_{numeric_sequence}")

        # Extended standards body patterns
        for pattern in self._extended_id_patterns:
            for match in pattern.findall(text):
                normalised = re.sub(r'\s+', '_', match.strip().upper())
                tokens.add(normalised)

        return sorted(list(tokens))


# ===========================================================================
# PARSER
# ===========================================================================
class ProductionMarkdownParser:
    """
    State machine parser for NIST AI markdown documents.

    Improvements over v1:
    - List buffering: consecutive bullet lines → one entry with list_items metadata
    - Paragraph buffering: consecutive prose lines → one entry per paragraph
    - Format A table detection: ## Table N headers (100-1 style)
    - Format B table detection: **Table N** bold headers (800-4 style)
    - Sibling IDs: table rows carry sibling_ids pointing to the other rows in their table
    - Footnote resolution: [^N] inline refs get footnote text appended to their chunk
    - Reference list skipping: - [47] Author... entries are not indexed
    - No synopsis: removed from all entries
    - metadata dict: every entry carries source, chapter, section, content_type
    - JSONL output: one JSON record per line (via BulkIngestionManager)
    """

    def __init__(self, token_extractor: ComplianceTokenExtractor) -> None:
        self.extractor = token_extractor

    def parse_file(self, file_path: str, file_id: int) -> Tuple[List[Document], Dict[str, dict]]:
        child_documents: List[Document]  = []
        parent_registry: Dict[str, dict] = {}

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except IOError as e:
            logger.error(f"Cannot read {file_path}: {e}")
            return [], {}

        # --- STATE ---
        doc_title          = "Unknown Framework"
        file_name          = os.path.basename(file_path)
        prose_section      = "Global Context"
        current_mode       = "Prose"
        active_object_name = ""
        is_parsing_table_rows = False
        table_headers: List[str] = []
        figure_buffer: List[str] = []

        # New state
        list_buffer:       List[str]       = []
        prose_buffer:      List[str]       = []
        table_row_ids:     List[str]       = []
        table_counter:     int             = 0
        element_counter:   int             = 0
        footnote_registry: Dict[str, str]  = {}

        # ------------------------------------------------------------------
        # Nested flush helpers (use closures to avoid long argument lists)
        # ------------------------------------------------------------------
        def flush_list() -> None:
            nonlocal element_counter
            if not list_buffer:
                return
            items    = [b.lstrip("-* ").strip() for b in list_buffer]
            full_text = (
                f"Document: {doc_title}\nChapter: {prose_section}\n"
                f"Section: {active_object_name}\nContent:\n"
                + "\n".join(list_buffer)
            )
            eid  = f"f{file_id}_l{element_counter}"
            refs = self.extractor.extract_and_normalize(" ".join(items))
            parent_registry[eid] = {
                "text": full_text,
                "refs": refs,
                "metadata": {
                    "source":       doc_title,
                    "chapter":      prose_section,
                    "section":      active_object_name,
                    "content_type": "list",
                    "list_size":    len(items),
                    "list_items":   items,
                },
            }
            child_documents.append(Document(
                page_content=full_text,
                metadata={"parent_link": eid, "source": doc_title, "content_type": "list"},
            ))
            element_counter += 1
            list_buffer.clear()

        def flush_prose() -> None:
            nonlocal element_counter
            if not prose_buffer:
                return
            combined = " ".join(prose_buffer)
            prose_buffer.clear()
            if len(combined) < 40:   # skip noise: page numbers, lone symbols
                return
            eid      = f"f{file_id}_p{element_counter}"
            full_text = (
                f"Document: {doc_title}\nChapter: {prose_section}\n"
                f"Content: {combined}"
            )
            refs = self.extractor.extract_and_normalize(combined)
            parent_registry[eid] = {
                "text": full_text,
                "refs": refs,
                "metadata": {
                    "source":       doc_title,
                    "chapter":      prose_section,
                    "section":      active_object_name,
                    "content_type": "prose",
                },
            }
            child_documents.append(Document(
                page_content=full_text,
                metadata={"parent_link": eid, "source": doc_title, "content_type": "prose"},
            ))
            element_counter += 1

        def flush_all() -> None:
            flush_list()
            flush_prose()

        def finalize_table_siblings() -> None:
            if len(table_row_ids) < 2:
                return
            total = len(table_row_ids)
            for rid in table_row_ids:
                if rid in parent_registry:
                    parent_registry[rid]["metadata"]["sibling_ids"] = [
                        r for r in table_row_ids if r != rid
                    ]
                    parent_registry[rid]["metadata"]["table_total"] = total

        # ------------------------------------------------------------------
        # Main parse loop
        # ------------------------------------------------------------------
        for line in lines:
            stripped_line = line.strip()

            # Paragraph boundary: flush prose buffer on blank line
            if not stripped_line:
                if current_mode == "Prose":
                    flush_prose()
                    if list_buffer:
                        flush_list()
                continue

            # Skip horizontal rules and image links
            if stripped_line == "---":
                continue
            if stripped_line.startswith("!["):
                continue

            # ----------------------------------------------------------
            # HEADER PROCESSING
            # ----------------------------------------------------------
            if stripped_line.startswith(("# ", "## ", "### ")):

                # Finalise any open table before mode change
                if current_mode == "Table" and table_row_ids:
                    finalize_table_siblings()
                    table_row_ids.clear()

                # Flush figure buffer before mode change
                if current_mode == "Figure" and figure_buffer:
                    element_counter = self._flush_figure_buffer(
                        figure_buffer, doc_title, prose_section,
                        active_object_name, file_id, element_counter,
                        child_documents, parent_registry,
                    )
                    figure_buffer.clear()

                # Flush prose/list buffers before mode change
                flush_all()

                if stripped_line.startswith("# "):
                    doc_title    = stripped_line.replace("#", "").strip()
                    current_mode = "Prose"

                elif stripped_line.startswith("## "):
                    section_title = stripped_line.replace("##", "").strip()
                    # Format A: ## Table N  (100-1 style)
                    if re.match(r'^Table\s+', section_title, re.IGNORECASE):
                        active_object_name    = section_title
                        current_mode          = "Table-Pending"
                        table_headers.clear()
                        table_row_ids.clear()
                        table_counter        += 1
                    else:
                        prose_section         = section_title
                        current_mode          = "Prose"
                        is_parsing_table_rows = False

                elif stripped_line.startswith("### "):
                    object_title          = stripped_line.replace("###", "").strip()
                    active_object_name    = object_title
                    is_parsing_table_rows = False

                    if "table" in object_title.lower():
                        current_mode = "Table-Pending"
                        table_headers.clear()
                        table_row_ids.clear()
                        table_counter += 1
                    elif "figure" in object_title.lower() or "diagram" in object_title.lower():
                        current_mode = "Figure"
                        figure_buffer.clear()
                    else:
                        current_mode  = "Prose"
                        prose_section = object_title
                continue

            # Dynamic transition: Table-Pending → Table on first pipe row
            if current_mode == "Table-Pending" and stripped_line.startswith("|"):
                current_mode          = "Table"
                is_parsing_table_rows = False

            # Strip blockquote markers (used inside figure technical descriptions)
            content_cleaned = stripped_line.lstrip("> ").strip()

            # ----------------------------------------------------------
            # ROUTE 1: PRE-TABLE DESCRIPTION (text before pipe headers)
            # ----------------------------------------------------------
            if current_mode == "Table-Pending":
                eid = f"f{file_id}_p{element_counter}"
                element_counter += 1
                text = (
                    f"Document: {doc_title}\nContext: {active_object_name}\n"
                    f"Description: {content_cleaned}"
                )
                parent_registry[eid] = {
                    "text": text,
                    "refs": self.extractor.extract_and_normalize(content_cleaned),
                    "metadata": {
                        "source":       doc_title,
                        "chapter":      prose_section,
                        "section":      active_object_name,
                        "content_type": "prose",
                    },
                }
                child_documents.append(Document(
                    page_content=text,
                    metadata={"parent_link": eid, "source": doc_title, "content_type": "prose"},
                ))
                continue

            # ----------------------------------------------------------
            # ROUTE 2: TABLE ROWS
            # ----------------------------------------------------------
            elif current_mode == "Table":
                if stripped_line.startswith("|"):
                    if "---" in stripped_line:
                        is_parsing_table_rows = True
                        continue

                    cell_values = [c.strip() for c in stripped_line.split("|")[1:-1]]
                    if not is_parsing_table_rows:
                        table_headers         = cell_values
                        is_parsing_table_rows = True
                        continue

                    if len(cell_values) == len(table_headers):
                        row_data_map   = dict(zip(table_headers, cell_values))
                        serialized_row = " | ".join(f"{k}: {v}" for k, v in row_data_map.items())
                        parent_text    = (
                            f"Document: {doc_title} | Chapter: {prose_section} | "
                            f"Table: {active_object_name} | {serialized_row}"
                        )

                        base_refs  = self.extractor.extract_and_normalize(stripped_line)
                        final_refs = sorted(set(base_refs))

                        eid = f"f{file_id}_t{table_counter}_r{element_counter}"
                        element_counter += 1
                        table_row_ids.append(eid)

                        parent_registry[eid] = {
                            "text": parent_text,
                            "refs": final_refs,
                            "metadata": {
                                "source":       doc_title,
                                "chapter":      prose_section,
                                "section":      active_object_name,
                                "content_type": "table",
                                "table_id":     f"{doc_title}::{active_object_name}",
                                "table_title":  active_object_name,
                                "table_row":    len(table_row_ids),
                                "table_total":  0,    # set by finalize_table_siblings
                                "sibling_ids":  [],   # set by finalize_table_siblings
                            },
                        }
                        primary_cell = row_data_map.get(
                            "Recommendations [R], Considerations [C], and Notes [N]",
                            row_data_map.get(
                                "Suggested Action",
                                row_data_map.get("Requirement", stripped_line[:120])
                            )
                        )
                        child_documents.append(Document(
                            page_content=(
                                f"Doc: {doc_title} | Table: {active_object_name} | {primary_cell}"
                            ),
                            metadata={"parent_link": eid, "source": doc_title, "content_type": "table"},
                        ))
                    continue

                else:
                    # Non-pipe line: table has ended
                    finalize_table_siblings()
                    table_row_ids.clear()
                    current_mode          = "Prose"
                    is_parsing_table_rows = False
                    # fall through to Route 4 for this line

            # ----------------------------------------------------------
            # ROUTE 3: FIGURE BUFFER
            # ----------------------------------------------------------
            if current_mode == "Figure":
                figure_buffer.append(content_cleaned)
                continue

            # ----------------------------------------------------------
            # ROUTE 4: PROSE — with paragraph/list buffering and
            #           bold header detection
            # ----------------------------------------------------------
            if current_mode == "Prose":

                # Skip numbered reference list entries (bibliographic, not content)
                if REF_ENTRY_RE.match(content_cleaned):
                    continue

                # Collect footnote definitions — resolve into chunks in post-processing
                fn_match = FOOTNOTE_DEF_RE.match(content_cleaned)
                if fn_match:
                    footnote_registry[fn_match.group(1)] = fn_match.group(2)
                    continue

                # Format B: **Table N** bold header (800-4, 600-1 style)
                if BOLD_TABLE_RE.match(content_cleaned):
                    flush_all()
                    active_object_name = content_cleaned.strip("*").strip()
                    current_mode       = "Table-Pending"
                    table_headers.clear()
                    table_row_ids.clear()
                    table_counter += 1
                    continue

                # Bold figure header
                if BOLD_FIGURE_RE.match(content_cleaned):
                    flush_all()
                    if figure_buffer:
                        element_counter = self._flush_figure_buffer(
                            figure_buffer, doc_title, prose_section,
                            active_object_name, file_id, element_counter,
                            child_documents, parent_registry,
                        )
                        figure_buffer.clear()
                    active_object_name = content_cleaned.strip("*").strip()
                    current_mode       = "Figure"
                    figure_buffer.clear()
                    continue

                # Bullet list buffering
                if _is_bullet(content_cleaned):
                    flush_prose()
                    list_buffer.append(content_cleaned)
                    continue

                # Non-bullet line: flush any open list buffer first
                if list_buffer:
                    flush_list()

                # Accumulate into paragraph buffer
                prose_buffer.append(content_cleaned)

        # ------------------------------------------------------------------
        # END OF FILE: flush all remaining open buffers
        # ------------------------------------------------------------------
        flush_list()
        flush_prose()

        if current_mode == "Figure" and figure_buffer:
            element_counter = self._flush_figure_buffer(
                figure_buffer, doc_title, prose_section,
                active_object_name, file_id, element_counter,
                child_documents, parent_registry,
            )

        if table_row_ids:
            finalize_table_siblings()

        # Post-processing: resolve inline footnote refs into their chunks
        self._resolve_footnotes(parent_registry, footnote_registry)

        return child_documents, parent_registry

    # ------------------------------------------------------------------
    # FLUSH HELPERS
    # ------------------------------------------------------------------
    def _flush_figure_buffer(
        self,
        buffer:       List[str],
        doc_title:    str,
        section:      str,
        object_name:  str,
        file_id:      int,
        counter:      int,
        child_list:   List[Document],
        registry:     Dict[str, dict],
    ) -> int:
        """Flush figure buffer as a single entry. Returns incremented counter."""
        aggregated = " ".join(buffer)
        eid        = f"f{file_id}_fig{counter}"
        full_text  = (
            f"Document: {doc_title} | Chapter: {section} | "
            f"Visual Element: {object_name} | Details: {aggregated}"
        )
        base_refs = (
            self.extractor.extract_and_normalize(aggregated)
            or self.extractor.extract_and_normalize(object_name)
        )
        registry[eid] = {
            "text": full_text,
            "refs": base_refs,
            "metadata": {
                "source":       doc_title,
                "chapter":      section,
                "section":      object_name,
                "content_type": "figure",
            },
        }
        child_list.append(Document(
            page_content=f"Doc: {doc_title} | Visual: {object_name} | {aggregated[:180]}",
            metadata={"parent_link": eid, "source": doc_title, "content_type": "figure"},
        ))
        return counter + 1

    def _resolve_footnotes(
        self,
        registry:          Dict[str, dict],
        footnote_registry: Dict[str, str],
    ) -> None:
        """Append resolved footnote text to any chunk that contains an inline [^N] reference."""
        if not footnote_registry:
            return
        for entry in registry.values():
            for fn_key, fn_text in footnote_registry.items():
                if f"[^{fn_key}]" in entry["text"]:
                    entry["text"] += f"\n[Note {fn_key}: {fn_text[:300]}]"


# ===========================================================================
# BULK INGESTION — JSONL OUTPUT
# ===========================================================================
class BulkIngestionManager:
    """Walks INPUT_DIRECTORY, parses each markdown file, writes JSONL output."""

    def __init__(self, parser: ProductionMarkdownParser) -> None:
        self.parser = parser

    def process_all_files(self) -> None:
        if not os.path.exists(INPUT_DIRECTORY):
            logger.error(f"Input directory not found: {INPUT_DIRECTORY}")
            return

        os.makedirs(OUTPUT_STORAGE_DIR, exist_ok=True)

        total_chunks = 0
        file_count   = 0

        logger.info(f"Scanning     : {INPUT_DIRECTORY}")
        logger.info(f"Parent store : {PARENT_REGISTRY_PATH}")
        logger.info(f"Child store  : {CHILD_STORE_PATH}")

        total_children = 0

        with open(PARENT_REGISTRY_PATH, "w", encoding="utf-8") as parent_file, \
             open(CHILD_STORE_PATH,     "w", encoding="utf-8") as child_file:

            for entry in sorted(os.listdir(INPUT_DIRECTORY)):
                if not entry.lower().endswith((".md", ".markdown")):
                    continue
                if "merged" in entry.lower():
                    logger.info(f"Skipping merged file: {entry}")
                    continue

                file_count += 1
                target_path = os.path.join(INPUT_DIRECTORY, entry)
                logger.info(f"Parsing file #{file_count}: {entry}")

                file_children, file_parents = self.parser.parse_file(target_path, file_count)

                for chunk_id, chunk_data in file_parents.items():
                    record = {"id": chunk_id, **chunk_data}
                    parent_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1

                for doc in file_children:
                    record = {
                        "id":           doc.metadata.get("parent_link"),
                        "page_content": doc.page_content,
                        "metadata":     doc.metadata,
                    }
                    child_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_children += 1

        logger.info(f"Done. {file_count} files → {total_chunks} parent chunks → {PARENT_REGISTRY_PATH}")
        logger.info(f"Done. {file_count} files → {total_children} child chunks  → {CHILD_STORE_PATH}")

        self._print_stats(PARENT_REGISTRY_PATH)

    @staticmethod
    def _print_stats(jsonl_path: str) -> None:
        from collections import Counter
        counts: Counter = Counter()
        total = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                counts[record.get("metadata", {}).get("content_type", "unknown")] += 1
                total += 1
        print("\n=== parent_store_v2.jsonl content type distribution ===")
        for ctype, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {ctype:<12}: {n:>5}  ({100*n//total}%)")
        print(f"  {'TOTAL':<12}: {total:>5}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    token_extractor = ComplianceTokenExtractor()
    parser          = ProductionMarkdownParser(token_extractor)
    manager         = BulkIngestionManager(parser)
    manager.process_all_files()
