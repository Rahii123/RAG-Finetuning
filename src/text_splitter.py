"""
text_splitter.py  –  Production-Grade Chunker
===============================================
Improvements over v1:
  ✅ Deduplicated metadata logic (no more repeated if-blocks)
  ✅ Richer metadata: guideline_name, section_header, section_type,
     chunk_position, total_chunks
  ✅ Section-aware splitting: detects numbered section headers (e.g., "1.0 DEFINITION")
     and prepends the section header to every child chunk for context
  ✅ Chunk size 800 chars ≈ 200 tokens  (overlap 100)
  ✅ Filters out boilerplate foreword / references chunks
"""

import os
import re
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================
# 1. PATHS
# =====================================================
INPUT_FOLDER  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
OUTPUT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chunks"))
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================================
# 2. METADATA REGISTRY  (clean, single source of truth)
# =====================================================

METADATA_REGISTRY = [
    # (keyword_list, year, disease_type, guideline_name)
    {
        "keywords": ["hypertension"],
        "year": 2013,
        "disease_type": "hypertension",
        "guideline_name": "Malaysian CPG Hypertension 2013",
    },
    {
        "keywords": ["dsa700"],
        "year": 2020,
        "disease_type": "hypertension",
        "guideline_name": "DSA700 Hypertension Guideline 2020",
    },
    {
        "keywords": ["t2dm", "diabetes", "cpg_t2dm"],
        "year": 2020,
        "disease_type": "diabetes",
        "guideline_name": "Malaysian CPG Type 2 Diabetes 2020",
    },
    {
        "keywords": ["type 2 diabetes", "type2 diabetes"],
        "year": 2020,
        "disease_type": "diabetes",
        "guideline_name": "Type 2 Diabetes in Children & Adolescents – PubMed 2020",
    },
    {
        "keywords": ["bacterial arthritis", "arthritis"],
        "year": 2023,
        "disease_type": "bacterial_arthritis",
        "guideline_name": "PIDS/IDSA Acute Bacterial Arthritis Guidelines 2023",
    },
    {
        "keywords": ["pneumonia"],
        "year": 2003,
        "disease_type": "pneumonia",
        "guideline_name": "Healthcare-Associated Pneumonia Guideline 2003",
    },
    {
        "keywords": ["hand-hygiene", "hand hygiene"],
        "year": 2020,
        "disease_type": "infection_control",
        "guideline_name": "Hand Hygiene Guidelines 2020",
    },
    {
        "keywords": ["covid", "antigen"],
        "year": 2023,
        "disease_type": "covid19",
        "guideline_name": "IDSA COVID-19 Antigen Testing Guidelines 2023",
    },
    {
        "keywords": ["immunization", "vaccine", "rr6007", "rr6210"],
        "year": 2025,
        "disease_type": "immunization",
        "guideline_name": "CDC Adult Immunization Schedule 2025",
    },
]

def get_metadata(document_id: str) -> dict:
    """
    Resolve rich metadata from document_id using the registry.
    Returns a dict with year (int), disease_type, and guideline_name.
    """
    clean_id = document_id.replace("\u00a0", " ").strip().lower()
    for entry in METADATA_REGISTRY:
        if any(kw in clean_id for kw in entry["keywords"]):
            return {
                "year": entry["year"],
                "disease_type": entry["disease_type"],
                "guideline_name": entry["guideline_name"],
            }
    return {
        "year": 0,
        "disease_type": "general_guideline",
        "guideline_name": document_id,
    }

# =====================================================
# 3. SECTION DETECTOR
# =====================================================

# Matches patterns like "1.0 DEFINITION", "8.1 HYPERTENSION AND DIABETES", "CHAPTER 5"
SECTION_HEADER_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+[A-Z][A-Z\s&/(),\-]{4,}|[A-Z][A-Z\s&/(),\-]{8,})$",
    re.MULTILINE,
)

BOILERPLATE_KEYWORDS = [
    "references", "bibliography", "disclosure statement",
    "sources of funding", "foreword", "acknowledgement",
    "table of contents", "list of tables", "working group",
    "chairperson", "external reviewers",
]

def is_boilerplate(text: str) -> bool:
    """Return True if this chunk is administrative / boilerplate (not clinical)."""
    lower = text.lower()
    # Filter out table of contents, index, section headers, empty formatting
    if len(lower.strip()) < 100:
        return True
    if any(kw in lower for kw in ["table of contents", "index", "foreword", "appendix", "references", "glossary"]):
        return True
    # If chunk is mostly formatting or headers (e.g., >60% lines are short)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and sum(len(l) < 30 for l in lines) / len(lines) > 0.6:
        return True
    return sum(1 for kw in BOILERPLATE_KEYWORDS if kw in lower) >= 2

def split_into_sections(content: str) -> list[tuple[str, str]]:
    """
    Split raw document text into (section_header, section_body) tuples.
    Falls back to a single section if no headers detected.
    """
    matches = list(SECTION_HEADER_PATTERN.finditer(content))
    if len(matches) < 3:
        # Document has no clear headers – treat as one section
        return [("General", content)]

    sections = []
    for i, match in enumerate(matches):
        header = match.group(0).strip()
        start  = match.end()
        end    = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body   = content[start:end].strip()
        if body:
            sections.append((header, body))
    return sections

# =====================================================
# 4. TEXT SPLITTER  (800 char ≈ 200 tokens, overlap 100)
# =====================================================

# Improved chunking: larger size, more overlap, avoid splitting lists/tables
class CustomTextSplitter(RecursiveCharacterTextSplitter):
    def split_text(self, text):
        # Avoid splitting inside bullet lists or tables
        # If a chunk ends in a bullet, extend to next non-bullet line
        chunks = super().split_text(text)
        merged_chunks = []
        buffer = ""
        for chunk in chunks:
            lines = chunk.splitlines()
            if lines and (lines[-1].strip().startswith(('-','*','•','·')) or lines[-1].strip().startswith('|')):
                buffer += chunk + "\n"
                continue
            if buffer:
                merged_chunks.append(buffer + chunk)
                buffer = ""
            else:
                merged_chunks.append(chunk)
        if buffer:
            merged_chunks.append(buffer)
        return merged_chunks

text_splitter = CustomTextSplitter(
    chunk_size=1000,  # increased chunk size
    chunk_overlap=180,  # increased overlap
    separators=["\n\n", "\n", ". ", " "],
)

# =====================================================
# 5. PROCESS ALL FILES
# =====================================================

total_docs   = 0
total_chunks = 0

for filename in sorted(os.listdir(INPUT_FOLDER)):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(INPUT_FOLDER, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    document_id = data.get("document_id", filename.replace(".json", ""))
    content     = data.get("content", "").strip()
    source      = data.get("source", "clinical_guideline")

    if not content:
        print(f"⚠️  Skipped (empty): {document_id}")
        continue

    print(f"\n📄 Processing: {document_id}")
    meta_info = get_metadata(document_id)

    # Split into clinical sections first
    sections = split_into_sections(content)
    print(f"   Detected {len(sections)} sections")

    output_chunks = []
    chunk_index   = 0

    for section_header, section_body in sections:
        # Determine section type from header keywords
        hdr_lower = section_header.lower()
        if any(k in hdr_lower for k in ["definition", "classif"]):
            section_type = "definition_classification"
        elif any(k in hdr_lower for k in ["diagnos", "assessment"]):
            section_type = "diagnosis"
        elif any(k in hdr_lower for k in ["pharmacol", "treatment", "management", "drug"]):
            section_type = "pharmacological_management"
        elif any(k in hdr_lower for k in ["non-pharmacol", "lifestyle", "weight", "sodium", "exercise"]):
            section_type = "non_pharmacological_management"
        elif any(k in hdr_lower for k in ["special", "elderly", "pregnan", "children", "diabetes and", "stroke"]):
            section_type = "special_populations"
        elif any(k in hdr_lower for k in ["key message", "summary", "recommendation"]):
            section_type = "recommendations"
        else:
            section_type = "general"

        # Prepend the section header to every sub-chunk for context
        prefixed_body = f"[Section: {section_header}]\n{section_body}"
        sub_chunks    = text_splitter.split_text(prefixed_body)

        for sub_chunk in sub_chunks:
            if is_boilerplate(sub_chunk):
                continue  # Skip administrative filler
            if len(sub_chunk.strip()) < 100:
                continue  # Skip tiny fragments

            output_chunks.append({
                "chunk_id": f"{document_id}_chunk_{chunk_index}",
                "document_id": document_id,
                "text": sub_chunk.strip(),
                "metadata": {
                    "disease_type":   meta_info["disease_type"],
                    "year":           meta_info["year"],
                    "guideline_name": meta_info["guideline_name"],
                    "section_header": section_header,
                    "section_type":   section_type,
                    "source":         source,
                    "chunk_position": chunk_index,
                },
            })
            chunk_index += 1

    # Write total_chunks back into each record's metadata
    for rec in output_chunks:
        rec["metadata"]["total_chunks"] = len(output_chunks)

    # Save output
    out_name  = filename.replace(".json", "_chunks.json")
    out_path  = os.path.join(OUTPUT_FOLDER, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_chunks, f, indent=2, ensure_ascii=False)

    print(f"   ✅  {len(output_chunks)} chunks saved → {out_name}")
    total_docs   += 1
    total_chunks += len(output_chunks)

print(f"\n🎯 Done! {total_docs} documents → {total_chunks} total chunks")
