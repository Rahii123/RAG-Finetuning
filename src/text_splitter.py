import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

# =====================================================
# 1️⃣ METADATA RULES (Keyword Based – Production Safe)
# =====================================================

def get_metadata(document_id: str):
    """
    Robust metadata matcher based on keyword detection.
    Avoids exact string dependency.
    """

    clean_id = document_id.replace("\u00a0", " ").strip().lower()

    # -------- Immunization --------
    if "immunization" in clean_id or "vaccine" in clean_id:
        return {
            "year": "2025",
            "disease_type": "immunization"
        }

    # -------- Hypertension --------
    if "hypertension" in clean_id:
        return {
            "year": "2013",
            "disease_type": "hypertension"
        }

    # -------- Diabetes --------
    if "diabetes" in clean_id or "t2dm" in clean_id:
        return {
            "year": "2024",
            "disease_type": "diabetes"
        }

    # -------- Arthritis --------
    if "arthritis" in clean_id:
        return {
            "year": "2023",
            "disease_type": "bacterial_arthritis"
        }

    # -------- Pneumonia --------
    if "pneumonia" in clean_id:
        return {
            "year": "2003",
            "disease_type": "pneumonia"
        }

    # -------- rr6007 --------
    if "rr6007" in clean_id:
        return {
            "year": "2011",
            "disease_type": "immunization"
        }
        # -------- dsa700 --------
    if "dsa700" in clean_id:
        return {
            "year": "2020",
            "disease_type": "hypertension"
        }

    # -------- Hand Hygiene --------
    if "hand-hygiene" in clean_id or "hand hygiene" in clean_id:
        return {
            "year": "2020",
            "disease_type": "infection_control"
        }

    # -------- COVID Antigen Testing --------
    if "covid" in clean_id and "antigen" in clean_id:
        return {
            "year": "2023",
            "disease_type": "covid19"
        }
        # -------- dsa700 --------
    if "dsa700" in clean_id:
        return {
            "year": "2020",
            "disease_type": "hypertension"
        }

    # -------- Hand Hygiene --------
    if "hand-hygiene" in clean_id or "hand hygiene" in clean_id:
        return {
            "year": "2020",
            "disease_type": "infection_control"
        }

    # -------- COVID Antigen Testing --------
    if "covid" in clean_id and "antigen" in clean_id:
        return {
            "year": "2023",
            "disease_type": "covid19"
        }
    # -------- dsa700 --------
    if "dsa700" in clean_id:
        return {
            "year": "2020",
            "disease_type": "hypertension"
        }

    # -------- Hand Hygiene --------
    if "hand-hygiene" in clean_id or "hand hygiene" in clean_id:
        return {
            "year": "2020",
            "disease_type": "infection_control"
        }

    # -------- COVID Antigen Testing --------
    if "covid" in clean_id and "antigen" in clean_id:
        return {
            "year": "2023",
            "disease_type": "covid19"
        }
    # -------- dsa700 --------
    if "dsa700" in clean_id:
        return {
            "year": "2020",
            "disease_type": "hypertension"
        }

    # -------- Hand Hygiene --------
    if "hand-hygiene" in clean_id or "hand hygiene" in clean_id:
        return {
            "year": "2020",
            "disease_type": "infection_control"
        }

    # -------- COVID Antigen Testing --------
    if "covid" in clean_id and "antigen" in clean_id:
        return {
            "year": "2023",
            "disease_type": "covid19"
        }
    

        # -------- dsa700 --------
    if "dsa700" in clean_id:
        return {
            "year": "2020",
            "disease_type": "hypertension"
        }

    # -------- Hand Hygiene --------
    if "hand-hygiene" in clean_id or "hand hygiene" in clean_id:
        return {
            "year": "2020",
            "disease_type": "infection_control"
        }

    # -------- COVID Antigen Testing --------
    if "covid" in clean_id and "antigen" in clean_id:
        return {
            "year": "2023",
            "disease_type": "covid19"
        }
 # -------- dsa700 --------
    if "dsa700" in clean_id:
        return {
            "year": "2020",
            "disease_type": "hypertension"
        }

    # -------- Hand Hygiene --------
    if "hand-hygiene" in clean_id or "hand hygiene" in clean_id:
        return {
            "year": "2020",
            "disease_type": "infection_control"
        }

    # -------- COVID Antigen Testing --------
    if "covid" in clean_id and "antigen" in clean_id:
        return {
            "year": "2023",
            "disease_type": "covid19"
        }

    # -------- Default --------
    return {
        "year": "unknown",
        "disease_type": "general_guideline"
    }


# =====================================================
# 2️⃣ PATHS
# =====================================================

INPUT_FOLDER = "data/processed"
OUTPUT_FOLDER = "data/chunks"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================================
# 3️⃣ TEXT SPLITTER
# =====================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

# =====================================================
# 4️⃣ PROCESS FILES
# =====================================================

for filename in os.listdir(INPUT_FOLDER):

    if filename.endswith(".json"):

        file_path = os.path.join(INPUT_FOLDER, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        document_id = data.get("document_id", "unknown_id")
        content = data.get("content", "")

        print(f"Processing: {document_id}")

        # 🔥 Robust metadata detection
        metadata_info = get_metadata(document_id)

        # Split text
        chunks = text_splitter.split_text(content)

        output_chunks = []

        for i, chunk in enumerate(chunks):
            output_chunks.append({
                "chunk_id": f"{document_id}_chunk_{i}",
                "document_id": document_id,
                "text": chunk,
                "metadata": {
                    "year": metadata_info["year"],
                    "disease_type": metadata_info["disease_type"],
                    "source": data.get("source", "clinical_guideline")
                }
            })

        # Save output
        output_file = os.path.join(
            OUTPUT_FOLDER,
            f"{filename.replace('.json', '')}_chunks.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_chunks, f, indent=4)

        print(f"✅ Saved {len(output_chunks)} chunks\n")

print("🎯 All documents processed successfully.")
