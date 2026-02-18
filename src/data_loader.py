import fitz  # PyMuPDF
import os
from tqdm import tqdm
import json

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a single PDF file.
    """
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def load_pdfs_to_json(input_folder, output_folder):
    """
    Read all PDFs in input_folder, extract text,
    and save each as a JSON file in output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    for file in tqdm(os.listdir(input_folder)):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, file)
            text = extract_text_from_pdf(pdf_path)

            # Save JSON
            json_path = os.path.join(output_folder, file.replace(".pdf", ".json"))
            data = {
                "document_id": file.replace(".pdf", ""),
                "source": "clinical_guideline",
                "content": text
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    print("All PDFs converted to JSON.")

if __name__ == "__main__":
    input_folder = "../data/raw_pdfs"
    output_folder = "../data/processed"
    load_pdfs_to_json(input_folder, output_folder)
    
