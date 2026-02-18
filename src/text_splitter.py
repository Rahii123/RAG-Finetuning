import os
import json
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text):
    """
    Basic cleaning for extracted PDF text.
    """
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text


def split_documents(input_folder, output_folder,
                    chunk_size=800,
                    chunk_overlap=100):
    """
    Load JSON documents, clean text,
    split into chunks, and save chunked output.
    """
    os.makedirs(output_folder, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for file in tqdm(os.listdir(input_folder)):
        if file.endswith(".json"):
            file_path = os.path.join(input_folder, file)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cleaned_text = clean_text(data["content"])
            chunks = splitter.split_text(cleaned_text)

            chunked_data = []

            for i, chunk in enumerate(chunks):
                chunked_data.append({
                    "document_id": data["document_id"],
                    "chunk_id": f"{data['document_id']}_chunk_{i}",
                    "content": chunk
                })

            output_path = os.path.join(output_folder, file)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunked_data, f, ensure_ascii=False, indent=2)

    print("All documents split into chunks.")


if __name__ == "__main__":
    input_folder = "../data/processed"
    output_folder = "../data/chunked"
    split_documents(input_folder, output_folder)
