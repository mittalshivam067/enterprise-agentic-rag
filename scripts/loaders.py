import os
import pandas as pd
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter  

def parse_document(file_path: str, filename: str) -> List[Document]:
    """
    Parses PDF, TXT, CSV, or Excel files into a list of LangChain Document objects
    with enriched metadata tags.
    """
    ext = filename.split('.')[-1].lower()
    docs = []

    if ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        docs.append(Document(page_content=text, metadata={"source": filename}))

    elif ext == 'pdf':
        reader = PdfReader(file_path)
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                docs.append(Document(page_content=text, metadata={"source": filename, "page": page_idx + 1}))

    elif ext in ['csv', 'xlsx', 'xls']:
        df = pd.read_csv(file_path) if ext == 'csv' else pd.read_excel(file_path)

        # 1. Summary Chunk: Column names, size, and header sample
        summary_text = (
            f"Dataset Name: {filename}\n"
            f"Columns: {', '.join(df.columns.astype(str))}\n"
            f"Total Rows: {len(df)}\n\n"
            f"Sample Data:\n{df.head(3).to_string(index=False)}"
        )
        docs.append(Document(page_content=summary_text, metadata={"source": filename, "type": "table_summary"}))

        # 2. Data Chunks: Group rows into blocks of 15-20 rows
        row_batch_size = 20
        for start_idx in range(0, len(df), row_batch_size):
            batch_df = df.iloc[start_idx:start_idx + row_batch_size]
            docs.append(Document(page_content=batch_df.to_string(index=False), metadata={"source": filename, "type": "table_data", "row_range": f"{start_idx + 1}-{min(start_idx + row_batch_size, len(df))}"}))

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return docs

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    """
    Splits text documents into overlapping semantic chunks while preserving metadata.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""])
    return splitter.split_documents(documents)

if __name__ == "__main__":
    # Quick sanity test
    test_file = "test.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Enterprise Knowledge Agent Test.\nThis is a sample document for RAG ingestion.")


    parsed = parse_document(test_file, test_file)
    chunks = chunk_documents(parsed)
    print(f"Loaded {len(parsed)} doc, created {len(chunks)} chunks.")
    print("Metadata:", chunks[0].metadata)
    os.remove(test_file)