# Enterprise Agentic RAG Knowledge & Decision Support System

An AI-powered enterprise knowledge assistant built using **Python, Streamlit, LangChain, Gemini, and ChromaDB**. It allows users to upload business documents and ask questions in natural language, receiving context-aware answers grounded in document content.

## Overview

This project demonstrates how Generative AI, Retrieval-Augmented Generation (RAG), and Agentic AI can work together to build an intelligent enterprise knowledge and decision support system.

Users can upload documents in multiple formats, retrieve relevant information through semantic search, and receive answers generated using an LLM.

## Key Features

- **Multi-format document ingestion:** Supports PDF, TXT, CSV, and Excel documents.
- **Document chunking:** Splits documents into smaller overlapping chunks for effective retrieval.
- **Vector database:** Uses ChromaDB to store and retrieve document embeddings.
- **Query decomposition:** Planner agent breaks complex questions into smaller search queries.
- **Semantic retrieval:** Finds relevant document content using similarity search.
- **Context-aware responses:** Generates answers based on retrieved document information.
- **Source attribution:** Provides references to supporting document content.
- **Hallucination detection:** Uses a grounding validation step to check generated answers against retrieved evidence.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| LLM | Google Gemini |
| AI Framework | LangChain |
| Vector Database | ChromaDB |
| Document Processing | PyPDF, Pandas, OpenPyXL |
| Environment Management | Python Virtual Environment |

## System Architecture

```text
                 User
                  |
                  v
          Streamlit Interface
               app.py
                  |
          +-------+-------+
          |               |
          v               v
    Document Upload    User Query
          |               |
          v               v
     loaders.py       Planner Agent
          |               |
          v               v
    Text Chunking    Search Queries
          |               |
          v               v
     Embeddings <--- Semantic Retrieval
          |               |
          v               v
       ChromaDB <---- Retrieved Context
                          |
                          v
                  Reasoning Agent
                          |
                          v
                  Grounding Validator
                          |
                          v
                 Answer with Sources
```

## Project Structure

```text
enterprise-rag/
│
├── app.py
├── agent.py
├── loaders.py
├── vectorstore.py
├── requirements.txt
├── DOCUMENTATION.md
├── README.md
├── .env.example
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-agentic-rag.git
cd enterprise-agentic-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Never commit your actual `.env` file or API key to GitHub.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## Agent Workflow

1. **Planner Agent:** Decomposes user questions into targeted search queries.
2. **Retrieval Coordinator:** Searches the vector database and collects relevant document chunks.
3. **Reasoning Agent:** Generates answers using retrieved context.
4. **Grounding Validator:** Checks whether the generated response is supported by the retrieved evidence.

## Example Use Cases

- Searching company policies and HR documents.
- Retrieving information from business reports.
- Answering questions about product documentation.
- Extracting relevant information from Excel and CSV files.
- Building an internal enterprise knowledge assistant.

## Challenges Addressed

- Processing structured and unstructured documents.
- Improving semantic retrieval accuracy.
- Handling complex multi-part questions.
- Reducing hallucinations through grounding validation.
- Preserving source information for answer traceability.

## Limitations and Future Enhancements

- Add OCR support for scanned PDF documents.
- Improve numerical analysis of large Excel datasets.
- Introduce user authentication and role-based access control.
- Support larger enterprise document collections.
- Explore cloud deployment and managed vector databases.

## Documentation

For detailed information about the system architecture, implementation, setup instructions, agent roles, challenges, and limitations, refer to **[DOCUMENTATION.md](DOCUMENTATION.md)**.

## Author

**Shivam Mittal**

Generative AI and Machine Learning Capstone Project.