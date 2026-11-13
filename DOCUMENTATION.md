# Enterprise Agentic RAG – Knowledge & Decision Support System

**Generative AI and Machine Learning Capstone Project**

---

## 1. Executive Summary

The Enterprise Agentic RAG (Retrieval-Augmented Generation) Knowledge & Decision Support System is an AI-powered application designed to help users retrieve information from enterprise documents and obtain context-aware answers using natural language.

The application allows users to upload documents in multiple formats, including PDF, TXT, CSV, and Excel. It processes these documents, converts their content into searchable vector representations, and stores them in a persistent ChromaDB vector database.

Unlike a basic RAG implementation, this project uses a multi-stage agentic workflow consisting of:

- **Planner Agent:** Breaks complex user questions into smaller search queries.
- **Retrieval Coordinator:** Retrieves relevant information from the vector database.
- **Reasoning Agent:** Generates answers based on retrieved document content.
- **Guardrail and Critic Agent:** Checks whether generated answers are supported by the retrieved evidence.

The application uses Streamlit for its user interface and Google's Gemini API for LLM-powered tasks.

The primary objective is to improve enterprise knowledge discovery, answer accuracy, source transparency, and hallucination control.

---

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Large Language Model | Gemini 3.6 Flash |
| LLM Integration | LangChain |
| Vector Database | ChromaDB |
| Document Processing | PyPDF, Pandas, OpenPyXL |
| Text Chunking | RecursiveCharacterTextSplitter |
| Embedding Model | Configured in vectorstore.py |
| Environment Management | Python Virtual Environment |
| Configuration | Python dotenv |

---

## 3. System Architecture

The application follows a modular architecture with separate components for document ingestion, vector storage, retrieval, reasoning, and response validation.

### Architecture Overview

```text
                  USER
                   |
                   v
       +------------------------+
       |   Streamlit UI         |
       |       app.py           |
       +-----------+------------+
                   |
          +--------+---------+
          |                  |
          v                  v
   DOCUMENT UPLOAD      USER QUESTION
          |                  |
          v                  v
   +--------------+   +------------------+
   | Document     |   | Planner Agent    |
   | Loaders      |   | plan_retrieval() |
   | loaders.py   |   +--------+---------+
   +------+-------+            |
          |                    v
          v            +------------------+
   +--------------+    | Retrieval        |
   | Text         |    | Coordinator      |
   | Chunking     |    | retrieve_context()|
   +------+-------+    +--------+---------+
          |                     |
          v                     v
   +--------------+    +------------------+
   | Embeddings   |<---| Semantic Search  |
   | Generation   |    | vectorstore.py   |
   +------+-------+    +--------+---------+
          |                     |
          v                     v
   +--------------+    +------------------+
   | ChromaDB     |--->| Retrieved Context|
   | Vector Store |    +--------+---------+
   +--------------+             |
                                v
                       +------------------+
                       | Reasoning Agent  |
                       | reason_and_generate()
                       +--------+---------+
                                |
                                v
                       +------------------+
                       | Guardrail Agent  |
                       | validate_grounding()
                       +--------+---------+
                                |
                                v
                       +------------------+
                       | Final Answer     |
                       | + Source Details |
                       +------------------+
```

### Core Components

**1. User Interface – `app.py`**

The Streamlit application provides:

- Multi-format document upload.
- Document processing and indexing.
- Conversational question-answering interface.
- Chat history management.
- Display of retrieved sources and validation results.
- Transparency into the agent workflow.

**2. Document Processing – `loaders.py`**

Responsible for extracting and preparing document content.

Supported formats include:

- PDF
- TXT
- CSV
- XLSX
- XLS

The document processing pipeline extracts content and divides it into smaller chunks using `RecursiveCharacterTextSplitter`.

Current chunking configuration:

- Chunk size: 1,000 characters.
- Chunk overlap: 150 characters.

Metadata is preserved where available to help identify the original document and its location.

**3. Vector Database – `vectorstore.py`**

Responsible for converting document chunks into embeddings and storing them in ChromaDB.

Key responsibilities:

- Generate vector embeddings using the configured embedding model.
- Store document vectors and associated metadata.
- Persist the vector database locally.
- Retrieve semantically similar document chunks.
- Apply similarity-score filtering to reduce irrelevant results.

**4. Agent Orchestration – `agent.py`**

Coordinates the multi-stage RAG workflow.

It uses Gemini through LangChain to perform query planning, answer generation, and grounding validation.

---

## 4. Agent Roles and Workflow

The application uses specialized stages to improve retrieval quality and response reliability.

### 4.1 Planner Agent – `plan_retrieval()`

The Planner Agent analyzes the incoming user question and identifies the most useful search terms.

For complex questions, it generates between one and three targeted sub-queries.

**Example:**

User question:

"What was the company's Q3 revenue growth compared with the previous year?"

Possible search queries:

- Q3 revenue growth rate.
- Third-quarter financial results revenue growth.
- Q3 revenue year-over-year percentage.

This helps retrieve relevant information from different sections of the uploaded documents.

### 4.2 Retrieval Coordinator – `retrieve_context()`

The Retrieval Coordinator executes semantic searches using the generated queries.

Its responsibilities include:

- Searching the ChromaDB vector database.
- Retrieving relevant document chunks.
- Filtering results based on similarity scores.
- Removing duplicate chunks.
- Preserving source metadata.

The retrieved information is passed to the Reasoning Agent.

### 4.3 Reasoning and Synthesis Agent – `reason_and_generate()`

The Reasoning Agent generates a natural-language answer using the retrieved document content.

Its key responsibilities are:

- Understand the retrieved context.
- Identify relevant facts.
- Combine information from multiple document chunks.
- Generate an answer grounded in the available evidence.
- Include source references where supported.
- Avoid making unsupported assumptions.

If sufficient information is unavailable, the application should communicate that limitation rather than inventing an answer.

### 4.4 Guardrail and Hallucination Auditor – `validate_grounding()`

The Guardrail Agent acts as an independent verification stage.

It compares the generated answer against the retrieved document context.

The validation response follows a structured format:

```text
VERDICT: PASS or FAIL
REASON: Explanation of the validation result
```

- **PASS:** The generated claims are supported by the retrieved context.
- **FAIL:** One or more claims are not adequately supported.

The application uses this validation result to identify potentially unsupported answers.

---

## 5. Project Structure

The project is organized into separate modules to maintain readability and simplify development.

```text
enterprise-rag/
│
├── app.py                 # Streamlit user interface
├── agent.py               # Agent orchestration and LLM workflow
├── loaders.py             # Document loading and text chunking
├── vectorstore.py         # Embeddings and ChromaDB operations
├── requirements.txt       # Python dependencies
├── .env                   # API key configuration (not submitted)
├── chroma_db/             # Persistent vector database
├── venv/                  # Python virtual environment
└── DOCUMENTATION.md       # Project documentation
```

---

## 6. Setup and Deployment Instructions

### 6.1 Prerequisites

Before running the application, ensure the following are installed:

- Python 3.10 or higher.
- pip package manager.
- A valid Google Gemini API key.
- Git (optional, for version control).

### 6.2 Clone or Extract the Project

Open PowerShell or a terminal and navigate to the project directory.

```powershell
cd C:\Users\shiva\enterprise-rag
```

### 6.3 Create a Virtual Environment

```powershell
python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 6.4 Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 6.5 Configure Environment Variables

Create a `.env` file in the project root directory.

Add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Do not commit or share the `.env` file because it contains sensitive credentials.

### 6.6 Launch the Application

Run the following command:

```powershell
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

Open this address in your browser to access the Enterprise Agentic RAG application.

### 6.7 Application Usage

1. Open the Streamlit application.
2. Upload one or more supported enterprise documents.
3. Allow the application to process and index the uploaded content.
4. Enter a question in the chat interface.
5. The Planner Agent generates relevant search queries.
6. The Retrieval Coordinator searches the vector database.
7. The Reasoning Agent generates an answer.
8. The Guardrail Agent checks the answer against the retrieved context.
9. Review the final answer and its source information.

---

## 7. Challenges Faced and Solutions

### 7.1 Processing Different Document Formats

**Challenge:**

Enterprise documents contain different types of content. PDF and TXT files contain unstructured text, whereas CSV and Excel files contain structured tabular information.

Using a single extraction strategy can lead to loss of useful information, especially in tabular documents.

**Solution:**

Implemented a modular document-loading pipeline in `loaders.py` to handle different file formats.

For tabular files, the processing pipeline converts structured data into text representations suitable for embedding and retrieval.

### 7.2 Improving Retrieval Accuracy

**Challenge:**

A simple semantic search may retrieve partially relevant or duplicate chunks, particularly when users ask broad or multi-part questions.

**Solution:**

Introduced query decomposition through the Planner Agent.

Multiple targeted searches are performed, and duplicate context chunks are removed before answer generation.

Similarity filtering is also applied to reduce irrelevant retrieval results.

### 7.3 Reducing Hallucinations

**Challenge:**

LLMs may generate plausible-sounding answers even when the uploaded documents do not contain sufficient evidence.

**Solution:**

Implemented a grounding-oriented generation prompt and a separate validation stage.

The Guardrail Agent evaluates the generated answer against the retrieved context and returns a PASS/FAIL verdict.

### 7.4 Handling LLM API Errors

**Challenge:**

During development, temporary API availability errors and quota limitations affected model execution.

Different Gemini models also returned response content in different formats, including strings and lists of content blocks.

**Solution:**

- Tested model availability using the Gemini API.
- Selected an available Gemini Flash model.
- Added response normalization to handle different response content formats.
- Added exception handling for API failures.
- Used clearer error messages to distinguish temporary service errors from quota exhaustion.

---

## 8. Current Limitations

Although the application provides an agentic RAG workflow, several limitations remain.

**1. Scanned PDFs and Complex Layouts**

The current PDF extraction approach may not reliably process scanned documents, embedded images, charts, and complex multi-column layouts. OCR or layout-aware document processing can improve this.

**2. Complex Numerical Analysis**

Semantic retrieval is effective for finding relevant information but is not a replacement for executing precise calculations across large datasets. A controlled Python/Pandas execution tool could improve numerical analysis.

**3. API Dependency**

The application depends on an external Gemini API. Temporary service outages, rate limits, and usage quotas may affect availability.

**4. Response Latency**

The sequential Planner, Reasoning, and Guardrail stages require multiple LLM requests. This can increase response time and API usage.

**5. Local Vector Database**

The current ChromaDB implementation is suitable for a prototype or local application. Larger enterprise deployments may require a managed vector database, authentication, access controls, and monitoring.

**6. Validation Limitations**

The Guardrail Agent uses an LLM-based evaluation process. Its PASS/FAIL result is not a mathematical guarantee that every generated claim is correct.

---

## 9. Future Enhancements

Potential improvements include:

- Integrating OCR and layout-aware PDF processing.
- Adding hybrid retrieval using keyword and semantic search.
- Introducing reranking models to improve retrieval relevance.
- Supporting more document formats.
- Adding role-based access control for enterprise users.
- Implementing document-level permissions.
- Supporting conversation memory and follow-up questions.
- Adding analytics and retrieval-quality evaluation.
- Introducing caching and request optimization to reduce API usage.
- Deploying the application using Docker and a cloud hosting platform.

---

## 10. Conclusion

The Enterprise Agentic RAG Knowledge & Decision Support System demonstrates how Retrieval-Augmented Generation and specialized AI agents can be combined to build an intelligent enterprise document question-answering application.

By combining document ingestion, semantic retrieval, query decomposition, grounded answer generation, and independent validation, the system provides a structured approach to enterprise knowledge discovery.

The project also highlights practical considerations involved in developing LLM-powered applications, including document preprocessing, vector search quality, API reliability, response validation, and deployment.

The modular architecture provides a foundation for future improvements and potential enterprise-scale adoption.

---

**Project:** Enterprise Agentic RAG – Knowledge & Decision Support System  
**Category:** Generative AI and Machine Learning Capstone Project  
**Primary LLM:** Google Gemini 3.6 Flash  
**Interface:** Streamlit  
**Vector Database:** ChromaDB