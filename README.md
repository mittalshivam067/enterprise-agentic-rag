# Enterprise Agentic RAG Knowledge & Decision Support System

An AI-powered enterprise knowledge assistant built using **Python, Streamlit, LangChain, Google Gemini, and ChromaDB**. The application enables users to upload business documents and ask questions in natural language, generating context-aware responses grounded in the uploaded content.

## 1. Project Overview

This project demonstrates how Generative AI, Retrieval-Augmented Generation (RAG), and Agentic AI can be combined to build an intelligent enterprise knowledge and decision support system.

The application allows users to upload documents in multiple formats, process and index their content, retrieve relevant information through semantic search, and generate answers using a Large Language Model (LLM).

An agent-based workflow enhances the RAG pipeline by decomposing complex questions, coordinating information retrieval, generating context-aware responses, and validating answers against retrieved evidence to reduce hallucinations.

## 2. Key Features

- **Multi-format Document Ingestion:** Supports PDF, TXT, CSV, and Excel (`.xlsx`) documents.
- **Document Chunking:** Splits documents into smaller, overlapping chunks to improve retrieval accuracy.
- **Vector Database:** Uses ChromaDB for persistent storage and retrieval of document embeddings.
- **Query Decomposition:** Planner agent breaks complex questions into smaller, targeted search queries.
- **Semantic Retrieval:** Retrieves relevant document content using vector similarity search.
- **Context-aware Response Generation:** Generates answers using the information retrieved from uploaded documents.
- **Source Attribution:** Provides references to supporting document content.
- **Hallucination Detection:** Uses a grounding validation step to verify whether generated answers are supported by retrieved evidence.
- **Persistent Knowledge Base:** Stores vector embeddings locally for reuse across application sessions.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Large Language Model | Google Gemini |
| AI Framework | LangChain |
| Vector Database | ChromaDB |
| Document Processing | PyPDF, Pandas, OpenPyXL |
| Embeddings | Configured embedding model |
| Environment Management | Python Virtual Environment |
| Configuration Management | Python Dotenv |

## 4. Project Structure

```text
enterprise-rag/
│
├── app.py
│
├── scripts/
│   ├── __init__.py
│   ├── agent.py
│   ├── loaders.py
│   └── vectorstore.py
│
├── sample_documents/
│   ├── 01_quarterly_business_report.pdf
│   ├── 02_hr_employee_policy.txt
│   ├── sales_data.csv
│   └── product_information.xlsx
│
├── storage/
│   └── chroma_db/
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── DOCUMENTATION.md
```

**Folder descriptions:**

- `app.py`: Main Streamlit application and user interface.
- `scripts/agent.py`: Implements agent planning, reasoning, retrieval coordination, and grounding validation.
- `scripts/loaders.py`: Handles document loading and text chunking.
- `scripts/vectorstore.py`: Handles embeddings, vector database creation, persistence, and similarity search.
- `sample_documents/`: Contains sample business documents used for testing.
- `storage/chroma_db/`: Stores the persistent ChromaDB vector database.
- `.env`: Stores the Gemini API key and other sensitive configuration.
- `.env.example`: Provides a template for environment variables.

## 5. System Architecture

The application follows an agentic RAG architecture consisting of document ingestion, vector indexing, query processing, retrieval, response generation, and grounding validation.

```text
                     User
                       |
                       v
              Streamlit Interface
                    app.py
                       |
              +--------+--------+
              |                 |
              v                 v
        Document Upload      User Query
              |                 |
              v                 v
       Document Loaders    Planner Agent
        loaders.py              |
              |                 v
              v            Search Queries
       Text Chunking            |
              |                 v
              v          Semantic Retrieval
          Embeddings             |
              |                 v
              v             ChromaDB
       Vector Database           |
         storage/                v
                       Retrieved Context
                              |
                              v
                       Reasoning Agent
                              |
                              v
                       Draft Response
                              |
                              v
                      Grounding Validator
                              |
                              v
                      Final Answer
                      with Sources
```

## 6. Installation and Setup

### Prerequisites

- Python 3.14 (or another compatible Python version).
- Google Gemini API key.
- Git (optional, for cloning the repository).

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-agentic-rag.git

cd enterprise-agentic-rag
```

### Step 2: Create a virtual environment

```powershell
python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 4: Configure the Gemini API key

Create a `.env` file in the project root directory.

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your actual Gemini API key.

**Security Note:** Never commit your actual `.env` file or API key to GitHub. Use `.env.example` as a template instead.

### Step 5: Run the application

From the project root directory, execute:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal:

```text
http://localhost:8501
```

## 7. Agent Workflow

The system uses multiple stages to process user queries and generate evidence-based responses.

### 1. Planner Agent

- Analyzes the user's question.
- Identifies the main information requirements.
- Decomposes complex questions into smaller search queries.

### 2. Retrieval Coordinator

- Executes semantic searches against ChromaDB.
- Retrieves relevant document chunks.
- Combines and prepares the retrieved context for the reasoning stage.

### 3. Reasoning Agent

- Processes the user's original question and retrieved context.
- Synthesizes information from relevant documents.
- Generates a context-aware draft response.

### 4. Grounding Validator

- Evaluates whether the generated response is supported by retrieved evidence.
- Identifies unsupported claims.
- Helps reduce hallucinations and improve answer reliability.

## 8. Example Use Cases

- Searching internal company policies and HR documents.
- Retrieving financial information from business reports.
- Answering questions about product documentation.
- Extracting information from Excel and CSV datasets.
- Comparing information across multiple business documents.
- Building an internal enterprise knowledge assistant.

## 9. Evaluation Questions & Expected Answers

The following evaluation dataset tests the application across three sample documents:

1. `01_quarterly_business_report.pdf`
2. `02_hr_employee_policy.txt`
3. `sales_data.csv`

The questions cover direct retrieval, quantitative reasoning, multi-document synthesis, and hallucination detection.

### A. Single-Document Fact Retrieval

| ID | Evaluation Question | Expected Answer |
|---|---|---|
| Q1 | What were Northstar Technologies' operating expenses and operating profit in Q3 2026? | Operating expenses: $8.1 million. Operating profit: $3.9 million. |
| Q2 | Which division generated the highest revenue in Q3 2026, and how much did it exceed the lowest-performing division? | Cloud Services generated $5.4 million, exceeding Consulting ($3.0 million) by $2.4 million. |
| Q3 | How many total days of paid leave does a full-time employee receive annually, and how far in advance must standard leave requests be submitted? | 30 days: 20 annual leave days and 10 sick leave days. Requests should be submitted at least five working days in advance when possible. |
| Q4 | Under what conditions can an employee work remotely, and what is the weekly limit? | Up to two days per week with manager approval, subject to role requirements and agreed working hours. |
| Q5 | Which city and product were associated with transaction ID TXN-1006, and what was its total revenue? | Bengaluru, Enterprise Suite, and 150,000 (3 units at 50,000 each). |

### B. Tabular Aggregation & Quantitative Reasoning

These questions evaluate numerical accuracy, grouping, filtering, and aggregation using `sales_data.csv`.

| ID | Evaluation Question | Expected Answer |
|---|---|---|
| Q6 | What is the total revenue across all transactions, and which region generated the highest revenue? | Total revenue: 818,000. South: 270,000; West: 224,000; North: 198,000; East: 126,000. |
| Q7 | Which product sold the highest total number of units, and which generated the highest revenue? | Cloud Basic sold 36 units. Enterprise Suite generated 350,000 revenue from 7 units. |
| Q8 | Rank the three months from highest to lowest total revenue. | August: 314,000; September: 268,000; July: 236,000. |
| Q9 | How many transactions occurred in the North region for Analytics Pro, and what was their combined revenue? | One transaction, TXN-1009, in Jaipur, generating 48,000. |

### C. Cross-Document Synthesis & Multi-Hop Reasoning

These questions evaluate whether the agent can retrieve and combine information from multiple documents.

| ID | Evaluation Question | Expected Answer |
|---|---|---|
| Q10 | If an employee wants to take a cloud migration certification course to support Northstar Technologies' Q4 priorities, what annual learning budget is available, and what approval is required? | $1,000 annual learning and development budget per employee. Manager approval is required before purchasing training. |
| Q11 | Compare Q3 2026 revenue reported in the quarterly business report with total revenue recorded in `sales_data.csv`. Why might the figures differ? | The report lists $12.0 million USD, while the CSV totals 818,000 across 12 Indian cities. The CSV may represent a regional subset, a different currency, or a sample of transactions. Its currency is not specified. |
| Q12 | What was the active enterprise customer count before new Q3 additions, assuming no churn? During what standard hours should support teams be reachable? | 222 customers (240 ending customers minus 18 additions). Monday to Friday, 9:00 AM to 6:00 PM. |

### D. Negative / Unanswerable Questions

These questions test hallucination prevention, grounding validation, and the ability to identify missing information.

| ID | Evaluation Question | Expected Answer |
|---|---|---|
| Q13 | How many weeks of paid parental leave does Northstar Technologies offer full-time employees? | The provided documents do not contain information about parental leave policies. |
| Q14 | What was Northstar Technologies' net profit after taxes in Q3 2026? | Not specified. The report provides operating profit of $3.9 million but does not mention net profit after taxes. |
| Q15 | Which salesperson closed the highest-value deal in Bengaluru in August 2026? | Cannot be determined because `sales_data.csv` does not contain a salesperson column. |
| Q16 | Why did Northstar Technologies decide to cut its Q4 technology investment budget by 20%, according to the Q3 report? | False premise. The report does not state that the budget was cut by 20%. It reports 20% year-over-year revenue growth and a maximum Q4 technology investment budget of $1.2 million. |

### Evaluation Dimensions

| Evaluation Area | Questions | What It Tests |
|---|---|---|
| Document Retrieval | Q1–Q5 | Relevant document and chunk retrieval |
| Numerical Reasoning | Q6–Q9 | Filtering, grouping, aggregation, and calculations |
| Multi-Document Reasoning | Q10–Q12 | Information synthesis across multiple sources |
| Hallucination Prevention | Q13–Q16 | Missing information detection and grounded responses |

**Important evaluation considerations:**

- Standard vector retrieval may not retrieve every CSV row required for numerical aggregation. Pandas-based analysis or a structured data query tool can improve accuracy.
- Verify source references against the original documents.
- Ensure unanswerable questions are handled without fabricated responses.
- Validate all expected answers against the actual sample documents before using them for formal evaluation.

## 10. Challenges Addressed

- Processing structured and unstructured business documents.
- Improving semantic search and retrieval accuracy.
- Handling complex multi-part questions through query decomposition.
- Combining information from multiple documents.
- Reducing hallucinations through grounding validation.
- Preserving source information for answer traceability.
- Maintaining a persistent vector database for repeated queries.

## 11. Limitations and Future Enhancements

Potential improvements include:

- Adding OCR support for scanned PDF documents.
- Improving numerical analysis of large Excel and CSV datasets.
- Integrating Pandas-based analytical tools for structured data queries.
- Introducing user authentication and role-based access control.
- Supporting larger enterprise document collections.
- Adding conversation memory and contextual follow-up questions.
- Exploring cloud deployment and managed vector database solutions.

## 12. Author

**Shivam Mittal**

Generative AI and Machine Learning Capstone Project.