import os
import tempfile
import streamlit as st

from loaders import parse_document, chunk_documents
from vectorstore import build_vector_store, load_existing_vector_store
from agent import EnterpriseAgent

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Enterprise AI Decision Support",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Enterprise Knowledge & Decision Support System")

st.caption(
    "AI-powered enterprise knowledge assistant using "
    "Gemini, RAG, ChromaDB, and hallucination guardrails."
)

# =========================================================
# 2. SESSION STATE INITIALIZATION
# =========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

# =========================================================
# 3. LOAD EXISTING VECTOR DATABASE
# =========================================================

if st.session_state.agent is None and os.path.exists("./chroma_db"):
    try:
        with st.spinner("Loading existing knowledge base..."):
            vector_db = load_existing_vector_store()
            st.session_state.agent = EnterpriseAgent(vector_db=vector_db)
            st.session_state.index_ready = True
            st.success("Knowledge base loaded successfully!")
    except Exception as e:
        st.session_state.agent = None
        st.session_state.index_ready = False

        st.sidebar.error(
            f"Unable to load existing knowledge base: {str(e)}"
        )

# =========================================================
# 4. SIDEBAR - DOCUMENT INGESTION
# =========================================================

with st.sidebar:

    st.header("📂 Document Ingestion")

    st.write(
        "Upload enterprise documents to build your knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, CSV, or Excel files",
        type=["pdf", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if st.button(
        "🚀 Process & Index Documents",
        type="primary",
        use_container_width=True
    ):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            try:
                with st.status("Processing and indexing documents...",
                               expanded=True) as status:
                    all_chunks = []

                    # -------------------------------------
                    # Parse and chunk documents
                    # -------------------------------------

                    for uploaded_file in uploaded_files:
                        st.write(
                            f"📄 Processing: {uploaded_file.name}"
                        )

                        suffix = os.path.splitext(uploaded_file.name)[1]

                        tmp_path = None

                        try:

                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=suffix
                            ) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name

                            documents = parse_document(
                                file_path=tmp_path,
                                filename=uploaded_file.name
                            )

                            chunks = chunk_documents(documents)

                            all_chunks.extend(chunks)

                            st.write(
                                f"Created {len(chunks)} chunks."
                            )

                        finally:

                            if tmp_path and os.path.exists(tmp_path):
                                os.remove(tmp_path)

                    # -------------------------------------
                    # Build ChromaDB
                    # -------------------------------------

                    if not all_chunks:
                        raise ValueError("No valid document chunks were created.")

                    st.write(
                        f"Creating embeddings for "
                        f"{len(all_chunks)} chunks..."
                    )

                    vector_db = build_vector_store(all_chunks)

                    # -------------------------------------
                    # Initialize Gemini Agent
                    # -------------------------------------

                    agent = EnterpriseAgent(vector_db=vector_db)

                    st.session_state.agent = agent
                    st.session_state.index_ready = True

                    status.update(
                        label="Knowledge Base Ready!",
                        state="complete",
                        expanded=False
                    )

                st.success(
                    f"Successfully indexed {len(all_chunks)} "
                    f"chunks from {len(uploaded_files)} documents."
                )

            except Exception as e:

                st.error("Document ingestion failed.")

                st.code(
                    f"{type(e).__name__}: {str(e)}"
                )

    st.divider()

    # -----------------------------------------
    # Knowledge Base Status
    # -----------------------------------------


    st.subheader("System Status")

    if st.session_state.index_ready:
        st.success("Knowledge Base: Ready")

    else:
        st.warning("Knowledge Base: Not Available")

    st.caption("Powered by Google Gemini API")

    st.divider()


    # -----------------------------------------
    # System Architecture
    # -----------------------------------------

    st.subheader("System Architecture")

    st.markdown("""
    1. **Planner** – Breaks complex queries into subqueries.
    2. **Retriever** – Retrieves relevant document chunks.
    3. **Reasoner** – Generates context-grounded answers.
    4. **Guardrail** – Checks answers against retrieved evidence.
    """)

    # =========================================================
    # 5. CHAT HISTORY DISPLAY
    # =========================================================  

    st.subheader("💬 Ask Your Enterprise Knowledge Assistant")

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            details = message.get("details")

            if message["role"] == "assistant" and details:

                with st.expander("🔍 Inspect Agent Execution & Citations"):

                    # Planned Search Queries
                    st.markdown("### 1. Planned Search Queries")

                    search_queries = details.get(
                        "search_queries", []
                    )

                    if search_queries:

                        for query in search_queries:
                            st.code(query)

                    else:
                        st.write("No search queries available.")

                    # Retrieved Context
                    st.markdown("### 2. Retrieved Sources")

                    retrieved_context = details.get(
                        "retrieved_context", []
                    )

                    st.write(
                        f"Retrieved {len(retrieved_context)} chunks."
                    )

                    for idx, item in enumerate(retrieved_context, 1):

                        with st.expander(f"Source {idx}"):
                            st.write("Source:", item.get("source", "Unknown"))
                            st.write("Page:", item.get("page", "N/A"))
                            st.write("Content:")
                            st.text(
                                item.get("content", str(item))
                            )

                    # Guardrail Evaluation
                    st.markdown("### 3. Guardrail Evaluation")

                    validation = details.get(
                        "grounding_evaluation", {}
                    )

                    if validation:

                        st.info(
                            validation.get(
                                "raw_evaluation",
                                str(validation)
                            )
                        )

                        if "is_grounded" in validation:
                            if validation["is_grounded"]:
                                st.success("Answer passed grounding validation.")
                            else:
                                st.warning(
                                    "Some claims could not be verified."
                                )

                    else:
                        st.write("No validation details available.")


    # =========================================================
    # 6. CHAT INPUT
    # =========================================================

    user_prompt = st.chat_input(
        "Ask a question about your uploaded documents..."
    )

    # =========================================================
    # 7. AGENT EXECUTION
    # =========================================================

    if user_prompt:

        if st.session_state.agent is None:

            st.error(
                "Please upload and index documents before asking questions."
            )

            st.stop()

        # Display and save user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_prompt
        })

        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Execute Enterprise Agent
        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Gemini is planning, retrieving, reasoning, "
                    "and validating..."
                ):
                    execution_result = st.session_state.agent.run(user_prompt)

                # -----------------------------------------
                # Read agent response
                # -----------------------------------------

                answer = execution_result.get(
                    "final_answer",
                    execution_result.get(
                        "draft_answer",
                        execution_result.get(
                            "answer",
                            "The agent did not generate an answer."
                        )
                    )
                )

                st.markdown(answer)

                # -----------------------------------------
                # Agent execution details
                # -----------------------------------------

                with st.expander(
                    "🔍 Inspect Agent Execution & Citations"
                ):
                    st.markdown("### 1. Search Plan")

                    for query in execution_result.get(
                        "search_queries",
                        execution_result.get("search_plan", [])
                    ):
                        st.code(query, language="text")

                    st.markdown("### 2. Retrieved Sources")

                    sources = execution_result.get(
                        "retrieved_context",
                        execution_result.get("sources", [])
                    )

                    st.write(f"Retrieved {len(sources)} chunks.")

                    for idx, item in enumerate(sources, 1):
                        with st.expander(f"Source {idx}"):
                            st.json(item)

                    st.markdown("### 3. Guardrail Evaluation")

                    validation = execution_result.get(
                        "grounding_evaluation",
                        execution_result.get("validation", {})
                    )

                    if validation:
                        st.info(
                            validation.get(
                                "raw_evaluation",
                                str(validation)
                            )
                        )

                        if validation.get("is_grounded") is True:
                            st.success("Grounding validation passed.")

                        elif validation.get("is_grounded") is False:

                            st.warning(
                                "Grounding validation did not pass."
                            )

                # -----------------------------------------
                # Save assistant response
                # -----------------------------------------

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "details": execution_result
                })

            except Exception as e:
                error_message = str(e)

                if "503" in error_message:
                    st.error(
                        "Gemini API is currently unavailable. "
                        "Please try again later."
                    )

                    st.warning(
                        "This is generally a server-side availability "
                        "or capacity issue. Please try again later."
                    )

                else:
                    st.error("An error occurred while processing your query.")

                st.code(
                    f"{type(e).__name__}: {error_message}"
                )

                st.caption(
                    "Your existing ChromaDB documents have not been deleted."
                )
