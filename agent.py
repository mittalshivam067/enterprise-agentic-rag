import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma

from vectorstore import similarity_search_with_filter

load_dotenv()

class EnterpriseAgent:
    def __init__(
        self,
        vector_db: Chroma,
        model_name: str = "gemini-3.6-flash",
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

        self.vector_db = vector_db

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            google_api_key=api_key
        )

    def _extract_text(self, content) -> str:
        """
        Converts Gemini's response content into a normal string.
        Gemini may return either a string or a list of content blocks.
        """

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = []

            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

                elif isinstance(block, str):
                    text_parts.append(block)

            return "\n".join(text_parts).strip()

        return str(content).strip()

    # ---------------------------------------------------------
    # 1. Planner Agent (Task 8)
    # ---------------------------------------------------------

    def plan_retrieval(self, query: str) -> List[str]:
        """
        Decomposes complex questions into 1 to 3 distinct
        search sub-queries to maximize vector retrieval coverage.
        """

        prompt = f"""You are a Strategic Planning AI Agent for an enterprise document search engine.
Analyze the user's inquiry and formulate 1 to 3 concise, keyword-rich sub-queries to retrieve all necessary evidence from the document store.

User Query: "{query}"

Output ONLY the search queries separated by a newline. Do not include numbered lists or extra punctuation."""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        content = self._extract_text(response.content)

        lines = [
            line.strip("- *0123456789. ")
            for line in content.split("\n")
            if line.strip()
        ]

        return lines[:3] if lines else [query]

    # ---------------------------------------------------------
    # 2. Retrieval Execution & Deduplication (Tasks 6 & 7)
    # ---------------------------------------------------------

    def retrieve_context(
        self,
        search_queries: List[str],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Executes similarity searches across all planned sub-queries,
        filtering noise and deduplicating chunks.
        """

        unique_chunks = {}

        for sub_query in search_queries:
            results = similarity_search_with_filter(
                self.vector_db,
                sub_query,
                top_k=top_k,
            )

            for doc, score in results:
                # Key by source and initial content snippet
                # to eliminate duplicate segments

                key = f"{doc.metadata.get('source', '')}_{doc.page_content[:60]}"

                if key not in unique_chunks:
                    unique_chunks[key] = {
                        "doc": doc,
                        "score": score,
                        "metadata": doc.metadata,
                        "content": doc.page_content
                    }

        return list(unique_chunks.values())

    # ---------------------------------------------------------
    # 3. Reasoning & Grounded Generation (Tasks 7 & 8)
    # ---------------------------------------------------------

    def reason_and_generate(
        self,
        query: str,
        context_items: List[Dict[str, Any]],
    ) -> str:
        """
        Synthesizes an answer using retrieved context
        with explicit citations.
        Adheres to strict truthfulness boundaries.
        """

        if not context_items:
            return (
                "The provided enterprise documents do not contain "
                "sufficient information to answer this question."
            )

        # Format context with citation metadata

        formatted_context = []

        for idx, item in enumerate(context_items, 1):
            meta = item["metadata"]
            source = meta.get("source", "Unknown")

            page_info = (
                f", Page {meta['page']}"
                if "page" in meta else ""
            )

            row_info = (
                f", Rows {meta['rows']}"
                if "rows" in meta else ""
            )

            formatted_context.append(
                f"[Source {idx}: {source}{page_info}{row_info}]\n"
                f"{item['content']}"
            )

        context_block = "\n\n---\n\n".join(formatted_context)

        system_instruction = """You are an Enterprise Knowledge Assistant.

Answer the user's question accurately and objectively using strictly the provided context.

- Cite the relevant source tags (e.g., [Source 1: report.pdf, Page 2]) when referencing facts.
- If the answer cannot be directly deduced from the provided context, state:
  'The provided documents do not contain sufficient information to answer this question.'
- Never guess or use external assumptions."""

        user_prompt = f"""Retrieved Context:
{context_block}

User Question:
{query}"""

        response = self.llm.invoke([
            HumanMessage(content=user_prompt)
        ])

        draft_answer = self._extract_text(response.content)

        return draft_answer

    # ---------------------------------------------------------
    # 4. Reliability & Guardrail Evaluation (Task 9)
    # ---------------------------------------------------------

    
    def validate_grounding(
        self,
        draft_answer: str,
        context_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        """
        Checks whether claims in the draft answer are strictly supported
        by the retrieved context to prevent hallucination.
        """

        # Ensure draft_answer is always a string
        draft_answer = self._extract_text(draft_answer)

        # If the model already refused due to missing info, validation passes
        if "insufficient information" in draft_answer.lower() or not context_items:
            return {
                "is_grounded": True,
                "reason": "Consistent refusal due to absent context."
            }

        context_summary = "\n".join(
            [item["content"][:300] for item in context_items]
        )

        guardrail_prompt = f"""You are a Fact-Checking and Hallucination Auditor.
    Verify if the claims made in the Draft Answer are completely supported by the Context.

    Context:
    {context_summary}

    Draft Answer:
    {draft_answer}

    Answer in this exact format:
    VERDICT: [PASS or FAIL]
    REASON: [One concise sentence explaining the verdict]"""

        # Get Gemini response
        response = self.llm.invoke([
            HumanMessage(content=guardrail_prompt)
        ])

        # Convert Gemini response into a normal string
        eval_output = self._extract_text(response.content)

        is_grounded = "VERDICT: PASS" in eval_output.upper()

        return {
            "is_grounded": is_grounded,
            "raw_evaluation": eval_output
        }

    # ---------------------------------------------------------
    # End-to-End Orchestrator
    # ---------------------------------------------------------
    def run(self, query: str) -> Dict[str, Any]:
        """
        Executes the full retrieval-augmented generation pipeline:
        1. Plan sub-queries
        2. Retrieve and deduplicate context
        3. Generate grounded answer
        4. Validate grounding
        """

        # Step 1: Plan sub-queries
        search_queries = self.plan_retrieval(query)

        # Step 2: Retrieve context
        context_items = self.retrieve_context(search_queries)

        # Step 3: Generate answer
        draft_answer = self.reason_and_generate(query, context_items)

        # Step 4: Validate grounding
        validation = self.validate_grounding(draft_answer, context_items)

        final_answer = draft_answer

        if not validation["is_grounded"]:
            final_answer += "\n\n⚠️ *Guardrail Notice: Some claims in this answer could not be fully substantiated by the indexed source documents.*"
        return {
            "query": query,
            "search_queries": search_queries,
            "retrieved_context": context_items,
            "draft_answer": draft_answer,
            "grounding_evaluation": validation
        }

if __name__ == "__main__":
    from vectorstore import load_existing_vector_store

    print("Testing Agent Workflow...")
    # Load the vector store created in Step 3
    vector_store = load_existing_vector_store()
    agent = EnterpriseAgent(vector_db=vector_store)

    # 1. Test grounded query
    query_1 = "What is the revenue growth rate reported for Q3?"
    res_1 = agent.run(query_1)
    print(f"\n--- Query: {query_1} ---")
    print("Plan:", res_1["search_queries"])
    print("Answer:\n", res_1["draft_answer"])
    print("Validation:", res_1["grounding_evaluation"]["raw_evaluation"])

    # 2. Test ungrounded / out-of-context query (Guardrail check)
    query_2 = "Who is the CEO of Apple?"
    res_2 = agent.run(query_2)
    print(f"\n--- Query: {query_2} ---")
    print("Answer:\n", res_2["draft_answer"])