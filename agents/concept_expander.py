import rag.rag as rag
from typing import Dict, Any
from tools.testing_decorator import measure_time

class ConceptExpanderAgent:

    @measure_time("ConceptExpander")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        doc_paths = [
            "../rag/docs/poster-rag-guide.md",
        ]
        query = state.get("user_input")
        if query is None:
            raise ValueError(f"State missing 'user_input': {state}")

        result, retriever = rag.rag_pipeline(query, doc_paths, embed_model="BAAI/bge-large-zh-v1.5")

        # Print source of retrieved document
        retrieved_docs = retriever.invoke(query)
        for doc in retrieved_docs:
            print(f"RAG file used: {doc.metadata.get('source')} | chunk_id={doc.metadata.get('chunk_id')}")

        print("----- LLM Response -----")
        print(result.content)

        return {
            "expanded_concept": result.content,
        }