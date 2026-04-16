import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise RuntimeError("Missing HF_API_TOKEN in environment variables")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_documents(paths: List[str]):
    docs = []
    for p in paths:
        loader = TextLoader(p, encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = p  # Add document source
        docs.extend(loaded)
    return docs

# 文档分割（附加 chunk_id）
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i  # 添加 chunk_id
    return chunks

# 在线 HuggingFace Embeddings
def build_embeddings(model_name: str = "BAAI/bge-large-zh-v1.5"):
    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=HF_API_TOKEN,
        model=model_name,
    )

# Build vector database and retriever
def build_vectorstore(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)

def build_retriever(vectorstore, top_k=5, use_mmr=True):
    if use_mmr:
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": max(20, top_k * 4)},
        )
    else:
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

# Build LLM prompt
def build_llm(model_name="deepseek-chat", temperature=0):
    llm = ChatOpenAI(
        model=model_name,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=temperature,
    )
    return llm

def build_prompt():
    system_prompt = (
        "You are a design and research assistant that creates marketing design briefs. "
        "Based on user requests and provided context, generate comprehensive briefs in JSON format.\n\n"

        "OUTPUT STRUCTURE:\n"
        "Return a JSON object with exactly these keys:\n"
        "- theme: Main subject/product (e.g., 'Seafood Noodle', 'Organic Coffee')\n"
        "- style: Design aesthetic (e.g., 'authentic Asian cuisine', 'modern minimalist')\n"
        "- tone: Emotional atmosphere (e.g., 'warm', 'elegant', 'playful')\n"
        "- emotion: Target feeling (e.g., 'satisfying umami', 'energizing freshness')\n"
        "- tagline: Memorable 3-6 word phrase\n"
        "- title: Main headline (can match theme or be creative)\n"
        "- description: 20-40 word detailed description highlighting key features\n"
        "- selling_points: Array of exactly 3 benefit statements\n"
        "- cta: Call-to-action (e.g., 'Order Now', 'Learn More', 'Shop Today')\n"
        "- badge: Promotional tag (e.g., 'Chef's Special', 'New Arrival', 'Limited Edition')\n\n"

        "EXAMPLE OUTPUT:\n"
        '{{\n'
        '  "theme": "Seafood Noodle",\n'
        '  "style": "authentic Asian cuisine",\n'
        '  "tone": "warm",\n'
        '  "emotion": "satisfying umami",\n'
        '  "tagline": "Fresh from the Ocean",\n'
        '  "title": "Seafood Noodle",\n'
        '  "description": "Rich seafood broth with fresh shrimp, scallops, and squid.",\n'
        '  "selling_points": ["Fresh Daily Seafood", "Rich Umami Broth", "Handmade Noodles"],\n'
        '  "cta": "Order Now",\n'
        '  "badge": "Chef\'s Special"\n'
        '}}\n\n'

        "RULES:\n"
        "- Return ONLY valid JSON (no markdown code blocks, no extra text)\n"
        "- All keys must be present\n"
        "- Use provided context when available\n"
        "- If context is missing, infer reasonable values\n"
        "- Ensure output is parseable by json.loads()"
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User query:\n{query}\n\nContext:\n{context}"),
    ])

# RAG chain
def build_rag_chain(retriever, llm, prompt):
    def join_docs(docs):
        return "\n\n".join([
            f"[{i}] (source={d.metadata.get('source')}, chunk_id={d.metadata.get('chunk_id')})\n{d.page_content}"
            for i, d in enumerate(docs)
        ])
    return (
        {"query": RunnablePassthrough(), "context": retriever | join_docs}
        | prompt
        | llm
    )

# Main pipeline
def rag_pipeline(query: str, doc_paths: List[str], embed_model="BAAI/bge-large-zh-v1.5"):
    documents = load_documents(doc_paths)
    chunks = split_documents(documents, chunk_size=600, chunk_overlap=80)

    embeddings = build_embeddings(model_name=embed_model)
    vectorstore = build_vectorstore(chunks, embeddings)
    retriever = build_retriever(vectorstore, top_k=5, use_mmr=True)

    llm = build_llm(model_name="deepseek-chat", temperature=0)
    prompt = build_prompt()
    rag_chain = build_rag_chain(retriever, llm, prompt)

    response = rag_chain.invoke(query)
    return response, retriever

if __name__ == "__main__":
    doc_paths = [
        "docs/poster-rag-guide.md",
    ]
    query = "复古，美食，海报"
    result, retriever = rag_pipeline(query, doc_paths, embed_model="BAAI/bge-large-zh-v1.5")

    retrieved_docs = retriever.invoke(query)
    for doc in retrieved_docs:
        print(f"RAG file used: {doc.metadata.get('source')} | chunk_id={doc.metadata.get('chunk_id')}")
    print("\n--- LLM Response ---\n")
    print(result.content)

