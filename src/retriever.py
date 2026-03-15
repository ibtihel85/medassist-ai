
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from src.config import CONFIG


def build_embedding_model() -> HuggingFaceEmbeddings:
    """Load the sentence-transformer embedding model on CPU."""
    return HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def build_vectorstore(chunks: list[Document],
                      embedding_model: HuggingFaceEmbeddings,
                      save_path: str = None) -> FAISS:
    """Embed all chunks and build a FAISS index."""
    print(f"🗄️  Building FAISS index for {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vectorstore.save_local(save_path)
        print(f"💾 Index saved → {save_path}")

    return vectorstore


def load_vectorstore(path: str,
                     embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    print(f"📂 Loading FAISS index from {path}...")
    return FAISS.load_local(path, embedding_model,
                            allow_dangerous_deserialization=True)


def get_retriever(vectorstore: FAISS):
    """Return a LangChain retriever from the vectorstore."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": CONFIG["top_k_retrieval"]},
    )