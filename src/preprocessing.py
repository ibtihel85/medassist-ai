from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from src.config import CONFIG


def load_pubmed_dataset(split: str = "train"):
    """Download and return the raw PubMed QA dataset."""
    print("📚 Loading PubMed QA dataset...")
    return load_dataset("pubmed_qa", "pqa_labeled", split=split)


def records_to_documents(dataset, max_samples: int = None) -> list[Document]:
    """Convert HuggingFace dataset records to LangChain Documents."""
    max_samples = max_samples or CONFIG["max_abstracts"]
    documents = []

    for i, record in enumerate(tqdm(dataset, desc="Processing")):
        if i >= max_samples:
            break

        ctx = record.get("context", {})
        text = " ".join(ctx.get("contexts", [])) if isinstance(ctx, dict) else str(ctx)

        if len(text.strip()) < 50:
            continue

        documents.append(Document(
            page_content=text,
            metadata={
                "doc_id"     : i,
                "question"   : record.get("question", ""),
                "gold_answer": record.get("long_answer", "")[:200],
                "decision"   : record.get("final_decision", ""),
                "source"     : f"PubMedQA_record_{i}",
            }
        ))

    print(f"✅ Loaded {len(documents)} documents")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for better retrieval granularity."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks