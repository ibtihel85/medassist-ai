CONFIG = {
    # Models
    "embedding_model"    : "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model"          : "mistralai/Mistral-7B-Instruct-v0.2",

    # Quantization
    "load_in_4bit"       : True,
    "bnb_4bit_quant_type": "nf4",

    # Data
    "max_abstracts"      : 500,
    "chunk_size"         : 512,
    "chunk_overlap"      : 64,

    # Retrieval
    "top_k_retrieval"    : 3,
    "faiss_index_path"   : "faiss_index/medassist_index",

    # Generation
    "max_new_tokens"     : 512,
    "temperature"        : 0.1,
    "repetition_penalty" : 1.15,
}