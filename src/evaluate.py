import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_rag(rag_chain, dataset, embedding_model, n_samples: int = 20) -> pd.DataFrame:
    """
    Run evaluation on n_samples from the dataset.

    Returns:
        DataFrame with per-sample metrics
    """
    results = []

    for i, record in enumerate(tqdm(dataset.select(range(n_samples)),
                                    desc="Evaluating")):
        question   = record["question"]
        gold       = record.get("long_answer", "")

        if not question or not gold:
            continue

        output     = rag_chain.invoke({"query": question})
        generated  = output.get("result", "")
        n_docs     = len(output.get("source_documents", []))

        sim = 0.0
        if generated and gold:
            g_emb  = embedding_model.embed_query(generated)
            gl_emb = embedding_model.embed_query(gold[:512])
            sim    = float(cosine_similarity([g_emb], [gl_emb])[0][0])

        results.append({
            "question"           : question[:80] + "...",
            "semantic_similarity": round(sim, 3),
            "docs_retrieved"     : n_docs,
            "answer_length"      : len(generated),
            "gold_decision"      : record.get("final_decision", ""),
        })

    df = pd.DataFrame(results)

    print("\n" + "="*55)
    print("📈 EVALUATION SUMMARY")
    print("="*55)
    print(f"  Samples evaluated        : {len(df)}")
    print(f"  Avg semantic similarity  : {df['semantic_similarity'].mean():.3f}")
    print(f"  Avg docs retrieved       : {df['docs_retrieved'].mean():.1f}")
    print(f"  Avg answer length (chars): {df['answer_length'].mean():.0f}")
    print("="*55)

    return df