from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate(
    template="""You are MedAssist AI, a precise medical research assistant.
Use ONLY the retrieved abstracts below to answer the question.
If the answer is not in the context, say:
"I do not have enough information in my database to answer this."

RETRIEVED ABSTRACTS:
{context}

QUESTION: {question}

ANSWER (factual, concise, cite the context):""",
    input_variables=["context", "question"],
)

TERMINOLOGY_DB = {
    "metformin"   : "First-line oral antidiabetic drug for type 2 diabetes; reduces hepatic glucose production.",
    "hypertension": "Blood pressure persistently ≥130/80 mmHg; major cardiovascular risk factor.",
    "rct"         : "Randomized Controlled Trial — gold standard study design; participants randomly assigned to treatment or control.",
    "placebo"     : "Inactive treatment given to the control group to measure psychological treatment effects.",
    "biomarker"   : "Measurable biological indicator (e.g., HbA1c, PSA) used to assess disease state or treatment response.",
    "comorbidity" : "Presence of two or more chronic conditions simultaneously in one patient.",
    "efficacy"    : "Ability of a treatment to produce the desired effect under ideal (controlled) conditions.",
    "cohort"      : "Group of patients sharing a characteristic, followed over time in an observational study.",
}

STATISTICS_DB = {
    "p-value"               : "P < 0.05 → result is statistically significant (< 5% probability of occurring by chance).",
    "confidence interval"   : "95% CI: if the study were repeated 100×, 95 intervals would contain the true population value.",
    "odds ratio"            : "OR > 1 → increased odds of outcome. OR = 2 → twice the odds vs reference group.",
    "hazard ratio"          : "HR in survival analysis. HR = 0.5 → 50% lower event risk at any time point vs reference.",
    "sensitivity"           : "TP / (TP + FN). High sensitivity → few missed cases (good for screening).",
    "specificity"           : "TN / (TN + FP). High specificity → few false positives (good for confirmation).",
    "number needed to treat": "NNT = patients needed to treat to prevent one adverse outcome. Lower = better.",
}


def build_tools(llm, retriever) -> list[Tool]:
    """
    Build and return all agent tools.

    Args:
        llm       : LangChain-wrapped LLM (HuggingFacePipeline)
        retriever : FAISS retriever from get_retriever()

    Returns:
        List of LangChain Tool objects
    """
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    def medical_search(query: str) -> str:
        result = rag_chain.invoke({"query": query})
        answer = result["result"]
        sources = [d.metadata.get("source", "?")
                   for d in result.get("source_documents", [])]
        src_str = "\n".join(f"  [{i+1}] {s}" for i, s in enumerate(sources))
        return f"ANSWER:\n{answer}\n\nSOURCES:\n{src_str}"

    def terminology_lookup(term: str) -> str:
        for key, definition in TERMINOLOGY_DB.items():
            if key in term.lower():
                return f"📖 {key.upper()}: {definition}"
        return f"Term '{term}' not in local dictionary. Try MedicalLiteratureSearch."

    def stats_helper(query: str) -> str:
        for key, explanation in STATISTICS_DB.items():
            if key in query.lower():
                return f"📊 {key.upper()}: {explanation}"
        return f"Concept not found. Supported: {', '.join(STATISTICS_DB.keys())}"

    return [
        Tool(
            name="MedicalLiteratureSearch",
            func=medical_search,
            description=(
                "Search PubMed abstracts to answer medical questions about "
                "treatments, diseases, drugs, or clinical outcomes. "
                "Input: a clear medical question."
            ),
        ),
        Tool(
            name="MedicalTerminologyExplainer",
            func=terminology_lookup,
            description=(
                "Look up the definition of a specific medical term or abbreviation. "
                "Input: a single term e.g. 'metformin', 'RCT', 'biomarker'."
            ),
        ),
        Tool(
            name="StudyStatisticsHelper",
            func=stats_helper,
            description=(
                "Explain a statistical concept from a medical paper. "
                "Input: a term like 'p-value', 'confidence interval', 'odds ratio'."
            ),
        ),
    ]