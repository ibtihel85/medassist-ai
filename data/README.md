

# 🤖 MedAssist AI — Medical Research Agent with RAG


> An intelligent medical research assistant that answers clinical questions grounded in real PubMed abstracts — powered by Retrieval-Augmented Generation (RAG) and a LangChain ReAct agent.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Sample Outputs](#-sample-outputs)
- [Evaluation](#-evaluation)
- [License](#-license)

---

## 🧠 Overview

**MedAssist AI** is an end-to-end AI agent that helps medical students, researchers, and curious users explore scientific medical literature through natural language. Built on top of 500 PubMed abstracts, it uses a **LangChain ReAct agent** backed by **FAISS vector search** and an open-source **HuggingFace LLM** (Mistral or Flan-T5) to deliver grounded, cited answers in seconds.

This project demonstrates a complete RAG pipeline from document ingestion and semantic chunking to agent orchestration, retrieval, and answer generation — entirely using open-source tools.

---

## 🎯 Problem Statement

Medical literature is vast, dense, and fragmented across thousands of papers. A student asking:

> *"What are the side effects of metformin in elderly patients?"*

would normally spend hours manually searching PubMed, reading abstracts, and synthesizing findings.

**MedAssist AI solves this by:**
- Retrieving the most semantically relevant PubMed abstracts for any query
- Generating a concise, grounded answer using an LLM
- Citing the exact papers used — no hallucinated references

This directly mirrors what companies like Elsevier, PubMed AI, and medical AI startups are building in 2025.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Search** | FAISS-powered retrieval finds relevant abstracts by meaning, not just keywords |
| 🤖 **ReAct Agent** | LangChain agent reasons step-by-step before deciding which tool to invoke |
| 📄 **Source Citations** | Every answer includes the title and source of the retrieved papers |
| 💬 **Multi-turn Queries** | Supports follow-up questions within a session |
| 📊 **Quantitative Evaluation** | Retrieval and answer quality measured against gold labels from PubMed QA |
| 🔓 **Fully Open-Source** | No paid APIs |

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│         LangChain Agent         │
│      (ReAct reasoning loop)     │
└────────────┬────────────────────┘
             │ decides which tool to use
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
[Tool 1]          [Tool 2]
RAG Retriever     Calculator /
    │             Date Tool
    ▼
FAISS Vector DB
    │
    ▼
Top-K Relevant
PubMed Abstracts
    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  HuggingFace    │
    │  LLM (Mistral / │
    │  Flan-T5)       │
    └────────┬────────┘
             │
             ▼
    Grounded Answer
    + Source Citations
```

**Step-by-step flow:**

1. User submits a medical question via CLI or notebook
2. The LangChain **ReAct agent** analyzes the query and selects a tool
3. The **RAG tool** queries FAISS and retrieves the top-3 most relevant PubMed abstracts
4. The retrieved context and original question are formatted into a prompt
5. The **HuggingFace LLM** generates a grounded, cited answer
6. The agent returns the final answer with source paper titles

---

## 📁 Project Structure

```
medassist-ai/
│
├── data/
│   └── pubmed_sample.jsonl        # 500 sampled PubMed QA abstracts
│
├── src/
│   ├── ingest.py                  # Load and chunk PubMed abstracts
│   ├── embed.py                   # Generate embeddings (HuggingFace)
│   ├── retriever.py               # FAISS index build + similarity search
│   ├── agent.py                   # LangChain ReAct agent + tool definitions
│   ├── tools.py                   # RAG tool, calculator tool
│   └── evaluate.py                # Retrieval + answer quality evaluation
│
├── notebooks/
│   └── MedAssist_AI.ipynb         # Full walkthrough notebook (Kaggle/Colab)
│
├── outputs/
│   └── sample_outputs.md          # Real example Q&A pairs from the agent
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📦 Dataset

**PubMed QA** (`pubmed_qa` / `pqa_labeled`)

| Field | Type | Description |
|---|---|---|
| `question` | `str` | Medical research question |
| `context.contexts` | `list[str]` | PubMed abstract sentences |
| `long_answer` | `str` | Gold expert answer |
| `final_decision` | `str` | `yes` / `no` / `maybe` |

- **Source:** [HuggingFace — pubmed_qa](https://huggingface.co/datasets/pubmed_qa)
- **License:** MIT
- **Size used:** 500 abstracts from the 1,000-sample labeled split

**Why PubMed QA?**
- Real PubMed abstracts — not synthetic data
- Gold-labeled answers enable quantitative evaluation
- Covers diverse medical topics (oncology, pharmacology, cardiology, etc.)
- Freely available, no scraping or licensing required

```python
from datasets import load_dataset
ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
```

---

## 🛠️ Technologies Used

| Technology | Role |
|---|---|
| **LangChain** | Agent framework, chains, retrievers, tool orchestration |
| **FAISS** | Vector database for fast approximate nearest-neighbor search |
| **HuggingFace Transformers** | Open-source LLMs (Mistral-7B / Flan-T5) and embedding models |
| **sentence-transformers** | Semantic text embeddings (`all-MiniLM-L6-v2`) |
| **PubMed QA** | Labeled biomedical QA dataset |
| **Python 3.10+** | Core language |
| **Kaggle** | Experimentation and presentation |

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/ibtihel85/medassist-ai.git
cd medassist-ai

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Build the FAISS index from PubMed abstracts
python src/ingest.py
python src/embed.py
```

> **Note:** The FAISS index and model weights are not committed to this repo (see `.gitignore`). Running the scripts above will generate them locally.

---

## 🚀 Usage

### Run the agent from the command line

```bash
python src/agent.py --query "What are the cardiovascular effects of aspirin in elderly patients?"
```

**Example output:**
```
Agent: Calling RAG tool...
Retrieved 3 abstracts.

Answer:
Based on the retrieved literature, aspirin use in elderly patients has been associated
with a reduced risk of myocardial infarction but carries an elevated risk of
gastrointestinal bleeding, particularly in patients over 75.

Sources:
  [1] "Aspirin and cardiovascular outcomes in older adults" — PMID 12345678
  [2] "Risk-benefit analysis of antiplatelet therapy in geriatric patients" — PMID 87654321
```

### Run in a notebook

Open `notebooks/MedAssist_AI.ipynb` in Kaggle and run all cells top to bottom.

---

## 📊 Evaluation

> Results will be populated after running `src/evaluate.py` on the full evaluation split.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---
