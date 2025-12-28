# OTT Runbook RAG (Productized)

A **production-style Retrieval Augmented Generation (RAG)** system built for OTT operational runbooks (DRM, CDN, Packager).
The system is designed to produce **auditable, citation-backed answers** with safeguards against hallucination.

This project is intentionally implemented as a **CLI + UI application**, not a notebook demo.

---

## Why this project exists

In incident triage and operations, answers must be:
- traceable to source documentation
- safe when information is missing
- resistant to hallucination

This project demonstrates how to build a **guarded RAG pipeline** that:
- retrieves evidence from runbooks
- forces **one citation per sentence**
- verifies semantic grounding
- refuses or falls back when evidence is insufficient

---

## Key Capabilities

### Ingestion
- Folder-based ingestion of `.md` / `.txt` runbooks
- Cleaning and chunking with configurable size and overlap
- Incremental indexing using a manifest file
- Vector storage using **Chroma**

### Retrieval
- Top-K semantic retrieval using sentence embeddings
- Score-based filtering for answerability
- Metadata preserved for traceability

### Answer Generation (LLM)
- OpenAI model used only for **language synthesis**
- Strict formatting: **one trailing citation per sentence**
- Automatic repair attempt if formatting is violated

### Guardrails
- Refusal modes:
  - `refuse_policy_gate`
  - `refuse_no_hits`
  - `refuse_not_answerable`
- Semantic groundedness verification:
  - each answer sentence is compared against its cited chunk
- Deterministic extractive fallback when LLM output fails validation

### Interfaces
- CLI for automation and scripting
- Minimal Gradio UI for interactive exploration
- Evidence inspection and downloadable answers

---

## Repository Structure

ott-rag-capstone/
├── app.py # CLI entry point
├── ui.py # Gradio UI
├── config.py # Central configuration
├── ingest.py # Folder ingestion + chunking + indexing
├── rag_pipeline.py # Retrieval, LLM synthesis, guardrails
├── requirements.txt
├── ARCHITECTURE.md
├── RUNBOOK.md
├── data/
│ └── runbooks/ # Synthetic runbooks only
└── .gitignore


---

## Quick Start (CLI)

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="YOUR_KEY"

python app.py index --dir data/runbooks
python app.py query --q "What are the immediate checks for DRM license failures?"

export OPENAI_API_KEY="YOUR_KEY"
python ui.py

In Colab, the UI provides a public gradio.live URL.

### What this project demonstrates

End-to-end RAG system design

Practical hallucination mitigation

Product-grade packaging (CLI, UI, config, docs)

Alignment with real OTT operational workflows

Disclaimer

All runbooks included are synthetic and contain no proprietary or customer data.
