# Architecture — OTT Runbook RAG

## High-level Design

The system follows a classic **RAG architecture**, extended with **validation and safety layers**.

Runbooks Folder
|
v
Ingestion Pipeline
(clean → chunk → embed)
|
v
Chroma Vector Store
|
v
User Query
|
v
Retriever (Top-K)
|
Runbooks Folder
|
v
Answerability Gate
(refuse if unsafe / insufficient)
|
v
LLM Synthesis
(format + citation rules)
|
v
Validation Layer
(citation format + grounding)
|
+--> Repair Attempt
|
+--> Extractive Fallback
|
v
Final Answer + Evidence


---

## Ingestion Flow

1. Read `.md` / `.txt` files from a folder
2. Clean text conservatively (preserve structure)
3. Chunk using configurable size and overlap
4. Generate embeddings
5. Store chunks in Chroma with metadata
6. Track changes via a manifest for incremental updates

---

## Query Flow

1. Embed user query
2. Retrieve top-K similar chunks
3. Decide if the question is answerable
4. If answerable:
   - Generate answer using LLM
   - Enforce one citation per sentence
5. Validate:
   - citation presence
   - semantic similarity between sentence and cited chunk
6. If validation fails:
   - attempt repair
   - else fall back to extractive answer
7. Return answer, mode, and evidence

---

## Design Principles

- **LLM is not the source of truth**
- **Retrieval drives correctness**
- **Validation gates trust**
- **Refusal is better than hallucination**
