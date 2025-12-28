# Operator Runbook — OTT Runbook RAG

## Prerequisites

- Python 3.10+
- Internet access
- OpenAI API key exported as environment variable

```bash
export OPENAI_API_KEY="YOUR_KEY"

Common Operations

Build or Update Index
    python app.py index --dir data/runbooks

This:
    adds new files
    reindexes modified files
    skips unchanged files

Force Full Rebuild
    python app.py index --dir data/runbooks --force

Use when:
    chunking parameters change
    embeddings model changes

Query the system
    python app.py query --q "What are the immediate checks for DRM license failures?"

Possible modes:
    llm_ok_first_try
    llm_repaired_second_try
    fallback_extractive
    refuse_no_hits
    refuse_not_answerable


UI Usage
    python ui.py

Capabilities:

    ask questions interactively
    inspect retrieved chunks
    view evidence per sentence
    download answers

Troubleshooting

LLM errors

    Check OPENAI_API_KEY
    Verify network access

Too many refusals
    Increase top_k
    Adjust chunk size / overlap
    Add more runbooks

Environment warnings (CUDA, cuDNN)
    Safe to ignore in Colab

Operational Notes

    Do not commit vector DBs
    Do not store real customer data
    Keep runbooks versioned externally

    
