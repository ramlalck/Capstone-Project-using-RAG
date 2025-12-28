from dataclasses import dataclass

@dataclass
class RAGConfig:
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    min_score: float = 0.38
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

CFG = RAGConfig()
