import argparse
from pathlib import Path
import json

import chromadb
from chromadb.utils import embedding_functions

from config import CFG
from ingest import index_folder
from rag_pipeline import query_rag_llm_robust

CHROMA_DIR = "/content/ott-rag-capstone/db/chroma"
COLLECTION_NAME = "ott_runbooks"
MANIFEST_PATH = "/content/ott-rag-capstone/db/index_manifest.json"


def get_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=CFG.embedding_model
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )
    return client, col


def cmd_index(args):
    runbooks_dir = Path(args.dir)
    if not runbooks_dir.exists():
        raise FileNotFoundError(runbooks_dir)

    client, col = get_collection()

    if args.force:
        client.delete_collection(COLLECTION_NAME)
        col = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=CFG.embedding_model
            )
        )
        Path(MANIFEST_PATH).unlink(missing_ok=True)

    stats = index_folder(
        folder_path=str(runbooks_dir),
        col=col,
        manifest_path=MANIFEST_PATH,
        include_ext=[".md", ".txt"],
        verbose=True
    )

    print("\n--- INDEX RESULT ---")
    print(json.dumps(stats, indent=2))
    print("Chroma count:", col.count())


def cmd_query(args):
    _, col = get_collection()

    out = query_rag_llm_robust(
        args.q,
        col=col,
        model=args.model,
        min_sim=args.min_sim
    )

    print("\n--- QUERY RESULT ---")
    print("MODE:", out["mode"])
    print("format_ok:", out.get("format_ok"))
    print("grounded_ok:", out.get("grounded_ok"))

    print("\nANSWER:\n", out["answer"])


def main():
    p = argparse.ArgumentParser("OTT RAG CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index")
    p_index.add_argument("--dir", required=True)
    p_index.add_argument("--force", action="store_true")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query")
    p_query.add_argument("--q", required=True)
    p_query.add_argument("--model", default="gpt-4.1-mini")
    p_query.add_argument("--min-sim", type=float, default=0.38)
    p_query.set_defaults(func=cmd_query)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
