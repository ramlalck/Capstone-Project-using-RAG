from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time
from typing import List, Dict, Any, Optional

from config import CFG


# -------------------------
# Text cleaning + chunking
# -------------------------
def clean_text(text: str) -> str:
    """
    Keep this conservative. We avoid aggressive normalization that could
    remove important runbook tokens (codes, URLs, headers).
    """
    if text is None:
        return ""
    # Normalize line endings
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces per line
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    # Collapse excessive blank lines (keep structure)
    t = "\n".join([ln for ln in t.split("\n")])
    return t.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple, stable token-free chunker (character-based) for Colab portability.
    This matches your earlier labs: size + overlap tradeoff.
    """
    t = (text or "").strip()
    if not t:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks = []
    start = 0
    n = len(t)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap

    return chunks


# -------------------------
# Manifest helpers
# -------------------------
def _sha256_file(path: Path, buf_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_manifest(manifest_path: str) -> Dict[str, Any]:
    p = Path(manifest_path)
    if not p.exists():
        return {"version": 1, "files": {}, "last_run_ts": None}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_manifest(manifest_path: str, manifest: Dict[str, Any]) -> None:
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# -------------------------
# Folder indexing
# -------------------------
def index_folder(
    folder_path: str,
    col,
    manifest_path: str,
    include_ext: Optional[List[str]] = None,
    verbose: bool = True,
):
    """
    Incremental folder ingestion:
      - reads .md/.txt
      - cleans + chunks using CFG
      - deletes old chunks for each changed doc_id
      - adds new chunks with stable ids: <relpath>::C#
      - updates manifest for incremental runs
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if include_ext is None:
        include_ext = [".md", ".txt"]
    include_ext = [e.lower() for e in include_ext]

    manifest = _load_manifest(manifest_path)
    seen = manifest.get("files", {})

    files = sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in include_ext])

    stats = {
        "total_files_seen": len(files),
        "added_or_updated": 0,
        "skipped_unchanged": 0,
        "deleted": 0,
        "errors": 0,
        "total_chunks_indexed": 0
    }

    current_relpaths = set()

    for path in files:
        rel = str(path.relative_to(folder)).replace("\\", "/")
        current_relpaths.add(rel)

        try:
            mtime = path.stat().st_mtime
            prev = seen.get(rel)

            # Fast skip by mtime (incremental)
            if prev and abs(prev.get("mtime", 0) - mtime) < 1e-6:
                stats["skipped_unchanged"] += 1
                continue

            raw = path.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned, CFG.chunk_size, CFG.chunk_overlap)

            doc_id = rel  # stable doc id based on relative path

            # Delete prior chunks for this doc
            # Chroma requires a proper where filter (one key)
            col.delete(where={"doc_id": doc_id})

            ids, docs, metas = [], [], []
            file_hash = _sha256_file(path)

            for i, ch in enumerate(chunks):
                chunk_id = f"{doc_id}::C{i+1}"
                ids.append(chunk_id)
                docs.append(ch)
                metas.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "source_file": path.name,
                    "source_relpath": rel,
                    "mtime": mtime,
                    "file_hash": file_hash
                })

            if ids:
                col.add(ids=ids, documents=docs, metadatas=metas)

            # Update manifest
            seen[rel] = {
                "doc_id": doc_id,
                "mtime": mtime,
                "file_hash": file_hash,
                "chunk_count": len(chunks)
            }

            stats["added_or_updated"] += 1
            stats["total_chunks_indexed"] += len(chunks)

            if verbose:
                print(f"Indexed: {rel} | chunks={len(chunks)}")

        except Exception as e:
            stats["errors"] += 1
            if verbose:
                print(f"ERROR indexing {rel}: {e}")

    # Handle deletions (files removed from folder)
    missing = set(seen.keys()) - current_relpaths
    for rel in sorted(missing):
        try:
            doc_id = seen[rel]["doc_id"]
            col.delete(where={"doc_id": doc_id})
            del seen[rel]
            stats["deleted"] += 1
            if verbose:
                print(f"Deleted from index: {rel}")
        except Exception as e:
            stats["errors"] += 1
            if verbose:
                print(f"ERROR deleting {rel}: {e}")

    manifest["files"] = seen
    manifest["last_run_ts"] = time.time()
    _save_manifest(manifest_path, manifest)

    if verbose:
        print("\n--- Index Summary ---")
        for k, v in stats.items():
            print(f"{k}: {v}")

    return stats
