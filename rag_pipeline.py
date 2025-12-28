import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from openai import OpenAI

from config import CFG


# =========================
# Vector math
# =========================
def cosine(a: List[float], b: List[float]) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


# =========================
# Sentence + citation parsing
# =========================
def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Simple and stable sentence splitter for English prose.
    # Keeps citation tokens because we do not modify sentences.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def extract_sentence_citations(sentence: str) -> List[str]:
    """
    Extract *one or more* citations from a sentence.
    We accept citations anywhere, but our validator enforces at least one per sentence.
    Expected citation token: [<chunk_id>] where chunk_id contains ::C#
    Example: [RB_DRM_License_Failure.md::C2]
    """
    return re.findall(r"\[([^\[\]]+::C\d+)\]", sentence)

def parse_trailing_citation(sentence: str) -> Optional[str]:
    """
    Your stricter parser logic: capture a bracketed citation at the END of a sentence,
    allowing trailing '.' or ')'.

    This matches your example:
      m = re.search(r"\[([^\[\]]+)\]\s*[\.\)]?\s*$", s)
    """
    s = (sentence or "").strip()
   
    m = re.search(r"\[([^][]+)\]\s*[.)]?\s*$", s)


    return m.group(1) if m else None

def validate_citations(answer_text: str, require_trailing: bool = True) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Enforce: one citation per sentence.
    - If require_trailing=True: citation must appear at sentence end (your stricter rule)
    - Else: citation can appear anywhere in the sentence
    Returns (ok, message, parsed_sentences)
    """
    parsed = []
    sents = split_sentences(answer_text)

    if not sents:
        return False, "Empty answer.", []

    for s in sents:
        if require_trailing:
            cid = parse_trailing_citation(s)
            if not cid:
                parsed.append({"sentence": s, "citation": None})
                return False, "Missing trailing citation on a sentence.", parsed
            parsed.append({"sentence": s, "citation": cid})
        else:
            cids = extract_sentence_citations(s)
            if not cids:
                parsed.append({"sentence": s, "citation": None})
                return False, "Missing citation on a sentence.", parsed
            parsed.append({"sentence": s, "citation": cids[0]})

    return True, "", parsed


# =========================
# Snippet matching for groundedness
# =========================
def best_snippet_from_chunk(sentence: str, chunk_text: str, embed_fn) -> Tuple[float, str]:
    """
    Your added method.
    Goal: compare sentence embedding to the best local snippet inside the chunk, not the whole chunk.
    This reduces false negatives when a chunk contains multiple ideas.
    """
    candidates = split_sentences(chunk_text)
    if not candidates:
        return -1.0, ""

    sent_emb = embed_fn([sentence])[0]
    best_sim, best_snip = -1.0, ""

    for snip in candidates:
        sn_emb = embed_fn([snip])[0]
        sim = cosine(sent_emb, sn_emb)
        if sim > best_sim:
            best_sim, best_snip = sim, snip

    return best_sim, best_snip


def groundedness_by_citation_semantic(
    answer_text: str,
    hits: List[Dict[str, Any]],
    embed_fn,
    min_sim: float
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    For each sentence:
      - find its (trailing) citation chunk_id
      - compute semantic similarity sentence vs best snippet in cited chunk
      - fail if similarity < min_sim
    """
    ok, msg, parsed = validate_citations(answer_text, require_trailing=True)
    if not ok:
        return False, [{"reason": "citation_format", "message": msg, "parsed": parsed}]

    chunk_map = {h["chunk_id"]: h for h in hits}
    failures = []

    for item in parsed:
        sent = item["sentence"]
        cid = item["citation"]
        if cid not in chunk_map:
            failures.append({
                "sentence": sent,
                "citation": cid,
                "reason": "citation_not_in_hits"
            })
            continue

        chunk_text = chunk_map[cid]["text"]
        sim, best_snip = best_snippet_from_chunk(sent, chunk_text, embed_fn)

        if sim < min_sim:
            failures.append({
                "sentence": sent,
                "citation": cid,
                "best_similarity": sim,
                "min_required": min_sim,
                "best_snippet": best_snip
            })

    return (len(failures) == 0), failures


# =========================
# Refusal and answerability logic
# =========================
def is_summary_intent(q: str) -> bool:
    q = (q or "").lower()
    return any(k in q for k in ["summarize", "summary", "overview", "in one paragraph", "brief", "consolidate"])

def is_policy_blocked(q: str) -> bool:
    q = (q or "").lower()
    # Keep conservative; you can expand later
    return any(k in q for k in ["password", "secret", "api key", "token", "credentials"])

def should_refuse(question: str, hits: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Your negative-case handler.
    Returns: (refuse?, mode)
    """
    if is_policy_blocked(question):
        return True, "refuse_policy_gate"
    if not hits:
        return True, "refuse_no_hits"

    # Not-answerable = no sufficiently strong hit AND not a summary request
    if not is_summary_intent(question):
        strong = [h for h in hits if (h.get("score") or 0.0) >= CFG.min_score]
        if len(strong) == 0:
            return True, "refuse_not_answerable"

    return False, ""


# =========================
# Fallback extractive (deterministic)
# =========================
def fallback_extractive(hits: List[Dict[str, Any]], max_sentences: int = 4) -> str:
    """
    Deterministic, always-cited fallback.
    We reuse chunk text sentences and append trailing citations [chunk_id].
    """
    out = []
    for h in hits:
        cid = h["chunk_id"]
        for s in split_sentences(h["text"]):
            s = s.strip()
            if len(s) < 35:
                continue
            # ensure trailing citation
            out.append(f"{s} [{cid}].")
            if len(out) >= max_sentences:
                return " ".join(out)
    return "I don't know based on the provided runbooks."


# =========================
# LLM calls + repair
# =========================
def _call_openai(prompt: str, model: str) -> str:
    client = OpenAI()
    resp = client.responses.create(
        model=model,
        input=prompt
    )
    return (resp.output_text or "").strip()


def _build_prompt(question: str, hits: List[Dict[str, Any]]) -> str:
    ctx = "\n\n".join([f"[{h['chunk_id']}]\n{h['text']}" for h in hits])
    return (
        "You are a runbook assistant. Answer using ONLY the provided context.\n"
        "Rules:\n"
        "1) Write short, operational sentences.\n"
        "2) EVERY sentence MUST end with exactly one citation in the format [file::C#].\n"
        "3) Do not add any information not present in the context.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question:\n{question}\n"
    )

def _repair_prompt(bad_answer: str) -> str:
    return (
        "Fix the answer to satisfy the rules:\n"
        "- EVERY sentence must end with exactly one citation like [file::C#]\n"
        "- Do not add any new information\n"
        "- Keep the same meaning\n\n"
        f"Bad answer:\n{bad_answer}\n"
    )


# =========================
# Retrieval wrapper
# =========================
def retrieve(col, question: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Returns hits with:
      text, score, chunk_id, meta
    Note: Chroma returns distances. For cosine distance, similarity ~= 1 - distance.
    """
    res = col.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    hits = []
    for i in range(len(docs)):
        meta = metas[i]
        chunk_id = meta.get("chunk_id") or meta.get("id") or meta.get("chunk")
        hits.append({
            "text": docs[i],
            "score": 1.0 - float(dists[i]),
            "chunk_id": chunk_id,
            "meta": meta
        })
    return hits


# =========================
# Main pipeline: query_rag_llm_robust
# =========================
def query_rag_llm_robust(
    question: str,
    col,
    model: str = "gpt-4.1-mini",
    top_k: Optional[int] = None,
    min_sim: Optional[float] = None,
    require_trailing_citation: bool = True
) -> Dict[str, Any]:
    """
    End-to-end robust query.

    Modes:
      refuse_policy_gate
      refuse_no_hits
      refuse_not_answerable
      llm_ok_first_try
      llm_repaired_second_try
      fallback_extractive
    """
    top_k = top_k or CFG.top_k
    min_sim = min_sim if min_sim is not None else CFG.min_score

    # 1) Retrieve
    hits = retrieve(col, question, top_k=top_k)

    # 2) Refusal decision
    refuse, refuse_mode = should_refuse(question, hits)
    if refuse:
        # Special-case summary: allow deterministic extractive summary if we have any hits
        if refuse_mode == "refuse_not_answerable" and is_summary_intent(question) and hits:
            return {
                "mode": "fallback_extractive",
                "answer": fallback_extractive(hits),
                "format_ok": True,
                "grounded_ok": True,
                "hits": hits
            }

        return {
            "mode": refuse_mode,
            "answer": "I don't know based on the provided runbooks.",
            "format_ok": True,
            "grounded_ok": True,
            "hits": hits
        }

    # 3) LLM try #1
    prompt = _build_prompt(question, hits)
    answer1 = _call_openai(prompt, model=model)

    fmt_ok, fmt_msg, _ = validate_citations(answer1, require_trailing=require_trailing_citation)
    if fmt_ok:
        candidate = answer1
        mode = "llm_ok_first_try"
    else:
        # 4) Repair try #2
        answer2 = _call_openai(_repair_prompt(answer1), model=model)
        fmt_ok2, fmt_msg2, _ = validate_citations(answer2, require_trailing=require_trailing_citation)
        if fmt_ok2:
            candidate = answer2
            mode = "llm_repaired_second_try"
        else:
            # 5) Fallback extractive (format could not be repaired)
            return {
                "mode": "fallback_extractive",
                "answer": fallback_extractive(hits),
                "format_ok": True,
                "grounded_ok": True,
                "hits": hits,
                "format_fail": {"first": fmt_msg, "second": fmt_msg2}
            }

    # 6) Groundedness check (semantic vs cited evidence)
    embed_fn = col._embedding_function  # SentenceTransformerEmbeddingFunction in your setup
    grounded_ok, details = groundedness_by_citation_semantic(candidate, hits, embed_fn, min_sim=min_sim)

    if not grounded_ok:
        # 7) Fallback extractive if still not grounded
        return {
            "mode": "fallback_extractive",
            "answer": fallback_extractive(hits),
            "format_ok": True,
            "grounded_ok": True,
            "hits": hits,
            "ground_details": details
        }

    return {
        "mode": mode,
        "answer": candidate,
        "format_ok": True,
        "grounded_ok": True,
        "hits": hits,
        "ground_details": details
    }
