import os
import re
from pathlib import Path
import gradio as gr

from config import CFG
from app import get_collection, MANIFEST_PATH, COLLECTION_NAME
from ingest import index_folder
from rag_pipeline import query_rag_llm_robust, split_sentences


CSS = """
:root { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }

#app-title { font-size: 30px; font-weight: 900; line-height: 1.15; margin-bottom: 6px; }
#app-subtitle { font-size: 14px; opacity: 0.85; margin-bottom: 14px; }

.section-title { font-size: 16px; font-weight: 800; margin: 6px 0 10px 0; }
.label-big label { font-size: 16px !important; font-weight: 800 !important; }
.box-big textarea, .box-big input { font-size: 15px !important; }
.mono textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important; }

.hint { font-size: 12px; opacity: 0.85; }

.answer-box textarea {
  font-size: 16px !important;     /* bigger answer text */
  line-height: 1.5 !important;
}

.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  padding: 12px;
}

hr.soft { border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 10px 0; }

/* HTML answer rendering (Markdown) */
.answer-list { font-size: 16px; line-height: 1.55; }
.answer-item { margin: 8px 0; }
.answer-sent { font-weight: 600; }
.answer-cite {
  font-size: 13px;
  opacity: 0.80;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  margin-left: 6px;
  white-space: nowrap;
}
.evidence-title { font-weight: 800; margin: 4px 0 8px 0; }
.evidence-item { margin: 10px 0; padding: 10px; border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; }
.evidence-head { display: flex; gap: 10px; align-items: baseline; margin-bottom: 6px; }
.evidence-cid { font-family: ui-monospace, monospace; font-size: 12px; opacity: 0.9; }
.evidence-src { font-size: 12px; opacity: 0.85; }
.evidence-text { font-size: 13px; line-height: 1.45; white-space: pre-wrap; opacity: 0.92; }
"""


# -------------------------
# Helpers
# -------------------------
CITE_RE = re.compile(r"\[([^\[\]]+::C\d+)\]\s*[.)]?\s*$")

def parse_trailing_citation(sentence: str):
    s = (sentence or "").strip()
    m = CITE_RE.search(s)
    return m.group(1) if m else None

def strip_trailing_citation(sentence: str) -> str:
    s = (sentence or "").strip()
    # Remove trailing citation token (and optional trailing . or ))
    return re.sub(r"\s*\[[^\[\]]+::C\d+\]\s*[.)]?\s*$", "", s).strip()

def format_answer_as_markdown_bullets(answer: str):
    """
    Return:
      - md: pretty bullets with sentence emphasis + smaller citation styling
      - items: list of dicts with {sentence_text, citation}
    """
    answer = (answer or "").strip()
    if not answer:
        return "", []

    sents = split_sentences(answer)
    items = []
    lines = ['<div class="answer-list">']
    for s in sents:
        cid = parse_trailing_citation(s)  # expects trailing citation
        sent_txt = strip_trailing_citation(s)
        items.append({"sentence": sent_txt, "citation": cid})

        # render: bullet + sentence bold-ish + cite small mono
        cite_html = f'<span class="answer-cite">[{cid}]</span>' if cid else '<span class="answer-cite">[MISSING]</span>'
        lines.append(
            f'<div class="answer-item">• <span class="answer-sent">{sent_txt}</span> {cite_html}</div>'
        )
    lines.append("</div>")
    return "\n".join(lines), items

def build_evidence_panel(items, hits, show: bool):
    """
    Build evidence HTML for each sentence using cited chunk_id -> chunk text.
    """
    if not show:
        return ""

    if not items:
        return '<div class="hint">No answer to show evidence for.</div>'

    # Map hits by chunk_id
    hit_map = {h["chunk_id"]: h for h in (hits or [])}

    blocks = ['<div class="evidence-title">Evidence (cited chunks)</div>']
    for i, it in enumerate(items, start=1):
        cid = it.get("citation")
        sent = it.get("sentence") or ""
        if not cid or cid not in hit_map:
            blocks.append(
                f'<div class="evidence-item"><div class="evidence-head">'
                f'<div class="evidence-cid">[{cid or "MISSING"}]</div>'
                f'<div class="evidence-src">Not found in retrieved hits</div>'
                f'</div><div class="evidence-text">{sent}</div></div>'
            )
            continue

        h = hit_map[cid]
        src = h.get("meta", {}).get("source_file", "")
        score = h.get("score", 0.0)
        text = (h.get("text") or "").strip()

        blocks.append(
            f'<div class="evidence-item">'
            f'<div class="evidence-head">'
            f'<div class="evidence-cid">[{cid}]</div>'
            f'<div class="evidence-src">source={src} · score={score:.3f}</div>'
            f'</div>'
            f'<div class="evidence-text">{text}</div>'
            f'</div>'
        )

    return "\n".join(blocks)


def write_download_file(meta_text: str, answer_plain: str) -> str:
    """
    Create a .txt file and return its path for Gradio File component.
    """
    out_path = "/content/ott-rag-capstone/last_answer.txt"
    content = meta_text.strip() + "\n\n" + answer_plain.strip() + "\n"
    Path(out_path).write_text(content, encoding="utf-8")
    return out_path


# -------------------------
# Actions
# -------------------------
def do_index(runbooks_dir: str, force_rebuild: bool):
    runbooks_dir = (runbooks_dir or "").strip()
    if not runbooks_dir:
        return "ERROR: Provide a folder path."

    folder = Path(runbooks_dir)
    if not folder.exists():
        return f"ERROR: Folder not found: {folder}"

    client, col = get_collection()

    if force_rebuild:
        client.delete_collection(COLLECTION_NAME)
        client, col = get_collection()
        Path(MANIFEST_PATH).unlink(missing_ok=True)

    stats = index_folder(
        folder_path=str(folder),
        col=col,
        manifest_path=MANIFEST_PATH,
        include_ext=[".md", ".txt"],
        verbose=False
    )

    return (
        "Index completed successfully.\n"
        f"Stats: {stats}\n"
        f"Chroma count: {col.count()}\n"
        f"Manifest: {MANIFEST_PATH}"
    )


def do_query(question: str, model: str, min_sim: float, top_k: int, show_evidence: bool):
    question = (question or "").strip()
    if not question:
        return "ERROR: Enter a question.", "", "", "", None

    if not os.environ.get("OPENAI_API_KEY"):
        return "ERROR: OPENAI_API_KEY not set in environment.", "", "", "", None

    _, col = get_collection()

    out = query_rag_llm_robust(
        question,
        col=col,
        model=model,
        top_k=int(top_k),
        min_sim=float(min_sim)
    )

    meta = (
        f"MODE: {out.get('mode')}\n"
        f"format_ok: {out.get('format_ok')}\n"
        f"grounded_ok: {out.get('grounded_ok')}"
    )

    hits_lines = []
    for h in (out.get("hits") or [])[: int(top_k)]:
        hits_lines.append(
            f"- score={h.get('score', 0):.3f} | {h['chunk_id']} | {h['meta'].get('source_file')}"
        )
    hits_text = "\n".join(hits_lines) if hits_lines else "(no hits)"

    answer_md, items = format_answer_as_markdown_bullets(out.get("answer", ""))
    evidence_html = build_evidence_panel(items, out.get("hits") or [], show_evidence)

    # Download content: keep plain text with original sentence+citation (one per line)
    # We reconstruct plain lines in a stable way:
    plain_lines = []
    for it in items:
        if it.get("citation"):
            plain_lines.append(f"{it['sentence']} [{it['citation']}]")
        else:
            plain_lines.append(it["sentence"])
    answer_plain = "\n".join(plain_lines)

    dl_path = write_download_file(meta, answer_plain)

    return meta, hits_text, answer_md, evidence_html, dl_path


# -------------------------
# UI layout
# -------------------------
def build_ui():
    with gr.Blocks(title="OTT Runbook RAG", css=CSS) as demo:
        gr.Markdown(
            """
<div id="app-title">OTT Runbook RAG</div>
<div id="app-subtitle">
Productized Retrieval-Augmented Generation for OTT operational runbooks (DRM / CDN / Packager).
Enforces one trailing citation per sentence and semantic groundedness checks.
</div>
""",
        )

        with gr.Row():
            # Left: Indexing
            with gr.Column(scale=4):
                with gr.Group():
                    gr.Markdown('<div class="section-title">Index Runbooks</div>')
                    runbooks_dir = gr.Textbox(
                        value="/content/ott-rag-capstone/data/runbooks",
                        label="Runbooks folder",
                        elem_classes=["label-big", "box-big"],
                    )
                    force = gr.Checkbox(value=False, label="Force rebuild (drop collection + re-index)")
                    gr.Markdown('<div class="hint">Tip: Add new .md/.txt files to the folder, then click Index. Incremental updates are supported via the manifest.</div>')
                    btn_index = gr.Button("Run Index", variant="primary")
                    index_out = gr.Textbox(label="Index output", lines=8, elem_classes=["mono", "box-big"])
                    btn_index.click(do_index, inputs=[runbooks_dir, force], outputs=[index_out])

            # Right: Querying
            with gr.Column(scale=6):
                with gr.Group():
                    gr.Markdown('<div class="section-title">Ask a Question</div>')

                    q = gr.Textbox(
                        label="Question",
                        placeholder="e.g., What are the immediate checks for DRM license failures?",
                        lines=2,
                        elem_classes=["label-big", "box-big"],
                    )

                    with gr.Row():
                        model = gr.Textbox(value="gpt-4.1-mini", label="OpenAI model", elem_classes=["label-big", "box-big"])
                        min_sim = gr.Slider(0.20, 0.80, value=float(CFG.min_score), step=0.01, label="Groundedness min similarity")
                        top_k = gr.Slider(1, 10, value=int(CFG.top_k), step=1, label="Top K")

                    with gr.Row():
                        show_evidence = gr.Checkbox(value=True, label="Show evidence (cited chunks)")

                    btnq = gr.Button("Ask", variant="primary")

                    meta = gr.Textbox(label="Mode + checks", lines=3, elem_classes=["mono", "box-big"])
                    hits = gr.Textbox(label="Retrieved chunks", lines=6, elem_classes=["mono", "box-big"])

                    # Render answer as Markdown/HTML so we can style citations smaller
                    answer_md = gr.Markdown(label="Answer", elem_classes=["answer-box"])

                    evidence = gr.HTML(label="Evidence", value="")

                    with gr.Row():
                        download = gr.File(label="Download answer (.txt)")

                    btnq.click(
                        do_query,
                        inputs=[q, model, min_sim, top_k, show_evidence],
                        outputs=[meta, hits, answer_md, evidence, download]
                    )

        gr.Markdown(
            """
<div class="hint">
Notes:
- Each answer sentence must end with a trailing citation like <code>[RB_DRM_License_Failure.md::C1]</code>.
- Groundedness is verified via semantic similarity between each sentence and the best evidence snippet inside the cited chunk.
</div>
""",
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True)
