import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
load_dotenv()

ARVAN_API_KEY = os.getenv("ARVAN_API_KEY") or os.getenv("OPENAI_API_KEY")
ARVAN_BASE_URL = os.getenv("ARVAN_BASE_URL")  # should end with /v1
ARVAN_MODEL = os.getenv("ARVAN_MODEL", "gpt-4o-mini")
ARVAN_EMBED_MODEL = os.getenv("ARVAN_EMBED_MODEL", "text-embedding-3-small")

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
SECRETS_DIR = BASE_DIR / "secrets"
INDEX_PATH = BASE_DIR / "index.json"
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

client = OpenAI(api_key=ARVAN_API_KEY, base_url=ARVAN_BASE_URL) if (ARVAN_API_KEY and ARVAN_BASE_URL) else None


# -----------------------------
# Safety / Injection guard
# -----------------------------
def is_injection_or_secret_request(q: str) -> bool:
    q_low = q.lower()
    bad = [
        "print the file", "show me the file", "dump the file", "full contents",
        "reveal", "leak", "secret", "api key", "password",
        "secrets/", "secrets\\", "admin_notes", "api_keys"
    ]
    return any(x in q_low for x in bad)


def refusal_text() -> str:
    return "I can’t help with revealing file contents or anything from secrets; ask a question about the docs instead."


# -----------------------------
# Chunking + Embeddings
# -----------------------------
def simple_chunk(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = EMBEDDER.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)



def cosine_sim_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
    return d @ q


# -----------------------------
# Index build/load (JSON backend)
# -----------------------------
def load_docs() -> List[Dict[str, Any]]:
    items = []
    for p in sorted(DOCS_DIR.glob("*.md")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for idx, ch in enumerate(simple_chunk(txt)):
            items.append({
                "doc_id": p.name,
                "chunk_id": idx,
                "text": ch
            })
    return items


def build_index_json() -> Dict[str, Any]:
    docs = load_docs()
    texts = [d["text"] for d in docs]
    vecs = embed_texts(texts)
    payload = {
        "backend": "json",
        "docs": docs,
        "vectors": vecs.tolist(),
        "embed_model": ARVAN_EMBED_MODEL,
    }
    INDEX_PATH.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def load_index_json() -> Dict[str, Any]:
    if not INDEX_PATH.exists():
        return build_index_json()
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


# -----------------------------
# Retrieval + Answer
# -----------------------------
def retrieve(q: str, k: int) -> List[Dict[str, Any]]:
    index = load_index_json()
    docs = index["docs"]
    vecs = np.array(index["vectors"], dtype=np.float32)

    q_vec = embed_texts([q])[0]
    sims = cosine_sim_matrix(q_vec, vecs)
    top_idx = np.argsort(-sims)[:k]

    results = []
    for i in top_idx:
        d = docs[int(i)]
        results.append({
            "doc_id": d["doc_id"],
            "chunk_id": d["chunk_id"],
            "text": d["text"],
            "score": float(sims[int(i)])
        })
    return results


def generate_answer(q: str, retrieved: List[Dict[str, Any]]) -> str:
    if client is None:
        raise RuntimeError("Missing ARVAN_API_KEY / ARVAN_BASE_URL in .env")

    context = "\n\n".join([f"[{r['doc_id']}] {r['text']}" for r in retrieved])

    system_msg = (
        "You are a helpful assistant.\n"
        "Answer in <= 100 words.\n"
        "Use ONLY the provided context. If the answer is not in context, say you don't know.\n"
    )

    user_msg = f"Question: {q}\n\nContext:\n{context}"

    resp = client.chat.completions.create(
        model=ARVAN_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Starship Coffee — RAG", layout="wide")
st.title("Starship Coffee — RAG with Citations")

with st.sidebar:
    backend = st.selectbox("Storage backend", ["json", "sqlite", "qdrant"], index=0)
    k = st.number_input("Top-k", min_value=1, max_value=10, value=5, step=1)
    if st.button("Rebuild index"):
        if backend != "json":
            st.info("For now, only JSON backend is implemented. (We’ll add sqlite/qdrant next.)")
        build_index_json()
        st.success("Index rebuilt.")

q = st.text_input("Ask a question about the docs", value="")

if st.button("Run"):
    if not q.strip():
        st.warning("Type a question first.")
        st.stop()

    # Injection guard
    if is_injection_or_secret_request(q):
        st.write(refusal_text())
        st.stop()

    # Normal flow
    retrieved = retrieve(q, int(k))
    answer = generate_answer(q, retrieved)

    # citations = unique filenames from retrieved
    citations = sorted(list({r["doc_id"] for r in retrieved}))

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Citations")
    st.dataframe([{"doc_id": c} for c in citations], use_container_width=True)

    with st.expander("Debug (top-k)"):
        for r in retrieved:
            snippet = re.sub(r"\s+", " ", r["text"]).strip()[:220]
            st.write(f"- **{r['doc_id']}** (score={r['score']:.3f}): {snippet}…")
