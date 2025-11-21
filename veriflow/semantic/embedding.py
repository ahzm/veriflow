# veriflow/semantic/embedding.py
"""
Embedding utilities for Veriflow semantic matcher.

This module is OPTIONAL. It is only used when:
    VERIFLOW_USE_EMB=1 (default) and
    an embedding backend is available:
       - OpenAI embeddings if OPENAI_API_KEY is set and openai package installed
       - sentence-transformers local model if installed

All imports are lazy to avoid hard dependencies.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set, Any
import os
import math

# Simple in-memory cache: text -> embedding vector
_EMB_CACHE: Dict[str, List[float]] = {}

# --- Persistent embedding cache (load on startup) ---
import pickle

CACHE_PATH = os.path.expanduser("~/.veriflow_emb_cache.pkl")

# Load persistent cache if exists
if os.path.isfile(CACHE_PATH):
    try:
        with open(CACHE_PATH, "rb") as f:
            _EMB_CACHE.update(pickle.load(f))
    except Exception:
        pass


def _save_cache():
    """Persist embedding cache to disk (best effort)."""
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(_EMB_CACHE, f)
    except Exception:
        pass

# Singleton for sentence-transformers model (lazy init)
_ST_MODEL = None


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity for embeddings."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def get_embedding(text: str) -> Optional[List[float]]:
    """
    Return embedding for text using available backend.
    If no backend configured, return None.
    """
    global _ST_MODEL

    text = (text or "").strip()
    if not text:
        return None

    if text in _EMB_CACHE:
        return _EMB_CACHE[text]

    # ---- Backend selection ----
    # Option 1: OpenAI embeddings (if OPENAI_API_KEY set)
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI  # lazy import
            client = OpenAI(api_key=api_key)
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            emb = resp.data[0].embedding
            _EMB_CACHE[text] = emb
            _save_cache()
            return emb
        except Exception:
            # fallback to local model
            pass

    # Option 2: sentence-transformers local model (if installed)
    try:
        if _ST_MODEL is None:
            from sentence_transformers import SentenceTransformer  # lazy import
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

        emb = _ST_MODEL.encode([text])[0].tolist()
        _EMB_CACHE[text] = emb
        _save_cache()
        return emb
    except Exception:
        return None

def get_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Batch embeddings using sentence-transformers backend.
    Falls back to individual calls if batch not available.
    """
    global _ST_MODEL

    # empty shortcut
    if not texts:
        return []

    # batch is only possible if ST is available
    try:
        if _ST_MODEL is None:
            from sentence_transformers import SentenceTransformer
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

        # ST supports efficient batch encode
        embs = _ST_MODEL.encode(texts)
        out = []
        for t, e in zip(texts, embs):
            v = e.tolist()
            _EMB_CACHE[t] = v
            out.append(v)
        return out
    except Exception:
        # fallback: slow per-text
        return [get_embedding(t) for t in texts]

def embedding_relevance_boost(
    prompt: str,
    nodes: List[dict],
    node_label_fn,
    relevant_nodes: Set[str],
    evidence: List[Dict[str, Any]],
    top_k: int = 3,
    sim_threshold: float = 0.55,
) -> Set[str]:
    """
    Softly expand relevant_nodes using embedding similarity.

    - Controlled by env switch VERIFLOW_USE_EMB (default 1).
    - Never removes relevance, only adds.

    Parameters:
        node_label_fn: function(node_dict)->str used to compute node texts.
    """
    # Hard switch to disable embedding relevance
    if os.environ.get("VERIFLOW_USE_EMB", "1") != "1":
        evidence.append({
            "rule": "embedding_relevance",
            "applicable": False,
            "ok": None,
            "note": "embedding disabled by VERIFLOW_USE_EMB=0"
        })
        return relevant_nodes

    p_emb = get_embedding(prompt)
    if p_emb is None:
        evidence.append({
            "rule": "embedding_relevance",
            "applicable": False,
            "ok": None,
            "note": "no embedding backend available; skipped"
        })
        return relevant_nodes

    # ---- only score candidate nodes (not already relevant) ----
    cand_nodes = []
    cand_ids = []
    cand_labels = []

    for n in nodes:
        nid = n.get("id")
        if nid is None or nid in relevant_nodes:
            continue
        cand_nodes.append(n)
        cand_ids.append(nid)
        cand_labels.append(node_label_fn(n))

    if not cand_labels:
        evidence.append({
            "rule": "embedding_relevance",
            "applicable": True,
            "ok": True,
            "note": "no candidate nodes to boost"
        })
        return relevant_nodes

    # ---- batch encode all candidate labels ----
    label_embs = get_embeddings_batch(cand_labels)

    scored: List[Tuple[str, float]] = []
    for nid, lab, emb in zip(cand_ids, cand_labels, label_embs):
        if emb is None:
            continue
        sim = cosine(p_emb, emb)
        scored.append((nid, sim))

    scored.sort(key=lambda x: x[1], reverse=True)

    boosted = []
    for nid, sim in scored[:top_k]:
        if sim >= sim_threshold:
            relevant_nodes.add(nid)
            boosted.append((nid, sim))

    evidence.append({
        "rule": "embedding_relevance",
        "applicable": True,
        "ok": True,
        "top_k": top_k,
        "threshold": sim_threshold,
        "boosted": [{"id": i, "sim": round(s, 3)} for i, s in boosted],
        "note": "soft relevance boost by embeddings (batch)"
    })

    return relevant_nodes