# veriflow/generator/genllm.py

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    # Only for type checkers; won't run at runtime
    from openai import OpenAI as OpenAIType  # type: ignore[import-not-found]

try:
    from openai import OpenAI as OpenAIRuntime
except Exception:
    OpenAIRuntime = None  # type: ignore[assignment]


def _get_client() -> Optional[Any]:
    """Return an OpenAI-like client instance, or None if unavailable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if OpenAIRuntime is None or not api_key:
        return None
    return OpenAIRuntime(api_key=api_key)

def load_prompts_file(path: Path) -> List[Tuple[str, str]]:
    # simple parser for W5.txt style:
    # W5_01\n<text...>\n\nW5_02\n<text...>...
    text = path.read_text(encoding="utf-8").strip()
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    pairs: List[Tuple[str, str]] = []
    for blk in blocks:
        lines = [l.strip() for l in blk.splitlines() if l.strip()]
        if not lines:
            continue
        case_id = lines[0]
        prompt = " ".join(lines[1:]) if len(lines) > 1 else ""
        pairs.append((case_id, prompt))
    return pairs


def generate_n8n_workflow(prompt: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Generate a single n8n-like workflow JSON from a natural language prompt.

    If no API key/client is available, returns a minimal stub workflow
    (so that tests won't crash).
    """
    client = _get_client()
    if client is None:
        # Fallback: minimal stub to keep the pipeline runnable
        return {
            "nodes": [],
            "connections": {},
            "meta": {"warning": "LLM client not available; stub workflow returned."},
        }

    sys_msg = (
        "You are an assistant that generates n8n workflows as pure JSON. "
        "Output ONLY a JSON object with keys 'nodes' and 'connections'. "
        "Nodes must contain fields: id (string), name (string), type (string). "
        "Connections must follow the n8n format."
    )

    user_msg = (
        "Generate an n8n workflow for the following task:\n"
        f"{prompt}\n\n"
        "Return ONLY the JSON object, no explanations."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=800,
    )

    content = resp.choices[0].message.content.strip()
    try:
        wf = json.loads(content)
    except Exception:
        # Very defensive: if parsing fails, wrap content
        return {
            "nodes": [],
            "connections": {},
            "meta": {"error": "LLM returned non-JSON content", "raw": content},
        }

    if not isinstance(wf, dict):
        wf = {"meta": {"error": "LLM did not return a JSON object"}, "raw": wf}

    wf.setdefault("nodes", [])
    wf.setdefault("connections", {})

    return wf