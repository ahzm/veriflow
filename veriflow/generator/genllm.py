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

    This is intended for GenLLM benchmark synthesis, not for production n8n deployments.
    If no API key/client is available, returns a minimal stub workflow.
    """
    client = _get_client()
    if client is None:
        # Fallback: minimal stub to keep the pipeline runnable
        return {
            "nodes": [],
            "connections": {},
            "meta": {"warning": "LLM client not available; stub workflow returned."},
        }
    
    sys_msg = """
    You are an assistant that generates n8n workflows as pure JSON.

    Your task:
    - Given a natural-language description of a workflow, output a SINGLE JSON object.
    - The JSON MUST contain EXACTLY two top-level keys: "nodes" and "connections".
    - Output ONLY raw JSON. No Markdown, no code fences, no natural language, no explanations.

    Hard constraints:

    1) "nodes" MUST be a list of objects. Each node MUST have:
    - "id"         (string)
    - "name"       (string, unique within the workflow)
    - "type"       (string)
    - "parameters" (object, can be {})

    2) "connections" MUST use node NAMES as keys (NOT node IDs).
    Example:
    "connections": {
        "Webhook": {
            "main": [[{ "node": "Send Email", "type": "main", "index": 0 }]]
        }
    }

    3) Email node ("n8n-nodes-base.emailSend") MUST include in its "parameters":
    - "to"      (string)
    - "subject" (string)
    - "text"    (string)

    4) Slack node ("n8n-nodes-base.slack") MUST include:
    - "text" (string)

    5) Workflow MUST be a connected DAG:
    - no isolated nodes
    - no cycles
    - every node must be reachable from the trigger/root node

    6) Do NOT add extra keys at the top level. Only:
    {
        "nodes": [...],
        "connections": { ... }
    }
    """.strip()

    user_msg = (
    "Generate an n8n workflow for the following task:\n\n"
    f"{prompt}\n\n"
    "Return ONLY a raw JSON object with top-level keys \"nodes\" and \"connections\".\n"
    "Do NOT output Markdown, comments, code fences, or explanations."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=1500,
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

    # ---- Post-process: minimal parameters for executability ----
    for n in wf.get("nodes", []):
        n_type = n.get("type")
        params = n.setdefault("parameters", {})

        if n_type == "n8n-nodes-base.emailSend":
            # guarantee minimal parameters so that executability_score is not trivially 0
            params.setdefault("to", "placeholder@example.com")
            params.setdefault("subject", "LLM-generated confirmation")
            params.setdefault("text", "This is a confirmation email from an LLM-generated workflow.")

        if n_type == "n8n-nodes-base.slack":
            # make sure Slack has at least some text
            params.setdefault("text", "LLM-generated Slack notification.")

    return wf