# veriflow/semantic/matcher.py
# All comments in English.

from typing import Dict, Any, Tuple, List
import networkx as nx
from veriflow.utils.graph import build_dag
from veriflow.semantic.intent_extractor import extract_intent_hybrid


def _find_nodes(nodes: List[dict], predicate) -> List[str]:
    """Return node ids whose (type + name) satisfies the given predicate."""
    ids: List[str] = []
    for n in nodes:
        t = (n.get("type", "") + " " + n.get("name", "")).lower()
        if predicate(t):
            ids.append(n["id"])
    return ids


def inspect_nodes(nodes: List[dict]) -> Dict[str, bool]:
    """Summarize presence of key capability nodes in the workflow."""
    summary = {
        "has_schedule": False,
        "has_email": False,
        "has_http": False,
        "has_slack": False,
        "has_telegram": False,
    }
    for n in nodes:
        t = (n.get("type", "") + " " + n.get("name", "")).lower()
        if any(k in t for k in ("schedule", "cron", "trigger", "webhook")):
            summary["has_schedule"] = True
        if "email" in t:
            summary["has_email"] = True
        if "http" in t or "request" in t:
            summary["has_http"] = True
        if "slack" in t:
            summary["has_slack"] = True
        if "telegram" in t:
            summary["has_telegram"] = True
    return summary


def order_ok_by_path(workflow: Dict[str, Any]) -> float:
    """Check if there exists an HTTP→Email path when both are required."""
    nodes = workflow.get("nodes", [])
    G = build_dag(workflow)

    http_ids = _find_nodes(nodes, lambda t: ("http" in t) or ("request" in t))
    email_ids = _find_nodes(nodes, lambda t: ("email" in t))

    # If the pair is not present, we do not penalize ordering.
    if not http_ids or not email_ids:
        return 1.0

    for h in http_ids:
        for e in email_ids:
            if h in G and e in G and nx.has_path(G, h, e):
                return 1.0
    return 0.0


def _desired_keys(intent: Dict[str, bool]) -> List[str]:
    """Return list of desired capability keys (those starting with 'need_')."""
    return [k for k, v in intent.items() if v and k.startswith("need_")]


def semantic_score(
    workflow: Dict[str, Any],
    prompt: str,
    use_llm: bool = False,
) -> Tuple[float, List[str], Dict[str, float]]:
    """
    Compute semantic consistency between extracted intent and workflow nodes.

    Returns:
        M (float): semantic score in [0,1]
        issues (List[str]): human-readable issue list
        detail (Dict[str, float]): sub-scores and intent meta, e.g.
            {
              "trigger": 1.0,
              "action": 1.0,
              "order": 1.0,
              "intent_conf": 0.92,
              "source": "rule" | "rule+llm"
            }
    """
    # 1) Extract intent (rule-based with optional LLM refinement)
    intent_res = extract_intent_hybrid(prompt, use_llm=use_llm)
    intent = intent_res.intent

    # 2) Inspect nodes present in the workflow
    nodes_summary = inspect_nodes(workflow.get("nodes", []))

    # 3) Trigger sub-score
    trigger_ok = 1.0 if (not intent.get("need_schedule", False) or nodes_summary["has_schedule"]) else 0.0

    # 4) Action coverage sub-score
    desired = _desired_keys(intent)
    matched = 0
    if intent.get("need_email") and nodes_summary["has_email"]:
        matched += 1
    if intent.get("need_http") and nodes_summary["has_http"]:
        matched += 1
    if intent.get("need_slack") and nodes_summary["has_slack"]:
        matched += 1
    if intent.get("need_telegram") and nodes_summary["has_telegram"]:
        matched += 1

    # Do not count schedule in the denominator for action coverage
    denom = len(desired) - (1 if intent.get("need_schedule") else 0)
    action_ok = matched / max(1, denom)

    # 5) Ordering sub-score (HTTP→Email path when both required)
    order_ok = order_ok_by_path(workflow) if (intent.get("need_http") and intent.get("need_email")) else 1.0

    # 6) Aggregate
    M = round((trigger_ok + action_ok + order_ok) / 3, 2)

    # 7) Issues
    issues: List[str] = []
    if trigger_ok < 1.0:
        issues.append("Intent requires scheduling, but no schedule/trigger node")
    if intent.get("need_email") and not nodes_summary["has_email"]:
        issues.append("Missing Email node for intent")
    if intent.get("need_http") and not nodes_summary["has_http"]:
        issues.append("Missing HTTP node for intent")
    if intent.get("need_slack") and not nodes_summary["has_slack"]:
        issues.append("Missing Slack node for intent")
    if intent.get("need_telegram") and not nodes_summary["has_telegram"]:
        issues.append("Missing Telegram node for intent")

    detail = {
        "trigger": float(trigger_ok),
        "action": float(action_ok),
        "order": float(order_ok),
        "intent_conf": float(intent_res.overall_confidence),
        # storing as float-compatible value is convenient when exporting
        "source": intent_res.source,            # keep as string
        "intent": intent,                       
    }
    # Keep the source string separately (not a float)
    # Caller can print in verbose mode
    detail["source"] = intent_res.source  # type: ignore

    return M, issues, detail