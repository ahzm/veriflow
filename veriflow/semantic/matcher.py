# veriflow/semantic/matcher.py

from typing import Dict, Any, Tuple, List
import networkx as nx
from veriflow.utils.graph import build_dag
from veriflow.semantic.intent_extractor import extract_intent_hybrid

# --- Helpers for semantic ordering ---

def _node_label(n: dict) -> str:
    """Return a lowercase label combining type + name."""
    return (n.get("type", "") + " " + n.get("name", "")).lower()

def _is_action_node(label: str) -> bool:
    """
    Return True if this node is a terminal/action node
    (we should not “pass through” it unless it is the target).
    """
    action_keywords = (
        "email", "slack", "telegram",
        "sms", "discord", "notification",
        "http request",
    )
    return any(k in label for k in action_keywords)

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
    """
    Check if there exists a semantically coherent HTTP->Email path.

    We accept multi-hop paths such as:
      HTTP -> IF -> Router -> Transform -> Email

    but we avoid going through unrelated terminal actions
    (e.g., HTTP -> Slack -> ... -> Email) when exploring paths.
    
    NOTE: currently specialized to HTTP→Email. Future versions may
    generalize to other (source, sink) pairs.
    """
    nodes = workflow.get("nodes", [])
    if not nodes:
        return 1.0

    G = build_dag(workflow)

    # Index nodes by id for quick lookup
    index: Dict[str, dict] = {n["id"]: n for n in nodes if "id" in n}

    http_ids = _find_nodes(nodes, lambda t: ("http" in t) or ("request" in t))
    email_ids = set(_find_nodes(nodes, lambda t: ("email" in t)))

    # If the pair is not present, we do not penalize ordering.
    if not http_ids or not email_ids:
        return 1.0

    max_depth = 6  # simple safety cut-off to avoid very long / noisy paths

    def _has_semantic_path(src: str) -> bool:
        """BFS from src with semantic constraints."""
        if src not in G:
            return False

        from collections import deque

        visited = {src}
        queue = deque([(src, 0)])

        while queue:
            cur, depth = queue.popleft()
            if depth > max_depth:
                continue

            # If we reached an Email node, we are happy.
            if cur in email_ids:
                return True

            for succ in G.successors(cur):
                if succ in visited:
                    continue
                visited.add(succ)

                node = index.get(succ, {})
                label = _node_label(node)

                # If successor is another terminal action (Slack, Telegram…): stop this branch
                if _is_action_node(label):
                    continue

                # Otherwise treat as passthrough (IF / Router / Transform / unknown node)
                queue.append((succ, depth + 1))

        return False

    # If any HTTP node has a valid semantic path to an Email node, ordering is OK.
    for h in http_ids:
        if _has_semantic_path(h):
            return 1.0

    return 0.0


def _desired_keys(intent: Dict[str, bool]) -> List[str]:
    """Return list of desired capability keys (those starting with 'need_')."""
    return [k for k, v in intent.items() if v and k.startswith("need_")]


def semantic_score(
    workflow: Dict[str, Any],
    prompt: str,
    use_llm: bool = False,
) -> Tuple[float, List[str], Dict[str, Any]]:
    """
    Compute semantic consistency between extracted intent and workflow nodes.

    Returns:
        M (float): semantic score in [0,1]
        issues (List[str]): human-readable issue list
        detail (Dict[str, Any]): sub-scores and intent meta, e.g.
            {
              "trigger": 1.0,
              "action": 1.0,
              "order": 1.0,
              "intent_conf": 0.92,
              "source": "rule" | "rule+llm",
              "intent": {
                  "need_schedule": true,
                  "need_email": true,
                  "need_http": true,
                  "need_slack": false,
                  "need_telegram": false
              },
              "intent_chain": [
                  "rule: matched schedule keywords or English time pattern",
                  "rule: matched email keywords",
                  "rule: matched http/api keywords"
              ],
              "irrelevant_nodes": ["2", "5"]
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

    # 8) Irrelevant nodes: nodes that do not implement any requested capability
    irrelevant_nodes: List[str] = []
    for n in workflow.get("nodes", []):
        t = (n.get("type", "") + " " + n.get("name", "")).lower()
        relevant = (
            (intent.get("need_schedule") and any(k in t for k in ("schedule", "cron", "trigger", "webhook")))
            or (intent.get("need_email") and "email" in t)
            or (intent.get("need_http") and ("http" in t or "request" in t))
            or (intent.get("need_slack") and "slack" in t)
            or (intent.get("need_telegram") and "telegram" in t)
        )

        node_id = n.get("id")
        if not relevant and node_id is not None:
            irrelevant_nodes.append(node_id)

    detail = {
        "trigger": float(trigger_ok),
        "action": float(action_ok),
        "order": float(order_ok),
        "intent_conf": float(intent_res.overall_confidence),
        # storing as float-compatible value is convenient when exporting
        "source": intent_res.source,            # Keep the source string separately (not a float)
        "intent": intent,
        "intent_chain": intent_res.meta.get("intent_chain", []),
        "irrelevant_nodes": irrelevant_nodes,                       
    }

    return M, issues, detail