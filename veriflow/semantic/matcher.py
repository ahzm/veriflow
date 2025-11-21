# veriflow/semantic/matcher.py

from typing import Dict, Any, Tuple, List, Set, Optional
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
        "has_db": False,
        "has_form": False,
        "has_condition": False,
        "has_transform": False,
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
        if any(k in t for k in ("airtable", "notion", "sheet", "spreadsheet", "database", "db")):
            summary["has_db"] = True
        if any(k in t for k in ("form", "webhook")):
            summary["has_form"] = True
        if any(k in t for k in ("if", "switch", "router", "branch", "condition")):
            summary["has_condition"] = True
        if any(k in t for k in ("function", "set", "merge", "transform", "map", "validate", "parse", "format")):
            summary["has_transform"] = True
    return summary

def _capability_relevant_label(label: str, intent: Dict[str, bool]) -> bool:
    """
    Check if a node label is directly relevant to at least one requested capability.
    This is your previous 'local' heuristic, factored out for reuse.
    """
    return (
        (intent.get("need_schedule") and any(k in label for k in ("schedule", "cron", "trigger", "webhook")))
        or (intent.get("need_email") and "email" in label)
        or (intent.get("need_http") and ("http" in label or "request" in label))
        or (intent.get("need_slack") and "slack" in label)
        or (intent.get("need_telegram") and "telegram" in label)
        or (intent.get("need_db") and any(k in label for k in ("airtable","notion","sheet","spreadsheet","database","db")))
        or (intent.get("need_form") and any(k in label for k in ("form","webhook")))
        or (intent.get("need_condition") and any(k in label for k in ("if","switch","router","branch","condition")))
        or (intent.get("need_transform") and any(k in label for k in ("function","set","merge","transform","map","validate","parse","format")))
    )

def _collect_semantic_path_nodes(
    G: nx.DiGraph,
    index: Dict[str, dict],
    sources: List[str],
    targets: Set[str],
    max_depth: int = 6,
) -> Tuple[bool, Set[str]]:
    """
    BFS from each source to any target, with semantic constraints:

      - we can traverse through intermediate nodes (IF / Router / Transform / etc.)
      - we stop a branch when we hit an unrelated terminal action
      - when a path to target is found, we record all nodes on that path

    Returns:
        (has_path, path_nodes)
          has_path: True if at least one path exists
          path_nodes: union of nodes lying on at least one valid path
    """
    found_any = False
    path_nodes: Set[str] = set()

    from collections import deque

    for src in sources:
        if src not in G:
            continue

        visited = {src}
        parents: Dict[str, Optional[str]] = {src: None}
        queue = deque([(src, 0)])

        while queue:
            cur, depth = queue.popleft()
            if depth > max_depth:
                continue

            if cur in targets:
                # reconstruct path src -> cur
                found_any = True
                p = cur
                while p is not None:
                    path_nodes.add(p)
                    p = parents.get(p)
                continue

            for succ in G.successors(cur):
                if succ in visited:
                    continue
                visited.add(succ)

                node = index.get(succ, {})
                label = _node_label(node)

                # if succ is an unrelated terminal action, cut this branch
                if succ not in targets and _is_action_node(label):
                    continue

                parents[succ] = cur
                queue.append((succ, depth + 1))

    return found_any, path_nodes

def order_ok_by_path(workflow: Dict[str, Any], intent: Dict[str, bool]) -> float:
    """
    Generic semantic ordering check based on intent.
    Only enforce ordering rules for capabilities explicitly required by intent.
    """
    nodes = workflow.get("nodes", [])
    if not nodes:
        return 1.0

    G = build_dag(workflow)
    index: Dict[str, dict] = {n["id"]: n for n in nodes if "id" in n}

    # Precompute capability node id sets
    cap_nodes = {
        "schedule": _find_nodes(nodes, lambda t: any(k in t for k in ("schedule","cron","trigger","webhook"))),
        "http": _find_nodes(nodes, lambda t: ("http" in t) or ("request" in t)),
        "email": _find_nodes(nodes, lambda t: "email" in t),
        "slack": _find_nodes(nodes, lambda t: "slack" in t),
        "telegram": _find_nodes(nodes, lambda t: "telegram" in t),
        "db": _find_nodes(nodes, lambda t: any(k in t for k in ("airtable","notion","sheet","spreadsheet","database","db"))),
        "form": _find_nodes(nodes, lambda t: any(k in t for k in ("form","webhook"))),
        "condition": _find_nodes(nodes, lambda t: any(k in t for k in ("if","switch","router","branch","condition"))),
        "transform": _find_nodes(nodes, lambda t: any(k in t for k in ("function","set","merge","transform","map","validate","parse","format"))),
    }

    # ORDER_RULES: (source_cap, target_cap) both must exist to require an ordering path
    ORDER_RULES = [
        ("schedule", "http"),
        ("schedule", "email"),
        ("schedule", "slack"),
        ("form", "transform"),
        ("transform", "db"),
        ("http", "db"),
        # ("db", "email"),
        ("condition", "slack"),  # slack typically after condition when needed
    ]

    def _cap_required(cap: str) -> bool:
        return intent.get(f"need_{cap}", False)

    required = []
    for src_cap, tgt_cap in ORDER_RULES:
        if _cap_required(src_cap) and _cap_required(tgt_cap):
            if cap_nodes[src_cap] and cap_nodes[tgt_cap]:
                required.append((cap_nodes[src_cap], set(cap_nodes[tgt_cap])))

    if not required:
        return 1.0

    ok = True
    for sources, targets in required:
        has_path, _ = _collect_semantic_path_nodes(G, index, sources, targets)
        ok = ok and has_path

    return 1.0 if ok else 0.0

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
    nodes = workflow.get("nodes", []) or []
    nodes_summary = inspect_nodes(nodes)

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
    action_keys = {"need_email", "need_http", "need_slack", "need_telegram"}
    denom = sum(1 for k in desired if k in action_keys)
    if denom <= 0:
        action_ok = 1.0
    else:
        action_ok = matched / denom

    # ---- Semantic coverage: relevant node set ----
    # 5) Ordering sub-score (generic ordering rules)
    order_ok = order_ok_by_path(workflow, intent)

    # Build graph + index once for relevance + path collection
    G = build_dag(workflow) if nodes else nx.DiGraph()
    index: Dict[str, dict] = {n["id"]: n for n in nodes if "id" in n}

    # Node groups by capability
    schedule_ids = _find_nodes(nodes, lambda t: any(k in t for k in ("schedule", "cron", "trigger", "webhook")))
    http_ids = _find_nodes(nodes, lambda t: ("http" in t) or ("request" in t))
    email_ids = _find_nodes(nodes, lambda t: "email" in t)
    slack_ids = _find_nodes(nodes, lambda t: "slack" in t)
    telegram_ids = _find_nodes(nodes, lambda t: "telegram" in t)
    db_ids = _find_nodes(nodes, lambda t: any(k in t for k in ("airtable","notion","sheet","spreadsheet","database","db")))
    form_ids = _find_nodes(nodes, lambda t: any(k in t for k in ("form","webhook")))
    transform_ids = _find_nodes(nodes, lambda t: any(k in t for k in ("function","set","merge","transform","map","validate","parse","format")))
    condition_ids = _find_nodes(nodes, lambda t: any(k in t for k in ("if","switch","router","branch","condition")))
 
    # 6) Compute a semantic relevant node set:
    #    - direct capability match
    #    - plus any node on a semantic path between key capabilities
    relevant_nodes: Set[str] = set()

    # 6.1 Direct capability relevance (local)
    for n in nodes:
        node_id = n.get("id")
        if node_id is None:
            continue
        label = _node_label(n)
        if _capability_relevant_label(label, intent):
            relevant_nodes.add(node_id)

    # 6.2 Semantic paths relevance (multi-hop)
    # Helper for adding paths:
    def _add_paths(sources: List[str], targets: List[str]) -> bool:
        if not sources or not targets or not G:
            return False
        has_path, path_nodes = _collect_semantic_path_nodes(G, index, sources, set(targets))
        if has_path:
            relevant_nodes.update(path_nodes)
        return has_path

    def _add_paths_optional(sources: List[str], targets: List[str]) -> None:
        """
        Optional (weak) path relevance:
        - we do NOT require a path to exist;
        - if a path exists, we add its nodes into relevant_nodes.
        """
        if not sources or not targets or not G:
            return
        _, path_nodes = _collect_semantic_path_nodes(G, index, sources, set(targets))
        relevant_nodes.update(path_nodes)

    # schedule -> http
    if intent.get("need_schedule") and intent.get("need_http"):
        _add_paths(schedule_ids, http_ids)

    # schedule -> email/slack/telegram
    if intent.get("need_schedule"):
        if intent.get("need_email"):
            _add_paths(schedule_ids, email_ids)
        if intent.get("need_slack"):
            _add_paths(schedule_ids, slack_ids)
        if intent.get("need_telegram"):
            _add_paths(schedule_ids, telegram_ids)

    # http -> email/slack/telegram
    if intent.get("need_http"):
        if intent.get("need_email"):
            _add_paths(http_ids, email_ids)
        if intent.get("need_slack"):
            _add_paths(http_ids, slack_ids)
        if intent.get("need_telegram"):
            _add_paths(http_ids, telegram_ids)
        if intent.get("need_db"):
            _add_paths_optional(http_ids, db_ids)

    # form -> transform -> db -> notify
    if intent.get("need_form"):
        if intent.get("need_transform"):
            _add_paths(form_ids, transform_ids)
        if intent.get("need_db"):
            _add_paths_optional(form_ids, db_ids)

    if intent.get("need_transform") and intent.get("need_db"):
        _add_paths_optional(transform_ids, db_ids)

    if intent.get("need_condition"):
        if intent.get("need_slack"):
            _add_paths(condition_ids, slack_ids)
        if intent.get("need_email"):
            _add_paths(condition_ids, email_ids)

    # --- DB paths: OPTIONAL weak constraints (only add if path exists) ---
    if intent.get("need_db"):
        if intent.get("need_email"):
            _add_paths_optional(db_ids, email_ids)
        if intent.get("need_slack"):
            _add_paths_optional(db_ids, slack_ids)
    

    # 7) Aggregate
    M = round((trigger_ok + action_ok + order_ok) / 3, 2)

    # 8) Issues
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
    if intent.get("need_db") and not nodes_summary["has_db"]:
        issues.append("Missing DB/storage node for intent")
    if intent.get("need_form") and not nodes_summary["has_form"]:
        issues.append("Missing Form/Webhook node for intent")
    if intent.get("need_condition") and not nodes_summary["has_condition"]:
        issues.append("Missing Condition/Branching node for intent")
    if intent.get("need_transform") and not nodes_summary["has_transform"]:
        issues.append("Missing Transform/Validation node for intent")

    # 9) Irrelevant nodes: nodes that are not on any capability or semantic path
    irrelevant_nodes: List[str] = []
    for n in nodes:
        node_id = n.get("id")
        if node_id is None:
            continue
        if node_id not in relevant_nodes:
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