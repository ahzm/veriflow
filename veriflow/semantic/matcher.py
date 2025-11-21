# veriflow/semantic/matcher.py

from typing import Dict, Any, Tuple, List, Set, Optional
import networkx as nx
import json
from veriflow.utils.graph import build_dag, build_scc_dag
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
        #"http request",
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

def flatten_keys(obj: Any, prefix: str = "") -> Set[str]:
    """
    Recursively collect JSON keys from dict/list structures.
    Returned keys are lowercased full paths like:
        "nodes.parameters.threshold.value"
    """
    keys: Set[str] = set()

    if isinstance(obj, dict):
        for k, v in obj.items():
            k2 = str(k).lower()
            path = f"{prefix}.{k2}" if prefix else k2
            keys.add(path)
            keys.update(flatten_keys(v, path))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(flatten_keys(item, prefix))

    return keys


def normalize_field_name(f: str) -> str:
    """
    Normalize intent field name to improve matching:
    - lower
    - remove spaces/hyphens
    """
    return f.strip().lower().replace(" ", "").replace("-", "_")

# Field synonyms for param-sem matching
# These are static small sets to reduce false negatives in real n8n workflows.
FIELD_SYNONYMS: Dict[str, Set[str]] = {
    # Generic identifiers
    "id": {"id", "uuid", "uid", "identifier", "key"},
    "user_id": {"user_id", "userid", "user", "userkey", "account_id", "accountid"},
    "customer_id": {"customer_id", "customerid", "client_id", "clientid", "buyer_id", "buyerid"},

    # Time / schedule
    "time": {"time", "timestamp", "ts", "datetime", "date", "created_at", "updated_at"},
    "cron": {"cron", "schedule", "interval", "every", "repeat", "timer"},

    # HTTP / API
    "url": {"url", "uri", "endpoint", "path", "link"},
    "method": {"method", "http_method", "verb", "request_method"},
    "headers": {"headers", "header", "http_headers"},
    "query": {"query", "querystring", "qs", "params", "query_params"},
    "body": {"body", "payload", "data", "json", "request_body"},
    "status": {"status", "code", "status_code", "http_status"},

    # Email / notification
    "email": {"email", "mail", "address", "recipient", "to", "cc", "bcc"},
    "subject": {"subject", "title", "headline"},
    "message": {"message", "text", "content", "body", "msg"},
    "channel": {"channel", "room", "chat", "slack_channel", "telegram_chat"},

    # DB / storage
    "db": {"db", "database", "table", "collection", "sheet", "spreadsheet"},
    "record": {"record", "row", "item", "entry", "document"},
    "field": {"field", "column", "property", "attribute", "key"},

    # Conditions / thresholds
    "threshold": {"threshold", "limit", "bound", "min", "max", "target"},
    "operator": {"operator", "op", "cmp", "compare", "comparison", "rule"},
    "value": {"value", "val", "amount", "number", "count", "qty"},

    # Transform / mapping
    "map": {"map", "mapping", "transform", "convert", "parse", "format", "derive"},
}

def field_aliases(f: str) -> Set[str]:
    """
    Generate aliases for an intent field, including:
      - normalized form (snake-ish)
      - no-underscore form
      - camelCase lowered
      - synonyms expansion (static)
    """
    aliases: Set[str] = set()

    f0 = normalize_field_name(f)  # e.g., "customer_id"
    aliases.add(f0)

    # Remove underscores: customer_id -> customerid
    no_us = f0.replace("_", "")
    aliases.add(no_us)

    # Camel case: customer_id -> customerId -> customerid (lowered)
    parts = f0.split("_")
    if parts:
        camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
        aliases.add(camel.lower())

    # Add static synonyms if known
    if f0 in FIELD_SYNONYMS:
        aliases.update(FIELD_SYNONYMS[f0])

    # Heuristic: if field ends with _id, also add generic id synonyms
    if f0.endswith("_id") and "id" in FIELD_SYNONYMS:
        aliases.update(FIELD_SYNONYMS["id"])

    return {a for a in aliases if a}


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

def _collect_semantic_path_nodes_scc(
    G0, Gc, comp_of, nodes_of_comp,
    index,
    sources, targets,
    max_depth=6
):
    """
    SCC-level BFS:
        - BFS in compressed DAG (Gc)
        - Expand components back to original nodes
    """
    from collections import deque

    source_comps = [comp_of[s] for s in sources if s in comp_of]
    target_comps = {comp_of[t] for t in targets if t in comp_of}

    if not source_comps or not target_comps:
        return False, set()

    found = False
    final_nodes = set()

    for sc in source_comps:
        visited = {sc}
        parents = {sc: None}
        q = deque([(sc, 0)])

        while q:
            cur, depth = q.popleft()
            if depth > max_depth:
                continue

            if cur in target_comps:
                found = True
                # Expand component-path into node ids
                pc = cur
                while pc is not None:
                    final_nodes.update(nodes_of_comp[pc])
                    pc = parents.get(pc)
                continue

            for nxt in Gc.successors(cur):
                if nxt in visited:
                    continue
                # Cut path if nxt contains unrelated action node
                if nxt not in target_comps:
                    cut = False
                    for nid in nodes_of_comp[nxt]:
                        label = _node_label(index.get(nid, {}))
                        if _is_action_node(label):
                            cut = True; break
                    if cut:
                        continue

                visited.add(nxt)
                parents[nxt] = cur
                q.append((nxt, depth + 1))

    return found, final_nodes

def order_ok_by_path(workflow: Dict[str, Any], intent: Dict[str, bool]) -> float:
    """
    Generic semantic ordering check based on intent.
    Only enforce ordering rules for capabilities explicitly required by intent.
    """
    nodes = workflow.get("nodes", [])
    if not nodes:
        return 1.0

    Gc, G0, comp_of, nodes_of_comp = build_scc_dag(workflow)
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
    _, edges_intent = build_intent_graph(intent)
    ORDER_RULES = [(a, b) for (a, b) in edges_intent if b != "action"]

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
        has_path, _ = _collect_semantic_path_nodes_scc(
                        G0, Gc, comp_of, nodes_of_comp,
                        index,
                        sources,
                        set(targets),
                        max_depth=6
                    )
        ok = ok and has_path

    return 1.0 if ok else 0.0

def _desired_keys(intent: Dict[str, bool]) -> List[str]:
    """Return list of desired capability keys (those starting with 'need_')."""
    return [k for k, v in intent.items() if v and k.startswith("need_")]

# --- build explicit intent graph ---
def build_intent_graph(intent: Dict[str, bool]) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """
    Build a simple capability graph from intent flags.
    Nodes: capability names
    Edges: semantic dependencies that should hold in workflow
    """
    caps: Set[str] = set()
    edges: List[Tuple[str, str]] = []

    # capability nodes
    if intent.get("need_schedule"):  caps.add("schedule")
    if intent.get("need_form"):      caps.add("form")
    if intent.get("need_http"):      caps.add("http")
    if intent.get("need_transform"): caps.add("transform")
    if intent.get("need_condition"): caps.add("condition")
    if intent.get("need_email"):     caps.add("email")
    if intent.get("need_slack"):     caps.add("slack")
    if intent.get("need_telegram"):  caps.add("telegram")
    if intent.get("need_db"):        caps.add("db")

    # edges == ORDER_RULES + conditional rule
    ORDER_RULES = [
        ("schedule", "http"),
        ("schedule", "email"),
        ("schedule", "slack"),
        ("schedule", "telegram"),
        ("form", "transform"),
        ("transform", "db"),
        ("http", "db"),
        ("condition", "email"),
        ("condition", "slack"),
        ("condition", "telegram"),
    ]

    for a, b in ORDER_RULES:
        if a in caps and b in caps:
            edges.append((a, b))

    # conditional notify means: condition -> action (explicit or implicit)
    if intent.get("need_conditional_notify") and "condition" in caps:
        edges.append(("condition", "action"))

    return caps, edges

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
    params_req = intent_res.meta.get("params", {}) or {}

    issues: List[str] = []
    evidence: List[Dict[str, Any]] = []

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

    # 4.5) Support capability sub-score (non-action capabilities only)
    support_caps = ["form", "transform", "condition", "db"]
    required_support = [c for c in support_caps if intent.get(f"need_{c}", False)]

    if not required_support:
        support_ok = 1.0
    else:
        matched_support = 0
        if intent.get("need_form") and nodes_summary["has_form"]:
            matched_support += 1
        if intent.get("need_transform") and nodes_summary["has_transform"]:
            matched_support += 1
        if intent.get("need_condition") and nodes_summary["has_condition"]:
            matched_support += 1
        if intent.get("need_db") and nodes_summary["has_db"]:
            matched_support += 1

        if intent.get("need_transform") and not nodes_summary["has_transform"]:
            support_ok = 0.0
            evidence.append({
                "rule": "support_transform_required",
                "from": "intent",
                "to": "transform",
                "applicable": True,
                "ok": False,
                "sources": [],
                "targets": [],
                "path": [],
                "note": "transform requested but missing; support score forced to 0"
            })
        else:
            support_ok = matched_support / len(required_support)

    # ---- Semantic coverage: relevant node set ----
    # 5) Ordering sub-score (generic ordering rules)
    order_ok = order_ok_by_path(workflow, intent)

    # Build graph + index once for relevance + path collection
    #G = build_dag(workflow) if nodes else nx.DiGraph()
    Gc, G0, comp_of, nodes_of_comp = build_scc_dag(workflow) if nodes else (nx.DiGraph(), nx.DiGraph(), {}, {})
    G = G0
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
    
    # collect requested action targets
    action_targets: List[str] = []
    if intent.get("need_email"):
        action_targets += email_ids
    if intent.get("need_slack"):
        action_targets += slack_ids
    if intent.get("need_telegram"):
        action_targets += telegram_ids

    workflow_action_targets: List[str] = email_ids + slack_ids + telegram_ids

    # --- intent graph -> workflow alignment report ---
    caps, edges = build_intent_graph(intent)

    relevant_nodes: Set[str] = set()

    cap_nodes_map = {
        "schedule": schedule_ids,
        "form": form_ids,
        "http": http_ids,
        "transform": transform_ids,
        "condition": condition_ids,
        "email": email_ids,
        "slack": slack_ids,
        "telegram": telegram_ids,
        "db": db_ids,
        # "action" is special (implicit), handled below
    }

    capability_matches: Dict[str, bool] = {}
    for c in caps:
        if c == "action":
            continue
        capability_matches[c] = bool(cap_nodes_map.get(c))

    edge_matches: List[Dict[str, Any]] = []
    for a, b in edges:
        if b == "action":
            # action can be any terminal notify node
            targets = set(email_ids + slack_ids + telegram_ids)
        else:
            targets = set(cap_nodes_map.get(b, []))

        sources = cap_nodes_map.get(a, [])
        ok_edge = False
        path_nodes: Set[str] = set()
        if sources and targets:
            ok_edge, path_nodes = _collect_semantic_path_nodes_scc(
                                    G0, Gc, comp_of, nodes_of_comp,
                                    index,
                                    sources,
                                    targets,
                                    max_depth=6
                                )

        if ok_edge:
            relevant_nodes.update(path_nodes)

        edge_matches.append({
            "from": a,
            "to": b,
            "ok": bool(ok_edge),
            "path_nodes": sorted(path_nodes),
        })

        evidence.append({
            "rule": "intent_edge",
            "from": a,
            "to": b,
            "applicable": bool(sources and targets),
            "ok": bool(ok_edge) if (sources and targets) else None,
            "sources": list(sources),
            "targets": sorted(list(targets)),
            "path": sorted(path_nodes),
            "note": "intent edge satisfied" if ok_edge else "intent edge missing"
        })

    # 6) Compute a semantic relevant node set:
    #    - direct capability match
    #    - plus any node on a semantic path between key capabilities

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
    def _split_rule(rule_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse names like 'schedule->http(required)'.
        Returns (from_cap, to_cap) if parseable.
        """
        if "->" not in rule_name:
            return None, None
        left, right = rule_name.split("->", 1)
        right = right.split("(", 1)[0]
        return left.strip(), right.strip()

    def _add_paths(sources: List[str], targets: List[str], rule_name: str) -> bool:
        from_cap, to_cap = _split_rule(rule_name)

        # Not applicable
        if not sources or not targets or Gc.number_of_nodes() == 0:
            evidence.append({
                "rule": rule_name,
                "from": from_cap,
                "to": to_cap,
                "applicable": False,
                "ok": None,
                "sources": list(sources),
                "targets": list(targets),
                "path": [],
                "note": "not applicable (sources/targets empty or graph missing)"
            })
            return False

        has_path, path_nodes = _collect_semantic_path_nodes_scc(
                                G0, Gc, comp_of, nodes_of_comp,
                                index,
                                sources,
                                set(targets),
                                max_depth=6
                            )
        if has_path:
            relevant_nodes.update(path_nodes)

        evidence.append({
            "rule": rule_name,
            "from": from_cap,
            "to": to_cap,
            "applicable": True,
            "ok": bool(has_path),
            "sources": list(sources),
            "targets": list(targets),
            "path": sorted(path_nodes),
            "note": "required semantic path found" if has_path else "required semantic path missing"
        })
        return has_path

    def _add_paths_optional(sources: List[str], targets: List[str], rule_name: str) -> None:
        from_cap, to_cap = _split_rule(rule_name)

        if not sources or not targets or Gc.number_of_nodes() == 0:
            evidence.append({
                "rule": rule_name,
                "from": from_cap,
                "to": to_cap,
                "applicable": False,
                "ok": None,
                "sources": list(sources),
                "targets": list(targets),
                "path": [],
                "note": "not applicable (optional)"
            })
            return

        has_path, path_nodes = _collect_semantic_path_nodes_scc(
                                G0, Gc, comp_of, nodes_of_comp,
                                index,
                                sources,
                                set(targets),
                                max_depth=6
                            )
        if has_path:
            relevant_nodes.update(path_nodes)

        evidence.append({
            "rule": rule_name,
            "from": from_cap,
            "to": to_cap,
            "applicable": True,
            "ok": bool(has_path),
            "sources": list(sources),
            "targets": list(targets),
            "path": sorted(path_nodes),
            "note": "optional semantic path found" if has_path else "optional semantic path not found"
        })

    # schedule -> http
    if intent.get("need_schedule") and intent.get("need_http"):
        _add_paths(schedule_ids, http_ids, "schedule->http(required)")

    # schedule -> email/slack/telegram
    if intent.get("need_schedule"):
        if intent.get("need_email"):
            _add_paths(schedule_ids, email_ids, "schedule->email(required)")
        if intent.get("need_slack"):
            _add_paths(schedule_ids, slack_ids, "schedule->slack(required)")
        if intent.get("need_telegram"):
            _add_paths(schedule_ids, telegram_ids, "schedule->telegram(required)")

    # http -> email/slack/telegram
    if intent.get("need_http"):
        if intent.get("need_email"):
            _add_paths(http_ids, email_ids, "http->email(required)")
        if intent.get("need_slack"):
            _add_paths(http_ids, slack_ids, "http->slack(required)")
        if intent.get("need_telegram"):
            _add_paths(http_ids, telegram_ids, "http->telegram(required)")
        if intent.get("need_db"):
            _add_paths_optional(http_ids, db_ids, "http->db(optional)")

    # form -> transform -> db -> notify
    if intent.get("need_form"):
        if intent.get("need_transform"):
            _add_paths(form_ids, transform_ids, "form->transform(required)")
        if intent.get("need_db"):
            _add_paths_optional(form_ids, db_ids, "form->db(optional)")

    if intent.get("need_transform") and intent.get("need_db"):
        _add_paths_optional(transform_ids, db_ids, "transform->db(optional)")
    
    if intent.get("need_condition"):
        if intent.get("need_slack"):
            _add_paths(condition_ids, slack_ids, "condition->slack(required)")
        if intent.get("need_email"):
            _add_paths(condition_ids, email_ids, "condition->email(required)")
    
    # If the prompt says “notify if ...” but does NOT specify email/slack/tg,
    # we still consider Condition -> (any workflow action) as relevant.
    if intent.get("need_conditional_notify") and not action_targets:
        _add_paths_optional(condition_ids, workflow_action_targets, "condition->action(optional)")

    # --- DB paths: OPTIONAL weak constraints (only add if path exists) ---
    if intent.get("need_db"):
        if intent.get("need_email"):
            _add_paths_optional(db_ids, email_ids, "db->email(optional)")
        if intent.get("need_slack"):
            _add_paths_optional(db_ids, slack_ids, "db->slack(optional)")
        if intent.get("need_telegram"):
            _add_paths_optional(db_ids, telegram_ids, "db->telegram(optional)")
    
    # --- Conditional conflict (conditional action) ---
    conditional_ok = 1.0

    def _normalize_op(op: str) -> List[str]:
        op = op.strip()
        mapping = {
            ">": [">", "gt", "greater", "greaterthan", "min_exclusive"],
            "<": ["<", "lt", "less", "lessthan", "max_exclusive"],
            ">=": [">=", "gte", "ge", "at_least", "min_inclusive"],
            "<=": ["<=", "lte", "le", "at_most", "max_inclusive"],
            "超过": [">", "gt", "greater", "greaterthan"],
            "大于": [">", "gt", "greater", "greaterthan"],
            "小于": ["<", "lt", "less", "lessthan"],
            "低于": ["<", "lt", "less", "lessthan"],
        }
        return mapping.get(op, [op.lower()])

    if intent.get("need_conditional_notify"):
        # Intent requires Condition -> Action path
        if condition_ids and not action_targets:
            conditional_ok = 1.0
        elif action_targets and condition_ids:
            has_path, _ = _collect_semantic_path_nodes_scc(
                G0, Gc, comp_of, nodes_of_comp,
                index,
                condition_ids,
                set(action_targets),
                max_depth=6
            )
            conditional_ok = 1.0 if has_path else 0.0
        else:
            conditional_ok = 0.0
    else:
        # Intent does NOT require conditional notify,
        # but workflow has Condition -> Action paths => weak penalty
        if condition_ids and workflow_action_targets:
            hp, pn = _collect_semantic_path_nodes_scc(
                G0, Gc, comp_of, nodes_of_comp,
                index,
                condition_ids,
                set(workflow_action_targets),
                max_depth=6
            )
            if hp:
                conditional_ok = 0.35
                evidence.append({
                    "rule": "extra_conditional_notify",
                    "from": "condition",
                    "to": "action",
                    "applicable": True,
                    "ok": False,
                    "sources": list(condition_ids),
                    "targets": list(workflow_action_targets),
                    "path": sorted(pn),
                    "note": "workflow has conditional notify but intent does not require it"
                })
            else:
                evidence.append({
                    "rule": "extra_conditional_notify",
                    "from": "condition",
                    "to": "action",
                    "applicable": True,
                    "ok": True,
                    "sources": list(condition_ids),
                    "targets": list(workflow_action_targets),
                    "path": [],
                    "note": "no extra conditional notify path"
                })

    # --- Param-sem check (fields/threshold align) ---
    param_sem_ok = 1.0

    # Threshold alignment: check condition node parameters
    if "threshold_value" in params_req:
        wanted = float(params_req["threshold_value"])
        wanted_op = params_req.get("threshold_op")
        found_value_match = False
        found_op_match = False

        op_aliases = _normalize_op(wanted_op) if wanted_op else []

        for cid in condition_ids:
            node = index.get(cid, {})
            p = node.get("parameters", {}) or {}
            blob_p = json.dumps(p, ensure_ascii=False).lower()
            flat_p_keys = flatten_keys(p)

            for key, v in p.items():
                # check value
                try:
                    val = float(v)
                    if abs(val - wanted) <= 1e-3:
                        found_value_match = True
                except Exception:
                    pass

                if wanted_op and (
                    any(a in blob_p for a in op_aliases) or
                    any(any(a in k for a in op_aliases) for k in flat_p_keys)
                ):
                    found_op_match = True

            if found_value_match and (not wanted_op or found_op_match):
                break

        if not found_value_match:
            param_sem_ok = 0.0
            issues.append(f"Threshold value {wanted} not found in Condition node parameters")
        
        elif wanted_op and not found_op_match:
            param_sem_ok = min(param_sem_ok, 0.5)
            issues.append(f"Threshold operator '{wanted_op}' not aligned with workflow parameters")
    
    # Field alignment (structured): match requested fields against flattened workflow keys/values
    if params_req.get("fields"):
        fields = params_req["fields"]

        flat_keys = flatten_keys(workflow)  # all key paths
        flat_blob = json.dumps(workflow, ensure_ascii=False).lower()  # fallback for value match

        missing_fields = []
        for f in fields:
            aliases = field_aliases(f)

            # key-path match: alias appears in some key path
            key_hit = any(any(a in k for a in aliases) for k in flat_keys)

            # fallback: old value-side substring match
            val_hit = any(a in flat_blob for a in aliases)

            if not (key_hit or val_hit):
                missing_fields.append(f)

        if missing_fields:
            param_sem_ok = 0.0 # min(param_sem_ok, 0.5)
            #ratio = (len(fields) - len(missing_fields)) / max(1, len(fields))
            #param_sem_ok = min(param_sem_ok, 0.5 * ratio)
            issues.append(f"Some requested fields are not reflected in workflow parameters: {missing_fields}")

    # intent-graph edge score (denominator = only edges where both capabilities exist)
    valid_edges = [e for e in edge_matches if e["path_nodes"] or e["ok"] or (
        cap_nodes_map.get(e["from"], []) and 
        (cap_nodes_map.get(e["to"], []) if e["to"] != "action" else workflow_action_targets)
    )]

    if valid_edges:
        intent_edge_ok = sum(e["ok"] for e in valid_edges) / len(valid_edges)
    else:
        # If no edges were actually applicable, treat it as perfect score
        intent_edge_ok = 1.0
    
    # 7) Aggregate
    M = round((trigger_ok + action_ok + support_ok + order_ok + conditional_ok + param_sem_ok + 0.5*intent_edge_ok) / 6.5, 2)

    # 8) Issues
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
    if intent.get("need_conditional_notify") and conditional_ok < 1.0:
        issues.append("Intent requires conditional notification, but no Condition->Action path found")
    if (not intent.get("need_conditional_notify")) and conditional_ok < 1.0:
        issues.append("Workflow contains conditional notification but intent does not require it")

    # 9) Irrelevant nodes: nodes that are not on any capability or semantic path
    irrelevant_nodes: List[str] = []
    for n in nodes:
        node_id = n.get("id")
        if node_id is None:
            continue
        if node_id not in relevant_nodes:
            irrelevant_nodes.append(node_id)

    missing_caps = [c for c, ok in capability_matches.items() if not ok]
    if intent.get("need_conditional_notify") and not workflow_action_targets:
        missing_caps.append("action")

    # alignment JSON for visualization
    workflow_edges = []
    try:
        for u, v in G.edges():
            workflow_edges.append({"source": u, "target": v})
    except Exception:
        workflow_edges = []

    # label + capability tagging
    def _guess_cap_for_node(label: str) -> Optional[str]:
        if any(k in label for k in ("schedule","cron","trigger","webhook")): return "schedule"
        if "http" in label or "request" in label: return "http"
        if "email" in label: return "email"
        if "slack" in label: return "slack"
        if "telegram" in label: return "telegram"
        if any(k in label for k in ("airtable","notion","sheet","spreadsheet","database","db")): return "db"
        if any(k in label for k in ("form","webhook")): return "form"
        if any(k in label for k in ("if","switch","router","branch","condition")): return "condition"
        if any(k in label for k in ("function","set","merge","transform","map","validate","parse","format")): return "transform"
        return None

    workflow_nodes = []
    for n in nodes:
        nid = n.get("id")
        if nid is None:
            continue
        label = n.get("name") or n.get("type") or nid
        cap = _guess_cap_for_node(_node_label(n))
        workflow_nodes.append({
            "id": nid,
            "label": label,
            "capability": cap,
            "relevant": nid in relevant_nodes
        })

    intent_edges_json = []
    for em in edge_matches:
        intent_edges_json.append({
            "from": em["from"],
            "to": em["to"],
            "ok": em["ok"],
            "path": em["path_nodes"]
        })

    alignment_graph = {
        "intent_caps": sorted(list(caps)),
        "intent_edges": intent_edges_json,
        "workflow_nodes": workflow_nodes,
        "workflow_edges": workflow_edges,
        "relevant_nodes": sorted(list(relevant_nodes)),
        "irrelevant_nodes": irrelevant_nodes,
        "missing_capabilities": missing_caps,
    }

    detail = {
        "trigger": float(trigger_ok),
        "action": float(action_ok),
        "support": float(support_ok),
        "order": float(order_ok),
        "conditional": float(conditional_ok),
        "param_sem": float(param_sem_ok),
        "intent_edge": float(intent_edge_ok),
        "intent_edge_note": "ratio of satisfied intent-graph edges",
        "params_req": params_req,
        "intent_conf": float(intent_res.overall_confidence),
        # storing as float-compatible value is convenient when exporting
        "source": intent_res.source,            # Keep the source string separately (not a float)
        "intent": intent,
        "intent_chain": intent_res.meta.get("intent_chain", []),
        "irrelevant_nodes": irrelevant_nodes,
        "intent_caps": sorted(caps),
        "intent_edges": edges,
        "missing_capabilities": missing_caps,
        "alignment": {
            "capability_matches": capability_matches,
            "edge_matches": edge_matches,
        },
        "score_weights": {
            "trigger": 1.0,
            "action": 1.0,
            "support": 1.0,
            "order": 1.0,
            "conditional": 1.0,
            "param_sem": 1.0,
            "intent_edge": 0.5
        }, 
        "evidence": evidence,
        "alignment_graph": alignment_graph,                  
    }

    return M, issues, detail