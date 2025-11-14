# veriflow/executable/dataflow.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Set
import re

import networkx as nx

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_dataflow(
    workflow: Dict[str, Any],
    G: nx.DiGraph,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Lightweight data–flow consistency check.

    Idea:
      - Extract JSON fields referenced via `{{$json[...]}}` templates
        from node parameters;
      - Approximate which fields are produced by triggers / HTTP nodes;
      - Propagate available fields along the DAG;
      - Report nodes that reference fields that are never available
        from any upstream producer.

    Returns (ok, issues, detail) where:
      - ok: True iff no missing-field issue is detected;
      - issues: human-readable messages;
      - detail: structured diagnostics (per-node info).
    """
    nodes = workflow.get("nodes", [])
    # Map both id and name to node definitions for convenience
    index = _build_node_index(nodes)

    graph_nodes = {str(n) for n in G.nodes}

    # 1) Extract required fields from templates
    required: Dict[str, Set[str]] = {}
    for key, node in index.items():
        if str(key) not in graph_nodes:
            continue
        fields = _extract_json_fields(node)
        if fields:
            required[key] = fields

    # 2) Infer produced fields per node (local)
    produced_local: Dict[str, Set[str]] = {}
    for key, node in index.items():
        if str(key) not in graph_nodes:
            continue
        produced_local[key] = _infer_produced_fields(node)

    # 3) Propagate available fields along topological order
    available: Dict[str, Set[str]] = {}
    try:
        order = list(nx.topological_sort(G))
    except Exception:
        # Fall back to arbitrary order if graph is not a DAG
        order = list(G.nodes)

    for n in order:
        # Fields produced locally by this node
        local = set()
        node_def = index.get(n)
        if node_def is not None:
            local = produced_local.get(n, set())
        # Fields coming from predecessors (union)
        pred_fields: Set[str] = set()
        for p in G.predecessors(n):
            pred_fields |= available.get(p, set())
        available[n] = pred_fields | local

    # 4) Check missing fields
    issues: List[str] = []
    per_node_missing: Dict[str, List[str]] = {}
    for n, needed in required.items():
        have = available.get(n, set())
        missing = sorted(f for f in needed if f not in have)
        if missing:
            per_node_missing[n] = missing
            node_name = _node_label(index.get(n), default=str(n))
            issues.append(
                f"Data-flow issue: node '{node_name}' references "
                f"$json fields {missing} that are not provided by upstream nodes"
            )

    ok = len(issues) == 0
    applicable = any(required.values())
    detail = {
        "applicable": applicable,
        "required_fields": {k: sorted(v) for k, v in required.items()},
        "available_fields": {k: sorted(v) for k, v in available.items()},
        "missing_fields": per_node_missing,
    }
    return ok, issues, detail


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_node_index(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build an index mapping both node id and name to the node definition.
    This allows us to handle graphs where edges use ids or names.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        nid = n.get("id")
        nname = n.get("name")
        if nid is not None:
            index[str(nid)] = n
        if nname:
            index[str(nname)] = n
    return index


# Match patterns like:
#   {{$json["foo"]}}
#   {{$json['foo']}}
#   {{$json.foo}}
_JSON_FIELD_RE = re.compile(
    r"\{\{\s*\$json\s*(?:\[\s*['\"]([^'\"]+)['\"]\s*\]|\.([a-zA-Z_][a-zA-Z0-9_]*))\s*\}\}"
)

def _extract_json_fields(node: Dict[str, Any]) -> Set[str]:
    """
    Extract JSON field names referenced via `{{$json[...]}}` templates
    from all parameter string values.
    """
    fields: Set[str] = set()

    def _scan_value(v: Any):
        if isinstance(v, str):
            for m in _JSON_FIELD_RE.finditer(v):
                f1, f2 = m.groups()
                fname = f1 or f2
                if fname:
                    fields.add(fname)
        elif isinstance(v, dict):
            for vv in v.values():
                _scan_value(vv)
        elif isinstance(v, list):
            for vv in v:
                _scan_value(vv)

    params = node.get("parameters") or {}
    _scan_value(params)
    return fields


def _infer_produced_fields(node: Dict[str, Any]) -> Set[str]:
    """
    Very small model of which JSON fields a node may produce.

    This is intentionally conservative but still useful:
      - Triggers produce 'timestamp';
      - HTTP requests produce 'statusCode', 'headers', 'body';
      - Other nodes are treated as pass-through (no new fields).
    """
    ntype = (node.get("type") or "").lower()
    fields: Set[str] = set()

    if any(t in ntype for t in ("cron", "trigger", "schedule", "webhook")):
        fields |= {"timestamp"}
    if "httprequest" in ntype or ("http" in ntype and "request" in ntype):
        fields |= {"statusCode", "headers", "body"}

    # For now, Slack/Telegram/email etc. are treated as not introducing
    # specific structured fields that will be referenced via $json[…];
    # they simply forward whatever was present in the incoming JSON.

    return fields


def _node_label(node: Dict[str, Any] | None, default: str) -> str:
    if not node:
        return default
    return str(node.get("name") or node.get("id") or default)