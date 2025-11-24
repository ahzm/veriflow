# veriflow/structural/metrics.py

import networkx as nx
from typing import Callable, Set, List, Dict, Any, Tuple, Optional
from collections import deque

def _extract_edges(workflow: Dict[str, Any]) -> List[tuple]:
    """
    Extract directed edges (src_name, dst_name) from an n8n-style 'connections' dict.
    Supports shapes like:
      connections[src]["main"] = [
         [ {"node": "B", "type": "main", "index": 0}, {"node": "C", ...} ],   # path with multiple targets
         [ {"node": "D", "type": "main", "index": 0} ]                         # single-hop path
      ]
    """
    edges: List[tuple] = []
    connections = workflow.get("connections") or {}
    for src_name, conns in connections.items():
        if not isinstance(conns, dict):
            continue
        for _stream, paths in conns.items():  # e.g., "main": [ [ {...}, ... ], [ {...} ] ]
            if not isinstance(paths, list):
                continue
            for path in paths:
                # Each 'path' should be a list of dict targets; be defensive
                if isinstance(path, dict):
                    # Rare shape: some tools directly put a dict instead of a list
                    dst = path.get("node")
                    if dst:
                        edges.append((src_name, dst))
                    continue
                if not isinstance(path, list):
                    continue
                for hop in path:
                    if isinstance(hop, dict):
                        dst = hop.get("node")
                        if dst:
                            edges.append((src_name, dst))
    # de-duplicate edges to avoid inflating edge counts
    edges = list(set(edges))
    return edges

def _extract_trigger_names(nodes: List[dict]) -> List[str]:
    """
    Heuristic trigger detection by node type/name.
    """
    triggers = []
    for n in nodes:
        label = str(n.get("type","")).lower()
        name  = str(n.get("name","")).lower()
        if any(k in label for k in ("trigger", "webhook", "schedule", "cron")) \
            or name.startswith(("schedule","cron")):
            triggers.append(n.get("name") or n.get("id"))
    return [str(t) for t in triggers if t is not None]


def find_unreachable_nodes(workflow: Dict[str, Any]) -> Set[str]:
    """
    Return set of node names/ids that are unreachable from any trigger.
    """
    nodes = workflow.get("nodes", []) or []
    edges = _extract_edges(workflow)

    # Build adjacency on node names
    names = []
    for n in nodes:
        nm = n.get("name") or n.get("id")
        if nm is not None:
            names.append(str(nm))

    adj = {nm: [] for nm in names}
    for u, v in edges:
        if u is None or v is None:
            continue
        u, v = str(u), str(v)
        adj.setdefault(u, []).append(v)

    triggers = _extract_trigger_names(nodes)
    if not triggers:
        # If no trigger found, treat all nodes unreachable (structural checker also reports missing trigger)
        return set(names)

    reachable: Set[str] = set()
    q = deque(triggers)
    reachable.update(triggers)

    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt not in reachable:
                reachable.add(nxt)
                q.append(nxt)

    return set(names) - reachable

def find_dead_end_chains(
    workflow: Dict[str, Any],
    action_predicate: Optional[Callable[[dict], bool]] = None
) -> List[List[str]]:
    """
    Find dead-end chains starting from reachable nodes:
      - node is reachable from trigger
      - out_degree == 0
      - node is NOT an action/terminal node
    Returns list of chains (each chain is a list of node names from ancestor->deadend).
    """
    nodes = workflow.get("nodes", []) or []
    edges = _extract_edges(workflow)

    # map name->node dict
    def name_of(n):
        nm = n.get("name") or n.get("id")
        return str(nm) if nm is not None else None

    index = {name_of(n): n for n in nodes if name_of(n) is not None}

    names = list(index.keys())

    # adjacency + reverse adjacency
    adj = {nm: [] for nm in names}
    radj = {nm: [] for nm in names}
    for u, v in edges:
        if u is None or v is None:
            continue
        u, v = str(u), str(v)
        adj.setdefault(u, []).append(v)
        radj.setdefault(v, []).append(u)

    triggers = _extract_trigger_names(nodes)
    if not triggers:
        return []

    # reachable first
    reachable: Set[str] = set()
    q = deque(triggers)
    reachable.update(triggers)
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt not in reachable:
                reachable.add(nxt)
                q.append(nxt)

    # default action predicate: email/slack/tg/http/db treated as terminal-ish
    def _default_is_action(node: dict) -> bool:
        label = (str(node.get("type", "")) + " " + str(node.get("name", ""))).lower()
        return any(k in label for k in ("email", "slack", "telegram", "http", "request", "db", "database"))

    is_action = action_predicate or _default_is_action

    deadends = []
    for nm in reachable:
        node = index.get(nm, {})
        if len(adj.get(nm, [])) == 0 and not is_action(node):
            deadends.append(nm)

    chains: List[List[str]] = []
    for de in deadends:
        chain = [de]
        cur = de
        # walk backwards until branching point or trigger
        while True:
            preds = [p for p in radj.get(cur, []) if p in reachable]
            if len(preds) != 1:
                break
            p = preds[0]
            if p in triggers:
                chain.append(p)
                break
            chain.append(p)
            cur = p

        chain.reverse()
        chains.append(chain)

    return chains

def compute_structural_metrics(workflow: Dict[str, Any], small_graph_floor: float = 0.3, weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    Compute structural metrics for an n8n-style workflow.
    Returns a dict of normalized metrics in [0,1].
    """
    nodes = workflow.get("nodes", [])
    edges = _extract_edges(workflow)

    # Build graph using node 'name' (fallback to 'id' when name is missing)
    names = []
    for n in nodes:
        nm = n.get("name") or n.get("id")
        if nm is not None:
            names.append(str(nm))
    
    # mapping name -> id
    name_to_id = { str(n.get("name") or n.get("id")): n.get("id") for n in nodes }

    G = nx.DiGraph()
    G.add_nodes_from(names)
    # Also ensure that any src/dst seen in edges but missing from nodes are added
    for (u, v) in edges:
        if u is not None and v is not None:
            if u not in G:
                G.add_node(u)
            if v not in G:
                G.add_node(v)
            G.add_edge(u, v)

    n_nodes = len(G.nodes)
    n_edges = len(G.edges)
    if n_nodes == 0:
        return {
            "n_nodes": 0, "n_edges": 0,
            "connected_ratio": 0.0,
            "acyclic": 1.0,
            "orphan_ratio": 1.0,
            "avg_out_norm": 0.0,
            "structural_score": 0.0,
        }

    # Connectivity: proportion of nodes in the largest weakly connected component
    largest_cc = max(nx.weakly_connected_components(G), key=len) if n_nodes > 0 else set()
    connected_ratio = len(largest_cc) / n_nodes

    # DAG property (acyclic)
    acyclic = 1.0 if nx.is_directed_acyclic_graph(G) else 0.0

    # Orphan nodes (no incoming and no outgoing)
    orphan_nodes = [n for n in G.nodes if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    orphan_ratio = len(orphan_nodes) / n_nodes

    # Average out-degree (normalized by max possible out-degree)
    avg_out = sum(dict(G.out_degree()).values()) / max(1, n_nodes)
    max_possible_out = max(1, n_nodes - 1)
    avg_out_norm = min(avg_out / max_possible_out, 1.0)

    if n_nodes <= 3:
        avg_out_norm = max(avg_out_norm, small_graph_floor)

    # Aggregate structural score (weighted)
    w = weights or {"C": 0.45, "A": 0.30, "O": 0.20, "D": 0.05}
    structural_score = round(
        w["C"]*connected_ratio + w["A"]*acyclic + w["O"]*(1-orphan_ratio) + w["D"]*avg_out_norm, 2
    )

    # unreachable + dead ends
    unreachable = find_unreachable_nodes(workflow)
    unreachable_ratio = len(unreachable) / n_nodes

    dead_end_chains = find_dead_end_chains(workflow)
    dead_end_ratio = len(dead_end_chains) / max(1, n_nodes)

    return {
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "connected_ratio": float(connected_ratio),
        "acyclic": float(acyclic),
        "orphan_ratio": float(orphan_ratio),
        "avg_out_norm": float(avg_out_norm),
        "structural_score": float(structural_score),
        "unreachable_nodes": [name_to_id.get(u, u) for u in unreachable],
        "unreachable_ratio": float(unreachable_ratio),
        "dead_end_chains": [
            [name_to_id.get(x, x) for x in chain]
            for chain in dead_end_chains
        ],
        "dead_end_ratio": float(dead_end_ratio),
        "_small_graph_floor": float(small_graph_floor),
        "_weights": w
    }