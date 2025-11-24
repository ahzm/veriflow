# veriflow/structural/metrics.py

import networkx as nx
from typing import Dict, Any, List

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
        nid = n.get("id")
        name = str(nid) if nid is not None else n.get("name")
        if name is not None:
            names.append(name)

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

    return {
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "connected_ratio": float(connected_ratio),
        "acyclic": float(acyclic),
        "orphan_ratio": float(orphan_ratio),
        "avg_out_norm": float(avg_out_norm),
        "structural_score": float(structural_score),
        "_small_graph_floor": float(small_graph_floor),
        "_weights": w
    }