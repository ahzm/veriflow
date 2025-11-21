# utils/graph.py
from typing import Dict, Any, List, Tuple
import networkx as nx

def build_dag(workflow: Dict[str, Any]) -> nx.DiGraph:
    """
    Build a DAG from either:
      1) n8n native json (nodes + connections)
      2) simplified bench format (nodes + edges)
    """
    G = nx.DiGraph()
    nodes = workflow.get("nodes", []) or []
    
    for n in nodes:
        nid = n.get("id")
        if nid is None:
            continue
        G.add_node(nid, **n)

    # Case A: simplified bench format
    edges = workflow.get("edges") or []
    if edges:
        for e in edges:
            src = e.get("source")
            tgt = e.get("target")
            if src is None or tgt is None:
                continue
            if src not in G:
                G.add_node(src)
            if tgt not in G:
                G.add_node(tgt)
            G.add_edge(src, tgt)
        return G
    
    # Case B: n8n native format
    conns = workflow.get("connections") or {}
    if not conns:
        return G
    
    id_by_name = {
        n.get("name"): n.get("id")
        for n in nodes
        if n.get("name") is not None and n.get("id") is not None
    }


    # n8n connections: connections[<nodeName>][<outputIndex>][<inputIndex>] -> list of {node: <name>, type: "main", index: 0}    
    for src_name, outs in conns.items():
        src_id = id_by_name.get(src_name)
        if src_id is None:
            continue
        if not isinstance(outs, dict):
            continue
        for _out_idx, paths in outs.items():
            if isinstance(paths, dict):
                iterable = paths.values()
            else:
                iterable = paths
            for edges_list in iterable:
                if not edges_list:
                    continue
                for e in edges_list:
                    tgt_name = e.get("node")
                    tgt_id = id_by_name.get(tgt_name)
                    
                    if tgt_id is not None:
                        if tgt_id not in G:
                            G.add_node(tgt_id)
                        G.add_edge(src_id, tgt_id)
    return G

def connectivity_ratio(G: nx.DiGraph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    connected = sum(1 for n in G.nodes if (G.in_degree(n) + G.out_degree(n)) > 0)
    return connected / G.number_of_nodes()

def has_trigger(nodes: List[dict]) -> bool:
    TRIGGER_KEYS = ("trigger", "webhook", "cron", "schedule", "interval")
    for n in nodes:
        t = (n.get("type","") + " " + n.get("name","")).lower()
        if any(k in t for k in TRIGGER_KEYS):
            return True
    return False

def exit_coverage(G: nx.DiGraph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    progressed = sum(1 for n in G.nodes if G.out_degree(n) > 0 or G.in_degree(n) > 0)
    return progressed / G.number_of_nodes()

# SCC compression
def build_scc_dag(workflow: Dict[str, Any]):
    """
    Build SCC-compressed DAG.

    Returns:
        Gc  (nx.DiGraph): compressed DAG
        G0  (nx.DiGraph): original graph
        comp_of (dict): node_id -> component_id
        nodes_of_comp (dict): component_id -> list[node_id]
    """
    # Original graph (may contain cycles)
    G0 = build_dag(workflow)

    # Compute SCCs
    sccs = list(nx.strongly_connected_components(G0))

    comp_of = {}
    nodes_of_comp = {}
    for cid, comp in enumerate(sccs):
        comp_nodes = sorted(list(comp))
        nodes_of_comp[cid] = comp_nodes
        for nid in comp_nodes:
            comp_of[nid] = cid

    # Build compressed DAG (no cycles)
    Gc = nx.DiGraph()
    for cid in nodes_of_comp:
        Gc.add_node(cid)

    for u, v in G0.edges():
        cu = comp_of[u]
        cv = comp_of[v]
        if cu != cv:
            Gc.add_edge(cu, cv)

    return Gc, G0, comp_of, nodes_of_comp