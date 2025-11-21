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
    id_by_name = {n["name"]: n["id"] for n in workflow["nodes"]}
    for n in workflow["nodes"]:
        G.add_node(n["id"], **n)

    nodes = workflow.get("nodes", []) or []
    for n in nodes:
        if "id" in n:
            G.add_node(n["id"], **n)

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
    
    id_by_name = {n.get("name"): n.get("id") for n in nodes if "name" in n and "id" in n}


    # n8n connections: connections[<nodeName>][<outputIndex>][<inputIndex>] -> list of {node: <name>, type: "main", index: 0}    
    for src_name, outs in conns.items():
        src_id = id_by_name.get(src_name)
        if src_id is None:
            continue
        for _out_idx, paths in outs.items():
            for _in_idx, edges_list in enumerate(paths):
                for e in edges_list:
                    tgt_name = e.get("node")
                    tgt_id = id_by_name.get(tgt_name)
                    if tgt_id is not None:
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