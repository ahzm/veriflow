# utils/graph.py
from typing import Dict, Any, List, Tuple
import networkx as nx

def build_dag(workflow: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()
    id_by_name = {n["name"]: n["id"] for n in workflow["nodes"]}
    for n in workflow["nodes"]:
        G.add_node(n["id"], **n)
    # n8n connections: connections[<nodeName>][<outputIndex>][<inputIndex>] -> list of {node: <name>, type: "main", index: 0}
    for src_name, outs in workflow["connections"].items():
        src_id = id_by_name.get(src_name)
        if src_id is None: 
            continue
        for _out_idx, paths in outs.items():
            for _in_idx, edges in enumerate(paths):
                for e in edges:
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