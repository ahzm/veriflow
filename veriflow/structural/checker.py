# veriflow/structural/checker.py

from typing import Dict, Any, List, Tuple
from jsonschema import validate, ValidationError

from .schema import N8N_MINIMAL_SCHEMA
from veriflow.structural.metrics import compute_structural_metrics
from veriflow.utils.graph import build_dag, has_trigger, exit_coverage


def structural_check(workflow: Dict[str, Any], small_graph_floor: float = 0.3, weights: dict = None) -> Tuple[float, List[str], Dict[str, Any]]: 
    """
    Compute the structural score S and collect human-readable issues.
    S aggregates: schema validity, graph quality (from metrics.py), trigger presence, exit coverage.

    Returns:
        S (float in [0,1]), issues (List[str]), details
    """
    issues: List[str] = []

    # 1) Schema validation (syntax)
    syntax_ok = 1.0
    try:
        validate(instance=workflow, schema=N8N_MINIMAL_SCHEMA)
    except ValidationError as e:
        syntax_ok = 0.0
        issues.append(f"[SCHEMA] Schema validation error: {e.message}")

    # 2) Graph-derived metrics (single source of truth for structure quality)
    m = compute_structural_metrics(workflow, small_graph_floor=small_graph_floor, weights=weights)
    conn     = float(m.get("connected_ratio", 0.0))   # largest weakly CC / n_nodes
    acyclic  = float(m.get("acyclic", 1.0))           # 1 if DAG else 0
    orphan   = float(m.get("orphan_ratio", 1.0))      # fraction of isolated nodes
    avg_out  = float(m.get("avg_out_norm", 0.0))      # normalized avg out-degree
    base_struct = float(m.get("structural_score", 0.0))

    # Issue collection from metrics (kept simple and interpretable)
    if conn < 1.0:
        issues.append(
            f"[STRUCTURE] Workflow not fully connected "
            f"(largest weakly connected component covers {conn:.2f} of nodes)"
        )
    if acyclic < 1.0:
        issues.append(
            "[STRUCTURE] Workflow contains cycles "
            "(may cause infinite loops or repeated execution)"
        )
    if orphan > 0.0:
        issues.append(
            f"[STRUCTURE] Orphan nodes detected (ratio={orphan:.2f}) "
            "(nodes with no incoming and no outgoing edges)"
        )

    unreachable_nodes = m.get("unreachable_nodes", []) or []
    dead_end_chains = m.get("dead_end_chains", []) or []

    if unreachable_nodes:
        issues.append(
            f"[REACHABILITY] Unreachable nodes from any trigger: {unreachable_nodes} "
            "(these nodes will never be executed)"
        )

    if dead_end_chains:
        issues.append(
            f"[REACHABILITY] Dead-end chains found (reachable but no continuation): "
            f"{dead_end_chains}"
        )

    # 3) Trigger presence and exit coverage
    trig = 1.0 if has_trigger(workflow.get("nodes", [])) else 0.0
    if trig == 0.0:
        issues.append(
            "[FLOW] Missing trigger node "
            "(workflow has no entry point such as schedule/webhook/trigger)"
        )

    # exit_coverage expects a DAG view of the workflow
    flow_computed = True
    try:
        G = build_dag(workflow)
        flow = float(exit_coverage(G))  # fraction of sinks reachable as intended
    except Exception:
        flow_computed = False
        flow = 0.0
        issues.append(
            "[FLOW] Exit coverage could not be computed (graph invalid or non-DAG)"
            
        )

    if flow_computed and m.get("n_nodes", 0) >= 2 and flow < 0.95:
        issues.append(
            f"[FLOW] Not all terminal nodes are covered by the main flow "
            f"(exit coverage={flow:.2f})"
        )

    unreachable_ratio = float(m.get("unreachable_ratio", 0.0))
    dead_end_ratio = float(m.get("dead_end_ratio", 0.0))

    k = 0.15
    penalty = 0.0
    if "unreachable_ratio" in m or "dead_end_ratio" in m:
        penalty = min(1.0, unreachable_ratio + dead_end_ratio)

    # 4) Aggregate structural score (weights can be tuned; sum to 1.0)
    #    We deliberately lean on base_struct from metrics.py and keep schema/trigger/flow visible.
    S = (
        0.40 * base_struct * (1-k*penalty) +   # composite graph quality from metrics.py
        0.25 * syntax_ok   +   # JSON schema validity
        0.20 * trig        +   # presence of a trigger
        0.15 * flow            # exit coverage
    )

    S = round(max(0.0, min(1.0, S)), 2)

    # 5) Detail payload for verbose/report
    detail = {
        "schema_ok": float(syntax_ok),
        "connected_ratio": conn,
        "acyclic": acyclic,
        "orphan_ratio": orphan,
        "avg_out_norm": avg_out,
        "exit_coverage": flow,
        "base_structural_score": base_struct,
        "final_S": S,
        "unreachable_ratio": unreachable_ratio,
        "dead_end_ratio": dead_end_ratio,
        "penalty": penalty,
        "k": k,
        "explanation": {
            "connected_ratio": "Fraction of nodes in the largest weakly connected component",
            "acyclic": "1.0 means no directed cycles; 0.0 means at least one cycle exists",
            "orphan_ratio": "Fraction of nodes with no incoming and no outgoing edges",
            "avg_out_norm": "Average out-degree normalized by maximum possible out-degree",
            "exit_coverage": "Fraction of terminal nodes that are reachable from triggers",
            "unreachable_ratio": "Fraction of nodes not reachable from any trigger",
            "dead_end_ratio": "Reachable nodes/chains that do not lead to terminal actions",
        },
    }

    return S, issues, detail