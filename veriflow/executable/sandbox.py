# veriflow/executable/sandbox.py

from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional
import re
import time

import networkx as nx

from veriflow.utils.graph import build_dag
from veriflow.executable.dataflow import check_dataflow
from veriflow.executable.faults import inject_fault


# ---------- Public API ----------

def validate_workflow(
    workflow: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    time_budget_sec: float = 2.0,
    enable_faults: bool = False,
    fault_profile: str = "medium",
) -> Tuple[float, List[str], Dict[str, Any]]:
    """
    Validate if a workflow is executable in a sandbox (no real side effects).
    Returns (E, issues, detail) where:
      - E in [0,1] is the executability score
      - issues is a list of human-readable problems
      - detail includes sub-scores and diagnostics

    Scoring:
      E = w_param * param_ok + w_path * path_ok + w_run * runtime_ok + w_dataflow * dataflow_ok
      defaults: w_param=0.4, w_path=0.25, w_run=0.15, w_dataflow = 0.2
    """
    w = weights or {"param": 0.4, "path": 0.25, "run": 0.15, "dataflow": 0.20}
    w_sum = max(sum(w.values()), 1e-9)
    # normalize in case caller passes non-1.0 sum
    w = {k: v / w_sum for k, v in w.items()}

    issues: List[str] = []
    nodes = workflow.get("nodes", [])
    G = build_dag(workflow)

    # 1) Parameter completeness (static)
    param_ok, missing_params, param_detail = _check_parameters(nodes)
    if not param_ok:
        for m in missing_params:
            issues.append(m)

    # 2) Path reachability from triggers (static graph reachability)
    path_ok, unreachable_nodes = _check_reachability(G, nodes)
    if not path_ok:
        for u in unreachable_nodes:
            issues.append(f"[PATH] Unreachable node: {u}")

    # 3) Data-flow consistency (static, new)
    dataflow_ok = 1.0
    dataflow_applicable = False
    df_detail: Dict[str, Any] = {}
    try:
        df_ok, df_issues, df_detail = check_dataflow(workflow, G)
        dataflow_applicable = df_detail.get("applicable", True)

        if dataflow_applicable:
            dataflow_ok = 1.0 if df_ok else 0.0
            if not df_ok:
                issues.extend(df_issues)
        else:
            dataflow_ok = 1.0  # vacuously satisfied
    except Exception as e:
        dataflow_ok = 0.0
        dataflow_applicable = False
        issues.append(f"[DATAFLOW] Data-flow analysis failed: {e}")

    # 4) Runtime simulation (safe, no side effects)
    runtime_ok, run_issues, exec_log = _simulate_runtime(G, nodes, time_budget_sec=time_budget_sec, enable_faults=enable_faults,
    fault_profile=fault_profile,)
    issues.extend(run_issues)

    E = round(
        w["param"] * (1.0 if param_ok else 0.0)
        + w["path"] * (1.0 if path_ok else 0.0)
        + w["run"]  * (1.0 if runtime_ok else 0.0)
        + w["dataflow"] * dataflow_ok,
        2,
    )

    detail = {
        "param_ok": 1.0 if param_ok else 0.0,
        "path_ok":  1.0 if path_ok else 0.0,
        "runtime_ok": 1.0 if runtime_ok else 0.0,
        "dataflow_ok":   dataflow_ok, 
        "weights": w,
        "missing_params": missing_params,
        "param_detail": param_detail, 
        "unreachable_nodes": unreachable_nodes,
        "dataflow_detail": df_detail,
        "executed_nodes": exec_log.get("executed_nodes", []),
        "executed_nodes_readable": exec_log.get("executed_nodes_readable", []),  
        "name_map": exec_log.get("name_map", {}),                                
        "node_results": exec_log.get("node_results", {}),
        "time_budget_sec": time_budget_sec,
        "faulted_nodes": exec_log.get("faulted_nodes", []),
        "fault_profile": exec_log.get("fault_profile", None),
        "faults_enabled": exec_log.get("faults_enabled", False),
    }
    return E, issues, detail


# ---------- Parameter checks ----------

def _check_parameters(nodes: List[Dict[str, Any]]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate minimum parameter completeness for common node types.
    This is conservative and easily extensible.
    """
    missing: List[str] = []
    detail: Dict[str, Any] = {}
    ok = True

    for n in nodes:
        ntype = (n.get("type") or "").lower()
        nname = n.get("name") or n.get("id") or "<unnamed>"
        params = n.get("parameters") or {}

        if "http" in ntype or "httprequest" in ntype:
            if not _has_nonempty(params, "url"):
                ok = False
                missing.append(f"[PARAM] HTTP Request '{nname}' missing parameter: url")
            elif not _looks_like_url(params.get("url")):
                ok = False
                missing.append(f"[PARAM] HTTP Request '{nname}' url does not look valid")

        if "email" in ntype and "send" in ntype:
            # n8n-nodes-base.emailSend
            if not _has_nonempty(params, "to"):
                ok = False
                missing.append(f"[PARAM] Email '{nname}' missing parameter: to")
            if not _has_nonempty(params, "subject"):
                ok = False
                missing.append(f"[PARAM] Email '{nname}' missing parameter: subject")
            # body is optional for our sandbox

        if "cron" in ntype or "schedule" in ntype or "trigger" in ntype:
            # accept cron-like expressions; keep it simple
            cron = params.get("cronExpression") or params.get("cron") or ""
            if not cron or not _looks_like_cron(cron):
                ok = False
                missing.append(f"[PARAM] Schedule '{nname}' has invalid or missing cron expression")

        if "slack" in ntype:
            # optional: channel/text minimal checks
            pass

        if "telegram" in ntype:
            # optional: chatId/text minimal checks
            pass

    detail["checked_nodes"] = len(nodes)
    return ok, missing, detail


def _has_nonempty(d: Dict[str, Any], key: str) -> bool:
    v = d.get(key)
    return v is not None and str(v).strip() != ""


def _looks_like_url(url: str) -> bool:
    t = (url or "").strip().lower()
    return t.startswith("http://") or t.startswith("https://")


_CRON_RE = re.compile(r"^([\d\*/,-]+)\s+([\d\*/,-]+)\s+([\d\*/,-]+)\s+([\d\*/,-]+)\s+([\d\*/,-]+)(\s+([\d\*/,-]+))?$")
def _looks_like_cron(expr: str) -> bool:
    return bool(_CRON_RE.match((expr or "").strip()))


# ---------- Reachability ----------
def _build_node_index(nodes):
    """Return a dict that maps BOTH id and name to the node definition."""
    idx = {}
    for n in nodes:
        nid = n.get("id")
        nname = n.get("name")
        if nid is not None:
            idx[str(nid)] = n
        if nname:
            idx[str(nname)] = n
    return idx


def _check_reachability(G: nx.DiGraph, nodes: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Check nodes reachable from any trigger-like node."""
    index = _build_node_index(nodes)

    # Discover triggers by inspecting node definitions if available; fall back to key heuristic.
    triggers = []
    for k in G.nodes:
        node = index.get(str(k))
        if node and _is_trigger_node(node):
            triggers.append(k)
        else:
            # fallback: infer from key when node def missing
            kl = str(k).lower()
            if any(t in kl for t in ("cron", "trigger", "webhook", "schedule")):
                triggers.append(k)

    if not triggers:
        all_nodes = list(G.nodes)
        return (len(all_nodes) == 0), ([] if not all_nodes else all_nodes)

    reachable = set()
    for t in triggers:
        reachable.add(t)
        for v in nx.algorithms.descendants(G, t):
            reachable.add(v)

    unreachable = [n for n in G.nodes if n not in reachable]
    return (len(unreachable) == 0), unreachable


def _is_trigger_node(node: Dict[str, Any]) -> bool:
    ntype = (node.get("type") or "").lower()
    nname = (node.get("name") or "").lower()
    return (
        "cron" in ntype or "trigger" in ntype or
        "webhook" in ntype or "schedule" in ntype or
        "cron" in nname or "trigger" in nname or "webhook" in nname
    )


# ---------- Runtime simulation ----------
def _simulate_runtime(
    G: nx.DiGraph,
    nodes: List[Dict[str, Any]],
    time_budget_sec: float = 2.0,
    enable_faults: bool = False,
    fault_profile: str = "medium",
) -> Tuple[bool, List[str], Dict[str, Any]]:
    start = time.time()
    issues: List[str] = []
    results: Dict[str, Dict[str, Any]] = {}
    executed: List[str] = []
    faulted_nodes: List[str] = [] 

    index = _build_node_index(nodes)

    # topo order (if cycles, fallback)
    try:
        order = list(nx.topological_sort(G))
    except Exception:
        order = list(G.nodes)

    # find triggers (same logic as reachability)
    triggers = []
    for k in order:
        node = index.get(str(k))
        if node and _is_trigger_node(node):
            triggers.append(k)
        else:
            kl = str(k).lower()
            if any(t in kl for t in ("cron", "trigger", "webhook", "schedule")):
                triggers.append(k)

    if not triggers and len(order) > 0:
        issues.append("[RUNTIME] No trigger node found for execution start")
        # still proceed in given order

    runner = NodeRunnerRegistry()

    queue = triggers[:] if triggers else order[:]
    seen_enqueued = set(queue)

    while queue:
        if time.time() - start > time_budget_sec:
            issues.append("[RUNTIME] Sandbox time budget exceeded")
            break

        curr = queue.pop(0)
        node = index.get(str(curr)) 
        if not node:
            results[str(curr)] = {"ok": False, "msg": "Missing node definition"}
            issues.append(f"[RUNTIME] Node '{curr}' missing definition in nodes[]")
            # still enqueue successors to keep traversal consistent
        else:            
            # (1) Fault injection before actual execution (optional)
            if enable_faults:
                f_ok, f_msg = inject_fault(node, profile_name=fault_profile)
            else:
                f_ok, f_msg = True, "faults disabled"

            if not f_ok:
                executed.append(str(curr))
                faulted_nodes.append(str(curr))
                results[str(curr)] = {"ok": False, "msg": f_msg}
                issues.append(f"[RUNTIME] Injected fault at '{curr}': {f_msg}")
                continue  # Do not run the real node

            # (2) Normal execution
            ok, msg = runner.run(node)
            results[str(curr)] = {"ok": ok, "msg": msg}
            executed.append(str(curr))
            if not ok:
                issues.append(f"[RUNTIME] Execution failed at '{curr}': {msg}")

        for succ in G.successors(curr):
            if succ not in seen_enqueued:
                queue.append(succ)
                seen_enqueued.add(succ)

    # runtime_ok = all nodes we *attempted* to evaluate are ok; if none evaluated, treat as False
    runtime_ok = (len(results) > 0) and all(v.get("ok", False) for v in results.values())

    # Build human-readable name mapping and executed name list
    name_map = {n.get("id"): n.get("name", n.get("id")) for n in nodes if n.get("id") is not None}
    executed_readable = [index.get(str(k), {}).get("name", str(k)) for k in executed]

    return runtime_ok, issues, {"node_results": results, 
                                "executed_nodes": executed, 
                                "executed_nodes_readable": executed_readable, 
                                "name_map": name_map,
                                "faulted_nodes": faulted_nodes,
                                "fault_profile": fault_profile,
                                "faults_enabled": enable_faults,
                                }

# ---------- Mock node runners ----------

class NodeRunnerRegistry:
    """
    Very small registry that simulates node execution.
    Extendable: add more handlers for other node types if needed.
    """
    def run(self, node: Dict[str, Any]) -> Tuple[bool, str]:
        ntype = (node.get("type") or "").lower()
        # Order matters for substring tests
        if "cron" in ntype or "trigger" in ntype or "schedule" in ntype:
            return self._run_cron(node)
        if "httprequest" in ntype or ("http" in ntype and "request" in ntype):
            return self._run_http(node)
        if "emailsend" in ntype or ("email" in ntype and "send" in ntype):
            return self._run_email(node)
        if "slack" in ntype:
            return self._run_slack(node)
        if "telegram" in ntype:
            return self._run_telegram(node)
        # default no-op
        return True, "noop"

    def _run_cron(self, node: Dict[str, Any]) -> Tuple[bool, str]:
        params = node.get("parameters") or {}
        expr = params.get("cronExpression") or params.get("cron") or ""
        if not _looks_like_cron(expr):
            return False, "invalid cron expression"
        return True, "cron ok"

    def _run_http(self, node: Dict[str, Any]) -> Tuple[bool, str]:
        params = node.get("parameters") or {}
        url = params.get("url")
        if not url or not _looks_like_url(url):
            return False, "invalid or missing url"
        # Do not perform a real request; pretend success
        return True, "http ok (mock)"

    def _run_email(self, node: Dict[str, Any]) -> Tuple[bool, str]:
        params = node.get("parameters") or {}
        to = params.get("to")
        subject = params.get("subject")
        if not to or not str(to).strip():
            return False, "missing 'to'"
        if not subject or not str(subject).strip():
            return False, "missing 'subject'"
        # No real sending
        return True, "email ok (mock)"

    def _run_slack(self, node: Dict[str, Any]) -> Tuple[bool, str]:
        # Minimal mock; extend with channel/text checks if needed
        return True, "slack ok (mock)"

    def _run_telegram(self, node: Dict[str, Any]) -> Tuple[bool, str]:
        # Minimal mock; extend with chatId/text checks if needed
        return True, "telegram ok (mock)"