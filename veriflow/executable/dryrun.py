#veriflow/executable/dryrun.py
from typing import Dict, Any, Tuple, List

REQUIRED_PARAMS = {
    "email": ["to", "subject"],   
    "http":  ["url"],             
}

def _node_ok(node: dict) -> bool:
    t = (node.get("type","") + " " + node.get("name","")).lower()
    params = node.get("parameters", {})
    if "email" in t:
        return all(k in params and params[k] for k in REQUIRED_PARAMS["email"])
    if "http" in t or "request" in t:
        return all(k in params and params[k] for k in REQUIRED_PARAMS["http"])
    # schedule/trigger 等宽松处理
    return True

def executability_score(workflow: Dict[str, Any]) -> Tuple[float, List[str]]:
    nodes = workflow.get("nodes", [])
    total = len(nodes) or 1
    ok = 0
    issues: List[str] = []
    for n in nodes:
        if _node_ok(n):
            ok += 1
        else:
            issues.append(f"Missing required parameters for node: {n.get('name','(unnamed)')}")
    return round(ok/total, 2), issues