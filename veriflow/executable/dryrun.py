#veriflow/executable/dryrun.py
from typing import Dict, Any, Tuple, List

REQUIRED_PARAMS = {
    "email": ["to", "subject"],   # minimal email fields
    "http":  ["url"],             # minimal HTTP request field
}

def _has_nonempty(params: Dict[str, Any], key: str) -> bool:
    """Return True if params[key] exists and is not an empty string."""
    if key not in params:
        return False
    v = params[key]
    # For URL / email / subject we expect a non-empty string
    return v is not None and str(v).strip() != ""


def _node_ok(node: Dict[str, Any]) -> bool:
    """
    Very lightweight executability check for a single node.
    Only checks a few common types (email, HTTP); others are treated as OK.
    """
    t = (str(node.get("type", "")) + " " + str(node.get("name", ""))).lower()
    params = node.get("parameters", {}) or {}

    # Email-like nodes
    if "email" in t and "send" in t:
        return all(_has_nonempty(params, k) for k in REQUIRED_PARAMS["email"])

    # HTTP request-like nodes
    if "httprequest" in t or ("http" in t and "request" in t):
        return all(_has_nonempty(params, k) for k in REQUIRED_PARAMS["http"])

    # Trigger / schedule nodes: we do not enforce any parameter here
    # (more detailed checks are handled in sandbox.py)
    return True

def executability_score(workflow: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Compute a very coarse executability score based on per-node parameter completeness.

    Returns:
        score in [0, 1], issues as a list of human-readable messages.
    """
    nodes = workflow.get("nodes", []) or []
    if not nodes:
        # Empty workflow: treat as trivially executable with no issues
        return 1.0, []

    total = len(nodes)
    ok = 0
    issues: List[str] = []

    for n in nodes:
        if _node_ok(n):
            ok += 1
        else:
            name = n.get("name") or n.get("id") or "(unnamed)"
            ntype = n.get("type") or ""
            issues.append(
                f"[DRYRUN] Missing required parameters for node '{name}' "
                f"(type='{ntype}')"
            )

    score = round(ok / total, 2)
    return score, issues