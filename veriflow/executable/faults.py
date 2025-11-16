# veriflow/executable/faults.py
from __future__ import annotations
from typing import Tuple, Dict, Any
import random

FAULT_PROFILES = {
    "minimal":  {"rate": 0.05, "api": True,  "data": False, "node": False},
    "medium":   {"rate": 0.15, "api": True,  "data": True,  "node": False},
    "chaos":    {"rate": 0.30, "api": True,  "data": True,  "node": True},
    "llm":      {"rate": 0.20, "api": False, "data": True,  "node": False},
    "prod":     {"rate": 0.10, "api": True,  "data": True,  "node": True},
}

DEFAULT_PROFILE = "medium"


def inject_fault(node: Dict[str, Any], profile_name: str | None = None) -> Tuple[bool, str]:
    """
    Possibly inject a runtime fault according to a fault profile.

    profile_name:
      - one of FAULT_PROFILES keys
      - if None or unknown, fall back to DEFAULT_PROFILE
    """
    profile = FAULT_PROFILES.get(profile_name or DEFAULT_PROFILE, FAULT_PROFILES[DEFAULT_PROFILE])
    rate = profile["rate"]

    # No fault in this run
    if random.random() > rate:
        return True, "no fault"

    ntype = (node.get("type", "")).lower()

    # API faults (HTTP / webhook)
    if profile["api"] and ("http" in ntype or "webhook" in ntype):
        faults = [
            "timeout",
            "connection reset",
            "HTTP 500",
            "HTTP 400",
            "rate limited (429)",
            "authentication failed (401)",
            "invalid JSON response",
        ]
        return False, f"injected API fault: {random.choice(faults)}"

    # Data faults
    if profile["data"]:
        faults = [
            "missing field",
            "unexpected null",
            "wrong data type",
            "schema mismatch",
            "index error",
            "key not found",
        ]
        return False, f"injected data fault: {random.choice(faults)}"

    # Node-level faults
    if profile["node"]:
        faults = [
            "node crash",
            "runtime exception",
            "resource exhausted",
            "unknown internal error",
        ]
        return False, f"injected node fault: {random.choice(faults)}"

    # If profile says nothing specific, treat as no fault
    return True, "no fault"