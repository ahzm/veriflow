import json
from pathlib import Path

import pytest

from veriflow.structural.checker import structural_check
from veriflow.structural.metrics import compute_structural_metrics
from veriflow.utils.graph import has_trigger


@pytest.mark.parametrize("case_dir", sorted((Path("bench") / "structural").glob("S*")))
def test_structural_bench(case_dir: Path):
    """
    Structural benchmark:
    - load workflow.json
    - load expect.json
    - run structural_check + compute_structural_metrics
    - check coarse-grained properties (connectivity, trigger, cycles, etc.)
    """
    wf_file = case_dir / "workflow.json"
    exp_file = case_dir / "expect.json"

    assert wf_file.exists(), f"Missing workflow.json in {case_dir}"
    assert exp_file.exists(), f"Missing expect.json in {case_dir}"

    with wf_file.open("r", encoding="utf-8") as f:
        workflow = json.load(f)

    with exp_file.open("r", encoding="utf-8") as f:
        expect = json.load(f)

    # Run structural analysis
    S, issues, detail = structural_check(workflow)
    metrics = compute_structural_metrics(workflow)

    asserts = (expect.get("assert") or {})

    # ---- has_trigger ----
    if "has_trigger" in asserts:
        expected = bool(asserts["has_trigger"])
        got = has_trigger(workflow.get("nodes", []) or [])
        assert got == expected, f"{case_dir.name}: has_trigger={got}, expected={expected}"

    # ---- connected ----
    if "connected" in asserts:
        expected = bool(asserts["connected"])
        connected = metrics.get("connected_ratio", 0.0) == pytest.approx(1.0)
        if expected:
            assert connected, f"{case_dir.name}: expected connected workflow"
        else:
            assert not connected, f"{case_dir.name}: expected disconnected workflow"

    # ---- has_cycles ----
    if "has_cycles" in asserts:
        expected = bool(asserts["has_cycles"])
        has_cycle = metrics.get("acyclic", 1.0) < 1.0
        assert has_cycle == expected, f"{case_dir.name}: has_cycles={has_cycle}, expected={expected}"

    # ---- has_orphans ----
    if "has_orphans" in asserts:
        expected = bool(asserts["has_orphans"])
        has_orphans = metrics.get("orphan_ratio", 0.0) > 0.0
        assert has_orphans == expected, f"{case_dir.name}: has_orphans={has_orphans}, expected={expected}"

    # ---- has_unreachable ----
    if "has_unreachable" in asserts:
        expected = bool(asserts["has_unreachable"])
        has_unreachable = metrics.get("unreachable_ratio", 0.0) > 0.0
        assert has_unreachable == expected, f"{case_dir.name}: has_unreachable={has_unreachable}, expected={expected}"

    # ---- has_dead_ends ----
    if "has_dead_ends" in asserts:
        expected = bool(asserts["has_dead_ends"])
        has_dead_ends = metrics.get("dead_end_ratio", 0.0) > 0.0
        assert has_dead_ends == expected, f"{case_dir.name}: has_dead_ends={has_dead_ends}, expected={expected}"

    # We don't assert on S itself here to stay robust against future scoring tweaks,
    # but we at least check it's within [0, 1].
    assert 0.0 <= S <= 1.0, f"{case_dir.name}: structural score S out of range: {S}"