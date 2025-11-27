# tests/test_executable_bench.py

import json
from pathlib import Path

import pytest

from veriflow.executable.sandbox import validate_workflow


@pytest.mark.parametrize(
    "case_dir",
    sorted((Path("bench") / "executable").glob("E*"))
)
def test_executable_bench(case_dir: Path):
    """
    Executable benchmark:
    - load workflow.json
    - load gold.json
    - run validate_workflow (no fault injection)
    - check coarse-grained properties:
        * global executability score E in [E_min, E_max]
        * param/path/runtime/dataflow sub-scores (bool)
        * presence of issue categories based on tags:
              [PARAM], [PATH], [RUNTIME], [DATAFLOW]
    """
    wf_file = case_dir / "workflow.json"
    gold_file = case_dir / "gold.json"

    assert wf_file.exists(), f"Missing workflow.json in {case_dir}"
    assert gold_file.exists(), f"Missing gold.json in {case_dir}"

    with wf_file.open("r", encoding="utf-8") as f:
        workflow = json.load(f)

    with gold_file.open("r", encoding="utf-8") as f:
        gold = json.load(f)

    asserts = gold.get("assert") or {}

    # --- run executable analysis (faults disabled for determinism) ---
    E, issues, detail = validate_workflow(
        workflow,
        enable_faults=False,
        fault_profile="medium",
    )

    # ------------- global score -------------
    if "E_min" in asserts:
        assert E >= float(asserts["E_min"]) - 1e-6, (
            f"{case_dir.name}: E={E}, expected E_min={asserts['E_min']}"
        )
    if "E_max" in asserts:
        assert E <= float(asserts["E_max"]) + 1e-6, (
            f"{case_dir.name}: E={E}, expected E_max={asserts['E_max']}"
        )

    # ------------- sub-scores (param/path/runtime/dataflow) -------------
    # detail["param_ok"], etc., are floats in {0.0, 1.0}
    def _as_bool(x):
        return bool(round(float(x or 0.0)))

    if "param_ok" in asserts:
        expected = bool(asserts["param_ok"])
        got = _as_bool(detail.get("param_ok", 0.0))
        assert got == expected, (
            f"{case_dir.name}: param_ok={got}, expected={expected}"
        )

    if "path_ok" in asserts:
        expected = bool(asserts["path_ok"])
        got = _as_bool(detail.get("path_ok", 0.0))
        assert got == expected, (
            f"{case_dir.name}: path_ok={got}, expected={expected}"
        )

    if "runtime_ok" in asserts:
        expected = bool(asserts["runtime_ok"])
        got = _as_bool(detail.get("runtime_ok", 0.0))
        assert got == expected, (
            f"{case_dir.name}: runtime_ok={got}, expected={expected}"
        )

    if "dataflow_ok" in asserts:
        expected = bool(asserts["dataflow_ok"])
        got = _as_bool(detail.get("dataflow_ok", 0.0))
        assert got == expected, (
            f"{case_dir.name}: dataflow_ok={got}, expected={expected}"
        )

    # ------------- issue category presence -------------
    # We use the standardized prefixes you added:
    #   [PARAM], [PATH], [RUNTIME], [DATAFLOW]
    has_param_issues = any("[PARAM]" in msg for msg in issues)
    has_path_issues = any("[PATH]" in msg for msg in issues)
    has_runtime_issues = any("[RUNTIME]" in msg for msg in issues)
    has_dataflow_issues = any("[DATAFLOW]" in msg for msg in issues)

    if "has_param_issues" in asserts:
        expected = bool(asserts["has_param_issues"])
        assert has_param_issues == expected, (
            f"{case_dir.name}: has_param_issues={has_param_issues}, "
            f"expected={expected}"
        )

    if "has_path_issues" in asserts:
        expected = bool(asserts["has_path_issues"])
        assert has_path_issues == expected, (
            f"{case_dir.name}: has_path_issues={has_path_issues}, "
            f"expected={expected}"
        )

    if "has_runtime_issues" in asserts:
        expected = bool(asserts["has_runtime_issues"])
        assert has_runtime_issues == expected, (
            f"{case_dir.name}: has_runtime_issues={has_runtime_issues}, "
            f"expected={expected}"
        )

    if "has_dataflow_issues" in asserts:
        expected = bool(asserts["has_dataflow_issues"])
        assert has_dataflow_issues == expected, (
            f"{case_dir.name}: has_dataflow_issues={has_dataflow_issues}, "
            f"expected={expected}"
        )