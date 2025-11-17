#!/usr/bin/env python3
# veriflow/cli.py

import json
from pathlib import Path
import typer
from typing import Optional

from veriflow.structural.checker import structural_check
from veriflow.semantic.matcher import semantic_score  # hybrid aware (use_llm flag)
from veriflow.executable.dryrun import executability_score
from veriflow.executable.sandbox import validate_workflow
from veriflow.executable.faults import FAULT_PROFILE_NAMES

app = typer.Typer(help="VeriFlow CLI - Verify LLM-generated (n8n) workflows")

@app.command()
def verify(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True, help="Path to n8n workflow JSON"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Natural-language task description"),
    alpha: float = typer.Option(1 / 3, "--alpha", help="Weight for StructuralScore"),
    beta: float = typer.Option(1 / 3, "--beta", help="Weight for SemanticScore"),
    gamma: float = typer.Option(1 / 3, "--gamma", help="Weight for ExecutabilityScore"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug info"),
    use_sandbox: bool = typer.Option(False, "--sandbox", help="Use sandbox runtime instead of dry-run"),
    path_sep: str = typer.Option(" -> ", "--path-sep", help="Separator for executed path when verbose"),
    use_llm: bool = typer.Option(False, "--use-llm", help="Enable LLM refinement for intent extraction"),
    report: Optional[Path] = typer.Option(None, "--report", help="Write a JSON report to this path"),
    small_graph_floor: float = typer.Option(0.3, "--small-graph-floor", help="Floor for avg_out_norm when n_nodes<=3"),
    w_c: float = typer.Option(0.45, "--wC", help="Weight for Structural Completeness (C): node definitions, required fields, config presence."),
    w_a: float = typer.Option(0.30, "--wA", help="Weight for Structural Alignment (A): correctness of edges, flow logic, and node connectivity."),
    w_o: float = typer.Option(0.20, "--wO", help="Weight for Structural Ordering (O): topological order, cycle handling, and trigger ordering."),
    w_d: float = typer.Option(0.05, "--wD", help="Weight for Structural Density (D): graph out-degree normalization detecting degenerate workflows."),
    faults: bool = typer.Option(False, "--faults", help="Enable fault injection during sandbox execution"),
    fault_profile: str = typer.Option("medium", "--fault-profile", help="Fault injection profile: minimal | medium | chaos | llm | prod"),
):
    """
    Verify a workflow against structural, semantic, and executability criteria with weighted overall score.
    This version passes `use_llm` to semantic_score (hybrid intent extraction).
    """
    wf = json.load(open(input, "r", encoding="utf-8"))
    fault_profile = fault_profile.lower()
    if fault_profile not in FAULT_PROFILE_NAMES:
        raise typer.BadParameter(f"Invalid fault profile '{fault_profile}'. "
                                f"Choose one of: {', '.join(FAULT_PROFILE_NAMES)}")

    # Structural
    weights = {"C": w_c, "A": w_a, "O": w_o, "D": w_d}
    _s = structural_check(wf, small_graph_floor=small_graph_floor, weights=weights)
    if isinstance(_s, tuple) and len(_s) == 3:
        s, s_issues, s_detail = _s
    else:
        s, s_issues = _s
        s_detail = None

    # Semantic (robust unpack: accepts 2- or 3-tuple; hybrid if use_llm=True)
    _sem = semantic_score(wf, prompt, use_llm=use_llm)
    if isinstance(_sem, tuple) and len(_sem) == 3:
        m, m_issues, m_detail = _sem
    else:
        m, m_issues = _sem
        m_detail = None

    # Executability
    if use_sandbox:
        e, e_issues, e_detail = validate_workflow(wf, enable_faults=faults, fault_profile=fault_profile,) # sandbox path
    else:
        e, e_issues = executability_score(wf)           # existing dry-run
        e_detail = {}                                   # keep shape consistent

    # Weighted overall (normalized)
    wsum = max(alpha + beta + gamma, 1e-9)
    overall = round((alpha * s + beta * m + gamma * e) / wsum, 2)

    print(f"StructuralScore:   {s}")
    print(f"SemanticScore:     {m}")
    print(f"ExecutabilityScore:{e}")
    print(f"Overall:           {overall}")

    # Collect issues before optional report writing
    issues = s_issues + m_issues + e_issues

    # Optional JSON report
    if report is not None:
        payload = {
            "input": str(input),
            "prompt": prompt,
            "scores": {"S": s, "M": m, "E": e, "Overall": overall},
            "issues": issues,
            "struct_detail": s_detail or {},  
            "semantic_detail": m_detail or {},
            "exec_detail": e_detail or {},
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        with open(report, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[ok] wrote report to {report}")

    # Human-readable issues
    if issues:
        print("Detected issues:")
        for it in issues:
            print(f"- {it}")

    # Verbose debug
    if verbose:
        if s_detail:
            print("[debug] structural detail:", s_detail)
        else:
            print("[debug] structural detail: <none>")

        if m_detail:
            print("[debug] semantic detail:", m_detail)
            src = m_detail.get("source") if isinstance(m_detail, dict) else None
            if src:
                print(f"[debug] intent source: {src}")
        else:
            print("[debug] semantic detail: <none>")

        if e_detail:
            print("[debug] executable detail:", e_detail)
            readable_path = e_detail.get("executed_nodes_readable")
            if readable_path:
                print("[debug] executed path:", path_sep.join(readable_path))
            else:
                # fallback to ids
                ids_path = e_detail.get("executed_nodes")
                if ids_path:
                    print("[debug] executed path:", " -> ".join(ids_path))
        else:
            print("[debug] executable detail: <none>")

@app.command()
def bench(
    glob: str = typer.Option("bench/*/gold.json", "--glob", help="Glob for workflow JSON files"),
    out: Path = typer.Option(Path("experiments/results/report.csv"), "--out", help="CSV path to write results"),
    alpha: float = typer.Option(1 / 3, "--alpha", help="Weight for StructuralScore"),
    beta: float = typer.Option(1 / 3, "--beta", help="Weight for SemanticScore"),
    gamma: float = typer.Option(1 / 3, "--gamma", help="Weight for ExecutabilityScore"),
    use_llm: bool = typer.Option(False, "--use-llm", help="Enable LLM refinement for intent extraction"),
    dump_details: bool = typer.Option(False, "--dump-details", help="Dump per-task JSON alongside CSV"),
    use_sandbox: bool = typer.Option(False, "--sandbox", help="Use sandbox runtime instead of dry-run"),
    small_graph_floor: float = typer.Option(0.3, "--small-graph-floor", help="Floor for avg_out_norm when n_nodes<=3"),
    w_c: float = typer.Option(0.45, "--wC", help="Weight for Structural Completeness (C): node definitions, required fields, config presence."),
    w_a: float = typer.Option(0.30, "--wA", help="Weight for Structural Alignment (A): correctness of edges, flow logic, and node connectivity."),
    w_o: float = typer.Option(0.20, "--wO", help="Weight for Structural Ordering (O): topological order, cycle handling, and trigger ordering."),
    w_d: float = typer.Option(0.05, "--wD", help="Weight for Structural Density (D): graph out-degree normalization detecting degenerate workflows."),
    faults: bool = typer.Option(False, "--faults", help="Enable fault injection in sandbox runs"),
    fault_profile: str = typer.Option("medium", "--fault-profile", help="Fault profile for sandbox runs"),
):
    """
    Batch verify workflows and export a CSV report.
    This version also accepts weights, sandbox, and the use_llm flag (hybrid intent).
    """
    import glob as _glob
    import pandas as pd

    fault_profile = fault_profile.lower()
    if fault_profile not in FAULT_PROFILE_NAMES:
        raise typer.BadParameter(
            f"Invalid fault profile '{fault_profile}'. "
            f"Choose one of: {', '.join(FAULT_PROFILE_NAMES)}"
        )

    rows = []
    for fp_str in _glob.glob(glob):
        fp = Path(fp_str)
        wf = json.load(open(fp, "r", encoding="utf-8"))

        # Structural (pass same knobs as verify)
        weights = {"C": w_c, "A": w_a, "O": w_o, "D": w_d}
        _s = structural_check(wf, small_graph_floor=small_graph_floor, weights=weights)
        if isinstance(_s, tuple) and len(_s) == 3:
            s, _, s_detail = _s
        else:
            s, _ = _s
            s_detail = None

        # Read prompt if exists
        prompt_path = fp.with_name("prompt.txt")
        prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

        # Semantic
        _sem = semantic_score(wf, prompt, use_llm=use_llm)
        if isinstance(_sem, tuple) and len(_sem) == 3:
            m, _, m_detail = _sem
            source = (m_detail or {}).get("source", "")
        else:
            m, _ = _sem
            m_detail = None
            source = ""

        # Executability (respect sandbox switch)
        if use_sandbox:
            e, _, e_detail = validate_workflow(wf, enable_faults=faults, fault_profile=fault_profile, )
        else:
            e, _ = executability_score(wf)
            e_detail = {}

        # Weighted overall (normalized)
        wsum = max(alpha + beta + gamma, 1e-9)
        overall = round((alpha * s + beta * m + gamma * e) / wsum, 2)

        rows.append({
            "id": fp.parent.name,
            "S": s,
            "M": m,
            "E": e,
            "Overall": overall,
            "IntentSource": source,  # "rule" | "rule+llm" or empty
        })

        # Optional per-task JSON dump
        if dump_details:
            detail_path = fp.parent / "veriflow_detail.json"
            payload = {
                "S": s, "M": m, "E": e, "Overall": overall,
                "struct_detail": s_detail or {},
                "semantic_detail": m_detail or {}, 
                "exec_detail": e_detail or {},
                "executed_path": (e_detail or {}).get("executed_nodes_readable", []),
            }
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[ok] wrote {out}")

if __name__ == "__main__":
    app()