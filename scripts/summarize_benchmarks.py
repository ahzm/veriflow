#!/usr/bin/env python3
# scripts/summarize_benchmarks.py
# Aggregate rule vs rule+LLM benchmark CSVs and compute improvements.

import pandas as pd
import os
import glob
import argparse


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV and ensure required columns exist."""
    df = pd.read_csv(path)
    # Sometimes the first column might not be named "id"
    if "id" not in df.columns:
        df.rename(columns={df.columns[0]: "id"}, inplace=True)

    # Some CSVs might miss "Overall" (older runs) -> recompute if needed
    needed = ["id", "S", "M", "E"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column '{col}'")

    if "Overall" not in df.columns:
        # Default equal weights
        df["Overall"] = (df["S"] + df["M"] + df["E"]) / 3.0

    return df[["id", "S", "M", "E", "Overall"]]


def compare_improvement(df_rule: pd.DataFrame, df_llm: pd.DataFrame, metric: str = "M") -> float:
    """Compute percentage of workflows where LLM improved a given metric."""
    merged = df_rule.merge(df_llm, on="id", suffixes=("_rule", "_llm"))
    if merged.empty:
        return 0.0
    improved = (merged[f"{metric}_llm"] > merged[f"{metric}_rule"]).sum()
    total = len(merged)
    return improved / total if total > 0 else 0.0


def summarize(name: str, df_rule: pd.DataFrame, df_llm: pd.DataFrame) -> pd.DataFrame:
    """Generate summary rows for one benchmark family (e.g., W10, W50, GenLLM_W50)."""
    means_rule = df_rule.mean(numeric_only=True)
    means_llm = df_llm.mean(numeric_only=True)

    improved_M = compare_improvement(df_rule, df_llm, "M") * 100.0
    improved_O = compare_improvement(df_rule, df_llm, "Overall") * 100.0

    # absolute improvement of means
    abs_M = means_llm["M"] - means_rule["M"]
    abs_O = means_llm["Overall"] - means_rule["Overall"]

    # relative improvement of means (in %)
    rel_M = (abs_M / means_rule["M"] * 100.0) if means_rule["M"] > 0 else None
    rel_O = (abs_O / means_rule["Overall"] * 100.0) if means_rule["Overall"] > 0 else None

    def r2(x):
        return round(x, 2)

    def r1(x):
        return round(x, 1)

    return pd.DataFrame(
        [
            [
                name,
                "rule",
                r2(means_rule["S"]),
                r2(means_rule["M"]),
                r2(means_rule["E"]),
                r2(means_rule["Overall"]),
                None,  # % improved M
                None,  # % improved Overall
                None,  # ΔM abs
                None,  # ΔM rel %
                None,  # ΔOverall abs
                None,  # ΔOverall rel %
            ],
            [
                name,
                "rule+LLM",
                r2(means_llm["S"]),
                r2(means_llm["M"]),
                r2(means_llm["E"]),
                r2(means_llm["Overall"]),
                r1(improved_M),
                r1(improved_O),
                r2(abs_M),
                r1(rel_M),
                r2(abs_O),
                r1(rel_O),
            ],
        ],
        columns=[
            "Benchmark",
            "Mode",
            "mean(S)",
            "mean(M)",
            "mean(E)",
            "mean(Overall)",
            "% improved M",
            "% improved Overall",
            "ΔM (abs)",
            "ΔM (%)",
            "ΔOverall (abs)",
            "ΔOverall (%)",
        ],
    )


def discover_pairs(base_dir: str):
    """
    Find pairs like:
        W10.csv <-> W10_rule.csv
        GenLLM_W50.csv <-> GenLLM_W50_rule.csv
    """
    pattern = os.path.join(base_dir, "*.csv")
    files = glob.glob(pattern)

    print(f"[debug] Searching CSVs in: {base_dir}")
    print(f"[debug] Found CSV files: {', '.join(os.path.basename(f) for f in files) or '<none>'}")

    pairs = {}

    for f in files:
        name = os.path.basename(f)
        base, _ = os.path.splitext(name)

        if base.endswith("_rule"):
            key = base[:-5]  # remove '_rule'
            pairs.setdefault(key, {})["rule"] = f
        else:
            key = base
            pairs.setdefault(key, {})["llm"] = f

    # debug: print detected pairs
    print("\n[debug] Detected pairs (key -> rule / llm):")
    for k, v in pairs.items():
        print(f"  {k}: {v}")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Summarize VeriFlow benchmarks (rule vs rule+LLM).")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="experiments/results",
        help="Directory containing CSV files (default: experiments/results)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="benchmark_summary.csv",
        help="Output CSV for summary (default: benchmark_summary.csv)",
    )
    args = parser.parse_args()

    pairs = discover_pairs(args.base_dir)
    all_rows = []

    for key, paths in pairs.items():
        if "rule" not in paths or "llm" not in paths:
            # skip incomplete pairs
            continue

        print(f"\n[info] Processing benchmark {key}...")
        print(f"       rule CSV: {paths['rule']}")
        print(f"       llm  CSV: {paths['llm']}")

        df_rule = load_csv(paths["rule"])
        df_llm = load_csv(paths["llm"])

        summary = summarize(key, df_rule, df_llm)
        all_rows.append(summary)

    if all_rows:
        final = pd.concat(all_rows, ignore_index=True)
        print("\n===== Benchmark Summary =====\n")
        print(final.to_string(index=False))

        out_path = args.out
        final.to_csv(out_path, index=False)
        print(f"\nSaved -> {out_path}")
    else:
        print("\n[warn] No complete (rule, rule+LLM) pairs found. "
              "Check base_dir and file naming (e.g., W10.csv + W10_rule.csv).")


if __name__ == "__main__":
    main()