#!/usr/bin/env python3
import os, json, glob, csv, argparse
from veriflow.semantic.matcher import semantic_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="bench/semantic", help="bench directory containing S*/")
    ap.add_argument("--out", default="experiments/results/semantic_bench.csv")
    ap.add_argument("--use-llm", action="store_true")
    args = ap.parse_args()

    cases = sorted(glob.glob(os.path.join(args.bench, "S*")))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["case","score","min_score","max_score","pass","issues"])
        for c in cases:
            with open(os.path.join(c,"prompt.txt"),"r",encoding="utf-8") as f:
                prompt = f.read().strip()
            with open(os.path.join(c,"workflow.json"),"r",encoding="utf-8") as f:
                wf = json.load(f)
            with open(os.path.join(c,"gold.json"),"r",encoding="utf-8") as f:
                gold = json.load(f)

            score, issues, detail = semantic_score(wf, prompt, use_llm=args.use_llm)
            mn = gold.get("min_score",0.0)
            mx = gold.get("max_score",1.0)
            ok = (score >= mn) and (score <= mx)
            w.writerow([os.path.basename(c), score, mn, mx, ok, "; ".join(issues)])

    print(f"Wrote {args.out} with {len(cases)} cases.")

if __name__ == "__main__":
    main()
