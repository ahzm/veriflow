import json, os, glob
import pytest
from veriflow.semantic.matcher import semantic_score

BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "bench", "semantic")

@pytest.mark.parametrize("case_dir", sorted(glob.glob(os.path.join(BENCH_DIR, "S*"))))
def test_semantic_bench_case(case_dir):
    prompt_path = os.path.join(case_dir, "prompt.txt")
    wf_path = os.path.join(case_dir, "workflow.json")
    gold_path = os.path.join(case_dir, "gold.json")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    with open(wf_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)
    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)

    score, issues, detail = semantic_score(workflow, prompt, use_llm=False)

    min_score = gold.get("min_score", 0.0)
    max_score = gold.get("max_score", 1.0)

    assert score >= min_score - 1e-6, f"{os.path.basename(case_dir)} score {score} < min_score {min_score}; issues={issues}"
    assert score <= max_score + 1e-6, f"{os.path.basename(case_dir)} score {score} > max_score {max_score}; issues={issues}"
