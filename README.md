# 🔍 Veriflow: Structural–Semantic–Executable Verification for LLM-based Low-Code Workflows

**Veriflow** is a lightweight verification framework for **llm-based low-code workflow systems** such as [n8n](https://n8n.io).  
It aims to bridge human-language task specifications and formal workflow validation through a hybrid pipeline combining structural analysis, semantic intent recognition, and sandbox-level executability simulation.
It provides **formal-inspired consistency checking** across three complementary dimensions:

- 🧩 **Structural** – graph integrity and soundness  
- 💡 **Semantic** – intent alignment and node-type adequacy (rule + LLM hybrid)  
- ⚙️ **Executable** – sandbox-based simulation and reachability validation  

The framework also includes **publication-ready visualization** tools for workflow DAGs and a **batch evaluation CLI** for large-scale benchmarks.

---

## 🌐 Overview

```
Natural Language Prompt
        ↓
Intent Extraction (LLM + Rule)
        ↓
Workflow Graph (n8n JSON)
        ↓
Structural / Semantic / Executable Analysis
        ↓
JSON Report + Visualization
```

**Veriflow** bridges low-code workflows and formal verification by:
- Extracting **directed acyclic graph (DAG)** structures from n8n workflows;
- Computing **multi-criteria structural metrics**;
- Checking **semantic alignment** via rule-based and LLM-assisted intent recognition;
- Executing workflows in a **safe sandbox** (no external API calls);
- Producing detailed **JSON reports** and **graphical DAG visualizations**.

---

## 🧠 Core Features

| Category                     | Description                                                                                     |
| ---------------------------- | ----------------------------------------------------------------------------------------------- |
| **Structural analysis**      | Connectivity, acyclicity, orphan-ratio, out-degree balance, and exit coverage.                  |
| **Semantic checking**        | Hybrid rule + LLM mode for intent extraction (trigger, action, order, etc.).                    |
| **Executability simulation** | Sandbox execution without network calls; detects missing parameters or unreachable nodes.       |
| **Hybrid scoring**           | Weighted aggregation `Overall = α·S + β·M + γ·E` with normalized weights.                       |
| **Batch benchmarking**       | Evaluate multiple workflows under `bench/`; export per-case reports and CSV summaries.          |
| **Visualization**            | Generate publication-quality DAGs with rounded nodes, shadows, and highlighted execution paths. |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/ahzm/veriflow.git
cd veriflow
```

### 2. Create environment
```bash
conda env create -f environment.yml
conda activate veriflow
```

#### Dependencies:
- python=3.10
- networkx, matplotlib, pandas, typer, rich, openai, tiktoken

## Usage
### 1. Verify a single workflow
```bash
python -m veriflow.cli verify \
  --input bench/T001/gold.json \
  --prompt "$(cat bench/T001/prompt.txt)" \
  --use-llm --sandbox --report experiments/results/T001_detail.json -v
```
#### Output Example
```
StructuralScore:   0.99
SemanticScore:     1.00
ExecutabilityScore:1.00
Overall:           1.00
[ok] wrote report to experiments/results/T001_detail.json
```

### 2. Visualize a workflow DAG
```bash
python scripts/plot_dag.py \
  -i bench/T001/gold.json \
  -o experiments/results/T001_dag.png
```

## 🧩 Architecture

```
veriflow/
├── veriflow/                   # Core framework
│   ├── cli.py                  # Main CLI entry (verify / bench commands)
│   ├── structural/             # Structural validation
│   │   ├── checker.py          # Static schema checking
│   │   └── schema.py           # Workflow schema definitions
│   ├── semantic/               # Semantic consistency
│   │   ├── intent_extractor.py # Intent extraction
│   │   └── matcher.py          # keyword matching
│   ├── executable/             # Executability validation
│   │   ├── sandbox.py          # validate workflow in sandbox
│   │   └── dryrun.py           # Dry-run simulation
│   └── utils/                  # Shared helpers
│       ├── io.py               # JSON & figure I/O utilities
│       ├── logger.py           # unified logging configuration
│       └── graph.py            # Build and traverse workflow DAG
│
├── bench/                      # VeriFlow-Bench dataset
│   └── T001/                   # Example task
│       ├── prompt.txt          # Natural language prompt
│       └── gold.json           # Ground-truth workflow
│
├── experiments/                # Experimental results & configs
│   └── results/
│       ├── report.csv          # Aggregated scores
│       └── score_plot.png      # Visualization
│
├── scripts/                    # Utility scripts
│   ├── plot_dag.py             # Plot DAG
│   └── plot_results.py         # Plot S/M/E/Overall charts
│
├── environment.yml             # Reproducible environment
├── Makefile                    # Shortcut commands
├── LICENSE                     # MIT License
└── README.md                   # Project overview and usage
```

## 📈 Example Report Structure
```json
{
  "scores": { "S": 0.99, "M": 1.0, "E": 1.0, "Overall": 1.0 },
  "issues": [],
  "struct_detail": {
    "connected_ratio": 1.0,
    "acyclic": 1.0,
    "orphan_ratio": 0.0,
    "final_S": 0.99
  },
  "semantic_detail": {
    "intent_conf": 0.92,
    "source": "rule+llm",
    "intent": { "need_email": true, "need_http": true, "need_schedule": true }
  },
  "exec_detail": {
    "executed_nodes_readable": ["Schedule Trigger", "HTTP Request", "Email"],
    "runtime_ok": 1.0
  }
}
```

## 🧭 Milestones (Implemented)
- ✅ Structural metrics with robustness for small DAGs
- ✅ Hybrid semantic mode (rule + LLM)
- ✅ Sandbox execution validator (parameter & reachability checks)
- ✅ Unified CLI with JSON export and verbose diagnostics
- ✅ Publication-grade DAG plotting with highlighted paths
- ✅ Logging and I/O utilities (veriflow.utils)
- ✅ Benchmark suite support (bench/*)
