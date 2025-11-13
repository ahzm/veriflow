#scripts/plot_results.py
# Plot S/M/E/Overall from CSV using matplotlib (no seaborn, single figure).

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot VeriFlow scores from CSV.")
    parser.add_argument("--csv", type=Path, default=Path("experiments/results/report.csv"), help="Input CSV path")
    parser.add_argument("--out", type=Path, default=Path("experiments/results/score_plot.png"), help="Output PNG path")
    parser.add_argument("--show", action="store_true", help="Show the plot window")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "id" not in df.columns:
        df["id"] = [f"T{idx+1:03d}" for idx in range(len(df))]

    x = np.arange(len(df))

    plt.figure(figsize=(10, 5))

    # Slight vertical jitter to avoid perfect overlap
    jitter = 0.005

    # Use distinct colors and markers (clear in grayscale print)
    plt.scatter(x, df["S"] + np.random.uniform(-jitter, jitter, len(df)),
                label="S (Structural)", marker="o", facecolors="none", edgecolors="tab:blue")
    plt.scatter(x, df["M"] + np.random.uniform(-jitter, jitter, len(df)),
                label="M (Semantic)", marker="^", color="tab:orange")
    plt.scatter(x, df["E"] + np.random.uniform(-jitter, jitter, len(df)),
                label="E (Executable)", marker="s", color="tab:green")

    # Plot Overall as a solid line for emphasis
    plt.plot(x, df["Overall"], label="Overall", color="tab:red", linewidth=2)

    plt.xticks(x, df["id"], rotation=45, ha="right")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Task ID")
    plt.ylabel("Score")
    plt.title("VeriFlow Scores (S/M/E/Overall)")
    plt.legend()
    plt.grid(alpha=0.2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()