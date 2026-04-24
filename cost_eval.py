"""
ECE 284 Project 1 — Optimize Caches for Performance/Cost

Evaluation function:
    EvalScore = alpha * CPI_norm + (1 - alpha) * Cost_norm

    Where:
      - CPI_norm  = (CPI - CPI_min)  / (CPI_max  - CPI_min)   [0=best, 1=worst]
      - Cost_norm = (Cost - Cost_min) / (Cost_max - Cost_min)  [0=best, 1=worst]
      - alpha = 0.5  (equal weight to performance and cost)

    Lower EvalScore = better design.

Input CSVs expected in the same directory:
    sweep_results_bzip2.csv
    sweep_results_hmmer.csv
    sweep_results_lbm.csv
    sweep_results_mcf.csv
    sweep_results_sjeng.csv

Outputs:
    fig_part5_scatter.png     — CPI vs Cost scatter with optimal points marked
    fig_part5_pareto.png      — Pareto front per benchmark
    fig_part5_alpha.png       — How optimal config changes with alpha
    fig_part5_bar.png         — Comparison: best CPI vs best balanced vs cheapest
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BENCH_CSVS = {
    "401.bzip2": os.path.join(SCRIPT_DIR, "sweep_results_bzip2.csv"),
    "456.hmmer": os.path.join(SCRIPT_DIR, "sweep_results_hmmer.csv"),
    "470.lbm":   os.path.join(SCRIPT_DIR, "sweep_results_lbm.csv"),
    "429.mcf":   os.path.join(SCRIPT_DIR, "sweep_results_mcf.csv"),
    "458.sjeng": os.path.join(SCRIPT_DIR, "sweep_results_sjeng.csv"),
}

BENCHMARKS = ["401.bzip2", "429.mcf", "456.hmmer", "458.sjeng", "470.lbm"]
COLORS     = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
BMAP       = dict(zip(BENCHMARKS, COLORS))

BASELINE_CPI = {
    "401.bzip2": 1.321,
    "429.mcf":   1.780,
    "456.hmmer": 1.003,
    "458.sjeng": 1.939,
    "470.lbm":   1.801,
}

ALPHA = 0.5   # weight for CPI vs cost, change this to explore tradeoffs

# Cost function
L1_COST_PER_KB = 4.0
L2_COST_PER_KB = 1.0
ASSOC_FACTOR   = 0.15
BLOCK_PENALTY  = 0.005

def to_kb(s):
    s = str(s).strip()
    if "MB" in s: return int(s.replace("MB","")) * 1024
    if "kB" in s: return int(s.replace("kB",""))
    return int(s)

def compute_cost(row):
    l1d_kb    = to_kb(row["l1d"])
    l1i_kb    = to_kb(row["l1i"])
    l2_kb     = to_kb(row["l2"])
    l1d_assoc = int(row["l1d_assoc"])
    l1i_assoc = int(row["l1i_assoc"])
    l2_assoc  = int(row["l2_assoc"])
    blk       = int(row["block_size"])

    cost_l1d = l1d_kb * L1_COST_PER_KB * (1 + ASSOC_FACTOR * np.log2(max(l1d_assoc,1)))
    cost_l1i = l1i_kb * L1_COST_PER_KB * (1 + ASSOC_FACTOR * np.log2(max(l1i_assoc,1)))
    cost_l2  = l2_kb  * L2_COST_PER_KB * (1 + ASSOC_FACTOR * np.log2(max(l2_assoc, 1)))
    block_cost = BLOCK_PENALTY * max(blk - 32, 0)
    return cost_l1d + cost_l1i + cost_l2 + block_cost

# Evaluation function
def eval_score(df, alpha=ALPHA):
    """Lower is better. Returns a series of scores."""
    cpi_min,  cpi_max  = df["CPI"].min(),  df["CPI"].max()
    cost_min, cost_max = df["cost"].min(), df["cost"].max()

    cpi_norm  = (df["CPI"]  - cpi_min)  / (cpi_max  - cpi_min  + 1e-12)
    cost_norm = (df["cost"] - cost_min) / (cost_max - cost_min + 1e-12)

    return alpha * cpi_norm + (1 - alpha) * cost_norm

# Load & compute data
print("Loading data...")
frames = {}
for bench, path in BENCH_CSVS.items():
    df = pd.read_csv(path)
    df["CPI"]        = df["CPI"].astype(float)
    df["block_size"] = df["block_size"].astype(int)
    df["l1d_assoc"]  = df["l1d_assoc"].astype(int)
    df["l1i_assoc"]  = df["l1i_assoc"].astype(int)
    df["l2_assoc"]   = df["l2_assoc"].astype(int)
    df["cost"]       = df.apply(compute_cost, axis=1)
    df["eval_score"] = eval_score(df)
    frames[bench]    = df

def save(name):
    path = os.path.join(SCRIPT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {name}")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — CPI vs Cost scatter with optimal points
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle(
    f"Part 5 — CPI vs Cost Tradeoff  (EvalScore = {ALPHA}×CPI_norm + {1-ALPHA}×Cost_norm)",
    fontsize=12, fontweight="bold"
)

for ax, bench in zip(axes, BENCHMARKS):
    df = frames[bench]

    best_cpi      = df.loc[df["CPI"].idxmin()]
    best_balanced = df.loc[df["eval_score"].idxmin()]
    best_cost     = df.loc[df["cost"].idxmin()]

    sc = ax.scatter(df["cost"], df["CPI"],
                    c=df["eval_score"], cmap="RdYlGn_r",
                    alpha=0.55, s=35, edgecolors="none")

    ax.scatter(best_cpi["cost"], best_cpi["CPI"],
               color="blue", s=120, zorder=6, marker="*", label="Best CPI")
    ax.scatter(best_balanced["cost"], best_balanced["CPI"],
               color="red", s=100, zorder=6, marker="D", label="Best Balanced")
    ax.scatter(best_cost["cost"], best_cost["CPI"],
               color="orange", s=100, zorder=6, marker="^", label="Cheapest")

    ax.set_title(bench.split(".")[1], fontsize=11, fontweight="bold")
    ax.set_xlabel("Cost (arb. units)", fontsize=9)
    ax.set_ylabel("CPI", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    plt.colorbar(sc, ax=ax, label="Eval Score")

plt.tight_layout()
save("fig_part5_scatter.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Pareto front (non-dominated configs)
# ══════════════════════════════════════════════════════════════════════════════
def pareto_front(df):
    """Return rows that are not dominated (lower CPI AND lower cost)."""
    pts = df[["cost","CPI"]].values
    mask = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if mask[i]:
            # dominated if someone else is <= on both axes and < on at least one
            dominated = np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)
            dominated[i] = False
            mask[dominated] = False
    return df[mask].sort_values("cost")

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("Part 5 — Pareto Front: CPI vs Cost",
             fontsize=13, fontweight="bold")

for ax, bench in zip(axes, BENCHMARKS):
    df     = frames[bench]
    pareto = pareto_front(df)
    best   = df.loc[df["eval_score"].idxmin()]

    ax.scatter(df["cost"], df["CPI"],
               color="lightgray", alpha=0.4, s=25, zorder=1, label="All configs")
    ax.scatter(pareto["cost"], pareto["CPI"],
               color=BMAP[bench], s=60, zorder=3, label="Pareto front")
    ax.plot(pareto["cost"], pareto["CPI"],
            color=BMAP[bench], linewidth=1.8, zorder=2)
    ax.scatter(best["cost"], best["CPI"],
               color="red", s=120, zorder=5, marker="D", label="Best balanced")

    ax.set_title(bench.split(".")[1], fontsize=11, fontweight="bold")
    ax.set_xlabel("Cost (arb. units)", fontsize=9)
    ax.set_ylabel("CPI", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

plt.tight_layout()
save("fig_part5_pareto.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — How optimal config changes with alpha
# ══════════════════════════════════════════════════════════════════════════════
alphas = np.linspace(0, 1, 21)  # 0=cost only, 1=perf only

fig, axes = plt.subplots(2, 5, figsize=(20, 7))
fig.suptitle("Part 5 — Optimal CPI and Cost vs Alpha Weight\n"
             "(alpha=0: cost-only, alpha=1: performance-only)",
             fontsize=12, fontweight="bold")

for col, bench in enumerate(BENCHMARKS):
    df  = frames[bench]
    ax_cpi  = axes[0][col]
    ax_cost = axes[1][col]

    best_cpis  = []
    best_costs = []

    for a in alphas:
        scores = eval_score(df, alpha=a)
        idx    = scores.idxmin()
        best_cpis.append(df.loc[idx, "CPI"])
        best_costs.append(df.loc[idx, "cost"])

    ax_cpi.plot(alphas, best_cpis, color=BMAP[bench], linewidth=2.5, marker="o",
                markersize=4, markerfacecolor="white", markeredgewidth=1.5)
    ax_cpi.axhline(BASELINE_CPI[bench], color="gray", linestyle="--",
                   linewidth=1, label="Baseline")
    ax_cpi.set_title(bench.split(".")[1], fontsize=11, fontweight="bold")
    ax_cpi.set_ylabel("Optimal CPI")
    ax_cpi.set_xlabel("Alpha (CPI weight)")
    ax_cpi.legend(fontsize=7)
    ax_cpi.grid(alpha=0.3)

    ax_cost.plot(alphas, best_costs, color=BMAP[bench], linewidth=2.5,
                 linestyle="--", marker="s", markersize=4,
                 markerfacecolor="white", markeredgewidth=1.5)
    ax_cost.set_ylabel("Optimal Cost")
    ax_cost.set_xlabel("Alpha (CPI weight)")
    ax_cost.grid(alpha=0.3)

plt.tight_layout()
save("fig_part5_alpha.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — 3-way bar: best CPI vs best balanced vs cheapest
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Part 5 — CPI and Cost for Three Design Strategies",
             fontsize=13, fontweight="bold")

x  = np.arange(len(BENCHMARKS))
w  = 0.25
short = [b.split(".")[1] for b in BENCHMARKS]

cpi_best_perf     = [frames[b].loc[frames[b]["CPI"].idxmin(),  "CPI"]  for b in BENCHMARKS]
cpi_best_balanced = [frames[b].loc[frames[b]["eval_score"].idxmin(), "CPI"]  for b in BENCHMARKS]
cpi_cheapest      = [frames[b].loc[frames[b]["cost"].idxmin(), "CPI"]  for b in BENCHMARKS]

cost_best_perf     = [frames[b].loc[frames[b]["CPI"].idxmin(),  "cost"] for b in BENCHMARKS]
cost_best_balanced = [frames[b].loc[frames[b]["eval_score"].idxmin(), "cost"] for b in BENCHMARKS]
cost_cheapest      = [frames[b].loc[frames[b]["cost"].idxmin(), "cost"] for b in BENCHMARKS]

# CPI subplot
ax = axes[0]
ax.bar(x - w, cpi_best_perf,     w, label="Best CPI",      color="#2196F3", alpha=0.85)
ax.bar(x,     cpi_best_balanced, w, label="Best Balanced",  color="#FF5722", alpha=0.85)
ax.bar(x + w, cpi_cheapest,      w, label="Cheapest",       color="#4CAF50", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
ax.set_ylabel("CPI"); ax.set_title("CPI Comparison")
ax.legend(); ax.grid(axis="y", alpha=0.3)

# Cost subplot
ax = axes[1]
ax.bar(x - w, cost_best_perf,     w, label="Best CPI",      color="#2196F3", alpha=0.85)
ax.bar(x,     cost_best_balanced, w, label="Best Balanced",  color="#FF5722", alpha=0.85)
ax.bar(x + w, cost_cheapest,      w, label="Cheapest",       color="#4CAF50", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
ax.set_ylabel("Cost (arb. units)"); ax.set_title("Cost Comparison")
ax.legend(); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save("fig_part5_bar.png")

# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print(f"PART 5 OPTIMAL CONFIGS  (alpha={ALPHA})")
print("="*72)
print(f"\n{'Benchmark':<14} {'CPI':>7} {'Cost':>8} {'EvalScore':>10} "
      f"{'L1D':>7} {'L1I':>7} {'L2':>5} "
      f"{'AssocD':>7} {'AssocI':>7} {'AssocL2':>8} {'Blk':>5}")
print("-"*100)

for bench in BENCHMARKS:
    df   = frames[bench]
    best = df.loc[df["eval_score"].idxmin()]
    print(f"{bench:<14} {best['CPI']:>7.4f} {best['cost']:>8.1f} "
          f"{best['eval_score']:>10.4f} "
          f"{best['l1d']:>7} {best['l1i']:>7} {best['l2']:>5} "
          f"{best['l1d_assoc']:>7} {best['l1i_assoc']:>7} "
          f"{best['l2_assoc']:>8} {best['block_size']:>5}")

print("\n" + "="*72)
print("COMPARISON: Best CPI vs Best Balanced vs Cheapest")
print("="*72)
print(f"\n{'Benchmark':<14} | {'Best CPI':^20} | {'Best Balanced':^20} | {'Cheapest':^20}")
print(f"{'':14} | {'CPI':>8} {'Cost':>10} | {'CPI':>8} {'Cost':>10} | {'CPI':>8} {'Cost':>10}")
print("-"*80)

for bench in BENCHMARKS:
    df = frames[bench]
    bp = df.loc[df["CPI"].idxmin()]
    bb = df.loc[df["eval_score"].idxmin()]
    bc = df.loc[df["cost"].idxmin()]
    print(f"{bench:<14} | {bp['CPI']:>8.4f} {bp['cost']:>10.1f} | "
          f"{bb['CPI']:>8.4f} {bb['cost']:>10.1f} | "
          f"{bc['CPI']:>8.4f} {bc['cost']:>10.1f}")

print("\nDone! All Part 5 figures saved.")