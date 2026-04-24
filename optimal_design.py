"""
ECE 284 Project 1 — Find Best Optimal Design per Benchmark
Considers both CPI and Cost using the evaluation function:

    EvalScore = alpha * CPI_norm + (1 - alpha) * Cost_norm

    Lower EvalScore = better design.

Put this script in the same folder as CSV files.
"""

import os
import numpy as np
import pandas as pd

# Settings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BENCH_CSVS = {
    "401.bzip2": os.path.join(SCRIPT_DIR, "sweep_results_bzip2.csv"),
    "456.hmmer": os.path.join(SCRIPT_DIR, "sweep_results_hmmer.csv"),
    "470.lbm":   os.path.join(SCRIPT_DIR, "sweep_results_lbm.csv"),
    "429.mcf":   os.path.join(SCRIPT_DIR, "sweep_results_mcf.csv"),
    "458.sjeng": os.path.join(SCRIPT_DIR, "sweep_results_sjeng.csv"),
}

BENCHMARKS = ["401.bzip2", "429.mcf", "456.hmmer", "458.sjeng", "470.lbm"]

BASELINE_CPI = {
    "401.bzip2": 1.321,
    "429.mcf":   1.780,
    "456.hmmer": 1.003,
    "458.sjeng": 1.939,
    "470.lbm":   1.801,
}

ALPHA = 0.5   # 0.5 = equal weight to CPI and cost
              # Change to 0.7 to prioritize performance, 0.3 to prioritize cost

# Cost function

L1_COST_PER_KB = 4.0
L2_COST_PER_KB = 1.0
ASSOC_FACTOR   = 0.15
BLOCK_PENALTY  = 0.005

def to_kb(s):
    s = str(s).strip()
    if "MB" in s: return int(s.replace("MB", "")) * 1024
    if "kB" in s: return int(s.replace("kB", ""))
    return int(s)

def compute_cost(row):
    l1d_kb    = to_kb(row["l1d"])
    l1i_kb    = to_kb(row["l1i"])
    l2_kb     = to_kb(row["l2"])
    l1d_assoc = int(row["l1d_assoc"])
    l1i_assoc = int(row["l1i_assoc"])
    l2_assoc  = int(row["l2_assoc"])
    blk       = int(row["block_size"])

    cost_l1d = l1d_kb * L1_COST_PER_KB * (1 + ASSOC_FACTOR * np.log2(max(l1d_assoc, 1)))
    cost_l1i = l1i_kb * L1_COST_PER_KB * (1 + ASSOC_FACTOR * np.log2(max(l1i_assoc, 1)))
    cost_l2  = l2_kb  * L2_COST_PER_KB * (1 + ASSOC_FACTOR * np.log2(max(l2_assoc,  1)))
    block_cost = BLOCK_PENALTY * max(blk - 32, 0)
    return cost_l1d + cost_l1i + cost_l2 + block_cost

# Evaluation function

def compute_eval_score(df, alpha=ALPHA):
    cpi_min,  cpi_max  = df["CPI"].min(),  df["CPI"].max()
    cost_min, cost_max = df["cost"].min(), df["cost"].max()

    cpi_norm  = (df["CPI"]  - cpi_min)  / (cpi_max  - cpi_min  + 1e-12)
    cost_norm = (df["cost"] - cost_min) / (cost_max - cost_min + 1e-12)

    return alpha * cpi_norm + (1 - alpha) * cost_norm

# Load & evaluate data

print("Loading and evaluating all benchmarks...\n")

frames = {}
for bench, path in BENCH_CSVS.items():
    df = pd.read_csv(path)
    df["CPI"]        = df["CPI"].astype(float)
    df["block_size"] = df["block_size"].astype(int)
    df["l1d_assoc"]  = df["l1d_assoc"].astype(int)
    df["l1i_assoc"]  = df["l1i_assoc"].astype(int)
    df["l2_assoc"]   = df["l2_assoc"].astype(int)
    df["cost"]       = df.apply(compute_cost, axis=1)
    df["eval_score"] = compute_eval_score(df)
    frames[bench]    = df

# Print results

SEP = "=" * 68

print(SEP)
print(f"  BEST OPTIMAL DESIGN PER BENCHMARK  (alpha = {ALPHA})")
print(f"  EvalScore = {ALPHA} x CPI_norm + {1-ALPHA} x Cost_norm")
print(SEP)

for bench in BENCHMARKS:
    df   = frames[bench]
    best = df.loc[df["eval_score"].idxmin()]

    cpi_improve  = (BASELINE_CPI[bench] - best["CPI"]) / BASELINE_CPI[bench] * 100

    # rank of this config by CPI alone and cost alone
    cpi_rank  = (df["CPI"]  <= best["CPI"]).sum()
    cost_rank = (df["cost"] <= best["cost"]).sum()

    print(f"\n  Benchmark   : {bench}")
    print(f"  {'─'*50}")
    print(f"  EvalScore   : {best['eval_score']:.4f}  (lower = better)")
    print(f"  CPI         : {best['CPI']:.4f}  "
          f"(baseline {BASELINE_CPI[bench]:.3f}, -{cpi_improve:.1f}% improvement)")
    print(f"  Cost        : {best['cost']:.1f} arb. units")
    print(f"  {'─'*50}")
    print(f"  L1D size    : {best['l1d']}")
    print(f"  L1I size    : {best['l1i']}")
    print(f"  L2 size     : {best['l2']}")
    print(f"  L1D assoc   : {best['l1d_assoc']}-way")
    print(f"  L1I assoc   : {best['l1i_assoc']}-way")
    print(f"  L2 assoc    : {best['l2_assoc']}-way")
    print(f"  Block size  : {best['block_size']}B")
    print(f"  {'─'*50}")
    print(f"  CPI rank    : #{df['CPI'].rank().loc[best.name]:.0f} "
          f"out of {len(df)} configs  (1 = best CPI)")
    print(f"  Cost rank   : #{df['cost'].rank().loc[best.name]:.0f} "
          f"out of {len(df)} configs  (1 = cheapest)")

# Summary comparison table

print(f"\n\n{SEP}")
print("  SUMMARY TABLE")
print(SEP)
print(f"\n  {'Benchmark':<14} {'CPI':>8} {'vs Base':>8} {'Cost':>8} "
      f"{'EvalScore':>10} {'L1D':>7} {'L1I':>7} {'L2':>5} "
      f"{'DA':>4} {'IA':>4} {'L2A':>5} {'Blk':>5}")
print(f"  {'-'*100}")

for bench in BENCHMARKS:
    df   = frames[bench]
    best = df.loc[df["eval_score"].idxmin()]
    imp  = (BASELINE_CPI[bench] - best["CPI"]) / BASELINE_CPI[bench] * 100
    print(f"  {bench:<14} {best['CPI']:>8.4f} {f'-{imp:.1f}%':>8} "
          f"{best['cost']:>8.1f} {best['eval_score']:>10.4f} "
          f"{best['l1d']:>7} {best['l1i']:>7} {best['l2']:>5} "
          f"{best['l1d_assoc']:>4} {best['l1i_assoc']:>4} "
          f"{best['l2_assoc']:>5} {best['block_size']:>5}")

print(f"\n  DA = L1D associativity, IA = L1I associativity, L2A = L2 associativity")
print(f"  Blk = block size in bytes\n")