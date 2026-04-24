"""
ECE 284 Project 1 — Cache DSE
Plotting script for visualizing results.

Input CSVs in the same directory as script:
    sweep_results.csv          — size sweep (all 5 benchmarks)
    sweep_results_bzip2.csv    — assoc/blocksize sweep for bzip2
    sweep_results_hmmer.csv    — assoc/blocksize sweep for hmmer
    sweep_results_lbm.csv      — assoc/blocksize sweep for lbm
    sweep_results_mcf.csv      — assoc/blocksize sweep for mcf
    sweep_results_sjeng.csv    — assoc/blocksize sweep for sjeng

Outputs (saved in same directory):
    fig1_l2_size_vs_cpi.png
    fig2_l1_size_vs_cpi.png
    fig3_blocksize_vs_cpi.png
    fig4_l1d_assoc_vs_cpi.png
    fig5_l2_assoc_vs_cpi.png
    fig6_baseline_vs_optimal.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SIZE_CSV = os.path.join(SCRIPT_DIR, "sweep_results.csv")

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

# Cost function
L1_COST_PER_KB  = 4.0   # arbitrary cost units per KB
L2_COST_PER_KB  = 1.0
ASSOC_FACTOR    = 0.15
BLOCK_PENALTY   = 0.005 # per byte of block size above 32B baseline

def compute_cost(row):
    def to_kb(s):
        s = str(s).strip()
        if "MB" in s: return int(s.replace("MB","")) * 1024
        if "kB" in s: return int(s.replace("kB",""))
        return int(s)

    l1d_kb = to_kb(row["l1d"])
    l1i_kb = to_kb(row["l1i"])
    l2_kb  = to_kb(row["l2"])
    blk    = int(row["block_size"])

    l1d_assoc = int(row["l1d_assoc"])
    l1i_assoc = int(row["l1i_assoc"])
    l2_assoc  = int(row["l2_assoc"])

    # Base size costs
    cost_l1d = l1d_kb * L1_COST_PER_KB
    cost_l1i = l1i_kb * L1_COST_PER_KB
    cost_l2  = l2_kb  * L2_COST_PER_KB

    # Associativity multiplier (log2 scaling — going 1→2 way costs same as 2→4)
    cost_l1d *= (1 + ASSOC_FACTOR * np.log2(max(l1d_assoc, 1)))
    cost_l1i *= (1 + ASSOC_FACTOR * np.log2(max(l1i_assoc, 1)))
    cost_l2  *= (1 + ASSOC_FACTOR * np.log2(max(l2_assoc,  1)))

    # Block size penalty
    block_cost = BLOCK_PENALTY * max(blk - 32, 0)

    return cost_l1d + cost_l1i + cost_l2 + block_cost

# Helpers

def to_kb(s):
    s = str(s).strip()
    if "MB" in s: return int(s.replace("MB","")) * 1024
    if "kB" in s: return int(s.replace("kB",""))
    return int(s)

def save(name):
    path = os.path.join(SCRIPT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {name}")

# Load data

print("Loading CSVs...")
size_df = pd.read_csv(SIZE_CSV)
size_df["CPI"] = size_df["CPI"].astype(float)
size_df["l1d_kb"]      = size_df["l1d"].apply(to_kb)
size_df["l1i_kb"]      = size_df["l1i"].apply(to_kb)
size_df["l2_kb"]       = size_df["l2"].apply(to_kb)
size_df["l1_total_kb"] = size_df["l1d_kb"] + size_df["l1i_kb"]

assoc_frames = {}
for bench, path in BENCH_CSVS.items():
    df = pd.read_csv(path)
    df["CPI"]        = df["CPI"].astype(float)
    df["block_size"] = df["block_size"].astype(int)
    df["l1d_assoc"]  = df["l1d_assoc"].astype(int)
    df["l1i_assoc"]  = df["l1i_assoc"].astype(int)
    df["l2_assoc"]   = df["l2_assoc"].astype(int)
    df["cost"]       = df.apply(compute_cost, axis=1)
    assoc_frames[bench] = df

assoc_df = pd.concat(assoc_frames.values(), ignore_index=True)

print("Generating plots...\n")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — L2 size vs CPI
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Fig 1 — Effect of L2 Cache Size on CPI",
             fontsize=13, fontweight="bold")

for ax, bench in zip(axes, BENCHMARKS):
    sub = size_df[size_df["benchmark"] == bench]
    grp = sub.groupby("l2_kb")["CPI"].mean().reset_index().sort_values("l2_kb")
    labels = [f"{v//1024}MB" for v in grp["l2_kb"]]
    ax.plot(labels, grp["CPI"], marker="o", color=BMAP[bench],
            linewidth=2.5, markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax.axhline(BASELINE_CPI[bench], color="gray", linestyle="--",
               linewidth=1.2, label="Baseline")
    ax.set_title(bench.split(".")[1], fontsize=11, fontweight="bold")
    ax.set_xlabel("L2 Size")
    ax.set_ylabel("CPI")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save("fig1_l2_size_vs_cpi.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — L1 total size vs CPI
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Fig 2 — Effect of Total L1 Cache Size on CPI",
             fontsize=13, fontweight="bold")

for ax, bench in zip(axes, BENCHMARKS):
    sub = size_df[size_df["benchmark"] == bench]
    grp = sub.groupby("l1_total_kb")["CPI"].mean().reset_index().sort_values("l1_total_kb")
    labels = [f"{v}kB" for v in grp["l1_total_kb"]]
    ax.plot(labels, grp["CPI"], marker="s", color=BMAP[bench],
            linewidth=2.5, markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax.axhline(BASELINE_CPI[bench], color="gray", linestyle="--",
               linewidth=1.2, label="Baseline")
    ax.set_title(bench.split(".")[1], fontsize=11, fontweight="bold")
    ax.set_xlabel("L1 Total Size")
    ax.set_ylabel("CPI")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save("fig2_l1_size_vs_cpi.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Block size vs CPI
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Fig 3 — Effect of Cache Block Size on CPI",
             fontsize=13, fontweight="bold")

for ax, bench in zip(axes, BENCHMARKS):
    sub = assoc_frames[bench]
    grp = sub.groupby("block_size")["CPI"].mean().reset_index().sort_values("block_size")
    labels = [f"{v}B" for v in grp["block_size"]]
    ax.plot(labels, grp["CPI"], marker="^", color=BMAP[bench],
            linewidth=2.5, markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax.set_title(bench.split(".")[1], fontsize=11, fontweight="bold")
    ax.set_xlabel("Block Size")
    ax.set_ylabel("CPI")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save("fig3_blocksize_vs_cpi.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — L1D associativity vs CPI
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Fig 4 — Effect of L1D Associativity on CPI",
             fontsize=13, fontweight="bold")

for ax, bench in zip(axes, BENCHMARKS):
    sub = assoc_frames[bench]
    grp = sub.groupby("l1d_assoc")["CPI"].mean().reset_index().sort_values("l1d_assoc")
    labels = [f"{v}-way" for v in grp["l1d_assoc"]]
    ax.plot(labels, grp["CPI"], marker="D", color=BMAP[bench],
            linewidth=2.5, markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax.set_title(bench.split(".")[1], fontsize=11, fontweight="bold")
    ax.set_xlabel("L1D Associativity")
    ax.set_ylabel("CPI")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save("fig4_l1d_assoc_vs_cpi.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — L2 associativity vs CPI
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Fig 5 — Effect of L2 Associativity on CPI",
             fontsize=13, fontweight="bold")

for ax, bench in zip(axes, BENCHMARKS):
    sub = assoc_frames[bench]
    grp = sub.groupby("l2_assoc")["CPI"].mean().reset_index().sort_values("l2_assoc")
    labels = [f"{v}-way" for v in grp["l2_assoc"]]
    ax.plot(labels, grp["CPI"], marker="o", color=BMAP[bench],
            linewidth=2.5, markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax.set_title(bench.split(".")[1], fontsize=11, fontweight="bold")
    ax.set_xlabel("L2 Associativity")
    ax.set_ylabel("CPI")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
save("fig5_l2_assoc_vs_cpi.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Baseline vs Optimal CPI bar chart
# ══════════════════════════════════════════════════════════════════════════════
optimal_cpi = {b: assoc_frames[b]["CPI"].min() for b in BENCHMARKS}

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(BENCHMARKS))
w = 0.35

bars1 = ax.bar(x - w/2, [BASELINE_CPI[b] for b in BENCHMARKS], w,
               label="Baseline (Part 2)", color="#B0BEC5",
               edgecolor="white", linewidth=1.2)
bars2 = ax.bar(x + w/2, [optimal_cpi[b] for b in BENCHMARKS], w,
               label="Optimized (Part 3)", color=COLORS,
               edgecolor="white", linewidth=1.2)

for b1, b2, bench in zip(bars1, bars2, BENCHMARKS):
    imp = (BASELINE_CPI[bench] - optimal_cpi[bench]) / BASELINE_CPI[bench] * 100
    ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.02,
            f"-{imp:.1f}%", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#2E7D32")

ax.set_xticks(x)
ax.set_xticklabels([b.split(".")[1] for b in BENCHMARKS], fontsize=11)
ax.set_ylabel("CPI", fontsize=12)
ax.set_title("Fig 6 — Baseline vs Optimized CPI per Benchmark",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(BASELINE_CPI.values()) * 1.15)

plt.tight_layout()
save("fig6_baseline_vs_optimal.png")

# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("OPTIMAL CONFIGURATIONS PER BENCHMARK")
print("="*72)
for bench in BENCHMARKS:
    df   = assoc_frames[bench]
    best = df.loc[df["CPI"].idxmin()]
    imp  = (BASELINE_CPI[bench] - best["CPI"]) / BASELINE_CPI[bench] * 100
    print(f"\n{bench}")
    print(f"  CPI         : {best['CPI']:.4f}  "
          f"(baseline {BASELINE_CPI[bench]:.3f}, -{imp:.1f}%)")
    print(f"  L1D / L1I   : {best['l1d']} / {best['l1i']}")
    print(f"  L2          : {best['l2']}")
    print(f"  Assoc D/I/L2: {best['l1d_assoc']} / "
          f"{best['l1i_assoc']} / {best['l2_assoc']}")
    print(f"  Block size  : {best['block_size']}B")
    print(f"  Cost        : {best['cost']:.1f}")

print("\nAll figures saved.")