import subprocess
import os
import itertools

GEM5_DIR = os.path.expanduser("~/gem5")
SPEC_DIR = os.path.expanduser("~/Downloads/Project1_SPEC-master")

benchmarks = {
    "401.bzip2": {
        "bin": f"{SPEC_DIR}/401.bzip2/src/benchmark",
        "arg": f"{SPEC_DIR}/401.bzip2/data/input.program",
    },
    "429.mcf": {
        "bin": f"{SPEC_DIR}/429.mcf/src/benchmark",
        "arg": f"{SPEC_DIR}/429.mcf/data/inp.in",
    },
    "456.hmmer": {
        "bin": f"{SPEC_DIR}/456.hmmer/src/benchmark",
        "arg": f"{SPEC_DIR}/456.hmmer/data/bombesin.hmm.new",
    },
    "458.sjeng": {
        "bin": f"{SPEC_DIR}/458.sjeng/src/benchmark",
        "arg": f"{SPEC_DIR}/458.sjeng/data/test.txt",
    },
    "470.lbm": {
        "bin": f"{SPEC_DIR}/470.lbm/src/benchmark",
        "arg": f"20 reference.dat 0 1 {SPEC_DIR}/470.lbm/data/100_100_130_cf_a.of",
    },
}

# Parameter sweep space
l1d_sizes  = ["64kB", "128kB", "256kB", "512kB"]
l1i_sizes  = ["32kB", "64kB", "128kB", "256kB"]
l2_sizes   = ["1MB", "2MB", "4MB"]
#l1_assocs  = [1, 2, 4, 8]
#l2_assocs  = [1, 2, 4, 8]
#block_sizes = [32, 64, 128]

l1_assocs  = [2]
l2_assocs  = [1]
block_sizes = [64]

MAX_INST = 500000000

results = []

for bench_name, bench in benchmarks.items():
    for l1d, l1i, l2, l1da, l1ia, l2a, blk in itertools.product(
        l1d_sizes, l1i_sizes, l2_sizes, l1_assocs, l1_assocs, l2_assocs, block_sizes
    ):
        # Enforce 512KB L1 budget
        l1d_kb = int(l1d.replace("kB",""))
        l1i_kb = int(l1i.replace("kB",""))
        if l1d_kb + l1i_kb > 512:
            continue

        out_dir = (f"{SPEC_DIR}/{bench_name}/m5out/"
                   f"l1d{l1d}_l1i{l1i}_l2{l2}_"
                   f"l1da{l1da}_l1ia{l1ia}_l2a{l2a}_blk{blk}")
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            f"{GEM5_DIR}/build/X86/gem5.opt", "-d", out_dir,
            f"{GEM5_DIR}/configs/deprecated/example/se.py",
            "-c", bench["bin"],
            "-o", bench["arg"],
            "-I", str(MAX_INST),
            "--cpu-type=TimingSimpleCPU",
            "--caches", "--l2cache",
            f"--l1d_size={l1d}",
            f"--l1i_size={l1i}",
            f"--l2_size={l2}",
            f"--l1d_assoc={l1da}",
            f"--l1i_assoc={l1ia}",
            f"--l2_assoc={l2a}",
            f"--cacheline_size={blk}",
        ]

        print(f"Running {bench_name} | l1d={l1d} l1i={l1i} l2={l2} "
              f"l1da={l1da} l1ia={l1ia} l2a={l2a} blk={blk}")
        subprocess.run(cmd)

        # Parse stats
        stats_file = f"{out_dir}/stats.txt"
        il1_miss = dl1_miss = l2_miss = sim_insts = 0
        with open(stats_file) as f:
            for line in f:
                if "system.cpu.icache.overallMisses::total" in line:
                    il1_miss = int(line.split()[1])
                elif "system.cpu.dcache.overallMisses::total" in line:
                    dl1_miss = int(line.split()[1])
                elif "system.l2.overallMisses::total" in line:
                    l2_miss = int(line.split()[1])
                elif "simInsts" in line:
                    sim_insts = int(line.split()[1])

        cpi = 1 + ((il1_miss + dl1_miss)*6 + l2_miss*50) / sim_insts

        results.append({
            "benchmark": bench_name,
            "l1d": l1d, "l1i": l1i, "l2": l2,
            "l1d_assoc": l1da, "l1i_assoc": l1ia, "l2_assoc": l2a,
            "block_size": blk,
            "IL1_misses": il1_miss, "DL1_misses": dl1_miss,
            "L2_misses": l2_miss, "CPI": round(cpi, 4)
        })

# Write CSV
import csv
with open("sweep_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Results saved to sweep_results.csv")