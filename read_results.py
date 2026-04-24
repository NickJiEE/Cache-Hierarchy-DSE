import csv

def find_optimal(csv_file):
    # Read all results
    results = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["CPI"] = float(row["CPI"])
            results.append(row)

    # Get unique benchmarks
    benchmarks = sorted(set(r["benchmark"] for r in results))

    print("=" * 70)
    print("OPTIMAL CONFIGURATIONS PER BENCHMARK")
    print("=" * 70)

    for bench in benchmarks:
        # Filter results for this benchmark
        bench_results = [r for r in results if r["benchmark"] == bench]

        # Find minimum CPI
        best = min(bench_results, key=lambda x: x["CPI"])

        print(f"\nBenchmark: {bench}")
        print(f"  CPI:          {best['CPI']}")
        print(f"  L1D size:     {best['l1d']}")
        print(f"  L1I size:     {best['l1i']}")
        print(f"  L2 size:      {best['l2']}")
        print(f"  L1D assoc:    {best['l1d_assoc']}")
        print(f"  L1I assoc:    {best['l1i_assoc']}")
        print(f"  L2 assoc:     {best['l2_assoc']}")
        print(f"  Block size:   {best['block_size']}")
        print(f"  IL1 misses:   {best['IL1_misses']}")
        print(f"  DL1 misses:   {best['DL1_misses']}")
        print(f"  L2 misses:    {best['L2_misses']}")
        print(f"  CPI vs baseline: {best['CPI']:.3f} (baseline from Part 2)")

    # Top 3 per benchmark
    print("\n" + "=" * 70)
    print("TOP 3 CONFIGS PER BENCHMARK")
    print("=" * 70)

    for bench in benchmarks:
        bench_results = [r for r in results if r["benchmark"] == bench]
        top3 = sorted(bench_results, key=lambda x: x["CPI"])[:3]

        print(f"\nBenchmark: {bench}")
        for i, r in enumerate(top3, 1):
            print(f"  #{i}: CPI={r['CPI']} | "
                  f"l1d={r['l1d']} l1i={r['l1i']} l2={r['l2']} | "
                  f"l1da={r['l1d_assoc']} l1ia={r['l1i_assoc']} "
                  f"l2a={r['l2_assoc']} blk={r['block_size']}")

if __name__ == "__main__":
    find_optimal("sweep_results_lbm.csv")