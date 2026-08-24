"""How much do the losses move when NOTHING changes? (TODO 4, first item.)

Rejection ABC keeps the trials with the smallest distance D. That only infers anything if a trial
scores well BECAUSE its parameters are good. But every trial also rolls dice -- SLiM's forward
mating, recapitation, and the msprime mutation overlay are all stochastic -- so part of every score
is luck. If the luck component is comparable to the difference good parameters actually make, the
pass selects lucky draws rather than correct ones and returns a confident-looking posterior built
out of coincidences. The acceptance threshold epsilon must sit ABOVE this floor.

Method: hold every parameter fixed and re-run the pipeline R times with different seeds, then look
at the spread of the per-statistic losses. Deliberately routed through the REAL production code --
AnalyzeTreeSeq.analyze_tree_sequence() and ABCAnalysisNoRedis.calculate_losses() -- so this is also
an end-to-end integration test of the path a CHTC trial actually takes.

What is and is not varied, and why:
  VARIED   SLiM forward seed (-s), recapitation seed, mutation-overlay seed. These are exactly the
           three dice production rolls: Main.main() passes no -s at all, so every real trial
           already draws a fresh SLiM seed. (analyze_tree_sequence takes no seed either, so its
           recapitation and mutation overlay draw fresh dice on every call -- which is precisely
           what we want to measure.)
  FIXED    clustering and the migration matrix. Both are deterministic in production too (KMeans is
           pinned at KMEANS_SEED=42 and the kernel is a deterministic function of the distances), so
           re-deriving them would add no variance -- and re-running KMeans needs sklearn, which is
           blocked on this box (3). data/cluster_data.csv + data/migration_rates.csv are reused.

--fixed-tree: PARTIAL FLOOR, and know what it does and does not cover. Device Guard blocks slim.exe
on the dev box (3 -- broader than the sklearn-only note there), so the forward phase cannot be
re-run locally. This mode reuses out/simTreeSeq.trees and varies ONLY the recapitation and mutation
seeds, giving a strict LOWER BOUND on the floor. The two fitted statistics are affected very
differently, so do not read one from the other:
  pi  -- well covered. ~97.6% of pairwise coalescent time is in the ancestral phase (branch_div
         26847 of a 27448 ceiling, 6.1.1), so recapitation is where pi's variance lives.
  Fst -- only PARTLY covered. Fst is a forward-phase ratio and is insensitive to the ancestral
         phase (<1% over a 3x Ne step, 6.2.1), so most of its run-to-run variance is forward
         genealogy, which this mode holds fixed. Treat the Fst number here as a floor on the floor.

Reads out, per statistic: mean, sd, min-max, and the mean pairwise |difference| between replicates
-- that last one is the most directly interpretable "two identical runs differ by this much".

NOTE: writes data/Output_Data/ in place, like production (10). Per-replicate copies are kept under
out/noise_floor/rep<N>/ so nothing has to be re-run to compute a different summary later.

Usage (run from diagnostics/ -- paths are ../data, ../out):
    python noise_floor.py --popmult 5000 --reps 3
"""
import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
import time
import platform
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Python_Code"))
import ABCAnalysisNoRedis as ABC   # noqa: E402  (real calculate_losses / readers / mask)
import AnalyzeTreeSeq             # noqa: E402  (the real recapitate->mutate->stats path)

LOSSES = ["pi_loss", "fst_loss", "ibd_loss", "dxy_loss", "genrel_loss"]
FITTED = {"pi_loss", "fst_loss"}

# Across-prior signal ranges, for scale. pi/fst from 6.1.1 at the calibrated mu; the prior is
# U(2000, 12000), so the relevant comparison is the 2000->saturation range, NOT the wider
# 500->5000 sweep range that earlier sections quote.
SIGNAL = {
    "pi_loss":  (0.0290, 0.0209, "POPMULT 2000 -> saturated (6.1.1 + 7.2.1 floor)"),
    "fst_loss": (0.01856, 0.00830, "POPMULT 2000 -> 5000 (6.1.1)"),
}


def run_slim(popmult, recomb, seed):
    script = Path("../SLiM_Code/CPBSampleSim"
                  + ("Win" if platform.system() == "Windows" else "Linux") + ".slim")
    t0 = time.perf_counter()
    subprocess.run(["slim", "-l", "0", "-s", str(seed),
                    "-d", f"POPMULT={popmult}", "-d", f"RECOMB={recomb!r}",
                    str(script)], check=True)
    dt = time.perf_counter() - t0
    mb = Path("../out/simTreeSeq.trees").stat().st_size / 1024 / 1024
    print(f"  [slim] seed={seed} {dt:.1f}s trees={mb:.1f}MB", flush=True)
    return dt


def read_sim_outputs():
    """Exactly what model() reads back out of data/Output_Data (ABCAnalysisNoRedis.py:259-264)."""
    d = {}
    for year in ["2015", "2019", "2023"]:
        d[f"{year}_diversity"] = ABC._read_vector(Path(f"../data/Output_Data/diversities_{year}.csv"))
        d[f"{year}_divergence"] = ABC._read_matrix(Path(f"../data/Output_Data/divergences_{year}.csv"))
        d[f"{year}_fst"] = ABC._read_matrix(Path(f"../data/Output_Data/fst_{year}.csv"))
        d[f"{year}_relatedness"] = ABC._read_matrix(Path(f"../data/Output_Data/relatedness_{year}.csv"))
    return d


def main(a):
    for p in (Path("../data/cluster_data.csv"), Path("../data/migration_rates.csv")):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- this script reuses the clustering on disk "
                                    f"rather than re-running KMeans (see module docstring)")

    if not a.fixed_tree:
        # Fail loudly and early rather than after the first 20-minute recapitation (10).
        try:
            subprocess.run(["slim", "-v"], check=True, capture_output=True)
        except Exception as e:
            raise SystemExit(
                f"slim is not runnable here ({e}).\n"
                "On the dev box Device Guard blocks slim.exe, so the FORWARD phase cannot be\n"
                "re-run and a full noise floor is not obtainable. Use --fixed-tree for the\n"
                "partial (recapitation+mutation) floor, or run this where slim works.")
    elif not Path("../out/simTreeSeq.trees").exists():
        raise FileNotFoundError("--fixed-tree needs out/simTreeSeq.trees")

    obs = ABC.getObservedData()
    outdir = Path("../out/noise_floor")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in range(a.reps):
        seed = a.seed0 + r
        print(f"\n=== replicate {r+1}/{a.reps}  seed={seed}  POPMULT={a.popmult}"
              f"{'  [FIXED TREE -- recap+mutation only]' if a.fixed_tree else ''} ===", flush=True)
        t0 = time.perf_counter()
        slim_s = None if a.fixed_tree else run_slim(a.popmult, a.recomb, seed)

        print(f"  [{time.strftime('%H:%M:%S')}] recapitate + mutate + stats ...", flush=True)
        # The REAL production call. Note it does NOT take a seed: recapitation and the mutation
        # overlay draw from the global RNG, so each replicate gets fresh dice there too -- which is
        # what production does and therefore what we want to measure.
        AnalyzeTreeSeq.analyze_tree_sequence(mutation_rate=a.mu, recombination_rate=a.recomb,
                                             ancestral_Ne=a.anc_ne)
        sim = read_sim_outputs()
        losses = ABC.calculate_losses(obs, sim)
        dt = time.perf_counter() - t0

        repdir = outdir / f"rep{r+1}"
        repdir.mkdir(parents=True, exist_ok=True)
        for year in ["2015", "2019", "2023"]:
            for stat in ["diversities", "divergences", "fst", "relatedness"]:
                shutil.copy2(Path(f"../data/Output_Data/{stat}_{year}.csv"),
                             repdir / f"{stat}_{year}.csv")

        row = {"rep": r + 1, "seed": seed, "slim_s": slim_s, "total_s": dt, **losses}
        rows.append(row)
        print("  " + "  ".join(f"{k}={losses[k]:.5g}" for k in LOSSES), flush=True)
        print(f"  [{time.strftime('%H:%M:%S')}] replicate done in {dt/60:.1f} min", flush=True)

    # ---- summary -----------------------------------------------------------------------------
    print("\n" + "=" * 84)
    print(f"NOISE FLOOR  --  {a.reps} replicates, identical parameters "
          f"(POPMULT={a.popmult}, mu={a.mu:g}, anc_Ne={a.anc_ne})")
    if a.fixed_tree:
        print("PARTIAL (--fixed-tree): recapitation + mutation only, forward phase held FIXED.")
        print("  -> a strict LOWER BOUND. Good coverage for pi, poor for Fst (see docstring).")
    print("=" * 84)
    print(f"{'statistic':<12} {'mean':>10} {'sd':>10} {'min':>10} {'max':>10} "
          f"{'mean|diff|':>11} {'CV%':>7}")
    summary = {}
    for k in LOSSES:
        v = np.array([row[k] for row in rows], dtype=float)
        pair = [abs(x - y) for x, y in itertools.combinations(v, 2)]
        s = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
             "min": float(v.min()), "max": float(v.max()),
             "mean_pairwise_abs_diff": float(np.mean(pair)) if pair else 0.0,
             "cv_pct": float(100 * v.std(ddof=1) / v.mean()) if len(v) > 1 else 0.0}
        summary[k] = s
        tag = "*" if k in FITTED else " "
        print(f"{tag}{k:<11} {s['mean']:>10.5f} {s['sd']:>10.5f} {s['min']:>10.5f} "
              f"{s['max']:>10.5f} {s['mean_pairwise_abs_diff']:>11.5f} {s['cv_pct']:>7.2f}")
    print("  (* = fitted; the others are diagnostics and do not enter D)")

    print("\n--- Is the floor small enough for the fitted statistics? ---")
    for k, (lo, hi, note) in SIGNAL.items():
        rng = abs(lo - hi)
        noise = summary[k]["mean_pairwise_abs_diff"]
        ratio = noise / rng if rng else float("inf")
        verdict = ("OK -- signal well above noise" if ratio < 0.2 else
                   "MARGINAL -- noise is a large fraction of the signal" if ratio < 0.5 else
                   "PROBLEM -- noise comparable to or larger than the signal")
        print(f"  {k}: run-to-run {noise:.5f}  vs  across-prior range {rng:.5f}  "
              f"({100*ratio:.0f}%)  -> {verdict}")
        print(f"      range is {note}")
        summary[k]["signal_range"] = rng
        summary[k]["noise_over_signal"] = ratio

    rec = {"popmult": a.popmult, "mu": a.mu, "recomb": a.recomb, "anc_ne": a.anc_ne,
           "reps": a.reps, "seed0": a.seed0, "fixed_tree": a.fixed_tree,
           "partial": a.fixed_tree, "rows": rows, "summary": summary}
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    with open(outdir / "replicates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rep", "seed", "slim_s", "total_s"] + LOSSES)
        w.writeheader()
        w.writerows(rows)
    print(f"\nDONE -> {a.out} and {outdir}/replicates.csv", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--popmult", type=float, default=5000)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed0", type=int, default=1001)
    p.add_argument("--mu", type=float, default=4.646e-7)
    p.add_argument("--recomb", type=float, default=2.75e-6)
    p.add_argument("--anc-ne", type=int, default=6700)
    p.add_argument("--fixed-tree", action="store_true",
                   help="reuse out/simTreeSeq.trees; vary recapitation+mutation only (PARTIAL "
                        "floor -- required where slim.exe is blocked, see docstring)")
    p.add_argument("--out", default="../out/noise_floor.jsonl")
    main(p.parse_args())
