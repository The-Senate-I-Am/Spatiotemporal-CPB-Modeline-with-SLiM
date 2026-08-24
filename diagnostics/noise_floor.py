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


def describe_inputs():
    """What landscape is actually on disk, and do the two files agree?

    The whole premise of this script is "nothing changed", so it must not silently inherit
    whatever cluster/migration files some earlier experiment left behind -- the pipeline
    overwrites data/ IN PLACE (CLAUDE.md 10). Read both inputs, cross-check them, and return them
    so every jsonl record says which landscape it was measured on.
    """
    cd = Path("../data/cluster_data.csv")
    mr = Path("../data/migration_rates.csv")
    for p in (cd, mr):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- this script reuses the clustering on disk "
                                    f"rather than re-running KMeans (see module docstring)")

    # cluster_data.csv has a header row; SLiM sets numSubpops = propDF.nrow (CPBSampleSim*.slim:27).
    with open(cd, newline="", encoding="utf-8") as f:
        n_demes = sum(1 for _ in csv.reader(f)) - 1

    # migration_rates.csv carries a header row AND an index column, both numeric-looking -- so a
    # naive loadtxt "works" and returns a matrix one bigger than the landscape. Strip both.
    with open(mr, newline="", encoding="utf-8") as f:
        M = np.array([r[1:] for r in list(csv.reader(f))[1:]], dtype=float)

    if M.shape != (n_demes, n_demes):
        raise ValueError(
            f"migration matrix is {M.shape} but cluster_data.csv has {n_demes} demes. The two "
            f"files are out of sync, so SLiM would run a landscape that does not match the "
            f"clustering. Regenerate both with Main.main() before measuring anything.")

    off = M.sum(1) - np.diag(M)
    tm = float(off.mean())
    if off.std() > 1e-3 * max(tm, 1e-12):
        raise ValueError(
            f"off-diagonal row sums are not constant (mean={tm:.6g}, sd={off.std():.3g}). Each "
            f"row must sum to exactly total_migration (CLAUDE.md 2) -- this matrix was not "
            f"produced by the current determine_migration_rates().")

    return {"n_demes": n_demes, "total_migration": tm}


def summarize(rows, meta):
    """Print the floor table and return it. Shared by the live run and --summarize."""
    print("\n" + "=" * 84)
    print(f"NOISE FLOOR  --  {len(rows)} replicates, identical parameters "
          f"(POPMULT={meta['popmult']}, mu={meta['mu']:g}, anc_Ne={meta['anc_ne']})")
    print(f"landscape: {meta['n_demes']} demes, total_migration={meta['total_migration']:.4g}")
    if meta.get("fixed_tree"):
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

    if len(rows) < 2:
        print("\n  Only one replicate -- sd and mean|diff| are meaningless. Pool more with:")
        print("      python noise_floor.py --summarize")
        return summary

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
    return summary


def do_summarize(a):
    """Pool replicate records across processes -- the 1-rep-per-job path (see --reps help)."""
    path = Path(a.out)
    if not path.exists():
        raise FileNotFoundError(f"{path} has no records yet")

    reps = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") != "replicate":
                continue                      # skip the per-process summary records
            if rec["popmult"] != a.popmult or bool(rec["fixed_tree"]) != bool(a.fixed_tree):
                continue
            reps.append(rec)
    if not reps:
        raise SystemExit(f"no replicate records in {path} at POPMULT={a.popmult}, "
                         f"fixed_tree={a.fixed_tree}")

    # Pooling only means anything if every replicate saw the same landscape and the same mu.
    for key in ("n_demes", "total_migration", "mu", "anc_ne"):
        vals = {r[key] for r in reps}
        if len(vals) > 1:
            raise SystemExit(f"refusing to pool: replicates disagree on {key} -> {sorted(vals)}")
    seeds = [r["seed"] for r in reps]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"refusing to pool: duplicate seeds {sorted(seeds)} -- those replicates "
                         f"re-draw the same genealogy, which would understate the floor")

    print(f"pooled {len(reps)} replicate records from {path} (seeds {sorted(seeds)})")
    summarize(reps, reps[0])


def main(a):
    meta = describe_inputs()
    print(f"landscape on disk: {meta['n_demes']} demes, "
          f"total_migration={meta['total_migration']:.4g}", flush=True)

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

    common = {"popmult": a.popmult, "mu": a.mu, "recomb": a.recomb, "anc_ne": a.anc_ne,
              "fixed_tree": a.fixed_tree, **meta}

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
        losses = ABC.calculate_losses(obs, sim)   # (observed, simulated) -- matches production
        dt = time.perf_counter() - t0

        repdir = outdir / f"rep{r+1}"
        repdir.mkdir(parents=True, exist_ok=True)
        for year in ["2015", "2019", "2023"]:
            for stat in ["diversities", "divergences", "fst", "relatedness"]:
                shutil.copy2(Path(f"../data/Output_Data/{stat}_{year}.csv"),
                             repdir / f"{stat}_{year}.csv")

        row = {"rep": r + 1, "seed": seed, "slim_s": slim_s, "total_s": dt, **losses}
        rows.append(row)

        # Append THIS replicate immediately. A 3-rep run is ~1.6 h at POPMULT=5000 and OSPool
        # evicts; writing only at the end would throw away every completed replicate when the job
        # dies on the last one. Each record stands alone, and --summarize pools them.
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "replicate", **common, **row}) + "\n")

        print("  " + "  ".join(f"{k}={losses[k]:.5g}" for k in LOSSES), flush=True)
        print(f"  [{time.strftime('%H:%M:%S')}] replicate done in {dt/60:.1f} min "
              f"-> appended to {a.out}", flush=True)

    summary = summarize(rows, common)

    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "summary", **common, "reps": a.reps, "seed0": a.seed0,
                            "partial": a.fixed_tree, "rows": rows, "summary": summary}) + "\n")

    with open(outdir / "replicates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rep", "seed", "slim_s", "total_s"] + LOSSES)
        w.writeheader()
        w.writerows(rows)
    print(f"\nDONE -> {a.out} and {outdir}/replicates.csv", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--popmult", type=float, default=5000)
    p.add_argument("--reps", type=int, default=3,
                   help="replicates in THIS process. On OSPool prefer --reps 1 across several "
                        "jobs (each ~30 min at POPMULT=5000) with a distinct --seed0 per job, "
                        "then pool with --summarize")
    p.add_argument("--seed0", type=int, default=1001,
                   help="first SLiM seed; replicate r uses seed0+r. MUST NOT overlap between "
                        "jobs -- repeated seeds re-draw the same genealogy and understate the floor")
    p.add_argument("--mu", type=float, default=4.646e-7)
    p.add_argument("--recomb", type=float, default=2.75e-6)
    p.add_argument("--anc-ne", type=int, default=6700)
    p.add_argument("--fixed-tree", action="store_true",
                   help="reuse out/simTreeSeq.trees; vary recapitation+mutation only (PARTIAL "
                        "floor -- required where slim.exe is blocked, see docstring)")
    p.add_argument("--summarize", action="store_true",
                   help="do not simulate; pool existing replicate records from --out and print "
                        "the floor table (filtered by --popmult and --fixed-tree)")
    p.add_argument("--out", default="../out/noise_floor.jsonl")
    args = p.parse_args()
    do_summarize(args) if args.summarize else main(args)
