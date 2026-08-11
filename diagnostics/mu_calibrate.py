"""Calibrate mu against the observed pi level, at a given POPMULT.

Why this exists (CLAUDE.md 6.1 / TODO 1): mu is not a mutation rate here. With ancestral_Ne
pinned at 6700 it is half of a calibration constant whose only meaningful content is
4*Ne_anc*mu, chosen so simulated pi reproduces observed pi. Setting 4*Ne*mu = 0.0122 does NOT
achieve that -- forward-phase coalescence pulls branch-mode diversity below 4*Ne_anc, by a
POPMULT-dependent factor. So mu must be calibrated AT the POPMULT actually intended for the pass.

Method, exploiting the pipeline's structure:

  1. Recapitate + simplify ONCE. This is ~97% of the cost and is completely mu-free.
  2. branch-mode diversity b_i (generations) is the mutation-free view of the demography.
     site pi_i ~= mu * b_i, so a first mu falls straight out of arithmetic -- no re-running.
  3. That identity is good to ~1-2% but not exact: multiple hits at a site make realised
     site pi run BELOW mu*b, and the deficit grows with mu*b. So iterate -- overlay mutations
     at mu_k, measure the real site pi, correct. Overlay on the simplified (~2.4k sample)
     tree costs seconds, so the loop is nearly free once step 1 is paid for.

The objective is the ACTUAL fitted loss, not a mean-matching proxy:

    pi_loss(mu) = mean_years( mean_i | log pi_sim,i(mu) - log pi_obs,i | )

Since log pi_sim,i = log mu + log b_i + O(1%), this is a weighted L1 problem in log mu, whose
exact minimiser is the WEIGHTED MEDIAN of (log pi_obs,i - log b_i) with weights 1/(3*n_year) --
the same year-normalisation calculate_losses uses. The fitted small-subpop mask (7.0) is applied
identically to both sides, so the calibration targets exactly what the ABC will score.

Also reported: the irreducible pi_loss floor at the optimum (the sim's own between-subpop log
spread, which no choice of mu can remove), and fst_loss at the calibrated mu -- fst is
mu-invariant, so this shows both fitted statistics at the same POPMULT in one run (6.2).

Usage (run from diagnostics/ or Python_Code/ -- paths are ../data, ../out):
    python mu_calibrate.py --popmult 5000                  # SLiM + recapitate + calibrate
    python mu_calibrate.py --popmult 5000 --skip-slim      # reuse out/simTreeSeq.trees
"""
import argparse
import gc
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import msprime
import pyslim
import tskit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Python_Code"))
import ABCAnalysisNoRedis as ABC  # noqa: E402  (for _read_vector / _read_matrix / get_keep_mask)
import CollectData, GenerateClusterData, GenerateSimulationParams  # noqa: E402

KMEANS_SEED = 42
TIMES = {"2015": 16, "2019": 8, "2023": 0}
ANCESTRAL_NE = 6700          # fixed, NOT inferred (6.1) -- calibration is conditional on it
DEFAULT_RECOMB = 2.75e-6

try:
    import psutil
    _PROC = psutil.Process()
except Exception:
    _PROC = None


def peak_mb():
    if _PROC is None:
        return None
    try:
        return _PROC.memory_info().peak_wset / 1024 / 1024
    except Exception:
        return None


def genome_indices(cluster_data, year):
    """Subpop index -> cluster row. Same construction as AnalyzeTreeSeq.py:93-120."""
    a = cluster_data[f"Genome Assignment {year}"]
    idx = [-1] * (int(max(a.dropna())) + 1)
    for i in range(len(a)):
        if not math.isnan(a[i]):
            k = int(a[i])
            if idx[k] == -1:
                idx[k] = i
    return idx


def weighted_median(values, weights):
    """Exact minimiser of sum_i w_i * |x - v_i|. Ties resolved to the midpoint of the flat
    interval, which is also a minimiser and is stable under reordering."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) / w.sum()
    i = int(np.searchsorted(c, 0.5))
    if i + 1 < len(v) and abs(c[i] - 0.5) < 1e-12:
        return 0.5 * (v[i] + v[i + 1])   # flat optimum -- take its midpoint
    return float(v[i])


def run_slim(popmult, num_clusters, recomb, total_migration, scale):
    """Mirror Main.main() lines 38-73 exactly: cluster -> migration rates -> SLiM -> .trees.

    Writes the same data/ files production does (it overwrites in place, 10).
    """
    warnings.filterwarnings("ignore")
    field_data = CollectData.read_csv(Path("../data/final_data_for_modeling.csv"))
    GenerateClusterData.cluster_coordinates(field_data, n_clusters=num_clusters, iters=2000,
                                            random_state=KMEANS_SEED)
    clusters = GenerateClusterData.populate_cluster_objects(field_data, estimate_data=True)
    GenerateClusterData.assign_genomes_to_clusters(clusters)
    distances = GenerateClusterData.create_cluster_distance_matrix(
        clusters, output_path=Path("../data/cluster_distances.csv"))
    GenerateClusterData.cluster_data_to_csv(clusters, output_path=Path("../data/cluster_data.csv"))
    GenerateSimulationParams.determine_migration_rates(
        distances, total_migration=total_migration, scale=scale,
        output_path=Path("../data/migration_rates.csv"))

    slim_script = Path("../SLiM_Code/CPBSampleSim"
                       + ("Win" if platform.system() == "Windows" else "Linux") + ".slim")
    t0 = time.perf_counter()
    subprocess.run(["slim", "-l", "0", "-d", f"POPMULT={popmult}", "-d", f"RECOMB={recomb!r}",
                    str(slim_script)], check=True)
    dt = time.perf_counter() - t0
    size = Path("../out/simTreeSeq.trees").stat().st_size / 1024 / 1024
    print(f"[slim] popmult={popmult} numClusters={num_clusters} {dt:.1f}s trees={size:.1f}MB",
          flush=True)
    return dt


def site_pi(ts, sample_sets, mu, seed):
    """Overlay mutations at rate mu and return per-subpop site-mode pi.

    Mutation model matches AnalyzeTreeSeq.py:150-155 exactly -- a different model would make
    the calibrated mu not transfer to production.
    """
    tsm = msprime.sim_mutations(
        ts, rate=mu,
        model=msprime.SLiMMutationModel(type=0, next_id=pyslim.next_slim_mutation_id(ts)),
        keep=True, random_seed=seed)
    return tsm, {y: np.asarray(tsm.diversity(sample_sets[y]), dtype=float) for y in TIMES}


def main(a):
    rec = {"popmult": a.popmult, "num_clusters": a.num_clusters, "anc_ne": ANCESTRAL_NE,
           "recomb": a.recomb, "total_migration": a.total_migration, "scale": a.scale,
           "seed": a.seed}

    if not a.skip_slim:
        rec["slim_s"] = run_slim(a.popmult, a.num_clusters, a.recomb, a.total_migration, a.scale)

    # ---- observed targets + fitted mask (identical to calculate_losses) --------------------
    obs, keep = {}, {}
    for y in TIMES:
        keep[y] = ABC.get_keep_mask(y)
        obs[y] = ABC._read_vector(Path(f"../data/empiricalStats/averaged_pi_{y}.csv"))[keep[y]]

    # ---- the expensive, mu-free half: recapitate + simplify, ONCE -------------------------
    cluster_data = pd.read_csv(Path("../data/cluster_data.csv"))
    gi = {y: genome_indices(cluster_data, y) for y in TIMES}

    t0 = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] recapitating popmult={a.popmult} "
          f"anc_ne={ANCESTRAL_NE} ...", flush=True)
    ts = tskit.load(Path("../out/simTreeSeq.trees"))
    ts = pyslim.recapitate(ts, recombination_rate=a.recomb, ancestral_Ne=ANCESTRAL_NE,
                           random_seed=a.seed)
    rec["recap_s"] = time.perf_counter() - t0
    rec["recap_peak_mb"] = peak_mb()
    print(f"[{time.strftime('%H:%M:%S')}] recap {rec['recap_s']:.1f}s "
          f"peak={rec['recap_peak_mb']:.0f}MB edges={ts.num_edges}", flush=True)

    ksamp = []
    for y in TIMES:
        for i in gi[y]:
            ksamp.extend(ts.samples(population=i, time=TIMES[y]))
    # filter_populations=False is REQUIRED -- the ts.samples(population=i) queries below use the
    # ORIGINAL cluster-row index (AnalyzeTreeSeq.py:146, commit c5963ae).
    ts = ts.simplify(samples=ksamp, filter_populations=False)
    gc.collect()
    print(f"[{time.strftime('%H:%M:%S')}] simplified -> {ts.num_samples} samples, "
          f"{ts.num_edges} edges", flush=True)

    # Sample sets in specifier-matrix order, then masked to the FITTED subpops (7.0).
    ss_all = {y: [ts.samples(population=i, time=TIMES[y]) for i in gi[y]] for y in TIMES}
    ss = {y: [s for s, k in zip(ss_all[y], keep[y]) if k] for y in TIMES}
    for y in TIMES:
        if len(ss[y]) != len(obs[y]):
            raise ValueError(f"{y}: {len(ss[y])} fitted sim subpops vs {len(obs[y])} observed")

    # ---- branch-mode diversity: the mu-free demography, and the analytic first guess -------
    branch = {y: np.asarray(ts.diversity(ss[y], mode="branch"), dtype=float) for y in TIMES}
    w = np.concatenate([np.full(len(obs[y]), 1.0 / (3 * len(obs[y]))) for y in TIMES])
    logobs = np.concatenate([np.log(obs[y]) for y in TIMES])
    logb = np.concatenate([np.log(branch[y]) for y in TIMES])

    mu = float(np.exp(weighted_median(logobs - logb, w)))
    rec["mu_analytic"] = mu
    rec["four_ne_mu_analytic"] = 4 * ANCESTRAL_NE * mu
    for y in TIMES:
        rec.setdefault("branch", {})[y] = {
            "n_fitted": len(branch[y]), "mean": float(branch[y].mean()),
            "sd": float(branch[y].std()), "cv": float(branch[y].std() / branch[y].mean()),
            "log_sd": float(np.log(branch[y]).std()),
            "frac_of_4Ne": float(branch[y].mean() / (4 * ANCESTRAL_NE)),
        }
    print(f"[calib] branch_div mean = "
          + " ".join(f"{y}:{branch[y].mean():.0f}" for y in TIMES)
          + f"  (4*Ne_anc = {4*ANCESTRAL_NE})")
    print(f"[calib] analytic mu = {mu:.6g}  (4*Ne*mu = {4*ANCESTRAL_NE*mu:.6g})", flush=True)

    # ---- cheap mu iteration: correct the multiple-hit deficit ------------------------------
    tsm = None
    for it in range(a.iters):
        tsm, pis = site_pi(ts, ss, mu, a.seed)
        logsim = np.concatenate([np.log(pis[y]) for y in TIMES])
        loss = float(np.mean([np.mean(np.abs(np.log(pis[y]) - np.log(obs[y]))) for y in TIMES]))
        ratio = float(np.exp(np.mean(logsim - (logb + math.log(mu)))))  # realised / (mu*b)
        step = weighted_median(logobs - logsim, w)
        print(f"[calib] iter {it}: mu={mu:.6g} pi_mean="
              + " ".join(f"{y}:{pis[y].mean():.5g}" for y in TIMES)
              + f" site/(mu*b)={ratio:.5f} pi_loss={loss:.5f} step={step:+.5f}", flush=True)
        rec.setdefault("iters", []).append(
            {"mu": mu, "pi_loss": loss, "site_over_mu_b": ratio, "step": step,
             "pi_mean": {y: float(pis[y].mean()) for y in TIMES}})
        if abs(step) < 1e-6:
            break
        mu = float(mu * math.exp(step))

    # ---- final evaluation at the calibrated mu ---------------------------------------------
    tsm, pis = site_pi(ts, ss, mu, a.seed)
    rec["mu_calibrated"] = mu
    rec["four_ne_mu_calibrated"] = 4 * ANCESTRAL_NE * mu
    rec["pi_loss"] = float(np.mean([np.mean(np.abs(np.log(pis[y]) - np.log(obs[y])))
                                    for y in TIMES]))
    for y in TIMES:
        rec.setdefault("pi", {})[y] = {
            "sim_mean": float(pis[y].mean()), "sim_geo": float(np.exp(np.log(pis[y]).mean())),
            "sim_log_sd": float(np.log(pis[y]).std()),
            "obs_geo": float(np.exp(np.log(obs[y]).mean())),
            "obs_log_sd": float(np.log(obs[y]).std()),
            "pi_loss_year": float(np.mean(np.abs(np.log(pis[y]) - np.log(obs[y])))),
        }

    # fst at the calibrated mu -- mu-invariant, so this is just "where does this POPMULT sit on
    # the OTHER fitted statistic" (6.2). Off-diagonal mean |sim - obs|, same masking as
    # calculate_losses.
    fst_terms = []
    for y in TIMES:
        k = len(ss[y])
        pairs = [(i, j) for i in range(k) for j in range(k) if i != j]
        F = np.zeros((k, k))
        for (i, j), v in zip(pairs, np.asarray(tsm.Fst(ss[y], indexes=pairs)).ravel()):
            F[i, j] = v
        O = ABC._read_matrix(Path(f"../data/empiricalStats/averaged_fst_{y}.csv"))[
            np.ix_(keep[y], keep[y])]
        m = ~np.eye(k, dtype=bool)
        fst_terms.append(float(np.nanmean(np.abs(F[m] - O[m]))))
        rec.setdefault("fst", {})[y] = {
            "sim_mean": float(np.nanmean(F[m])), "obs_mean": float(np.nanmean(O[m])),
            "fst_loss_year": fst_terms[-1]}
    rec["fst_loss"] = float(np.mean(fst_terms))

    rec["total_s"] = time.perf_counter() - t0
    rec["peak_mb"] = peak_mb()

    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\nDONE popmult={a.popmult}  mu_calibrated={mu:.6g}  "
          f"4*Ne*mu={4*ANCESTRAL_NE*mu:.6g}  pi_loss={rec['pi_loss']:.5f}  "
          f"fst_loss={rec['fst_loss']:.5f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--popmult", type=float, required=True)
    p.add_argument("--num-clusters", type=int, default=33)
    p.add_argument("--recomb", type=float, default=DEFAULT_RECOMB)
    p.add_argument("--total-migration", type=float, default=0.05)
    p.add_argument("--scale", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--skip-slim", action="store_true")
    p.add_argument("--out", default="../out/mu_calibration.jsonl")
    main(p.parse_args())
