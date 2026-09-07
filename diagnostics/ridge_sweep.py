"""Sweep ancestral_Ne and mu together along the 4*Ne*mu = const ridge, on ONE .trees file.

Answers the two questions in CLAUDE.md 6.2:

  Test 1 -- does site pi actually stay put along the ridge?
    "pi = 4*Ne_anc*mu" assumes essentially ALL coalescence happens in the recapitated
    ancestral phase. The forward SLiM phase contributes a part that scales with mu but
    NOT with Ne_anc, so if that part is non-trivial, trading Ne up and mu down does not
    leave pi invariant. Check: site_pi_mean constant, branch_div_mean proportional to Ne_anc.

  Test 2 -- does pi stop carrying information about POPMULT?
    Between-subpop variation in pi comes from the forward phase. As Ne_anc rises, that
    variation becomes a vanishing fraction of a total dominated by the ancestral phase.
    Check: does the coefficient of variation of branch_div / site_pi collapse toward 0?
    If it does, pi_loss stops ranking parameter draws and must be demoted to a
    calibration check (see 7.1 for the same move applied to IBD).

Runs ONE point per invocation and appends a JSON line to out/ridge_sweep.jsonl, so an
OOM at high Ne_anc costs only that point rather than the whole sweep.

Deliberately does NOT use qdriver.py, which has drifted from production (simplificationRatio=INF,
slim_rho=1e-8, and a determine_migration_rates() call whose signature no longer exists).

Usage:
    python ridge_sweep.py --setup --popmult 500          # once: build .trees via the production path
    python ridge_sweep.py --anc-ne 6700                  # then one of these per ridge point
"""
import argparse, json, math, platform, subprocess, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tskit, msprime, pyslim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Python_Code"))
import CollectData, GenerateClusterData, GenerateSimulationParams  # noqa: E402

# The ridge is pinned to the OBSERVED per-site pi, so every point on it SHOULD reproduce
# the empirical diversity level if Test 1 holds. mu = TARGET_PI / (4*Ne).
#   Ne=6700    -> mu=4.55e-7  (current config, ~217x biological)
#   Ne=1.452e6 -> mu=2.10e-9  (proposed config, biological rate)
TARGET_PI = 0.0122
KMEANS_SEED = 42
TIMES = {"2015": 16, "2019": 8, "2023": 0}

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


def setup(popmult, num_clusters, recomb):
    """Mirror Main.main() lines 38-73: cluster -> migration rates -> SLiM -> out/simTreeSeq.trees.

    Writes the same data/ files the production pipeline does (it overwrites in place, 10).
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
        distances, total_migration=0.05, scale=0.0001,
        output_path=Path("../data/migration_rates.csv"))

    slim_script = Path("../SLiM_Code/CPBSampleSim"
                       + ("Win" if platform.system() == "Windows" else "Linux") + ".slim")
    t0 = time.perf_counter()
    subprocess.run(["slim", "-l", "0", "-d", f"POPMULT={popmult}", "-d", f"RECOMB={recomb!r}",
                    str(slim_script)], check=True)
    dt = time.perf_counter() - t0
    size = Path("../out/simTreeSeq.trees").stat().st_size / 1024 / 1024
    print(f"SETUP done: popmult={popmult} numClusters={num_clusters} "
          f"slim_time={dt:.1f}s trees={size:.1f}MB")


def run_point(anc_ne, mu, recomb, seed, outfile, popmult):
    cluster_data = pd.read_csv(Path("../data/cluster_data.csv"))
    gi = {y: genome_indices(cluster_data, y) for y in TIMES}

    # popmult is recorded for labelling only -- it describes the .trees already on disk, so
    # mixing POPMULTs in one results file would be meaningless without it.
    rec = {"popmult": popmult, "anc_ne": anc_ne, "mu": mu,
           "four_ne_mu": 4 * anc_ne * mu, "seed": seed}
    print(f"[{time.strftime('%H:%M:%S')}] recapitating popmult={popmult} anc_ne={anc_ne:g} "
          f"mu={mu:.3g} ...", flush=True)

    t0 = time.perf_counter()
    ts = tskit.load(Path("../out/simTreeSeq.trees"))
    ts = pyslim.recapitate(ts, recombination_rate=recomb, ancestral_Ne=anc_ne, random_seed=seed)
    rec["recap_s"] = time.perf_counter() - t0
    rec["recap_peak_mb"] = peak_mb()
    print(f"[{time.strftime('%H:%M:%S')}] recap done in {rec['recap_s']:.1f}s "
          f"peak={rec['recap_peak_mb']:.0f}MB edges={ts.num_edges}", flush=True)

    keep = []
    for y in TIMES:
        for i in gi[y]:
            keep.extend(ts.samples(population=i, time=TIMES[y]))
    # filter_populations=False is REQUIRED -- the ts.samples(population=i) queries below use the
    # original cluster-row index (AnalyzeTreeSeq.py:146, commit c5963ae).
    ts = ts.simplify(samples=keep, filter_populations=False)

    ts_mut = msprime.sim_mutations(
        ts, rate=mu, model=msprime.SLiMMutationModel(type=0, next_id=pyslim.next_slim_mutation_id(ts)),
        keep=True, random_seed=seed)

    for y in TIMES:
        ss = [ts.samples(population=i, time=TIMES[y]) for i in gi[y]]
        # branch mode is in generations and mu-free: site_pi = mu * branch_div exactly.
        branch = np.array(ts.diversity(ss, mode="branch"))
        site = np.array(ts_mut.diversity(ss, mode="site"))
        k = len(ss)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        # Hudson, matching AnalyzeTreeSeq.py -- NOT ts.Fst (CLAUDE.md 6.7, invariant 9).
        dxy = np.asarray(ts_mut.divergence(ss, indexes=pairs)).ravel()
        F = np.array([0.0 if d == 0 else 1.0 - 0.5 * (site[i] + site[j]) / d
                      for (i, j), d in zip(pairs, dxy)])
        rec[y] = {
            "n_subpops": k,
            "branch_div_mean": float(branch.mean()),
            "branch_div_sd": float(branch.std()),
            "branch_div_cv": float(branch.std() / branch.mean()),
            "site_pi_mean": float(site.mean()),
            "site_pi_sd": float(site.std()),
            "site_pi_cv": float(site.std() / site.mean()),
            "fst_mean": float(np.nanmean(F)),
            "fst_sd": float(np.nanstd(F)),
        }

    rec["total_s"] = time.perf_counter() - t0
    rec["peak_mb"] = peak_mb()

    with open(outfile, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print("DONE " + json.dumps({k: v for k, v in rec.items()}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--setup", action="store_true")
    p.add_argument("--popmult", type=float, default=500)
    p.add_argument("--num-clusters", type=int, default=33)
    p.add_argument("--anc-ne", type=float)
    p.add_argument("--mu", type=float, help="default: TARGET_PI/(4*anc_ne), i.e. stay on the ridge")
    p.add_argument("--recomb", type=float, default=2.75e-6)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="../out/ridge_sweep.jsonl")
    a = p.parse_args()

    if a.setup:
        setup(a.popmult, a.num_clusters, a.recomb)
    else:
        if a.anc_ne is None:
            p.error("--anc-ne is required unless --setup")
        run_point(a.anc_ne, a.mu or TARGET_PI / (4 * a.anc_ne), a.recomb, a.seed, a.out, a.popmult)
