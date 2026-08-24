"""Does whole-deme sampling BIAS Fst (or pi) relative to the year's real per-site counts?

Why this exists (CLAUDE.md 6.6, rewritten). AnalyzeTreeSeq.py builds every sample set as
ts.samples(population=i, time=t) -- the WHOLE deme, 301-714 diploids -- while the observed
statistics come from 4-19 diploids per site. The original 6.6 read that as a violation of 8
invariant 1 and prescribed subsampling the simulated side to match. That prescription is wrong
for pi and unproven for Fst, and the two cases are different in kind:

  pi  is UNBIASED at any n >= 2. Whole-deme and n=5 estimates target the same number; only the
      precision differs. pi_loss is a plain L1 distance with no variance-matching term, so
      degrading the precise side would add a near-constant penalty to every draw (7.2.1 measured
      exactly this: uncorrelated scatter scored WORSE than no scatter) while inflating the
      Monte-Carlo variance of each trial -- i.e. it would raise the noise floor the pass has to
      clear, for nothing. There is no need to replicate sampling noise.

  Fst is a RATIO, so its estimator carries an O(1/n) bias whose size depends on the true level.
      That is a systematic offset, not noise: it does NOT average out over replicates, and it
      shifts the very quantity the fit is reading (simulated 0.0079 against observed 0.00645 is
      what puts POPMULT near 6000 -- 6.2). Nobody has measured it.

So this script measures the two separately, and reports pi's number only to size the noise, not
to justify changing it:

  1. Recapitate + simplify + mutate ONCE at the calibrated mu (the expensive, and here entirely
     shared, half). --save-ts/--load-ts make re-runs cost seconds.
  2. Whole-deme baseline: per-year pi vector and Fst matrix exactly as production computes them.
  3. R replicates: draw each deme's real n_i diploid INDIVIDUALS (not nodes -- the empirical unit
     is a diploid) and RECOMPUTE pi and Fst on those sample sets from scratch. Never slice a
     bigger matrix (8 invariant 2); tskit recomputes per sample-set pair anyway, but the sample
     sets themselves must be rebuilt, which is what this does.
  4. Report bias (mean over replicates minus whole-deme) separately from spread (sd over
     replicates), in the units of fst_loss and pi_loss.

The decision-relevant number is `fst_loss` under subsampling vs whole-deme: if it moves by much
less than the sim-vs-obs gap it is fitting against, whole-deme sampling stays and 6.6 closes.

NOT measured here, and not measurable from the tree sequence: the empirical side uses pixy's
Weir-Cockerham (SNP-weighted mean, 5.2) while the simulated side uses tskit's Hudson-style ratio.
That estimator mismatch is independent of sample size and survives whatever this finds.

Usage (run from diagnostics/ or Python_Code/ -- paths are ../data, ../out):
    python fst_subsample.py --reps 100 --save-ts ../out/fst_sub_mutated.trees
    python fst_subsample.py --reps 100 --load-ts ../out/fst_sub_mutated.trees
"""
import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import msprime
import pyslim
import tskit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Python_Code"))
import ABCAnalysisNoRedis as ABC  # noqa: E402  (real _read_vector/_read_matrix/get_keep_mask)

TIMES = {"2015": 16, "2019": 8, "2023": 0}
ANCESTRAL_NE = 6700
DEFAULT_RECOMB = 2.75e-6
DEFAULT_MU = 4.646e-7   # calibrated at POPMULT=5000 (6.1.1); mu-invariance of Fst is the point

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
    """Subpop index -> cluster row. Same construction as AnalyzeTreeSeq.py:106-120."""
    a = cluster_data[f"Genome Assignment {year}"]
    idx = [-1] * (int(max(a.dropna())) + 1)
    for i in range(len(a)):
        if not math.isnan(a[i]):
            k = int(a[i])
            if idx[k] == -1:
                idx[k] = i
    return idx


def site_names(year):
    """Site names in specifier-matrix row order (col 0) -- canonical subpop ordering (4)."""
    names = []
    with open(Path(f"../data/Genetic_Data/specifier_matrix_{year}.csv"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                names.append(line.split(",")[0].strip())
    return names


def site_counts(year):
    """Diploid individuals per site from the popfile, keyed by site name."""
    counts = {}
    with open(Path(f"../data/Genetic_Data/popFile{year}"), encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                counts[parts[1].strip()] = counts.get(parts[1].strip(), 0) + 1
    return counts


def build_mutated_ts(recomb, mu, seed, save_path=None):
    """Recapitate -> simplify -> overlay mutations, mirroring AnalyzeTreeSeq.py:123-155."""
    cluster_data = pd.read_csv(Path("../data/cluster_data.csv"))
    gi = {y: genome_indices(cluster_data, y) for y in TIMES}

    t0 = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] recapitating anc_ne={ANCESTRAL_NE} ...", flush=True)
    ts = tskit.load(Path("../out/simTreeSeq.trees"))
    ts = pyslim.recapitate(ts, recombination_rate=recomb, ancestral_Ne=ANCESTRAL_NE,
                           random_seed=seed)
    print(f"[{time.strftime('%H:%M:%S')}] recap {time.perf_counter()-t0:.1f}s "
          f"peak={peak_mb():.0f}MB edges={ts.num_edges}", flush=True)

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

    ts = msprime.sim_mutations(
        ts, rate=mu,
        model=msprime.SLiMMutationModel(type=0, next_id=pyslim.next_slim_mutation_id(ts)),
        keep=True, random_seed=seed)
    print(f"[{time.strftime('%H:%M:%S')}] mutated -> {ts.num_sites} sites "
          f"({time.perf_counter()-t0:.1f}s total)", flush=True)

    if save_path:
        ts.dump(Path(save_path))
        print(f"[{time.strftime('%H:%M:%S')}] saved {save_path}", flush=True)
    return ts, cluster_data, gi


def fst_matrix(ts, sets):
    """Off-diagonal Fst for a list of sample sets, one batched traversal (AnalyzeTreeSeq.py:47)."""
    k = len(sets)
    pairs = [(i, j) for i in range(k) for j in range(k) if i != j]
    F = np.zeros((k, k))
    for (i, j), v in zip(pairs, np.asarray(ts.Fst(sets, indexes=pairs)).ravel()):
        F[i, j] = v
    return F


def main(a):
    rec = {"reps": a.reps, "mu": a.mu, "recomb": a.recomb, "anc_ne": ANCESTRAL_NE,
           "seed": a.seed, "loaded_ts": a.load_ts}

    # ---- observed targets + the REAL fitted mask (identical to calculate_losses) ------------
    obs_pi, obs_fst, keep = {}, {}, {}
    for y in TIMES:
        keep[y] = ABC.get_keep_mask(y)
        obs_pi[y] = ABC._read_vector(Path(f"../data/empiricalStats/averaged_pi_{y}.csv"))[keep[y]]
        obs_fst[y] = ABC._read_matrix(
            Path(f"../data/empiricalStats/averaged_fst_{y}.csv"))[np.ix_(keep[y], keep[y])]

    # ---- the expensive, shared half --------------------------------------------------------
    t0 = time.perf_counter()
    if a.load_ts and Path(a.load_ts).exists():
        print(f"[{time.strftime('%H:%M:%S')}] loading {a.load_ts}", flush=True)
        ts = tskit.load(Path(a.load_ts))
        cluster_data = pd.read_csv(Path("../data/cluster_data.csv"))
        gi = {y: genome_indices(cluster_data, y) for y in TIMES}
    else:
        ts, cluster_data, gi = build_mutated_ts(a.recomb, a.mu, a.seed,
                                                save_path=a.save_ts or a.load_ts)
    rec["setup_s"] = time.perf_counter() - t0

    # ---- sample sets in specifier order, masked to the FITTED subpops (7.0) -----------------
    ss_all = {y: [ts.samples(population=i, time=TIMES[y]) for i in gi[y]] for y in TIMES}
    ss = {y: [s for s, k in zip(ss_all[y], keep[y]) if k] for y in TIMES}
    for y in TIMES:
        if len(ss[y]) != len(obs_pi[y]):
            raise ValueError(f"{y}: {len(ss[y])} fitted sim subpops vs {len(obs_pi[y])} observed")

    # ---- group each deme's nodes into diploid INDIVIDUALS -----------------------------------
    # The empirical sampling unit is a diploid, not a genome: drawing 2n random nodes would
    # break the pairing and quietly change what "n individuals" means.
    node_ind = ts.tables.nodes.individual
    inds = {}
    for y in TIMES:
        per_deme = []
        for nodes in ss[y]:
            d = {}
            for u in nodes:
                d.setdefault(int(node_ind[u]), []).append(int(u))
            groups = list(d.values())
            if any(len(g) != 2 for g in groups):
                raise ValueError(f"{y}: non-diploid individual in a deme sample set")
            per_deme.append(np.array(groups, dtype=np.int32))   # (n_ind, 2)
        inds[y] = per_deme

    # ---- the year's real per-site diploid counts, in the same (masked) order ---------------
    n_obs, names = {}, {}
    for y in TIMES:
        counts = site_counts(y)
        names[y] = [n for n, k in zip(site_names(y), keep[y]) if k]
        n_obs[y] = [int(counts[n]) for n in names[y]]
        for nm, want, have in zip(names[y], n_obs[y], inds[y]):
            if want > len(have):
                raise ValueError(f"{y}/{nm}: need {want} diploids, deme has {len(have)}")
        rec.setdefault("design", {})[y] = {
            "sites": names[y], "obs_n": n_obs[y],
            "sim_n_whole_deme": [int(len(g)) for g in inds[y]]}

    print("\n[design] simulated diploids per deme vs observed n:")
    for y in TIMES:
        sn = rec["design"][y]["sim_n_whole_deme"]
        print(f"  {y}: sim {min(sn)}-{max(sn)}   obs {min(n_obs[y])}-{max(n_obs[y])}   "
              f"ratio {min(sn)/max(n_obs[y]):.0f}-{max(sn)/min(n_obs[y]):.0f}x", flush=True)

    # ---- whole-deme baseline: exactly what production computes ------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] whole-deme baseline ...", flush=True)
    t1 = time.perf_counter()
    pi_full = {y: np.asarray(ts.diversity(ss[y]), dtype=float) for y in TIMES}
    fst_full = {y: fst_matrix(ts, ss[y]) for y in TIMES}
    print(f"[{time.strftime('%H:%M:%S')}] baseline {time.perf_counter()-t1:.1f}s", flush=True)

    def losses(pi_d, fst_d):
        pl, fl = [], []
        for y in TIMES:
            pl.append(float(np.mean(np.abs(np.log(pi_d[y]) - np.log(obs_pi[y])))))
            m = ~np.eye(len(ss[y]), dtype=bool)
            fl.append(float(np.nanmean(np.abs(fst_d[y][m] - obs_fst[y][m]))))
        return float(np.mean(pl)), float(np.mean(fl)), pl, fl

    pi_loss_full, fst_loss_full, pl_full, fl_full = losses(pi_full, fst_full)
    rec["whole_deme"] = {
        "pi_loss": pi_loss_full, "fst_loss": fst_loss_full,
        "pi_loss_year": dict(zip(TIMES, pl_full)), "fst_loss_year": dict(zip(TIMES, fl_full)),
        "fst_mean": {y: float(np.nanmean(fst_full[y][~np.eye(len(ss[y]), dtype=bool)]))
                     for y in TIMES},
        "pi_geo": {y: float(np.exp(np.log(pi_full[y]).mean())) for y in TIMES},
        "pi_log_sd": {y: float(np.log(pi_full[y]).std()) for y in TIMES},
    }
    print(f"[baseline] pi_loss={pi_loss_full:.5f}  fst_loss={fst_loss_full:.5f}  "
          + " ".join(f"fst_{y}:{rec['whole_deme']['fst_mean'][y]:.5f}" for y in TIMES), flush=True)

    # ---- replicates at the real per-site counts ---------------------------------------------
    rng = np.random.default_rng(a.seed)
    pi_reps = {y: [] for y in TIMES}
    fst_reps = {y: [] for y in TIMES}
    pil_reps, fstl_reps = [], []

    print(f"\n[{time.strftime('%H:%M:%S')}] {a.reps} subsampled replicates ...", flush=True)
    t2 = time.perf_counter()
    for r in range(a.reps):
        sub = {}
        for y in TIMES:
            sets = []
            for g, n in zip(inds[y], n_obs[y]):
                pick = rng.choice(len(g), size=n, replace=False)
                sets.append(np.sort(g[pick].ravel()))
            sub[y] = sets
        pi_s = {y: np.asarray(ts.diversity(sub[y]), dtype=float) for y in TIMES}
        fst_s = {y: fst_matrix(ts, sub[y]) for y in TIMES}
        for y in TIMES:
            pi_reps[y].append(pi_s[y])
            fst_reps[y].append(fst_s[y])
        p, f, _, _ = losses(pi_s, fst_s)
        pil_reps.append(p)
        fstl_reps.append(f)
        if (r + 1) % max(1, a.reps // 10) == 0:
            print(f"  rep {r+1}/{a.reps}  pi_loss={p:.5f} fst_loss={f:.5f} "
                  f"({time.perf_counter()-t2:.0f}s)", flush=True)
    rec["reps_s"] = time.perf_counter() - t2

    # ---- separate BIAS (systematic, survives replication) from SPREAD (noise) ---------------
    print("\n" + "=" * 78)
    print("Fst -- BIAS is the question: does subsampling shift the LEVEL?")
    print("=" * 78)
    for y in TIMES:
        k = len(ss[y])
        m = ~np.eye(k, dtype=bool)
        A = np.stack(fst_reps[y])                       # (reps, k, k)
        mean_sub = A.mean(axis=0)
        full = fst_full[y]
        bias = (mean_sub - full)[m]
        se = A.std(axis=0)[m] / math.sqrt(a.reps)
        lvl_full = float(np.nanmean(full[m]))
        lvl_sub = float(np.nanmean(mean_sub[m]))
        rec.setdefault("fst_bias", {})[y] = {
            "level_whole_deme": lvl_full, "level_subsampled": lvl_sub,
            "level_bias": lvl_sub - lvl_full,
            "level_bias_rel": (lvl_sub - lvl_full) / lvl_full,
            "level_bias_mc_se": float(np.nanmean(se)) / math.sqrt(max(1, bias.size)),
            "pair_bias_mean_abs": float(np.nanmean(np.abs(bias))),
            "pair_sd_across_reps": float(np.nanmean(A.std(axis=0)[m])),
            "obs_level": float(np.nanmean(obs_fst[y][m])),
        }
        d = rec["fst_bias"][y]
        print(f"  {y}: whole-deme {lvl_full:.6f} -> subsampled {lvl_sub:.6f}   "
              f"bias {d['level_bias']:+.6f} ({100*d['level_bias_rel']:+.2f}%)   "
              f"obs {d['obs_level']:.6f}")
        print(f"        per-pair sd across reps {d['pair_sd_across_reps']:.6f}   "
              f"|bias| per pair {d['pair_bias_mean_abs']:.6f}")

    rec["fst_loss_subsampled"] = {
        "mean": float(np.mean(fstl_reps)), "sd": float(np.std(fstl_reps)),
        "p2.5": float(np.percentile(fstl_reps, 2.5)),
        "p97.5": float(np.percentile(fstl_reps, 97.5))}
    d = rec["fst_loss_subsampled"]
    gap = fst_loss_full
    print(f"\n  fst_loss: whole-deme {fst_loss_full:.5f}  ->  subsampled "
          f"{d['mean']:.5f} +- {d['sd']:.5f}  (95% {d['p2.5']:.5f}-{d['p97.5']:.5f})")
    print(f"  shift = {d['mean']-gap:+.5f}, i.e. {100*(d['mean']-gap)/gap:+.1f}% of the "
          f"sim-vs-obs distance the fit is reading")

    print("\n" + "=" * 78)
    print("pi -- NOISE is the question: how much of the 0.0209 flat floor is sampling noise?")
    print("=" * 78)
    for y in TIMES:
        A = np.stack(pi_reps[y])
        L = np.log(A)
        bias = float(np.mean(L.mean(axis=0) - np.log(pi_full[y])))
        sd = float(np.mean(L.std(axis=0)))
        rec.setdefault("pi_noise", {})[y] = {
            "log_bias": bias, "log_sd_across_reps": sd,
            "implied_mean_abs": sd * math.sqrt(2 / math.pi),
            "obs_log_sd": float(np.log(obs_pi[y]).std()),
            "sim_log_sd_whole_deme": float(np.log(pi_full[y]).std())}
        p = rec["pi_noise"][y]
        print(f"  {y}: log bias {bias:+.5f}   sampling sd {sd:.5f} "
              f"(-> {p['implied_mean_abs']:.5f} in pi_loss units)   "
              f"obs between-site log sd {p['obs_log_sd']:.5f}")

    rec["pi_loss_subsampled"] = {
        "mean": float(np.mean(pil_reps)), "sd": float(np.std(pil_reps)),
        "p2.5": float(np.percentile(pil_reps, 2.5)),
        "p97.5": float(np.percentile(pil_reps, 97.5))}
    d = rec["pi_loss_subsampled"]
    print(f"\n  pi_loss: whole-deme {pi_loss_full:.5f}  ->  subsampled "
          f"{d['mean']:.5f} +- {d['sd']:.5f}  (95% {d['p2.5']:.5f}-{d['p97.5']:.5f})")
    print(f"  flat-simulation floor (7.2.1) = 0.02094 for reference")

    rec["total_s"] = time.perf_counter() - t0
    rec["peak_mb"] = peak_mb()
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\nDONE -> {a.out}  ({rec['total_s']:.0f}s, peak {rec['peak_mb']:.0f}MB)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--mu", type=float, default=DEFAULT_MU)
    p.add_argument("--recomb", type=float, default=DEFAULT_RECOMB)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-ts", default=None, help="dump the mutated tree for cheap re-runs")
    p.add_argument("--load-ts", default=None, help="reuse a dumped mutated tree (builds+saves if absent)")
    p.add_argument("--out", default="../out/fst_subsample.jsonl")
    main(p.parse_args())
