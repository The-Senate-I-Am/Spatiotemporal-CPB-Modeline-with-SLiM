"""Does simulated pi covary with observed pi SITE BY SITE? (CLAUDE.md 7.2 impl. 3, TODO 1)

Why this matters. pi_loss is element-wise:

    pi_loss = mean_years( mean_i | log pi_sim,i - log pi_obs,i | )

and mu is calibrated to the exact weighted-median minimiser of that objective at each POPMULT
(6.1.1). So the LEVEL is optimal by construction and every difference in pi_loss across POPMULT
comes from the SHAPE of the pi vector -- nothing else. Meanwhile the simulated between-site log
spread collapses with POPMULT (0.10-0.13 at 500, 0.028 at 2000, 0.010 at 5000) while pi_loss falls
alongside it (0.067 -> 0.029 -> 0.021).

Under L1, if the simulated deviations are UNCORRELATED with the observed ones, a flat vector at
the right level is the optimal predictor of a scattered target: a sim with the correct amount of
scatter placed on the wrong sites scores strictly WORSE than a sim with no scatter at all. So two
explanations are observationally tangled:

  (1) large POPMULT genuinely fits better -- the sim puts diversity on the right sites;
  (2) large POPMULT is merely flatter, and flat mechanically wins L1 against a scattered target.

If (2), pi's apparent preference for large POPMULT is an artifact of the distance's geometry and
must not be read as evidence about population size.

Three tests, in increasing order of how directly they decide the question:

  A. CORRELATION of log pi_sim against log pi_obs within each year, against a site-label
     permutation null (9999 perms). Sites are few (22/17/20 after the 7.0 mask) and are coupled by
     migration, so an OLS/t p-value would overstate -- permuting site labels is the exchangeable
     unit, same argument as the Mantel test in 7.1. Also run against branch-mode diversity, which
     is the mu-free shape and carries no mutation Monte-Carlo noise at all.

  B. FLAT-SIM COUNTERFACTUAL. Recompute pi_loss with the simulated vector replaced by a single
     constant, that constant chosen by the SAME weighted-median rule the calibration uses. This is
     "what would pi_loss be if the sim had no between-site structure whatsoever, calibrated
     identically?" If the real sim cannot beat it, its structure is not helping.

  C. SHUFFLE NULL, expressed in the units of the actual objective. Permute the simulated vector
     within each year, RE-CALIBRATE mu by the same weighted median (so the null sim keeps its own
     optimal level), and recompute pi_loss. This gives the distribution of pi_loss under "same
     simulated structure, wrong sites". Where the real pi_loss sits in that distribution is the
     decision-relevant answer: mid-distribution means the site assignment carries no information.

Also probed: whether the sim's own pi variation tracks deme size (its only mechanism for making
one deme more diverse than another), whether observed pi tracks sample size, and whether any
apparent agreement survives dropping the two isolate sites that 7.2 argues are within-site
sampling artifacts (Mortensen9-2015, H41-2023) rather than landscape structure.

Reads the per-subpop vectors stored by mu_calibrate.py. Usage:
    python pi_covary.py                      # latest record holding vectors
    python pi_covary.py --popmult 5000       # latest record at that POPMULT
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mu_calibrate import weighted_median  # noqa: E402  (identical semantics to the calibration)

YEARS = ["2015", "2019", "2023"]
ISOLATES = {"2015": "Mortensen9", "2023": "H41"}   # 7.2 -- lowest pi + highest self-relatedness


def load(path, popmult):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "vectors" in r and (popmult is None or r["popmult"] == popmult):
                    recs.append(r)
    if not recs:
        raise SystemExit(f"no record with per-subpop vectors in {path}"
                         + (f" at popmult={popmult}" if popmult else "")
                         + "\nRe-run mu_calibrate.py -- it stores them since 2026-08-12.")
    return recs[-1]


def spearman(x, y):
    """Pearson on ranks. Ties averaged; there are none here (continuous pi), but be safe."""
    def rank(v):
        o = np.argsort(np.argsort(v))
        return o.astype(float)
    return pearson(rank(x), rank(y))


def pearson(x, y):
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


def perm_corr(x, y, stat, rng, B=9999):
    """Two-sided site-label permutation p for a correlation statistic."""
    obs = stat(x, y)
    if not np.isfinite(obs):
        return obs, np.nan
    hits = sum(1 for _ in range(B) if abs(stat(x, rng.permutation(y))) >= abs(obs) - 1e-15)
    return obs, (1 + hits) / (1 + B)


def pi_loss(logsim, logobs, w):
    """The fitted objective, at the optimal level for `logsim` -- i.e. mu re-calibrated by the
    same weighted median the calibration uses. Level is never what is being tested here."""
    shift = weighted_median(logobs - logsim, w)
    return float(np.sum(w * np.abs((logsim + shift) - logobs)))


def main(a):
    rec = load(Path(a.jsonl), a.popmult)
    V = rec["vectors"]
    rng = np.random.default_rng(a.seed)

    print(f"POPMULT={rec['popmult']:.0f}  numClusters={rec['num_clusters']}  "
          f"mu={rec['mu_calibrated']:.4g}  seed={rec['seed']}")
    print(f"stored pi_loss = {rec['pi_loss']:.5f}\n")

    # ---- A. correlation, per year -----------------------------------------------------------
    print("A. site-by-site correlation of simulated vs observed pi (log space)")
    print(f"{'year':<6}{'n':>4}{'pearson':>10}{'p':>8}{'spearman':>11}{'p':>8}"
          f"{'branch r':>11}{'p':>8}{'no-isolate r':>14}")
    pooled = {"sim": [], "obs": []}
    for y in YEARS:
        v = V[y]
        ls = np.log(v["pi_sim"])
        lo = np.log(v["pi_obs"])
        lb = np.log(v["branch_div"])
        r, pr = perm_corr(ls, lo, pearson, rng, a.perms)
        s, ps = perm_corr(ls, lo, spearman, rng, a.perms)
        rb, pb = perm_corr(lb, lo, pearson, rng, a.perms)

        keep = np.ones(len(ls), dtype=bool)
        if y in ISOLATES:
            for i, nm in enumerate(v["sites"]):
                if nm.startswith(ISOLATES[y]):
                    keep[i] = False
        rn = pearson(ls[keep], lo[keep]) if keep.sum() < len(keep) else np.nan

        print(f"{y:<6}{len(ls):>4}{r:>10.3f}{pr:>8.3f}{s:>11.3f}{ps:>8.3f}"
              f"{rb:>11.3f}{pb:>8.3f}"
              + (f"{rn:>14.3f}" if np.isfinite(rn) else f"{'-':>14}"))
        pooled["sim"].append(ls - ls.mean())
        pooled["obs"].append(lo - lo.mean())

    ps_, po_ = np.concatenate(pooled["sim"]), np.concatenate(pooled["obs"])
    rp = pearson(ps_, po_)
    # Pooled null must permute WITHIN year -- across-year shuffling would leak the year effect
    # back in even though residuals are year-centred.
    sizes = [len(V[y]["pi_sim"]) for y in YEARS]
    hits = 0
    for _ in range(a.perms):
        perm = np.concatenate([rng.permutation(c) for c in np.split(po_, np.cumsum(sizes)[:-1])])
        if abs(pearson(ps_, perm)) >= abs(rp) - 1e-15:
            hits += 1
    print(f"{'pooled':<6}{len(ps_):>4}{rp:>10.3f}{(1+hits)/(1+a.perms):>8.3f}\n")

    # ---- B/C. the objective itself ----------------------------------------------------------
    w = np.concatenate([np.full(len(V[y]["pi_obs"]), 1.0 / (3 * len(V[y]["pi_obs"])))
                        for y in YEARS])
    logsim = np.concatenate([np.log(V[y]["pi_sim"]) for y in YEARS])
    logobs = np.concatenate([np.log(V[y]["pi_obs"]) for y in YEARS])

    actual = pi_loss(logsim, logobs, w)
    flat = pi_loss(np.zeros_like(logsim), logobs, w)

    print("B. flat-simulation counterfactual (identical calibration rule)")
    print(f"   pi_loss, real simulation      = {actual:.5f}")
    print(f"   pi_loss, FLAT sim at best mu  = {flat:.5f}")
    print(f"   structure buys                = {flat - actual:+.5f} "
          f"({100*(flat-actual)/flat:+.1f}% vs flat)\n")

    null = np.empty(a.perms)
    for k in range(a.perms):
        perm = np.concatenate([rng.permutation(c)
                               for c in np.split(logsim, np.cumsum(sizes)[:-1])])
        null[k] = pi_loss(perm, logobs, w)
    p_lower = (1 + np.sum(null <= actual + 1e-15)) / (1 + a.perms)

    print(f"C. shuffle null -- same simulated structure, wrong sites ({a.perms} perms)")
    print(f"   null pi_loss  mean {null.mean():.5f}   sd {null.std():.5f}   "
          f"[{np.percentile(null, 2.5):.5f}, {np.percentile(null, 97.5):.5f}]")
    print(f"   real pi_loss  {actual:.5f}   percentile {100*np.mean(null <= actual):.1f}"
          f"   p(one-sided, lower) = {p_lower:.4f}\n")

    # ---- mechanism probes --------------------------------------------------------------------
    print("D. mechanism probes")
    print(f"{'year':<6}{'r(branch,demesize)':>20}{'r(obs pi,obs n)':>18}"
          f"{'sim log-sd':>12}{'obs log-sd':>12}")
    for y in YEARS:
        v = V[y]
        rd = pearson(np.log(v["branch_div"]), np.log(v["deme_rel_size"]))
        rn = pearson(np.log(v["pi_obs"]), np.log(v["obs_n"]))
        print(f"{y:<6}{rd:>20.3f}{rn:>18.3f}"
              f"{np.log(v['pi_sim']).std():>12.4f}{np.log(v['pi_obs']).std():>12.4f}")

    if a.dump:
        for y in YEARS:
            v = V[y]
            print(f"\n{y}: site, sim pi, obs pi, branch, deme size, obs n")
            order = np.argsort(np.log(v["pi_obs"]))
            for i in order:
                print(f"  {v['sites'][i]:<18}{v['pi_sim'][i]:>11.6f}{v['pi_obs'][i]:>11.6f}"
                      f"{v['branch_div'][i]:>11.0f}{v['deme_rel_size'][i]:>9.2f}"
                      f"{v['obs_n'][i]:>5d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="../out/mu_calibration.jsonl")
    p.add_argument("--popmult", type=float, default=None)
    p.add_argument("--perms", type=int, default=9999)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dump", action="store_true", help="print the per-site table")
    main(p.parse_args())
