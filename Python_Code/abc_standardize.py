"""
Offline standardization + ranking for the rejection-ABC pass (CLAUDE.md 7).

Runs AFTER the big pass, on the accumulated results CSV. It does NOT run any simulations.

Offline because sigma can only be measured from the spread of losses across the whole run set,
which is its own pilot batch. This script:

  1. reads the results CSV,
  2. computes sigma_j = 1.4826 * MAD for each FITTED statistic,
  3. combines them into one standardized distance D per run,
  4. writes a ranked CSV and freezes the sigmas to a JSON file.

FITTED (enter D):      pi_loss (log-space), fst_loss
DIAGNOSTIC (not in D): ibd_loss, dxy_loss, genrel_loss   -- posterior-predictive checks only.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
RESULTS_CSV   = Path("../out/abc_results.csv")          # input: the pass results
RANKED_CSV    = Path("../out/abc_results_ranked.csv")   # output: results + D, sorted
SIGMAS_JSON   = Path("../out/abc_sigmas.json")          # output: frozen sigmas
FITTED_STATS  = ["pi_loss", "fst_loss"]                 # statistics that enter the distance D
# Set from batch 1 (2,495 trials) by diagnostics/collect_batch.py, NOT equal.
# Rule: weight by the share of each statistic's batch spread that is DEMOGRAPHIC signal --
# the unique rank-space R2 of pop + total_migration + m + numClusters.
#
#   statistic   R2_total   demographic   mu-nuisance   unexplained
#   pi_loss        0.240        0.0893        0.1533         0.760
#   fst_loss       0.622        0.6243        0.0004         0.378
#
# pi_loss's spread is MAJORITY mu-draw: 63% of the variance the parameters explain comes from
# mutation_rate and only 10% from pop. mu is a deliberate nuisance dimension (kept free to absorb
# the calibration's own uncertainty; only theta=4Nmu is ever reported), so weighting on that
# variance would put the mu draw into D. fst_loss takes 0.000 from mu -- the empirical
# confirmation of 5.3's claim that F_st is denominator-invariant.
#
# The 7.3 replicate-noise rule was tried first and REJECTED: it gives 0.488/0.512, near-equal,
# because mu-driven spread is not replicate noise and so counts as signal under it. These weights
# need no noise floor at all, which also sidesteps the deferred Nei->Hudson floor caveat (TODO 4).
WEIGHTS       = {"pi_loss": 0.125, "fst_loss": 0.875}   # None -> equal weights
ACCEPT_FRAC   = 0.20            # fraction of runs to flag as 'accepted' (top by smallest D)
# ------------------------------------------------------------------


def robust_sigma(values):
    """sigma = 1.4826 * MAD (median absolute deviation), ignoring NaN. Robust to the
    2-individual-subpop outliers (plan §1). Returns np.nan if <2 finite values or MAD==0."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan
    mad = np.median(np.abs(v - np.median(v)))
    return 1.4826 * mad if mad > 0 else np.nan


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results CSV not found: {RESULTS_CSV.resolve()} "
                                f"(run the pass first).")
    df = pd.read_csv(RESULTS_CSV)

    missing = [s for s in FITTED_STATS if s not in df.columns]
    if missing:
        raise ValueError(f"Results CSV missing fitted-stat columns: {missing}. "
                         f"Columns present: {list(df.columns)}")

    # 1) robust per-statistic scale, frozen
    sigmas = {}
    for stat in FITTED_STATS:
        s = robust_sigma(df[stat].values)
        if not np.isfinite(s):
            raise ValueError(f"Could not compute a finite sigma for '{stat}' "
                             f"(all-NaN, <2 values, or zero MAD). Inspect the pass output.")
        sigmas[stat] = float(s)

    # 2) weights
    if WEIGHTS is None:
        w = {stat: 1.0 / len(FITTED_STATS) for stat in FITTED_STATS}
    else:
        total = float(sum(WEIGHTS[s] for s in FITTED_STATS))
        w = {stat: WEIGHTS[stat] / total for stat in FITTED_STATS}

    # 3) combined standardized distance:  D = sqrt( sum_j w_j * (loss_j / sigma_j)^2 )
    sq = np.zeros(len(df))
    for stat in FITTED_STATS:
        z = df[stat].values / sigmas[stat]
        sq = sq + w[stat] * np.square(z)
    df["D"] = np.sqrt(sq)

    # 4) rank + flag acceptance (smaller D = better). NaN D sinks to the bottom.
    df = df.sort_values("D", kind="mergesort", na_position="last").reset_index(drop=True)
    n_accept = max(1, int(np.floor(ACCEPT_FRAC * np.isfinite(df["D"]).sum())))
    df["accepted"] = False
    df.loc[df.index[:n_accept], "accepted"] = np.isfinite(df["D"].iloc[:n_accept]).values

    # 5) write outputs
    RANKED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RANKED_CSV, index=False)
    SIGMAS_JSON.write_text(json.dumps(
        {"sigmas": sigmas, "weights": w, "fitted_stats": FITTED_STATS,
         "accept_frac": ACCEPT_FRAC, "n_runs": int(len(df)), "n_accepted": int(n_accept)},
        indent=2))

    # 6) report
    print(f"Runs: {len(df)}  |  finite D: {int(np.isfinite(df['D']).sum())}")
    print("Frozen sigmas (1.4826*MAD):")
    for stat in FITTED_STATS:
        print(f"  {stat:10s} sigma={sigmas[stat]:.6g}  weight={w[stat]:.3f}")
    print(f"Accepted (top {ACCEPT_FRAC:.0%} by D): {n_accept} runs")
    print(f"  best D = {df['D'].iloc[0]:.4g}   acceptance threshold eps = {df['D'].iloc[n_accept-1]:.4g}")
    print(f"Wrote: {RANKED_CSV}")
    print(f"Wrote: {SIGMAS_JSON}")
    print("\nNOTE (plan §4): measure the noise floor before trusting eps -- simulate >=2 "
          "replicates at identical params and compute D between them. If eps is below that "
          "floor you are selecting on coalescent noise.")


if __name__ == "__main__":
    main()
