"""Summarize out/mu_calibration.jsonl: the calibrated mu vs POPMULT, and what it costs.

Reads the JSONL written by mu_calibrate.py and prints:

  * the calibrated mu at each POPMULT, with 4*Ne_anc*mu,
  * branch_div against its hard ceiling 2*(gens + 2*Ne_anc) -- the ceiling is why mu SATURATES
    rather than drifting without bound as POPMULT grows,
  * both fitted losses at the calibrated mu, so the 6.2 pi-vs-Fst tension can be read off
    directly,
  * the irreducible pi_loss floor: the sim's own between-subpop log spread. No mu can remove it,
    because mu only shifts log pi by a constant.

Usage:  python mu_calibrate_summary.py [--in ../out/mu_calibration.jsonl]
"""
import argparse
import json
from pathlib import Path

import numpy as np

YEARS = ["2015", "2019", "2023"]
FORWARD_GENS = 324          # CPBSampleSim*.slim run length (6.1)
ANC_NE = 6700
CEILING = 2 * (FORWARD_GENS + 2 * ANC_NE)   # 2*E[T_pair] with no forward coalescence


def main(path):
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    recs.sort(key=lambda r: r["popmult"])

    print(f"branch_div ceiling = 2*({FORWARD_GENS} + 2*{ANC_NE}) = {CEILING}")
    print(f"=> mu floor (pi=0.0122) = {0.0122 / CEILING:.4g}\n")

    hdr = (f"{'POPMULT':>8} {'subpopN':>8} {'branch':>8} {'/ceil':>6} "
           f"{'mu_calib':>11} {'4Ne*mu':>9} {'pi_loss':>8} {'fst_loss':>9} "
           f"{'simFst':>8} {'obsFst':>8} {'recap_s':>8} {'peakMB':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        b = np.mean([r["branch"][y]["mean"] for y in YEARS])
        sf = np.mean([r["fst"][y]["sim_mean"] for y in YEARS])
        of = np.mean([r["fst"][y]["obs_mean"] for y in YEARS])
        # subpop size = Average Count * POPMULT / numSubpops; total N ~ 3.33*POPMULT (3.1)
        n_sub = 3.33 * r["popmult"] / r["num_clusters"]
        print(f"{r['popmult']:>8.0f} {n_sub:>8.0f} {b:>8.0f} {b/CEILING:>6.3f} "
              f"{r['mu_calibrated']:>11.4g} {r['four_ne_mu_calibrated']:>9.5f} "
              f"{r['pi_loss']:>8.4f} {r['fst_loss']:>9.5f} {sf:>8.4f} {of:>8.5f} "
              f"{r['recap_s']:>8.0f} {r.get('peak_mb') or 0:>7.0f}")

    # Each mu iteration re-draws mutations, so every iterate carries its own Monte-Carlo noise and
    # the loop settles into a small oscillation rather than a fixed point. mu_calibrated is just
    # the LAST iterate; the spread over the iterates is the honest precision of the calibration.
    print("\nMonte-Carlo precision of mu (spread over the mu iterates, excluding the analytic seed):")
    print(f"{'POPMULT':>8} {'mu_last':>11} {'mu_median':>11} {'spread':>8}  iterates")
    for r in recs:
        mus = [it["mu"] for it in r.get("iters", [])][1:]
        if not mus:
            continue
        spread = (max(mus) - min(mus)) / np.median(mus)
        print(f"{r['popmult']:>8.0f} {r['mu_calibrated']:>11.4g} {np.median(mus):>11.4g} "
              f"{spread:>7.2%}  " + " ".join(f"{m:.4g}" for m in mus))

    print("\nper-year detail at the calibrated mu (log-space spread is the pi_loss floor):")
    print(f"{'POPMULT':>8} {'year':>6} {'sim_geo_pi':>11} {'obs_geo_pi':>11} "
          f"{'sim_log_sd':>11} {'obs_log_sd':>11} {'pi_loss_y':>10}")
    for r in recs:
        for y in YEARS:
            d = r["pi"][y]
            print(f"{r['popmult']:>8.0f} {y:>6} {d['sim_geo']:>11.5g} {d['obs_geo']:>11.5g} "
                  f"{d['sim_log_sd']:>11.4f} {d['obs_log_sd']:>11.4f} "
                  f"{d['pi_loss_year']:>10.4f}")

    if len(recs) >= 2:
        # branch_div approaches CEILING from below, so model the DEFICIT d = 1 - b/CEILING as a
        # power law in POPMULT and fit it to the measured points. (An exp(-P/tau) form was tried
        # first and rejected: fitted to the largest POPMULT it under-predicts the smallest by 40%,
        # i.e. it cannot fit both ends. The power law is a fit, NOT a derivation -- there is no
        # theory behind it, so treat any value outside the measured range as indicative only.)
        P = np.array([r["popmult"] for r in recs], dtype=float)
        B = np.array([np.mean([r["branch"][y]["mean"] for y in YEARS]) for r in recs])
        d = 1.0 - B / CEILING
        k, logc = np.polyfit(np.log(P), np.log(d), 1)
        print(f"\nsaturation: deficit (1 - branch/ceiling) ~ {np.exp(logc):.3g} * POPMULT^{k:.3f}"
              f"   [fit over {len(P)} measured points, POPMULT {P.min():.0f}-{P.max():.0f}]")
        print(f"{'POPMULT':>8} {'branch':>8} {'mu':>10}   source")
        for p in [500, 2000, 5000, 8000, 12000]:
            hit = [r for r in recs if abs(r["popmult"] - p) < 1e-9]
            if hit:
                b = np.mean([hit[0]["branch"][y]["mean"] for y in YEARS])
                print(f"{p:>8} {b:>8.0f} {hit[0]['mu_calibrated']:>10.4g}   MEASURED")
            else:
                bp = CEILING * (1 - np.exp(logc) * p ** k)
                tag = "extrapolated" if p > P.max() else "interpolated"
                print(f"{p:>8} {bp:>8.0f} {0.0122 / bp:>10.4g}   {tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../out/mu_calibration.jsonl")
    main(ap.parse_args().inp)
