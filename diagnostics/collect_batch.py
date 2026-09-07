"""Concatenate a CHTC batch into one results CSV, and decide the fitted-statistic set.

Two jobs, deliberately in one script because the second depends on the first being clean.

1. CONCATENATE. Each job writes its own header (ABCAnalysisNoRedis.py:495), so a plain `cat`
   interleaves N header rows with the data. This dedupes them and adds a `job_id` column
   recovered from the filename -- job_id is NOT in the CSV, and `iteration` restarts at 0 in
   every job, so without it the only per-job provenance in the batch is the filename, and it is
   gone the moment the files are concatenated.

   That provenance is what the landing checks in TODO 4 need. A failed trial writes NO row
   (ABCAnalysisNoRedis.py:548-550 catches and continues), so a short file is a silent failure
   rather than skipped work; an OOM kills the process and loses the whole job. Both bias the
   pool, and rejection ABC pools whatever survived. If short/absent jobs cluster at high `pop`,
   that is the 3.1 memory extrapolation being wrong -- and the resulting posterior is biased
   toward the small draws that finished, with nothing in the output saying so.

2. DECIDE THE STATISTICS. This batch is the pilot that the deferred F_st noise floor (TODO 4)
   would otherwise have provided. abc_standardize.py builds

       D = sqrt( sum_j w_j * (loss_j / sigma_j)^2 ),   sigma_j = 1.4826 * MAD

   so MAD-standardization equalises each statistic's SPREAD across the batch. That is exactly
   the wrong thing to do blind: it inflates a statistic whose spread is mostly noise up to
   parity with one whose spread is signal. 7.3 measured that risk directly -- fst_loss is 2%
   noise, pi_loss is 30% -- so equal weights would seat a 30%-noise statistic beside a 2%-noise
   one. This script reports, per statistic:

     - sigma_j across the batch (the number abc_standardize.py will freeze),
     - sigma_j against the 7.3 replicate noise floor -> the signal fraction,
     - rank correlation against each PARAMETER -> does it carry information about `pop` at all,
     - rank correlation against the other losses -> 7.2's double-counting risk, now measurable
       across the prior rather than inferred from the observed matrices.

CAVEAT ON THE NOISE FLOOR (TODO 4, deferred deliberately). It was measured on the OLD Nei
estimator for fst_loss and ibd_loss. Only ONE conversion point exists (fst_loss 0.008715 ->
0.015714, x1.80), and one point is not a floor. Those two rows are therefore an ARGUMENT, not a
measurement, and are printed flagged. pi_loss/dxy_loss/genrel_loss are unaffected by 6.7 and
stand as measured.

Usage:
    python collect_batch.py                        # concatenate + full report
    python collect_batch.py --no-write             # report only, touch nothing
    python collect_batch.py --raw-dir ../out/batch2_raw --out ../out/abc_results_b2.csv
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python_Code"))
from abc_standardize import robust_sigma  # noqa: E402  -- the EXACT sigma the pass will freeze

# The column layout ABCAnalysisNoRedis.py:488 writes. A file whose header differs is from a
# different code revision and must not be pooled (TODO 0: never pool across the fitted-set change).
EXPECTED_FIELDS = ["iteration", "m", "total_migration", "pop", "numClusters",
                   "mutation_rate", "recombination_rate",
                   "pi_loss", "fst_loss", "ibd_loss", "dxy_loss", "genrel_loss"]

LOSSES = ["pi_loss", "fst_loss", "ibd_loss", "dxy_loss", "genrel_loss"]
FITTED = ["pi_loss", "fst_loss"]                       # 7: what currently enters D
PARAMS = ["pop", "total_migration", "m", "numClusters", "mutation_rate"]

# 7.3 replicate noise floor: run-to-run mean|diff| over 3 reps at POPMULT=5000, identical params.
NOISE_FLOOR = {"pi_loss": 0.00240, "fst_loss": 0.00017, "ibd_loss": 0.00014,
               "dxy_loss": 0.00005, "genrel_loss": 0.00001}
# fst_loss and ibd_loss above are on the OLD Nei statistic (6.7). Scaled by the single measured
# conversion point (fst_loss x1.80); ibd_loss doubles by the same algebra since the IBD slope
# regresses on F_st/(1-F_st). ASSUMPTION, not a measurement.
NEI_SCALED = {"fst_loss": 1.80, "ibd_loss": 1.80}

PRIOR_POP = (2000.0, 25000.0)      # ABCAnalysisNoRedis.py -- U(2000, 25000) since 2026-08-26
CLUSTER_MULT = 33                  # numClusters in the CSV is the raw draw; actual count is x33


# ------------------------------------------------------------------ rank statistics

def _rank(x):
    """Ranks with ties averaged."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(len(x), dtype=float)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(x, y):
    """Spearman rho + two-sided p via the Fisher-z normal approximation.

    A normal approximation is fine here and a permutation test is NOT needed -- unlike 7.1/7.2.1,
    where the units were few, coupled sites. These are independent prior draws, n ~ 2500.
    """
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 5:
        return float("nan"), float("nan")
    rx, ry = _rank(x[m]), _rank(y[m])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan"), float("nan")
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if abs(rho) >= 1.0:
        return rho, 0.0
    z = math.atanh(rho) * math.sqrt(n - 3)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return rho, p


def stars(p):
    if not np.isfinite(p):
        return "   "
    return "***" if p < 1e-3 else ("** " if p < 1e-2 else ("*  " if p < 0.05 else "   "))


# ------------------------------------------------------------------ loading

def load_jobs(raw_dir, prefix):
    """Read every {prefix}<id>.csv. Returns (jobs, problems).

    jobs: {job_id: [row dicts]} -- rows in file order, header rows stripped.
    """
    jobs, problems = {}, []
    paths = sorted(raw_dir.glob(f"{prefix}*.csv"))
    if not paths:
        raise FileNotFoundError(f"No {prefix}*.csv under {raw_dir.resolve()}")

    width = len(EXPECTED_FIELDS)
    for path in paths:
        stem = path.stem[len(prefix):]
        if not stem.lstrip("-").isdigit():
            problems.append(f"{path.name}: filename tail {stem!r} is not an integer job id -- SKIPPED")
            continue
        jid = int(stem)

        with open(path, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]

        hdr_idx = [i for i, r in enumerate(rows) if r == EXPECTED_FIELDS]
        if not hdr_idx:
            got = rows[0] if rows else []
            problems.append(f"{path.name}: no header matching the expected layout (first row: {got}) "
                            f"-- DIFFERENT CODE REVISION? not pooled")
            continue
        if hdr_idx[0] != 0:
            problems.append(f"{path.name}: header is at line {hdr_idx[0]}, not line 0")
        if len(hdr_idx) > 1:
            problems.append(f"{path.name}: {len(hdr_idx)} header rows -- the file was appended to "
                            f"more than once (a re-run wrote into an existing file)")

        skip = set(hdr_idx)
        data = []
        for i, r in enumerate(rows):
            if i in skip:
                continue
            if len(r) != width:
                problems.append(f"{path.name} line {i}: {len(r)} fields, expected {width} -- row dropped")
                continue
            data.append(dict(zip(EXPECTED_FIELDS, r)))

        if jid in jobs:
            problems.append(f"job id {jid} seen twice -- rows CONCATENATED. Duplicate ids re-draw "
                            f"identical parameters (np.random.seed(job_id)) and bias the pool.")
            jobs[jid].extend(data)
        else:
            jobs[jid] = data
    return jobs, problems


def to_arrays(jobs):
    """Flatten to {column: np.array}, plus a job_id column."""
    cols = {c: [] for c in EXPECTED_FIELDS}
    jid_col = []
    for jid in sorted(jobs):
        for row in jobs[jid]:
            jid_col.append(jid)
            for c in EXPECTED_FIELDS:
                cols[c].append(row[c])
    out = {"job_id": np.asarray(jid_col, dtype=float)}
    for c in EXPECTED_FIELDS:
        out[c] = np.asarray([float(v) if v not in ("", "nan", "NaN") else np.nan
                             for v in cols[c]], dtype=float)
    return out


# ------------------------------------------------------------------ report sections

def report_inventory(jobs, problems, expect_trials):
    print("=" * 78)
    print("1. INVENTORY")
    print("=" * 78)
    ids = sorted(jobs)
    counts = {j: len(jobs[j]) for j in ids}
    total = sum(counts.values())
    print(f"  files parsed        : {len(ids)}")
    print(f"  job id range        : {ids[0]} .. {ids[-1]}")
    print(f"  data rows total     : {total}")
    print(f"  rows/job expected   : {expect_trials}")

    missing = [j for j in range(ids[0], ids[-1] + 1) if j not in counts]
    short = sorted(j for j in ids if counts[j] < expect_trials)
    over = sorted(j for j in ids if counts[j] > expect_trials)

    hist = {}
    for n in counts.values():
        hist[n] = hist.get(n, 0) + 1
    print(f"  rows/job histogram  : " +
          ", ".join(f"{n} rows x{hist[n]} jobs" for n in sorted(hist)))

    lost = len(missing) * expect_trials + sum(expect_trials - counts[j] for j in short)
    ideal = (ids[-1] - ids[0] + 1) * expect_trials
    print(f"  trials lost         : {lost} of {ideal} ({100.0 * lost / ideal:.2f}%)")

    if missing:
        print(f"\n  !! {len(missing)} job id(s) ABSENT -- no file at all: {missing}")
        print("     A whole job died (OOM kills the process outright) or its transfer failed.")
        print("     These are INVISIBLE to the `pop` analysis below: no rows means no recorded")
        print("     parameters, so a high-`pop` OOM cannot be distinguished from a random")
        print("     eviction. Check the HTCondor log for these ids before concluding.")
    if short:
        print(f"\n  !! {len(short)} job(s) SHORT of {expect_trials} rows: "
              f"{[(j, counts[j]) for j in short]}")
        print("     A failed trial writes no row and the job continues, so these are per-trial")
        print("     failures, not evictions. See section 2 for whether they track `pop`.")
    if over:
        print(f"\n  !! {len(over)} job(s) with MORE than {expect_trials} rows: "
              f"{[(j, counts[j]) for j in over]} -- a re-run appended to an existing file.")
    if not missing and not short and not over:
        print(f"\n  OK: every job in range wrote exactly {expect_trials} rows.")

    if problems:
        print(f"\n  PARSE PROBLEMS ({len(problems)}):")
        for p in problems[:40]:
            print(f"    - {p}")
        if len(problems) > 40:
            print(f"    ... and {len(problems) - 40} more")
    return counts, missing, short


def report_failures(A, jobs, counts, short, expect_trials):
    print()
    print("=" * 78)
    print("2. DID FAILURES TRACK `pop`?  (the 3.1 memory signature)")
    print("=" * 78)
    if not short:
        print("  No short jobs -- no per-trial failures to attribute. Nothing to test.")
        print("  (Absent jobs, if any, remain untestable here -- see section 1.)")
        return
    # Compare the pop distribution of rows from complete vs short jobs. A per-trial failure
    # concentrated at high pop leaves the SURVIVING rows of those jobs skewed low.
    jid = A["job_id"]
    pop = A["pop"]
    short_set = set(short)
    m_short = np.array([j in short_set for j in jid])
    if m_short.sum() == 0:
        print("  Short jobs contributed no rows at all.")
        return
    print(f"  rows from complete jobs: {int((~m_short).sum())}   "
          f"median pop = {np.median(pop[~m_short]):.0f}")
    print(f"  rows from short jobs   : {int(m_short.sum())}   "
          f"median pop = {np.median(pop[m_short]):.0f}")
    rho, p = spearman(np.array([float(counts[j]) for j in sorted(counts)]),
                      np.array([np.median([float(r["pop"]) for r in jobs[j]]) if jobs[j] else np.nan
                                for j in sorted(counts)]))
    print(f"  spearman(rows written, job median pop) = {rho:+.3f}  p={p:.3g} {stars(p)}")
    print("  A strong NEGATIVE rho means big draws failed -> the 3.1 extrapolation is wrong and")
    print("  the pool is biased toward small POPMULT. Re-size request_memory before batch 2.")


def report_coverage(A):
    print()
    print("=" * 78)
    print("3. PRIOR COVERAGE")
    print("=" * 78)
    pop = A["pop"]
    lo, hi = PRIOR_POP
    n = len(pop)
    print(f"  pop ~ U({lo:.0f}, {hi:.0f}) | n={n}  min={pop.min():.0f}  "
          f"max={pop.max():.0f}  median={np.median(pop):.0f}")
    edges = np.linspace(lo, hi, 11)
    cnt, _ = np.histogram(pop, bins=edges)
    exp = n / 10.0
    print(f"  deciles of the prior (expected {exp:.1f} each):")
    for k in range(10):
        z = (cnt[k] - exp) / math.sqrt(exp)
        bar = "#" * int(round(40.0 * cnt[k] / max(cnt.max(), 1)))
        flag = "  <-- DEFICIT" if z < -3 else ""
        print(f"    {edges[k]:7.0f}-{edges[k+1]:7.0f} {cnt[k]:5d}  z={z:+5.2f} {bar}{flag}")
    print("  A deficit in the TOP decile is the memory signature: draws near the ceiling died.")
    print("  Note this can only show PER-TRIAL failures -- a job lost whole leaves no row here.")

    print()
    for p_ in ["total_migration", "m", "mutation_rate"]:
        v = A[p_]
        print(f"  {p_:16s} min={v.min():.4g}  median={np.median(v):.4g}  max={v.max():.4g}")
    nc = A["numClusters"]
    vals, cts = np.unique(nc[np.isfinite(nc)], return_counts=True)
    print(f"  {'numClusters':16s} " +
          ", ".join(f"{int(v)} (={int(v) * CLUSTER_MULT} demes): {c}" for v, c in zip(vals, cts)))


def report_statistics(A):
    print()
    print("=" * 78)
    print("4. PER-STATISTIC SCALE vs THE 7.3 NOISE FLOOR")
    print("=" * 78)
    print("  sigma = 1.4826*MAD across the batch -- the value abc_standardize.py freezes.")
    print("  floor = 7.3 run-to-run mean|diff| at identical parameters.")
    print("  signal frac = 1 - (floor/sigma)^2 : the share of the batch spread that is NOT noise.")
    print()
    print(f"  {'statistic':12s} {'median':>10s} {'sigma':>10s} {'floor':>10s} "
          f"{'floor/sig':>10s} {'sig.frac':>9s}  note")
    rows = {}
    for s in LOSSES:
        v = A[s]
        sig = robust_sigma(v)
        floor = NOISE_FLOOR[s]
        note = ""
        if s in NEI_SCALED:
            floor *= NEI_SCALED[s]
            note = f"floor SCALED x{NEI_SCALED[s]} (Nei->Hudson, ASSUMED)"
        frac = 1.0 - (floor / sig) ** 2 if np.isfinite(sig) and sig > 0 else np.nan
        rows[s] = {"sigma": sig, "floor": floor, "frac": frac}
        print(f"  {s:12s} {np.median(v):10.6f} {sig:10.6f} {floor:10.6f} "
              f"{floor / sig:10.3f} {frac:9.3f}  {note}")
    print()
    print("  Read this as: two statistics with equal sigma contribute equally to D after")
    print("  standardization, no matter how much of that sigma is noise.")
    return rows


def report_information(A):
    print()
    print("=" * 78)
    print("5. DOES EACH STATISTIC CARRY PARAMETER INFORMATION?  (spearman rho, p)")
    print("=" * 78)
    print("  The decisive column is `pop` -- 7.2.1 put essentially the whole inference on it.")
    print(f"  {'statistic':12s} " + " ".join(f"{p:>18s}" for p in PARAMS))
    for s in LOSSES:
        cells = []
        for p_ in PARAMS:
            rho, p = spearman(A[s], A[p_])
            cells.append(f"{rho:+.3f}{stars(p):>4s}".rjust(18))
        print(f"  {s:12s} " + " ".join(cells))
    print("  *** p<0.001  ** p<0.01  * p<0.05")

    print()
    print("=" * 78)
    print("6. ARE THE LOSSES REDUNDANT?  (7.2 double-counting)")
    print("=" * 78)
    print("  pi and F_st are strongly correlated in the OBSERVED data (r = -0.72 in 2015,")
    print("  -0.92 in 2023). This is the across-prior version -- measured, not inferred.")
    print(f"  {'':12s} " + " ".join(f"{s:>12s}" for s in LOSSES))
    for a in LOSSES:
        cells = []
        for b in LOSSES:
            if a == b:
                cells.append(f"{'--':>12s}")
            else:
                rho, _ = spearman(A[a], A[b])
                cells.append(f"{rho:+12.3f}")
        print(f"  {a:12s} " + " ".join(cells))


def _zrank(v):
    r = _rank(v)
    return (r - r.mean()) / r.std()


def report_decomposition(A):
    """Rank-space variance decomposition of each loss onto the parameters.

    Section 4's "signal fraction" (1 - (floor/sigma)^2) is NOT the right weighting input, and
    this section is why. It only separates batch spread from REPLICATE noise -- it counts
    variance driven by the mu nuisance draw as signal. mu is a nuisance dimension deliberately
    (TODO 1a): it is kept free to absorb the calibration's own uncertainty, and only theta=4Nmu
    is ever reported. Variance a statistic derives from mu is not information about demography;
    weighting on it puts the mu draw into D.
    """
    print()
    print("=" * 78)
    print("7. VARIANCE DECOMPOSITION -- what actually drives each loss")
    print("=" * 78)
    print("  Rank-space multiple regression of each loss on all five parameters.")
    print("  R2_tot = share of batch spread the parameters explain at all; the remainder is")
    print("  replicate noise plus nonlinearity. Columns are UNIQUE R2 (what dropping that")
    print("  parameter costs), so they do not sum to R2_tot when parameters are correlated.")
    print()
    n = len(A["pop"])
    X = np.column_stack([np.ones(n)] + [_zrank(A[p]) for p in PARAMS])
    hdr = f"  {'loss':12s} {'R2_tot':>7s} " + " ".join(f"{p[:10]:>11s}" for p in PARAMS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    out = {}
    for s in LOSSES:
        y = _zrank(A[s])
        sst = float(np.sum((y - y.mean()) ** 2))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1.0 - float(np.sum((y - X @ beta) ** 2)) / sst
        uniq = {}
        for k, p_ in enumerate(PARAMS):
            cols = [0] + [j + 1 for j in range(len(PARAMS)) if j != k]
            Xr = X[:, cols]
            b2, *_ = np.linalg.lstsq(Xr, y, rcond=None)
            uniq[p_] = r2 - (1.0 - float(np.sum((y - Xr @ b2) ** 2)) / sst)
        demog = sum(max(0.0, uniq[p_]) for p_ in PARAMS if p_ != "mutation_rate")
        out[s] = {"r2": r2, "uniq": uniq, "demographic": demog,
                  "nuisance": max(0.0, uniq["mutation_rate"])}
        print(f"  {s:12s} {r2:7.3f} " + " ".join(f"{uniq[p_]:11.3f}" for p_ in PARAMS))

    print()
    print(f"  {'loss':12s} {'demographic':>12s} {'mu-nuisance':>12s} {'unexplained':>12s}")
    for s in LOSSES:
        d = out[s]
        print(f"  {s:12s} {d['demographic']:12.4f} {d['nuisance']:12.4f} {1 - d['r2']:12.3f}")
    print()
    print("  'demographic' = unique R2 of pop + total_migration + m + numClusters. THIS is the")
    print("  share of a statistic's spread that carries information the pass is trying to infer.")
    return out


def report_gradient(A):
    """Median loss by pop decile -- does the fitted set actually resolve POPMULT, and does its
    preference run into the ceiling? TODO 4 asks whether the N marginal escapes the prior; a
    gradient still falling in the top decile is the warning sign, since 6.7 already had to raise
    the ceiling once for exactly this reason."""
    print()
    print("=" * 78)
    print("8. LOSS GRADIENT IN `pop`  (median per decile, raw units)")
    print("=" * 78)
    lo, hi = PRIOR_POP
    edges = np.linspace(lo, hi, 11)
    idx = np.clip(np.digitize(A["pop"], edges) - 1, 0, 9)
    print(f"  {'pop bin':>17s} {'n':>5s} " + " ".join(f"{s[:10]:>11s}" for s in LOSSES))
    meds = {s: [] for s in LOSSES}
    for k in range(10):
        m = idx == k
        row = " ".join(f"{np.median(A[s][m]):11.6f}" for s in LOSSES)
        for s in LOSSES:
            meds[s].append(float(np.median(A[s][m])))
        print(f"  {edges[k]:7.0f}-{edges[k+1]:7.0f} {int(m.sum()):5d} {row}")
    print()
    print("  Decile medians carry their own error: SE(median) ~ 1.2533*sigma/sqrt(n). Differences")
    print("  smaller than that are not a turning point, however suggestive the ordering looks.")
    for s in FITTED:
        v = np.asarray(meds[s])
        nper = len(A[s]) / 10.0
        se = 1.2533 * scale_sigma(A[s]) / math.sqrt(nper)
        best = int(np.argmin(v))
        flat = [k + 1 for k in range(10) if v[k] - v.min() < se]
        print(f"  {s:10s} min at decile {best + 1} ({edges[best]:.0f}-{edges[best+1]:.0f}), "
              f"SE(median)~{se:.6f}")
        print(f"  {'':10s} deciles within 1 SE of the minimum: {flat}")
        tail = "FALLING at the ceiling -- prior may still truncate" if v[-1] < v[-2] - se else \
               "flat/rising at the ceiling -- no evidence the prior truncates"
        print(f"  {'':10s} top decile: {tail}")
    print("  A minimum in decile 10 with the curve STILL FALLING means the prior is truncating")
    print("  the N marginal -- the failure 6.7 caught at POPMULT_MAX=12000. A flat tail instead")
    print("  means the ceiling is adequate but the marginal is broad, not sharply peaked.")


def scale_sigma(v):
    s = robust_sigma(v)
    return s if np.isfinite(s) else float(np.nanstd(v))


def report_recommendation(scale_rows, decomp, A):
    print()
    print("=" * 78)
    print("9. SUGGESTED WEIGHTS for abc_standardize.py")
    print("=" * 78)
    print("  Rule: weight each fitted statistic by the share of its batch spread that is")
    print("  DEMOGRAPHIC signal (5b), not merely non-noise (4). The two disagree whenever a")
    print("  statistic loads on the mu nuisance draw, which is exactly pi's situation.")
    print()
    dem = {s: decomp[s]["demographic"] for s in FITTED}
    tot = sum(dem.values())
    print(f"  FITTED_STATS = {FITTED}")
    if tot <= 0:
        print("  !! No fitted statistic carries demographic signal. Do not run the pass on this.")
        return
    print("  WEIGHTS = {")
    for s in FITTED:
        print(f"      {s!r}: {dem[s] / tot:.3f},        # demographic R2 {dem[s]:.4f}, "
              f"sigma {scale_rows[s]['sigma']:.6f}")
    print("  }")
    print()
    naive = {s: max(0.0, scale_rows[s]["frac"]) for s in FITTED}
    ntot = sum(naive.values())
    print("  For contrast, the two weightings this batch rules out:")
    print(f"    equal weights            : " +
          "  ".join(f"{s}={1.0 / len(FITTED):.3f}" for s in FITTED))
    if ntot > 0:
        print(f"    noise-floor rule only (4): " +
              "  ".join(f"{s}={naive[s] / ntot:.3f}" for s in FITTED))
    print("  Both over-weight pi, for the same reason: they treat its mu-driven spread as signal.")
    print()
    print("  Caveat kept: the fst_loss noise floor in section 4 is SCALED, not measured")
    print("  (6.7 / TODO 4). It does not enter this rule -- these weights come from 5b, which")
    print("  needs no floor at all. That is a further reason to prefer them.")


# ------------------------------------------------------------------ main

def write_concat(A, out_path):
    n = len(A["job_id"])
    fields = ["job_id"] + EXPECTED_FIELDS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for i in range(n):
            w.writerow([int(A["job_id"][i]), int(A["iteration"][i])] +
                       [repr(float(A[c][i])) for c in EXPECTED_FIELDS[1:]])
    print(f"\nWrote {n} rows + header -> {out_path}")
    print("  Columns: job_id (recovered from filename) + the 12 written by ABCAnalysisNoRedis.py.")
    print("  abc_standardize.py reads ../out/abc_results.csv and ignores the extra column.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="../out/batch1_raw")
    p.add_argument("--out", default="../out/abc_results.csv")
    p.add_argument("--prefix", default="abc_results_")
    p.add_argument("--expect-trials", type=int, default=5)
    p.add_argument("--no-write", action="store_true", help="report only; write nothing")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)

    jobs, problems = load_jobs(raw_dir, args.prefix)
    counts, missing, short = report_inventory(jobs, problems, args.expect_trials)
    A = to_arrays(jobs)
    report_failures(A, jobs, counts, short, args.expect_trials)
    report_coverage(A)
    scale_rows = report_statistics(A)
    report_information(A)
    decomp = report_decomposition(A)
    report_gradient(A)
    report_recommendation(scale_rows, decomp, A)

    if args.no_write:
        print("\n--no-write: nothing written.")
    else:
        if out_path.exists():
            print(f"\n!! {out_path} already exists -- OVERWRITING. TODO 0: never pool results "
                  f"across the 07-28/07-29 fitted-set change or the 6.7 F_st fix.")
        write_concat(A, out_path)


if __name__ == "__main__":
    main()
