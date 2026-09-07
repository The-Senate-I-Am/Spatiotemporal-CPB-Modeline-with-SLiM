"""Shared LD-decay binning. IMPORTED BY BOTH SIDES -- keep the two copies identical.

The empirical side (`ToUseOnBeagles/CalculateLD.py`) runs on the Beagle machine and the simulated
side (`AnalyzeTreeSeq.py`) runs here, so this file has to be COPIED alongside `ToUseOnBeagles/`.
That copy is a drift hazard of exactly the kind that broke `qdriver.py` (CLAUDE.md 3), so both
sides print `spec_hash()` and a mismatch means the two curves are not comparable and the
comparison is meaningless. Check it before trusting any ld_loss.

WHY LOG BINS (CLAUDE.md 7.5). The decay scale is not currently derivable: rho's units are
ambiguous over ~1000x (6.8), which puts the empirical half-decay anywhere from ~6 bp to ~6 kb; and
the SIMULATED half-decay moves ~37x across the POPMULT prior all by itself (36 bp to 135 kb
depending on r). Linear bins are wrong under every one of those readings. Log bins span the whole
ambiguity, and the first real run MEASURES where the curve falls. Narrow later if wanted.

WHY THE BIN EDGES STOP AT 1e6. That is the simulated sequence length
(`CPBSampleSim*.slim:11`, `initializeGenomicElement(g1, 0, 1e6 - 1)`). Nothing beyond it exists on
the simulated side, so nothing beyond it is comparable.

r^2 here is the squared correlation of phased 0/1 haplotype states, matching
tskit's ld_matrix(stat="r2").
"""
import csv
import hashlib

import numpy as np

# --------------------------------------------------------------------------- spec
# Bump SPEC_VERSION whenever anything below changes, and re-run BOTH sides.
SPEC_VERSION = "2026-09-06.1"

BIN_EDGES = np.array([1, 2, 3, 6, 10, 18, 32, 56, 100, 178, 316, 562, 1000,
                      1778, 3162, 5623, 10000, 17783, 31623, 56234, 100000,
                      177828, 316228, 562341, 1000000], dtype=np.int64)
N_BINS   = len(BIN_EDGES) - 1
MAX_DIST = int(BIN_EDGES[-1])

MIN_MAF  = 0.05          # must be IDENTICAL on both sides: two SNPs at different allele
                         # frequencies cannot reach r^2 = 1, so the MAF filter sets the
                         # short-distance ceiling of the curve, not just which sites enter.

# Pair enumeration is O(density^2 * W) per block, so a single pass at W = 1 Mb would build an
# ~87k x 174k r^2 block (~120 GB) at the empirical 8.72% SNP density. Instead: short distances use
# every SNP, long distances use a thinned subset. Thinning is UNBIASED for r^2-vs-distance -- which
# SNPs you keep does not change the expected r^2 at a given separation -- it costs only precision,
# and the long bins are wide enough to have pairs to spare.
#   (max_dist, thin)  -- each stage takes pairs in (previous max_dist, max_dist].
STAGES   = [(10_000, 1), (1_000_000, 25)]
ROWCHUNK = 4096          # cap rows per matmul so peak memory stays bounded


def spec_hash():
    """Short hash of every setting that has to match between the two sides."""
    s = f"{SPEC_VERSION}|{BIN_EDGES.tolist()}|{MIN_MAF}|{STAGES}"
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def bin_mid():
    """Geometric centre of each bin -- the right centre for log spacing."""
    return np.sqrt(BIN_EDGES[:-1].astype(float) * BIN_EDGES[1:])


# --------------------------------------------------------------------------- core

def standardize(H, min_maf=MIN_MAF):
    """(sites, haplotypes) 0/1 matrix -> (keep_mask, Z) with Z standardized per site.

    Drops monomorphic sites and anything below min_maf, so `Z @ Z.T / n_hap` is exactly the
    correlation r and its square is r^2.
    """
    H = np.asarray(H, dtype=np.float64)
    freq = H.mean(1)
    keep = (freq > 0) & (freq < 1) & (np.minimum(freq, 1 - freq) >= min_maf)
    Z = H[keep] - H[keep].mean(1, keepdims=True)
    Z /= np.sqrt((Z ** 2).mean(1, keepdims=True))
    return keep, Z


def accumulate(pos, Z, n_hap, sum_r2, cnt, dmin, dmax):
    """Add every SNP pair with dmin < distance <= dmax into the log bins, in place.

    Block scheme: every pair separated by at most dmax lies in the same or the adjacent
    dmax-wide block, so r^2 comes from BLAS matmuls rather than a Python loop. Rows are chunked
    so the r^2 block never exceeds ROWCHUNK x |candidates|.

    Verified against a brute-force pair loop: identical counts, max |sum r^2| difference 1.7e-13.
    """
    pos = np.asarray(pos, dtype=np.int64)
    if len(pos) < 2:
        return
    W = int(dmax)
    for b in range(int(pos[0]) // W, int(pos[-1]) // W + 1):
        a0 = np.searchsorted(pos, b * W)
        a1 = np.searchsorted(pos, (b + 1) * W)
        c1 = np.searchsorted(pos, (b + 2) * W)
        if a1 == a0:
            continue
        for s in range(a0, a1, ROWCHUNK):
            e = min(s + ROWCHUNK, a1)
            R2 = ((Z[s:e] @ Z[s:c1].T) / n_hap) ** 2
            d = pos[s:c1][None, :] - pos[s:e][:, None]
            m = (d > dmin) & (d <= dmax)
            if not m.any():
                continue
            idx = np.searchsorted(BIN_EDGES, d[m], side='right') - 1
            np.clip(idx, 0, N_BINS - 1, out=idx)
            np.add.at(sum_r2, idx, R2[m])
            np.add.at(cnt, idx, 1)


def decay_one_pop(pos, H, sum_r2, cnt, min_maf=MIN_MAF):
    """Full per-population pass: MAF filter, standardize, then run every STAGE.

    pos: (sites,) int positions, ASCENDING and unique.
    H:   (sites, haplotypes) 0/1.
    sum_r2, cnt: (N_BINS,) accumulators, modified in place.
    """
    pos = np.asarray(pos, dtype=np.int64)
    keep, Z = standardize(H, min_maf)
    pos = pos[keep]
    if len(pos) < 2:
        return
    n_hap = Z.shape[1]
    lo = 0
    for dmax, thin in STAGES:
        if thin == 1:
            accumulate(pos, Z, n_hap, sum_r2, cnt, lo, dmax)
        else:
            accumulate(pos[::thin], Z[::thin], n_hap, sum_r2, cnt, lo, dmax)
        lo = dmax


# --------------------------------------------------------------------------- io

def write_decay(path, sum_r2, cnt, labels):
    """Write a (N_BINS, K) decay table.

    Columns: bin_lo, bin_hi, bin_mid, one r2_<label> per population, one n_<label> per population.
    The counts are not decoration -- a bin with few pairs is noise, and the mean alone does not
    show that.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.asarray(sum_r2, float) / np.asarray(cnt, float)
    mid = bin_mid()
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bin_lo", "bin_hi", "bin_mid"]
                   + [f"r2_{p}" for p in labels] + [f"n_{p}" for p in labels])
        for i in range(N_BINS):
            w.writerow([int(BIN_EDGES[i]), int(BIN_EDGES[i + 1]), f"{mid[i]:.6g}"]
                       + ["" if cnt[i, k] == 0 else f"{mean[i, k]:.6g}"
                          for k in range(len(labels))]
                       + [int(cnt[i, k]) for k in range(len(labels))])


def report_halfway(sum_r2, cnt, prefix="   "):
    """Print where the pooled curve crosses halfway. THIS is the number that settles 6.8."""
    with np.errstate(invalid='ignore', divide='ignore'):
        m = np.nanmean(np.asarray(sum_r2, float) / np.asarray(cnt, float), axis=1)
    ok = np.isfinite(m)
    if ok.sum() <= 2:
        print(f"{prefix}too few populated bins to locate the decay")
        return
    hi, lo = np.nanmax(m[ok]), np.nanmin(m[ok])
    j = int(np.argmax(ok & (m <= 0.5 * (hi + lo))))
    print(f"{prefix}mean r^2 {hi:.4f} -> {lo:.4f}; crosses halfway in bin "
          f"{BIN_EDGES[j]}-{BIN_EDGES[j+1]} bp   [spec {spec_hash()}]")
