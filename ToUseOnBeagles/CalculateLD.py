#!/usr/bin/env python3
"""Per-year LD decay (mean r^2 vs physical distance) for each subpopulation.

Empirical half of the LD statistic (CLAUDE.md 7.5). All binning, filtering and pair enumeration
live in `ld_common.py`, which the SIMULATED side imports too -- reimplementing either half is the
drift that broke `qdriver.py` (CLAUDE.md 3).

`ld_common.py` must be COPIED next to this file on the Beagle machine. Both sides print
`spec_hash()`; if they disagree the two curves are not comparable and any ld_loss is meaningless.

RE-BINNED TO LOG SPACING 2026-09-06. The previous 100 linear 1-kb bins out to 100 kb cannot be
used under either reading of the recombination rate: at r = 2.75e-6 the whole simulated decay
falls inside the old bin 1, and at r = 2.75e-8 the low-POPMULT corner decays at 135 kb, past the
old MAX_DIST. Since rho's units are ambiguous over ~1000x (CLAUDE.md 6.8) the scale is not
derivable -- so THIS RUN MEASURES IT. Look at the "crosses halfway" line.

PARALLEL ACROSS CHROMOSOMES. One process per chromosome, so each VCF is read exactly once and
all three years are computed from that single read. Profiling put the serial cost at matmul 29%,
binning 46%, masking/distance 25% -- no single hot spot to optimise away, so the win has to come
from parallelism.

  * `spec_hash()` is UNCHANGED by this: parallelism touches none of BIN_EDGES / MIN_MAF / STAGES,
    so results stay comparable to any serial run, and a run already in flight stays valid.
  * Results are summed in CHROMOSOME ORDER, not completion order -- float addition is not
    associative, so pooling in arrival order would make the output non-reproducible.
  * MEMORY IS THE LIMIT, NOT CPU. Each worker holds one chromosome's genotypes (~2 GB int8 for
    chr1 at 6.5M SNPs x ~300 haplotypes) and `allel.read_vcf` peaks well above its final size
    while parsing. 16 workers can therefore want 50+ GB. Lower LD_WORKERS if the machine swaps
    -- swapping will be far slower than running fewer workers.

Usage:
    python CalculateLD.py                 # min(16, cpu_count) workers
    LD_WORKERS=8 python CalculateLD.py    # fewer, if memory is tight
    LD_WORKERS=1 python CalculateLD.py    # serial, for debugging

Output: ld_out/averaged_ldDecay_{year}.csv (pooled over chromosomes) plus one file per
chromosome. Copy the pooled files to data/empiricalStats/.
"""
import os

# Set BEFORE numpy is imported anywhere, including via ld_common. One BLAS thread per worker: the
# r^2 matmul has an inner dimension of only n_haplotypes (~14), which BLAS cannot usefully thread,
# and 16 processes each spawning their own thread pool would oversubscribe the machine badly.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import glob, re, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np, allel

# ld_common.py sits next to this file on the Beagle machine; in the repo it lives in Python_Code/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "Python_Code"))
import ld_common as ldc

VCF_FILES = sorted(glob.glob("chr*_cpb.vcf.gz"))       # your 17 chromosome files
POPFILES  = {2015: "popFile2015", 2019: "popFile2019", 2023: "popFile2023"}
OUTDIR    = "ld_out"
N_WORKERS = int(os.environ.get("LD_WORKERS", min(16, os.cpu_count() or 1)))


def load_pop(path):
    """sample_id -> population label (popfile: 'sampleID<TAB>pop')."""
    d = {}
    for line in open(path):
        sid, pop = line.split()[:2]
        d[sid] = pop
    return d


def pops_of(popfile):
    """Population labels in popfile order.

    NOTE (CLAUDE.md 7.5e): this is POPFILE order. Verified 2026-09-06 to be identical to
    specifier-matrix order in all three years, which is what every other matrix in the project
    uses -- but that is a checked fact, not a guarantee. Re-check if the popfiles are regenerated.
    """
    return list(dict.fromkeys(load_pop(popfile).values()))


def haplotypes(vcf):
    """Sorted unique positions and a (n_sites, 2*n_samples) 0/1 haplotype matrix for biallelic
    SNPs on one chromosome, plus the sample list."""
    c = allel.read_vcf(vcf, fields=['variants/POS', 'variants/REF',
                                    'variants/ALT', 'calldata/GT', 'samples'])
    ref = c['variants/REF'].astype(str)
    alt = c['variants/ALT'].astype(str)
    snp = (np.char.str_len(ref) == 1) & (np.char.str_len(alt[:, 0]) == 1)
    if alt.shape[1] > 1:                       # exclude multiallelic
        snp &= (np.char.str_len(alt[:, 1:]) == 0).all(axis=1)
    gt = c['calldata/GT'][snp]                 # (n_snp, n_samples, 2), int8
    H = gt.reshape(gt.shape[0], -1)            # haplotypes: sample s -> cols 2s, 2s+1
    pos = c['variants/POS'][snp].astype(np.int64)
    # ld_common.accumulate assumes ascending unique positions.
    pos, first = np.unique(pos, return_index=True)
    # DELIBERATELY left as int8. A float64 copy of a whole chromosome is ~15 GB at chr1's
    # 6.5M SNPs x ~300 haplotypes; ld_common.standardize() casts the 14-column per-population
    # slice instead, which is ~700 MB. Do not "tidy" this into an .astype(float) here.
    return pos, H[first], list(c['samples'])


def process_chromosome(vcf):
    """One worker task: read ONE chromosome once, do every year and every population on it.

    Returns (chrom, {year: (sum_r2, cnt)}). Keeping all three years inside one task is the whole
    point -- the VCF read is the expensive serial part, and splitting by (year, chromosome) would
    pay it three times.
    """
    chrom = re.search(r"chr[^0-9]*([0-9]+)", os.path.basename(vcf)).group(1)
    t0 = time.perf_counter()
    pos, H, samples = haplotypes(vcf)
    idx = {s: i for i, s in enumerate(samples)}
    t_read = time.perf_counter() - t0

    out = {}
    for year, popfile in POPFILES.items():
        samp2pop = load_pop(popfile)
        pops = pops_of(popfile)
        s_sum = np.zeros((ldc.N_BINS, len(pops)))
        s_cnt = np.zeros((ldc.N_BINS, len(pops)))
        for k, p in enumerate(pops):
            cols = [idx[s] for s in samp2pop if samp2pop[s] == p and s in idx]
            if not cols:
                continue
            hap_cols = np.ravel([[2 * c, 2 * c + 1] for c in cols])
            ldc.decay_one_pop(pos, H[:, hap_cols], s_sum[:, k], s_cnt[:, k])
        ldc.write_decay(f"{OUTDIR}/chr{chrom}_{year}_ldDecay.csv", s_sum, s_cnt, pops)
        out[year] = (s_sum, s_cnt)

    print(f"  chr{chrom}: {len(pos):,} sites | read {t_read:.0f}s | "
          f"total {time.perf_counter() - t0:.0f}s", flush=True)
    return chrom, out


def main():
    print(f"ld_common spec {ldc.spec_hash()} (version {ldc.SPEC_VERSION}) -- "
          f"the simulated side must print the same hash")
    if not VCF_FILES:
        raise SystemExit("No chr*_cpb.vcf.gz found in the working directory.")
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"{len(VCF_FILES)} chromosomes, {N_WORKERS} worker process(es), "
          f"1 BLAS thread each. Memory, not CPU, is the limit -- "
          f"lower LD_WORKERS if the machine swaps.")

    t0 = time.perf_counter()
    results = {}
    if N_WORKERS <= 1:
        for vcf in VCF_FILES:
            ch, out = process_chromosome(vcf)
            results[ch] = out
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            for ch, out in ex.map(process_chromosome, VCF_FILES):
                results[ch] = out

    # Pool in CHROMOSOME order, never completion order: float addition is not associative, so
    # arrival-order pooling would make the pooled file differ run to run.
    for year, popfile in POPFILES.items():
        pops = pops_of(popfile)
        g_sum = np.zeros((ldc.N_BINS, len(pops)))
        g_cnt = np.zeros((ldc.N_BINS, len(pops)))
        for ch in sorted(results, key=lambda c: int(c)):
            s, c = results[ch][year]
            g_sum += s; g_cnt += c
        ldc.write_decay(f"{OUTDIR}/averaged_ldDecay_{year}.csv", g_sum, g_cnt, pops)
        print(f"averaged_ldDecay_{year}.csv: {len(pops)} pops, {ldc.N_BINS} log bins, "
              f"{int(g_cnt.sum())} SNP pairs")
        ldc.report_halfway(g_sum, g_cnt, prefix="    ")

    print(f"\ntotal wall time {(time.perf_counter() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
