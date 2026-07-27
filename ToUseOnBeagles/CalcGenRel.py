#!/usr/bin/env python3
"""Per-year pairwise genetic relatedness between subpopulations.

Reproduces tskit's genetic_relatedness(mode="site") with its default options.
For two subpopulations i, j the value is

    GR(i, j) = [ sum_b (p_i,b - pbar_b)(p_j,b - pbar_b) ] / D

where p_k,b is the ALT-allele frequency of subpop k at site b, pbar_b is the
mean of those frequencies across the year's subpops, and D is the normalising
denominator set by NORMALISE_BY. Writes a per-chromosome matrix plus a pooled
genome-wide matrix for each year.

DENOMINATOR (DIVERSITY_SCALE_ISSUE.md 1.5)
------------------------------------------
Relatedness is a sum of centred allele-frequency products -- a level, not a
ratio -- so unlike Fst its magnitude depends on which sites it is normalised
over. tskit applies span_normalise=True by default, making the simulated side
per base pair over the full 1e6 bp of simulated sequence, invariant sites
included. Normalising the empirical side by segregating sites instead puts the
two on different scales by roughly the SNP density.

Note this script already pooled correctly across chromosomes (summed numerators
over summed denominators); the denominator itself was the only defect.
"""
import glob, os, re, csv
import numpy as np, allel

from CallableSites import CALLABLE_SITES, require_callable_sites

VCF_FILES = sorted(glob.glob("chr*_cpb.vcf.gz"))        # your 17 chromosome files
POPFILES  = {2015: "popFile2015", 2019: "popFile2019", 2023: "popFile2023"}
OUTDIR    = "genRel_out"
SPECIFIER = "specifier_matrix_{year}.csv"   # defines subpopulation ROW ORDER

# "callable"    -- per callable site, matching tskit's span_normalise=True.
#                  Use this for anything compared against the simulated side.
# "segregating" -- per segregating site. Keeps the estimator check against
#                  tskit reproducible (CLAUDE.md 8 records agreement to ~1e-18
#                  on identical input); do not use it for production matrices.
NORMALISE_BY = "callable"


def load_pop(path):
    """Map each sample ID to its population label (popfile: 'sampleID<TAB>pop')."""
    samp2pop = {}
    for line in open(path):
        sid, pop = line.split()[:2]
        samp2pop[sid] = pop
    return samp2pop


def load_site_order(year):
    """Subpopulation order for one year, taken from the specifier matrix.

    This is the ordering the whole project indexes by: subpop i is row i of
    specifier_matrix_{year}.csv, which is also what AverageData.py writes the
    pi/dxy/Fst matrices in and what the simulated matrices use (CLAUDE.md 4).
    Ordering by sorted(popfile labels) instead -- as this script used to --
    permutes the rows relative to every other matrix in the pipeline, silently
    pairing the wrong subpopulations in the ABC comparison.
    """
    with open(SPECIFIER.format(year=year)) as f:
        return [line.strip().split(',')[0] for line in f if line.strip()]


def write_matrix(path, M, pops, labels=True):
    """Write a population-by-population matrix to CSV.

    labels=False writes bare numbers, matching averaged_dxy/averaged_fst and
    what ABCAnalysisNoRedis._read_matrix expects -- it parses every field as a
    float, so a header row or label column makes it fail.
    """
    w = csv.writer(open(path, "w", newline=""))
    if labels:
        w.writerow([""] + pops)
        for label, row in zip(pops, M):
            w.writerow([label] + [f"{v:.8g}" for v in row])
    else:
        for row in M:
            w.writerow([f"{v:.8g}" for v in row])


def chunk_contribution(gt, subpops, pops):
    """Contribution of one block of variants to the relatedness numerator and
    the segregating-site count."""
    # ALT-allele count and total called alleles per population, per site
    ac = gt.count_alleles_subpops(subpops, max_allele=1)
    ALT = np.array([ac[p][:, 1] for p in pops], float)        # (K, n_sites)
    AN  = np.array([ac[p].sum(axis=1) for p in pops], float)  # (K, n_sites)

    # keep only sites that vary across the year's samples pooled together
    keep = (ALT.sum(0) > 0) & (ALT.sum(0) < AN.sum(0))

    P = ALT[:, keep] / AN[:, keep]    # per-population ALT frequency, (K, n_keep)
    C = P - P.mean(0)                 # centre each site across the populations
    return C @ C.T, int(keep.sum())   # (K, K) cross-products, and n_sites kept


def chrom_matrices(vcf, subpops, pops):
    """Sum the numerator and segregating-site count over one chromosome,
    reading the VCF in chunks to keep memory bounded."""
    K = len(pops)
    num, seg = np.zeros((K, K)), 0
    _, _, _, chunks = allel.iter_vcf_chunks(vcf, fields=["calldata/GT"])
    for chunk in chunks:
        gt = allel.GenotypeArray(chunk[0]["calldata/GT"])
        n, s = chunk_contribution(gt, subpops, pops)
        num += n
        seg += s
    return num, seg


def denominator(chrom, seg):
    """Normalising denominator for one chromosome, per NORMALISE_BY."""
    if NORMALISE_BY == "segregating":
        return seg
    if NORMALISE_BY == "callable":
        return CALLABLE_SITES[int(chrom)]
    raise SystemExit(f"NORMALISE_BY must be 'callable' or 'segregating', got {NORMALISE_BY!r}")


def process_year(year, popfile, vcf_files, sample_pos, outdir):
    """Per-chromosome and pooled genome-wide relatedness matrices for one year."""
    samp2pop = load_pop(popfile)
    pops = load_site_order(year)

    # The popfile and the specifier matrix must name the same subpopulations,
    # or the rows are being ordered by a list that does not describe this year.
    inPopfile = set(samp2pop.values())
    if inPopfile != set(pops):
        raise SystemExit(
            f"{year}: popfile and specifier matrix disagree.\n"
            f"  only in specifier matrix: {sorted(set(pops) - inPopfile)}\n"
            f"  only in popfile:          {sorted(inPopfile - set(pops))}"
        )

    # VCF column indices belonging to each population
    subpops = {p: [sample_pos[s] for s in samp2pop if samp2pop[s] == p]
               for p in pops}
    K = len(pops)

    gnum, gden, gseg = np.zeros((K, K)), 0, 0     # genome-wide running totals
    for vcf in vcf_files:
        chrom = re.search(r"chr[^0-9]*([0-9]+)", os.path.basename(vcf)).group(1)
        num, seg = chrom_matrices(vcf, subpops, pops)
        den = denominator(chrom, seg)
        write_matrix(f"{outdir}/chr{chrom}_{year}_genRel.csv", num / den, pops)
        gnum += num
        gden += den
        gseg += seg

    # pooled numerators / pooled denominators = correct genome-wide value.
    # Headerless: this file feeds the ABC directly (see write_matrix).
    write_matrix(f"{outdir}/averaged_genRel_{year}.csv", gnum / gden, pops, labels=False)
    # Both counts are reported because their ratio is the SNP density -- the
    # factor DIVERSITY_SCALE_ISSUE.md 4a asks you to check the old empirical
    # matrices against.
    print(f"averaged_genRel_{year}.csv: {K}x{K}, normalised by {NORMALISE_BY}, "
          f"{gden} denominator sites, {gseg} segregating sites")


def main():
    if NORMALISE_BY == "callable":
        require_callable_sites()
    os.makedirs(OUTDIR, exist_ok=True)
    # sample order, read once from the first VCF (assumed identical across files)
    _, samples, _, _ = allel.iter_vcf_chunks(VCF_FILES[0], fields=["variants/POS"])
    sample_pos = {s: i for i, s in enumerate(samples)}
    for year, popfile in POPFILES.items():
        process_year(year, popfile, VCF_FILES, sample_pos, OUTDIR)


if __name__ == "__main__":
    main()