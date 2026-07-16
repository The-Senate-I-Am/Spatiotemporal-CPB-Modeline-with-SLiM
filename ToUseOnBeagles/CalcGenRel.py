#!/usr/bin/env python3
"""Per-year pairwise genetic relatedness between subpopulations.

Reproduces tskit's genetic_relatedness(mode="site") with its default options.
For two subpopulations i, j the value is

    GR(i, j) = [ sum_b (p_i,b - pbar_b)(p_j,b - pbar_b) ] / S

where p_k,b is the ALT-allele frequency of subpop k at site b, pbar_b is the
mean of those frequencies across the year's subpops, and S is the number of
segregating sites. Writes a per-chromosome matrix plus a pooled genome-wide
matrix for each year.
"""
import glob, os, re, csv
import numpy as np, allel

VCF_FILES = sorted(glob.glob("chr*_cpb.vcf.gz"))        # your 17 chromosome files
POPFILES  = {2015: "popFile2015", 2019: "popFile2019", 2023: "popFile2023"}
OUTDIR    = "genRel_out"


def load_pop(path):
    """Map each sample ID to its population label (popfile: 'sampleID<TAB>pop')."""
    samp2pop = {}
    for line in open(path):
        sid, pop = line.split()[:2]
        samp2pop[sid] = pop
    return samp2pop


def write_matrix(path, M, pops):
    """Write a labelled population-by-population matrix to CSV."""
    w = csv.writer(open(path, "w", newline=""))
    w.writerow([""] + pops)
    for label, row in zip(pops, M):
        w.writerow([label] + [f"{v:.8g}" for v in row])


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


def process_year(year, popfile, vcf_files, sample_pos, outdir):
    """Per-chromosome and pooled genome-wide relatedness matrices for one year."""
    samp2pop = load_pop(popfile)
    pops = sorted(set(samp2pop.values()))
    # VCF column indices belonging to each population
    subpops = {p: [sample_pos[s] for s in samp2pop if samp2pop[s] == p]
               for p in pops}
    K = len(pops)

    gnum, gseg = np.zeros((K, K)), 0          # genome-wide running totals
    for vcf in vcf_files:
        chrom = re.search(r"chr[^0-9]*([0-9]+)", os.path.basename(vcf)).group(1)
        num, seg = chrom_matrices(vcf, subpops, pops)
        write_matrix(f"{outdir}/chr{chrom}_{year}_genRel.csv", num / seg, pops)
        gnum += num
        gseg += seg

    # pooled numerators / pooled segregating sites = correct genome-wide value
    write_matrix(f"{outdir}/averaged_genRel_{year}.csv", gnum / gseg, pops)
    print(f"averaged_genRel_{year}.csv: {K}x{K}, {gseg} segregating sites")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # sample order, read once from the first VCF (assumed identical across files)
    _, samples, _, _ = allel.iter_vcf_chunks(VCF_FILES[0], fields=["variants/POS"])
    sample_pos = {s: i for i, s in enumerate(samples)}
    for year, popfile in POPFILES.items():
        process_year(year, popfile, VCF_FILES, sample_pos, OUTDIR)


if __name__ == "__main__":
    main()