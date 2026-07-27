"""Pool pixy's per-chromosome output into genome-wide per-site statistics.

Two corrections over the previous version of this script (DIVERSITY_SCALE_ISSUE.md):

1.4  Chromosomes are POOLED, not averaged. pi, dxy and Fst are ratios, so a
     genome-wide value is sum(numerators) / sum(denominators). Taking a plain
     mean of 17 per-chromosome ratios over-weights sparse chromosomes.

1.1  pi and dxy are converted from per-SNP to PER-SITE. pixy ran on a
     variant-sites-only VCF, so its count_comparisons covers SNPs only. The
     denominator is extended over all callable sites (see CallableSites.py).

Fst needs neither correction in principle -- it is a ratio of variance
components, so per-site scaling cancels -- but see the FST POOLING note below
for what this script can and cannot do about pooling it.

Run from the directory holding the statsChr*_{year}/ folders.
"""
import os

from CallableSites import NUM_CHRS, require_callable_sites

# --- CONFIG ---------------------------------------------------------------
YEARS = ("2015", "2019", "2023")
OUTDIR = "finalStats"

# 0-based column offsets in the pixy output files.
#
# VERIFY THESE against the header line of your own output before trusting the
# numbers -- they shift between files because pi has one population column
# while dxy and fst have two, and they can change between pixy versions.
#
#   pi:  pop  chromosome  window_pos_1  window_pos_2  avg_pi  no_sites
#        count_diffs  count_comparisons  count_missing
#   dxy: pop1 pop2  chromosome  window_pos_1  window_pos_2  avg_dxy  no_sites
#        count_diffs  count_comparisons  count_missing
#   fst: pop1 pop2  chromosome  window_pos_1  window_pos_2  avg_wc_fst  no_snps
PI_POP, PI_NO_SITES, PI_DIFFS, PI_COMPARISONS = 0, 5, 6, 7
DXY_POP1, DXY_POP2, DXY_NO_SITES, DXY_DIFFS, DXY_COMPARISONS = 0, 1, 6, 7, 8
FST_POP1, FST_POP2, FST_VALUE, FST_NO_SNPS = 0, 1, 5, 6
# --------------------------------------------------------------------------


def read_pixy_rows(path):
    """Yield the split fields of each data row, skipping the header.

    The header of every pixy file starts with 'pop' ('pop' or 'pop1'), which no
    population label does.
    """
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("pop") or not line.strip():
                continue
            yield line.strip().split()


def number(value):
    """Parse a pixy field, mapping its NA sentinel to None."""
    if value == "NA":
        return None
    return float(value)


def per_site_denominator(count_comparisons, no_sites, callable_sites):
    """Extend a SNPs-only comparison count over every callable site.

    pixy's count_comparisons is summed over the sites present in the VCF, which
    here are all variant. The mean number of pairwise comparisons per site is
    count_comparisons / no_sites; over the full callable sequence the
    denominator is that mean times the callable-site count.

    The numerator needs no adjustment: invariant sites contribute zero
    differences, so count_diffs is already the correct per-site numerator.

    Assumes missingness at invariant sites matches missingness at variant
    sites. That is the standard assumption when back-correcting a SNPs-only
    VCF, and it is the reason an all-sites VCF is the better fix if one ever
    becomes available.
    """
    if not no_sites:
        return 0.0
    return (count_comparisons / no_sites) * callable_sites


def main():
    callable_sites = require_callable_sites(NUM_CHRS)
    os.makedirs(OUTDIR, exist_ok=True)

    for year in YEARS:
        # Site order is fixed by the specifier matrix so that every statistic
        # is written in the same order the simulated matrices use.
        siteNames = []
        with open(f"specifier_matrix_{year}.csv", 'r') as f:
            for line in f:
                siteNames.append(line.strip().split(',')[0])
        numSites = len(siteNames)
        index = {name: i for i, name in enumerate(siteNames)}

        # Running numerators and denominators, pooled across chromosomes.
        piDiffs = [0.0] * numSites
        piDenom = [0.0] * numSites
        dxyDiffs = [[0.0] * numSites for _ in range(numSites)]
        dxyDenom = [[0.0] * numSites for _ in range(numSites)]
        fstWeighted = [[0.0] * numSites for _ in range(numSites)]
        fstWeights = [[0.0] * numSites for _ in range(numSites)]

        for chrom in range(1, NUM_CHRS + 1):
            folderName = f"statsChr{chrom}_{year}"
            nCallable = callable_sites[chrom]

            for values in read_pixy_rows(os.path.join(folderName, f"chr{chrom}_{year}_pi.txt")):
                diffs = number(values[PI_DIFFS])
                comparisons = number(values[PI_COMPARISONS])
                noSites = number(values[PI_NO_SITES])
                if diffs is None or comparisons is None or noSites is None:
                    continue
                i = index[values[PI_POP]]
                piDiffs[i] += diffs
                piDenom[i] += per_site_denominator(comparisons, noSites, nCallable)

            for values in read_pixy_rows(os.path.join(folderName, f"chr{chrom}_{year}_dxy.txt")):
                diffs = number(values[DXY_DIFFS])
                comparisons = number(values[DXY_COMPARISONS])
                noSites = number(values[DXY_NO_SITES])
                if diffs is None or comparisons is None or noSites is None:
                    continue
                i, k = index[values[DXY_POP1]], index[values[DXY_POP2]]
                denom = per_site_denominator(comparisons, noSites, nCallable)
                dxyDiffs[i][k] += diffs
                dxyDiffs[k][i] += diffs
                dxyDenom[i][k] += denom
                dxyDenom[k][i] += denom

            # FST POOLING. Correct pooling (Bhatia et al. 2013) is
            # sum(a) / sum(a + b) over the Weir-Cockerham variance components,
            # but pixy emits only avg_wc_fst and no_snps -- the components are
            # not in the file, so that ratio cannot be reconstructed here.
            # Weighting each chromosome by its SNP count is the closest
            # available approximation and is strictly better than the plain
            # mean this script used to take. To pool Fst properly, recompute it
            # from the VCFs with allel.weir_cockerham_fst, which does return the
            # components. The residual error is a few percent, and Fst is
            # unaffected by the per-site denominator issue, so this is the
            # lowest-stakes of the three statistics.
            for values in read_pixy_rows(os.path.join(folderName, f"chr{chrom}_{year}_fst.txt")):
                fst = number(values[FST_VALUE])
                nSnps = number(values[FST_NO_SNPS])
                if fst is None or nSnps is None:
                    continue
                i, k = index[values[FST_POP1]], index[values[FST_POP2]]
                fstWeighted[i][k] += fst * nSnps
                fstWeighted[k][i] += fst * nSnps
                fstWeights[i][k] += nSnps
                fstWeights[k][i] += nSnps

        pooledPi = [piDiffs[i] / piDenom[i] if piDenom[i] else 0.0
                    for i in range(numSites)]
        pooledDxy = [[0.0 if k == i or not dxyDenom[i][k]
                      else dxyDiffs[i][k] / dxyDenom[i][k]
                      for k in range(numSites)] for i in range(numSites)]
        pooledFst = [[0.0 if k == i or not fstWeights[i][k]
                      else fstWeighted[i][k] / fstWeights[i][k]
                      for k in range(numSites)] for i in range(numSites)]

        with open(f"{OUTDIR}/averaged_pi_{year}.csv", 'w') as f:
            for pi in pooledPi:
                f.write(f"{pi}\n")

        for name, matrix in (("dxy", pooledDxy), ("fst", pooledFst)):
            with open(f"{OUTDIR}/averaged_{name}_{year}.csv", 'w') as f:
                for row in matrix:
                    f.write(",".join(str(v) for v in row) + "\n")

        meanPi = sum(pooledPi) / numSites
        print(f"{year}: {numSites} sites, mean per-site pi = {meanPi:.6g}")


if __name__ == "__main__":
    main()
