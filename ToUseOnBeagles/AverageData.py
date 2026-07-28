"""Pool pixy's per-chromosome output into genome-wide per-site statistics.

Three corrections over the original version of this script:

1.4  Chromosomes are POOLED, not averaged. pi, dxy and Fst are ratios, so a
     genome-wide value is sum(numerators) / sum(denominators). Taking a plain
     mean of 17 per-chromosome ratios over-weights sparse chromosomes.

1.1  pi and dxy are converted from per-SNP to PER-SITE. pixy ran on a
     variant-sites-only VCF, so its count_comparisons covers SNPs only. The
     denominator is extended over all callable sites (see CallableSites.py).

1.5  The comparison denominator is rebuilt from sample sizes instead of being
     read out of pixy's count_comparisons, which OVERFLOWS a signed 32-bit
     integer for large populations. See the COMPARISON DENOMINATOR note below.
     pixy's value is still read, but only to cross-check the analytic one.

Fst needs none of these corrections in principle -- it is a ratio of variance
components, so per-site scaling cancels, and its file carries no comparison
count to overflow -- but see the FST POOLING note below for what this script
can and cannot do about pooling it.

Run from the directory holding the statsChr*_{year}/ folders.
"""
import os
from math import comb

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

# Largest value pixy's count_comparisons field can represent. Past this the
# field is destroyed; see the COMPARISON DENOMINATOR note below.
INT32_MAX = 2 ** 31 - 1
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


def read_specifier(year):
    """Site names in matrix order, and the diploid sample count of each.

    col0 is the site name, cols 1 and 2 the coordinates, and every non-empty
    cell from col 3 on is one sampled individual. Row order fixes the order of
    every statistic this script writes, so that the empirical matrices line up
    element-wise with the simulated ones.

    The sample counts are what the comparison denominators are rebuilt from, so
    they are cross-checked against pixy's own count_comparisons on every row
    read (see check_comparisons) rather than trusted blind.
    """
    siteNames, sampleCounts = [], []
    with open(f"specifier_matrix_{year}.csv", 'r') as f:
        for line in f:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split(',')
            siteNames.append(fields[0].strip())
            sampleCounts.append(sum(1 for v in fields[3:] if v.strip() != ""))

    for name, count in zip(siteNames, sampleCounts):
        if count < 1:
            raise ValueError(
                f"{year}: site {name} has no sampled individuals in "
                f"specifier_matrix_{year}.csv, so its denominator cannot be built."
            )
    return siteNames, sampleCounts


# COMPARISON DENOMINATOR.
#
# pixy reports count_comparisons, but that field cannot represent a value past
# INT32_MAX, and (comparisons per site) * no_sites crosses it at 14 diploid
# individuals on the largest chromosome (~6.5e6 SNPs). Observed on this dataset,
# the field saturates to INT32_MIN (-2147483648) -- the "integer indefinite"
# value an x86 double-to-int32 cast yields when the double is out of range --
# rather than wrapping modularly.
#
# H53-2015 (19 individuals, 703 comparisons per site) saturates on 15 of 17
# chromosomes, which drove its denominator negative and produced a NEGATIVE pi.
# Rows that saturate on only SOME chromosomes are the more dangerous case: they
# stay positive and are merely inflated by whatever fraction of the denominator
# was lost. Alsum25-2015 x {Refuge,H15} saturate on 1 of 17 and came out ~18%
# high -- large enough to matter, small enough to look like real data.
#
# The count does not need to be read at all. count_missing is 0 on every row
# (Beagle imputes everything), so comparisons per site is a constant fixed by
# sample size alone, and the genome-wide denominator is that constant times the
# callable-site count. count_diffs is unaffected -- it peaks around 4e7 per
# chromosome, three orders of magnitude below the overflow point.

def comparisons_per_site_pi(nIndividuals):
    """Pairwise comparisons at a single site within one population: C(2n, 2)."""
    return comb(2 * nIndividuals, 2)


def comparisons_per_site_dxy(nOne, nTwo):
    """Pairwise comparisons at a single site between two populations: (2n1)(2n2)."""
    return (2 * nOne) * (2 * nTwo)


def per_site_denominator(comparisonsPerSite, callableSites):
    """Extend a per-site comparison count over every callable site.

    The numerator needs no matching adjustment: invariant sites contribute zero
    differences, so count_diffs is already the correct per-site numerator.

    Assumes missingness at invariant sites matches missingness at variant sites.
    That is the standard assumption when back-correcting a SNPs-only VCF, and it
    is the reason an all-sites VCF is the better fix if one ever becomes
    available. Here it is not merely assumed -- check_comparisons verifies on
    every row that pixy saw exactly this many comparisons.
    """
    return comparisonsPerSite * callableSites


def check_comparisons(reported, comparisonsPerSite, noSites, label, repairs):
    """Cross-check pixy's count_comparisons against the analytic expectation.

    Exact agreement is expected: with count_missing = 0 the number of
    comparisons is fully determined by sample size, so pixy's total must be
    exactly (comparisons per site) * no_sites.

    Disagreement is tolerated only where pixy could not have got it right: once
    the expected value exceeds INT32_MAX the field is unrepresentable and
    whatever it holds is garbage. The test is on representability rather than on
    any particular corrupted value, so it does not depend on how the overflow
    manifests (this dataset saturates to INT32_MIN; a modular wrap would be
    caught identically). Such rows are recorded and ignored, since the analytic
    denominator is used either way.

    Disagreement on a row whose expected value FITS in an int32 is a different
    matter: pixy had no reason to be wrong, so an assumption behind the analytic
    denominator must be false -- nonzero missingness, wrong column offsets, or
    specifier sample counts that do not match the populations pixy was given.
    That raises rather than silently emitting numbers on an unknowable scale.
    """
    if reported is None:
        return
    expected = comparisonsPerSite * int(noSites)
    if expected == int(reported):
        return
    if expected > INT32_MAX:
        repairs.append((label, expected, int(reported)))
        return
    raise ValueError(
        f"{label}: pixy reports count_comparisons = {int(reported)} but the "
        f"sample sizes imply {expected}, which is small enough for pixy to have "
        f"represented exactly ({expected} <= INT32_MAX). This is not the known "
        f"overflow. The analytic denominator cannot be trusted here -- check "
        f"count_missing, the column offsets at the top of this script, and the "
        f"specifier matrix sample counts."
    )


def check_diffs(diffs, label):
    """count_diffs is far too small to overflow; a negative one means trouble."""
    if diffs < 0:
        raise ValueError(
            f"{label}: count_diffs = {diffs:.0f} is negative. This field is not "
            f"expected to overflow, so the column offsets are probably wrong."
        )


def main():
    callable_sites = require_callable_sites(NUM_CHRS)
    os.makedirs(OUTDIR, exist_ok=True)

    for year in YEARS:
        siteNames, sampleCounts = read_specifier(year)
        numSites = len(siteNames)
        index = {name: i for i, name in enumerate(siteNames)}

        # Comparison counts per site, fixed by sample size (see the note above).
        piPerSite = [comparisons_per_site_pi(n) for n in sampleCounts]
        dxyPerSite = [[comparisons_per_site_dxy(sampleCounts[i], sampleCounts[k])
                       for k in range(numSites)] for i in range(numSites)]

        # Running numerators and denominators, pooled across chromosomes.
        piDiffs = [0.0] * numSites
        piDenom = [0.0] * numSites
        dxyDiffs = [[0.0] * numSites for _ in range(numSites)]
        dxyDenom = [[0.0] * numSites for _ in range(numSites)]
        fstWeighted = [[0.0] * numSites for _ in range(numSites)]
        fstWeights = [[0.0] * numSites for _ in range(numSites)]

        repairs = []

        for chrom in range(1, NUM_CHRS + 1):
            folderName = f"statsChr{chrom}_{year}"
            nCallable = callable_sites[chrom]

            for values in read_pixy_rows(os.path.join(folderName, f"chr{chrom}_{year}_pi.txt")):
                diffs = number(values[PI_DIFFS])
                noSites = number(values[PI_NO_SITES])
                if diffs is None or noSites is None:
                    continue
                i = index[values[PI_POP]]
                label = f"{year} chr{chrom} pi {values[PI_POP]}"
                check_diffs(diffs, label)
                check_comparisons(number(values[PI_COMPARISONS]), piPerSite[i],
                                  noSites, label, repairs)
                piDiffs[i] += diffs
                piDenom[i] += per_site_denominator(piPerSite[i], nCallable)

            for values in read_pixy_rows(os.path.join(folderName, f"chr{chrom}_{year}_dxy.txt")):
                diffs = number(values[DXY_DIFFS])
                noSites = number(values[DXY_NO_SITES])
                if diffs is None or noSites is None:
                    continue
                i, k = index[values[DXY_POP1]], index[values[DXY_POP2]]
                label = f"{year} chr{chrom} dxy {values[DXY_POP1]} x {values[DXY_POP2]}"
                check_diffs(diffs, label)
                check_comparisons(number(values[DXY_COMPARISONS]), dxyPerSite[i][k],
                                  noSites, label, repairs)
                denom = per_site_denominator(dxyPerSite[i][k], nCallable)
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

        # pi and dxy are counts over counts; neither can legitimately be
        # negative. Catch it here rather than letting a negative target reach
        # the ABC, where log(pi) simply masks the entry out and the affected
        # population disappears from the fit without warning.
        for i, pi in enumerate(pooledPi):
            if pi < 0:
                raise ValueError(f"{year}: pooled pi for {siteNames[i]} is {pi}, which is "
                                 f"impossible; the denominator correction has failed.")
        for i in range(numSites):
            for k in range(numSites):
                if pooledDxy[i][k] < 0:
                    raise ValueError(f"{year}: pooled dxy for {siteNames[i]} x {siteNames[k]} "
                                     f"is {pooledDxy[i][k]}, which is impossible; the "
                                     f"denominator correction has failed.")

        with open(f"{OUTDIR}/averaged_pi_{year}.csv", 'w') as f:
            for pi in pooledPi:
                f.write(f"{pi}\n")

        for name, matrix in (("dxy", pooledDxy), ("fst", pooledFst)):
            with open(f"{OUTDIR}/averaged_{name}_{year}.csv", 'w') as f:
                for row in matrix:
                    f.write(",".join(str(v) for v in row) + "\n")

        meanPi = sum(pooledPi) / numSites
        print(f"{year}: {numSites} sites, mean per-site pi = {meanPi:.6g}")
        if repairs:
            print(f"  {len(repairs)} row(s) exceeded what pixy's count_comparisons can hold; "
                  f"the analytic denominator was used instead. Affected rows:")
            for label, expected, reported in repairs:
                print(f"    {label}: expected {expected}, pixy reported {reported}")


if __name__ == "__main__":
    main()
