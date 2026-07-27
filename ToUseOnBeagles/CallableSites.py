"""Callable-site counts per chromosome.

These are the denominators that turn per-SNP statistics into per-site ones. Both
AverageData.py and CalcGenRel.py import from here so the two cannot drift apart.

WHY THIS IS NEEDED
------------------
The Beagle-derived VCFs contain only polymorphic sites, and pixy was run with
--bypass_invariant_check (PixyTheFiles.py:28). pixy's count_comparisons therefore
counts only comparisons at SNPs, making avg_pi a per-SNP heterozygosity rather
than a per-site nucleotide diversity. The simulated side is per-site over the
1e6 bp of simulated sequence, so the two are on different scales until the
empirical denominator is extended over every callable site.

See DIVERSITY_SCALE_ISSUE.md 1.1 for the evidence.

WHAT "CALLABLE" MEANS HERE
--------------------------
Sites that survived the upstream coverage/quality filtering, i.e. sites where a
genotype could have been called had one been variant. This is NOT the same as
chromosome length -- positions dropped by filtering are neither variant nor
callable, so using raw chromosome length over-corrects (it inflates the
denominator and pushes pi down).

HOW TO GET THE NUMBERS
----------------------
The honest source is the upstream filtering: the mask/BED or the per-chromosome
site count that came out of the variant-calling pipeline before Beagle.

If that is not recoverable, the fallback is chromosome length from the reference
assembly, which gives a lower bound on pi. Record which of the two you used in
the SOURCE string below -- the distinction matters when interpreting the result.

The SNP counts pixy actually used are in the no_sites column of the pixy output
and are read automatically by AverageData.py, so they do not need to go here.
"""

NUM_CHRS = 17

# Where the numbers below came from. Update this whenever CALLABLE_SITES changes.
SOURCE = (
    "PROVISIONAL -- pixy window_pos_2 (position of the last SNP on each "
    "chromosome), read from statsChr{i}_2015/chr{i}_2015_pi.txt on 2026-07-27. "
    "True callable-site counts are not recoverable: the Beagle files contain "
    "only variant sites, so the filtering upstream of them was never recorded. "
    "window_pos_2 is a LOWER bound on chromosome length and therefore an upper "
    "bound on pi; assembly chromosome lengths would give the other bound. The "
    "gap between the two is order 10-20%, against a ~2400x scale error being "
    "corrected here, so it does not affect any conclusion drawn downstream."
)

# chromosome number (1..NUM_CHRS) -> number of callable sites.
# Trailing comment is the SNP density (no_sites / window_pos_2) for that
# chromosome; genome-wide it is 81,141,632 / 930,522,190 = 8.72%.
CALLABLE_SITES = {
    1:  74_792_276,   #  8.76%
    2:  72_572_827,   #  8.18%
    3:  68_289_378,   #  8.71%
    4:  67_057_556,   #  9.02%
    5:  64_605_101,   #  8.58%
    6:  63_723_733,   #  6.26%
    7:  63_538_103,   #  9.21%
    8:  58_296_307,   #  9.58%
    9:  56_337_286,   #  8.92%
    10: 53_245_715,   #  8.86%
    11: 50_404_610,   #  9.70%
    12: 49_841_970,   #  9.67%
    13: 46_212_843,   #  7.75%
    14: 44_328_247,   #  9.51%
    15: 34_124_746,   #  7.35%
    16: 32_402_737,   # 10.13%
    17: 30_748_755,   #  8.59%
}


def require_callable_sites(num_chrs=NUM_CHRS):
    """Return the callable-site table, or fail loudly if it is incomplete.

    Deliberately fatal rather than falling back to a default: a silent fallback
    would reproduce the per-SNP scale that this table exists to correct, and the
    output CSVs carry no record of which scale they are on.
    """
    missing = [c for c in range(1, num_chrs + 1) if c not in CALLABLE_SITES]
    if missing:
        raise SystemExit(
            "CallableSites.CALLABLE_SITES is missing chromosomes: "
            + ", ".join(str(c) for c in missing)
            + "\nFill the table in before recomputing per-site statistics; see "
              "the module docstring for where the numbers come from."
        )
    return CALLABLE_SITES


def genome_callable_sites(num_chrs=NUM_CHRS):
    """Total callable sites across all chromosomes."""
    return sum(require_callable_sites(num_chrs).values())
