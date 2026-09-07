from pathlib import Path

import tskit
import msprime
import pyslim
import pandas as pd
import math
import numpy as np
import csv

import ld_common as ldc

# LD is OFF until (a) the empirical side has been run so the bin scale is confirmed (CLAUDE.md
# 6.8) and (b) its per-trial cost is measured -- this runs inside every ABC trial, where
# recapitation already takes ~97% of 1764 s at POPMULT=5000 (3.1). Flip to True, or set the
# COMPUTE_LD environment variable, once both are settled.
COMPUTE_LD = bool(int(__import__("os").environ.get("COMPUTE_LD", "0")))


def _real_sample_sizes(year):
    '''Diploid individuals actually sequenced per site, in SPECIFIER-MATRIX ROW ORDER.

    Same source and same order as ABCAnalysisNoRedis.get_keep_mask -- popFile{year} counted by
    site name, indexed by specifier row (CLAUDE.md 4). Reimplemented here rather than imported
    because ABCAnalysisNoRedis lazily imports Main, which imports this module.
    Raises rather than silently defaulting if a specifier site is missing from the popfile.
    '''
    year = str(year)
    names = []
    with open(Path(f"../data/Genetic_Data/specifier_matrix_{year}.csv"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                names.append(line.split(",")[0].strip())

    counts = {}
    with open(Path(f"../data/Genetic_Data/popFile{year}"), encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                counts[parts[1].strip()] = counts.get(parts[1].strip(), 0) + 1

    missing = [s for s in names if s not in counts]
    if missing:
        raise ValueError(f"{year}: specifier sites absent from popFile{year}: {missing}")
    return [counts[s] for s in names]


def _subsample_nodes(ts, pop_idx, time, n_diploid, rng):
    '''Draw n_diploid whole INDIVIDUALS from one deme and return their sample nodes.

    Individuals, not nodes: the empirical unit is a diploid, and drawing loose nodes would mix
    haplotypes from different beetles (same argument as diagnostics/fst_subsample.py).
    Returns every node if the deme holds fewer individuals than requested.
    '''
    nodes = ts.samples(population=pop_idx, time=time)
    by_ind = {}
    for nd in nodes:
        ind = ts.node(nd).individual
        by_ind.setdefault(ind, []).append(nd)
    inds = np.array(sorted(by_ind), dtype=np.int64)
    if len(inds) > n_diploid:
        inds = rng.choice(inds, size=n_diploid, replace=False)
    return np.array(sorted(nd for i in inds for nd in by_ind[i]), dtype=np.int64)


def calculate_ld_decay(ts, genome_indicies, time, year, output_path, rng=None):
    '''Per-deme LD decay (mean r^2 vs physical distance), written as bins x demes.

    THE ONE STATISTIC HERE THAT MUST BE SUBSAMPLED. The other four run on whole demes (301-714
    diploids), which 6.6 measured as correct for pi and ~1% on F_st. r^2 is different in kind: its
    small-sample bias is ~1/n_haplotypes (Hill 1981, CLAUDE.md 11), i.e. ~0.06-0.10 at the real
    n = 5-8 diploids -- plausibly larger than the signal. So the simulated demes are cut to each
    site's real n_i and the identical bias appears on both sides (invariant 1). Do NOT repoint the
    other four statistics at these sample sets: 6.6 measured that subsampling makes pi_loss worse.

    Binning, MAF filter and pair enumeration all come from ld_common, which the empirical side
    imports too. Writes ld_{year}.csv with rows = distance bins, columns = demes in
    specifier-matrix row order, mirroring the empirical file's layout (7.5).
    '''
    if rng is None:
        # np.random is seeded per job by ABCAnalysisNoRedis (np.random.seed(job_id)), so this is
        # reproducible per trial and varies across them, like the rest of the pipeline.
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))

    n_real = _real_sample_sizes(year)
    if len(n_real) != len(genome_indicies):
        raise ValueError(f"{year}: {len(n_real)} specifier rows vs "
                         f"{len(genome_indicies)} demes -- ordering is broken (CLAUDE.md 4)")

    K = len(genome_indicies)
    sum_r2 = np.zeros((ldc.N_BINS, K))
    cnt = np.zeros((ldc.N_BINS, K))

    all_pos = ts.tables.sites.position.astype(np.int64)
    for k, idx in enumerate(genome_indicies):
        nodes = _subsample_nodes(ts, idx, time, n_real[k], rng)
        if len(nodes) < 4:                      # need >=2 diploids for any r^2 at all
            continue
        # Per deme, never genome-wide: the full matrix is 255k sites x 70k samples (1.8e10).
        # Here it is 2*n_i haplotypes x 255k sites, which is trivial.
        H = ts.genotype_matrix(samples=nodes)
        biallelic = H.max(axis=1) <= 1          # SLiMMutationModel can stack states at a site
        # Left in its native small dtype; ld_common.standardize() casts. Same reasoning as
        # CalculateLD.haplotypes() -- the cast belongs next to the MAF filter, not here.
        pos_k, H = all_pos[biallelic], H[biallelic]
        # msprime places mutations on integer positions, so several can share one. accumulate()
        # requires ascending UNIQUE positions, and a d = 0 pair is not a distance.
        pos_k, first = np.unique(pos_k, return_index=True)
        ldc.decay_one_pop(pos_k, H[first], sum_r2[:, k], cnt[:, k])

    ldc.write_decay(output_path, sum_r2, cnt, [str(i) for i in range(K)])
    return sum_r2, cnt

def calculate_diversity_and_divergence(ts, genome_indicies, time, output_diversities_path,
                                       output_divergences_path, output_fst_path,
                                       output_relatedness_path):
    '''
    Calculate diversity, divergence, Fst and genetic relatedness for given genome indices at a
    specific time from a tree sequence and output them to csv files.

    Parameters:
    ts: tree sequence object
    genome_indicies: list of population indices to sample from
    time: time point to sample
    output_diversities_path: file path to save diversities (pi, per subpop)
    output_divergences_path: file path to save divergences (d_xy, pairwise) -- diagnostic only
    output_fst_path: file path to save Fst (pairwise, relative differentiation)
    output_relatedness_path: file path to save genetic relatedness (pairwise, per-year centred)
    '''

    #Get all the nodes we are trying to sample from
    pop_samples = []
    for idx in genome_indicies:
        pop_samples.append(ts.samples(population=idx, time=time))

    K = len(pop_samples)
    pairs = [(i, j) for i in range(K) for j in range(K) if i != j]

    #Diversity (pi) per subpop -- all sample sets in a single traversal.
    diversities = np.asarray(ts.diversity(pop_samples), dtype=float)

    #Divergence (d_xy) -- ALL pairs in one traversal via indexes=. Diagonal stays 0.
    #Fst is HUDSON, 1 - Hw/d_xy with Hw = (pi_X + pi_Y)/2. Do NOT switch to ts.Fst: that returns
    #Nei/Slatkin, ~half of Hudson, and the empirical target is pixy's Weir-Cockerham, which
    #matches Hudson (CLAUDE.md 6.7, invariant 9).
    divergences = np.zeros((K, K))
    fsts = np.zeros((K, K))
    if pairs:
        div_flat = ts.divergence(pop_samples, indexes=pairs)
        for (i, j), dv in zip(pairs, div_flat):
            divergences[i, j] = dv
            # dv == 0 only if the pair has no variation at all; 0 beats a NaN in fst_loss.
            fsts[i, j] = 0.0 if dv == 0 else 1.0 - 0.5 * (diversities[i] + diversities[j]) / dv

    #Relatedness is centred across THIS year's subpops -- one call with all sample sets.
    #Never slice a larger matrix to get a smaller one (CLAUDE.md 8#2).
    gr_indexes = [(i, j) for i in range(K) for j in range(K)]
    gr = ts.genetic_relatedness(pop_samples, indexes=gr_indexes)
    relatedness = np.asarray(gr, dtype=float).reshape(K, K)

    #write the data
    with open(output_diversities_path, "w", newline="") as f:
        writer = csv.writer(f)
        for item in diversities:
            writer.writerow([item])

    with open(output_divergences_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(divergences.tolist())

    with open(output_fst_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(fsts.tolist())

    with open(output_relatedness_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(relatedness.tolist())



def analyze_tree_sequence(mutation_rate=None, recombination_rate=None, ancestral_Ne=6700):
    '''
    This function analyzes the tree sequence file generated by the SLiM simulation by
    calculating diversity and divergence statistics before and after recapitation and mutation addition.
    It outputs the results to CSV files.

    mutation_rate, recombination_rate: REQUIRED, no defaults. See the guard below.

    ancestral_Ne: effective size of the panmictic ancestral population used in recapitation.
    Fixed empirical point estimate (6700); exposed here for sensitivity analysis only, NOT
    inferred -- it is confounded with mu via pi = 4*Ne*mu. See CLAUDE.md 5.1.
    '''
    # No defaults: these set the diversity and linkage scale of every output file, and the files
    # record no scale, so a silent fallback is worse than a crash (CLAUDE.md 10.1).
    if mutation_rate is None or recombination_rate is None:
        raise ValueError(
            "analyze_tree_sequence() requires explicit mutation_rate and recombination_rate. "
            "Use ABCAnalysisNoRedis.DEFAULT_MUTATION_RATE (4.646e-7) and "
            "DEFAULT_RECOMBINATION_RATE (2.75e-6). See CLAUDE.md 6.1.1 and 6.3.")

    
    # Load the cluster_data CSV file
    cluster_data = pd.read_csv(Path("../data/cluster_data.csv"))

    assignments_2015 = cluster_data['Genome Assignment 2015']
    assignments_2019 = cluster_data['Genome Assignment 2019']
    assignments_2023 = cluster_data['Genome Assignment 2023']

    total_assignments_2015 = int(max(assignments_2015.dropna()))
    total_assignments_2019 = int(max(assignments_2019.dropna()))
    total_assignments_2023 = int(max(assignments_2023.dropna()))

    genome_indicies_2015 = [-1] * (total_assignments_2015+1)
    genome_indicies_2019 = [-1] * (total_assignments_2019+1)
    genome_indicies_2023 = [-1] * (total_assignments_2023+1)
    
    
    for i in range(len(assignments_2015)):
        if math.isnan(assignments_2015[i]) == False:
            index = int(assignments_2015[i])
            if genome_indicies_2015[index] == -1:
                genome_indicies_2015[index] = i
    for i in range(len(assignments_2019)):
        if math.isnan(assignments_2019[i]) == False:
            index = int(assignments_2019[i])
            if genome_indicies_2019[index] == -1:
                genome_indicies_2019[index] = i
    for i in range(len(assignments_2023)):
        if math.isnan(assignments_2023[i]) == False:
            index = int(assignments_2023[i])
            if genome_indicies_2023[index] == -1:
                genome_indicies_2023[index] = i

    # Load the tree sequence file
    ts = tskit.load(Path("../out/simTreeSeq.trees"))
    

    ts = pyslim.recapitate(ts, recombination_rate=recombination_rate, ancestral_Ne=ancestral_Ne)
    
    
    print("Simplifying tree sequence...")
    samplesToKeep = []
    for idx in genome_indicies_2015:
        samplesToKeep.extend(ts.samples(population=idx, time=16))
    for idx in genome_indicies_2019:
        samplesToKeep.extend(ts.samples(population=idx, time=8))
    for idx in genome_indicies_2023:
        samplesToKeep.extend(ts.samples(population=idx, time=0))
    
    
    # filter_populations=False is REQUIRED. The default renumbers surviving populations, which
    # silently misaligns the ts.samples(population=idx) queries below (CLAUDE.md 2).
    ts = ts.simplify(samples=samplesToKeep, filter_populations=False)
    
    next_id = pyslim.next_slim_mutation_id(ts)
    print("Simulating mutations...")
    ts = msprime.sim_mutations(
            ts,
            rate=mutation_rate,
            model=msprime.SLiMMutationModel(type=0, next_id=next_id),
            keep=True,
    )

    calculate_diversity_and_divergence(
        ts, genome_indicies_2023, time=0,
        output_diversities_path=Path("../data/Output_Data/diversities_2023.csv"),
        output_divergences_path=Path("../data/Output_Data/divergences_2023.csv"),
        output_fst_path=Path("../data/Output_Data/fst_2023.csv"),
        output_relatedness_path=Path("../data/Output_Data/relatedness_2023.csv"))

    calculate_diversity_and_divergence(
        ts, genome_indicies_2019, time=8,
        output_diversities_path=Path("../data/Output_Data/diversities_2019.csv"),
        output_divergences_path=Path("../data/Output_Data/divergences_2019.csv"),
        output_fst_path=Path("../data/Output_Data/fst_2019.csv"),
        output_relatedness_path=Path("../data/Output_Data/relatedness_2019.csv"))

    calculate_diversity_and_divergence(
        ts, genome_indicies_2015, time=16,
        output_diversities_path=Path("../data/Output_Data/diversities_2015.csv"),
        output_divergences_path=Path("../data/Output_Data/divergences_2015.csv"),
        output_fst_path=Path("../data/Output_Data/fst_2015.csv"),
        output_relatedness_path=Path("../data/Output_Data/relatedness_2015.csv"))

    # LD decay (CLAUDE.md 7.5). Kept OFF by default until the empirical side has been run and the
    # bin scale confirmed (6.8) -- and until its per-trial cost is measured, since this runs inside
    # every ABC trial. Enable with COMPUTE_LD=1.
    if COMPUTE_LD:
        print(f"LD decay (ld_common spec {ldc.spec_hash()}) -- "
              f"the empirical side must print the same hash...")
        for year, gidx, t in [("2023", genome_indicies_2023, 0),
                              ("2019", genome_indicies_2019, 8),
                              ("2015", genome_indicies_2015, 16)]:
            s, c = calculate_ld_decay(
                ts, gidx, time=t, year=year,
                output_path=Path(f"../data/Output_Data/ld_{year}.csv"))
            ldc.report_halfway(s, c, prefix=f"    {year}: ")
    
