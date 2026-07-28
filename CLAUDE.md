# CLAUDE.md

Context for Claude Code sessions on this project. Read this first. Remaining work is in
**`TODO.md`**.

Status markers:
- **[VERIFIED]** — checked by reading the code and/or running it. Trust it.
- **[INFERRED]** — strong argument, not directly confirmed.
- **[OPEN]** — unresolved. Do not assume either way.

---

## 1. The project

Colorado Potato Beetle (*Leptinotarsa decemlineata*) population genetics across the Wisconsin
landscape.

- **Empirical.** Sequenced beetle genomes, phased/imputed (Beagle) VCFs, 17 chromosomes.
  Three sampling years — **2015, 2019, 2023** — each with a *different* set of subpopulations
  ("sites") and different per-subpop sample sizes.
- **Simulated.** SLiM (forward) → pyslim recapitation → msprime mutation overlay → tskit tree
  sequence. Entrypoint: `Python_Code/Main.py::main()`.

**Goal:** Approximate Bayesian Computation to infer simulator parameters from the empirical data.
The distance lives in `Python_Code/ABCAnalysisNoRedis.py::calculate_losses`.

> **Read §6 before touching the ABC.** Several parameters in the prior do not reach or move the
> simulation. Tuning the distance before fixing that is wasted effort.

---

## 2. How the simulation pipeline works [VERIFIED]

`Main.py::main(num_clusters, migration_rates_modifier, population_modifier, total_migration=0.05,
mutation_rate, recombination_rate, ancestral_Ne=6700)`:

1. `GenerateClusterData.cluster_coordinates(..., random_state=KMEANS_SEED)` — KMeans over field
   coordinates from `data/final_data_for_modeling.csv`. Seed pinned at 42.
2. Writes `data/cluster_data.csv` and `data/cluster_distances.csv` — **overwriting in place.**
3. `GenerateSimulationParams.determine_migration_rates(distances, total_migration, scale, ...)`
   → `data/migration_rates.csv`.
4. `slim -d POPMULT=<pop> SLiM_Code/CPBSampleSim{Win,Linux}.slim` → `out/simTreeSeq.trees`.
5. `AnalyzeTreeSeq.analyze_tree_sequence(..., ancestral_Ne=6700)` — recapitate → simplify →
   overlay mutations → write, per year, `diversities`, `divergences` (d_xy), `fst`, and
   `relatedness` matrices under `data/Output_Data/`. Pairwise stats use **single batched
   `indexes=` traversals** (verified bit-identical to per-pair loops, ~24× faster at k=8).

**Key architectural fact:** SLiM runs with `initializeMutationRate(0)`. It is a pure neutral
tree-sequence recorder. **All mutations are overlaid afterwards** by `msprime.sim_mutations()`,
*after* `simplify()`.

Consequences:
- Mutation rate has **zero** effect on SLiM's memory or runtime.
- Site-level π is **exactly linear in μ**: `π = μ × (branch-mode diversity)`. Branch-mode
  diversity (`ts.diversity(mode="branch")`) is `2·E[T_pair]` in generations and is the
  mutation-free view of the demography. Use it when diagnosing.

**Migration parameterization.** `total_migration` is the total immigration fraction per subpop per
generation (a real bounded rate); `scale` (the old `migration_rates_modifier`) is only the
dispersal-kernel decay, reshaping *where* migrants come from. The kernel is built over sources
(self excluded), normalized to 1, then scaled by `total_migration`, so each row's off-diagonals
sum to exactly `total_migration`; SLiM fills retention as `1 − total_migration`. [VERIFIED]

**`ts.simplify()` renumbers populations.** It defaults to `filter_populations=True`, which drops
unreferenced populations and renumbers survivors contiguously — while
`ts.samples(population=idx, ...)` uses the *original* cluster-row index. This produced the CHTC
"2-of-5 rows" bug (`Sample sets must contain at least one element`) at `numClusters=3` (×33 = 99
demes), where many demes go unreferenced. Fixed with `filter_populations=False`
(`AnalyzeTreeSeq.py:146`, commit `c5963ae`). **This was a correctness bug, not just a crash:**
when filtering occurred but the index stayed in range, the old code silently returned the *wrong
deme's* statistics. Treat any pre-`c5963ae` results as suspect. [VERIFIED]

---

## 3. Environment [VERIFIED]

**`cpb-env` (from `environment3.yml`): SLiM 5.1, pyslim 1.1.1, tskit 1.0.2, msprime 1.4.1,
python 3.12.** Internally consistent SLiM-5 stack; the pipeline runs end to end. Invoke via
`conda run -n cpb-env python ...` from `Python_Code/` (paths are relative to that dir).

> An older note claimed "SLiM 4.3 tree sequences need pyslim 1.0.x; pyslim 1.1+ fails." That
> described the *old* env. Do **not** downgrade pyslim on this env.

Per-run cost [VERIFIED, POPMULT=2000, fsync-timed]: SLiM forward sim ~3 s; `analyze_tree_sequence`
~250 s, of which **recapitation is ~246 s (~98%)**. The batched π/d_xy/F_st/relatedness stats are
~1 s total. Recapitation (coalescing ~42k lineages at r=2.75e-6 over 1e6 bp) is the throughput
bottleneck, **not** the statistics. Roughly **4 min/trial** at POPMULT≈2000. Simplify-before-
recapitate was considered and **declined** (unsafe without `keep_input_roots=True`; benefit is
local-only, since CHTC parallelizes).

Total simulated N is **not** POPMULT. Subpop size is `Average Count × POPMULT / numSubpops`, so
`total N ≈ POPMULT × mean(Average Count) ≈ 3.33 × POPMULT`. An older OOM (POPMULT≈40000 crashed a
128 GB machine) still stands as a caution; `simplificationRatio=INF` (§6.4) is unfixed, so keep
POPMULT modest.

---

## 4. Data layout

```
data/empiricalStats/averaged_{pi,dxy,fst,genRel}_{2015,2019,2023}.csv   # observed targets
data/Output_Data/{diversities,divergences,fst,relatedness}_{year}.csv   # simulated
data/cluster_data.csv          # Cluster ID, Latitude, Longitude, Average Count, Genome Assignment {year}
data/cluster_distances.csv     # 66x66 pairwise geographic distance (metres), + header row & ID col
data/Genetic_Data/specifier_matrix_{year}.csv  # headerless; row = subpop index; col0=site name, col1=lat, col2=lon
data/final_data_for_modeling.csv
diagnostics/qdriver.py         # controlled rescaling/sweep harness
diagnostics/qpost.py           # post-process an existing .trees -> branch div, site pi, Fst
```

**Subpop ordering — the single most important convention. [VERIFIED]** For year Y, subpop index
`i` is **row `i` of `specifier_matrix_Y.csv`**, whose col0 is the site name and cols 1/2 the real
lat/lon. **Every matrix in the project must use this ordering** — simulated stats, empirical
stats, and the IBD coordinate source. Rows == matrix dimension (24/17/20). This is what makes
element-wise within-year comparison meaningful. Across years the matrices still cannot be aligned
(different subpop sets).

> Ordering by `sorted(labels)` instead is **not** equivalent — the two disagree in all three years
> (2015 and 2023 differ at row 0). This was a real bug in `CalcGenRel.py`, now fixed. Any new
> script producing a per-subpop matrix must read the specifier matrix for its order.

Subpop counts per year: **2015 → 24, 2019 → 17, 2023 → 20.** All four empirical matrices and
`Genome Assignment {year}` agree. [VERIFIED]

Notes that have already caused bugs:
- Subpop sizes range 2–19 diploid individuals. Small ones are very noisy, and the large ones are
  not free either — at **≥14 individuals** pixy's comparison counter overflows (§6.5). Only
  `H53-2015` (19) has ever crossed it; 2023's largest is 11.
- A **typo-duplicated `Arlington` label** produces two near-identical subpopulations. [OPEN —
  decide deliberately whether to merge.]
- Subpop sets differ between years → matrices have different dimensions and **cannot be aligned
  element-wise across years.**

---

## 5. The empirical statistics pipeline (`ToUseOnBeagles/`)

Runs on the machine holding the Beagle VCFs and pixy output, **not** in this repo. All paths are
relative to that working directory. Its outputs are copied into `data/empiricalStats/`.

```
ConvertBeagleToVCF.py  Beagle -> VCF (one record per line)
PixyTheFiles.py        runs pixy per chromosome per year -> statsChr{i}_{year}/
CallableSites.py       per-chromosome callable-site denominators (shared config)
AverageData.py         pools pixy output -> averaged_{pi,dxy,fst}_{year}.csv
                       denominators are ANALYTIC, not read from pixy (5.1)
CalcGenRel.py          per-year relatedness from VCFs -> averaged_genRel_{year}.csv
CalculateLD,py         per-year, per-subpop LD decay (note the comma in the filename)
```

One command runs the recalculation:

```bash
python AverageData.py && python CalcGenRel.py
```

### 5.1 π and d_xy are per-SNP unless corrected [VERIFIED]

`PixyTheFiles.py:28` runs pixy with `--bypass_invariant_check` on a **variant-sites-only** VCF, so
`count_comparisons` counts comparisons at SNPs only and `avg_pi` is per-SNP heterozygosity, not
per-site nucleotide diversity. Confirmed directly from pixy's own output: chr1 `no_sites =
6,549,657`, exactly the VCF record count, with zero invariant sites.

`AverageData.py` corrects this by extending the denominator over callable sites:

```
comparisons_per_site = C(2n, 2)        for pi     <- ANALYTIC, from sample size
                     = (2n_i)(2n_k)    for d_xy
denominator          = comparisons_per_site * callable_sites
```

The numerator needs no adjustment — invariant sites contribute zero differences.

**The missingness assumption is exactly satisfied.** [VERIFIED] `count_missing = 0` and
`count_comparisons / no_sites` divides to exact integers: 91 = C(14,2) → 7 diploid individuals,
231 = C(22,2) → 11. Beagle imputes everything, so comparisons-per-site is constant and the
extrapolation is exact, not approximate.

That last fact is *why* the denominator is computed from sample size rather than read out of pixy's
`count_comparisons`, which overflows int32 for large subpops (§6.5). `AverageData.py` still reads
the field to cross-check and **raises** if it disagrees where pixy could have been right — do not
soften that guard into a warning.

**Callable sites are provisional.** True callable counts are unrecoverable — the Beagle files hold
only variant sites, so the upstream filtering was never recorded. `CallableSites.py` uses pixy's
`window_pos_2` (position of the last SNP per chromosome), a *lower* bound on length and therefore
an *upper* bound on π. Assembly chromosome lengths would give the other bound; the gap is order
10–20%, against the ~2400× scale error being corrected, so no downstream conclusion turns on it.

Genome-wide: **81,141,632 SNPs over 930,522,190 bp = 8.72% density.** Corrected mean π ≈ **0.0122**
(from 0.1403 uncorrected) — an ordinary insect value.

### 5.2 Pool across chromosomes, never average [VERIFIED]

These statistics are ratios, so genome-wide = `Σ numerators / Σ denominators`. Averaging 17
per-chromosome ratios over-weights sparse chromosomes. `AverageData.py` pools; `CalcGenRel.py`
always did.

**F_st is the exception, by necessity.** Proper pooling (Bhatia et al. 2013) is `Σa / Σ(a+b)` over
the Weir–Cockerham variance components, but pixy's `fst.txt` emits only `avg_wc_fst` and `no_snps`
— the components are not in the file. `AverageData.py` uses a SNP-count-weighted mean, strictly
better than a plain mean. To pool properly, recompute from the VCFs with
`allel.weir_cockerham_fst`, which returns the components. Error is a few percent, and F_st is
scale-invariant anyway.

### 5.3 Which statistics the denominator affects

**The dividing line is whether the statistic is a ratio.** [VERIFIED by reasoning]

| statistic | fitted? | affected by per-SNP denominator? | why |
|---|---|---|---|
| π | **yes** | **yes** | level; scales with diversity |
| F_st | **yes** | no | ratio of variance components; scaling cancels |
| IBD slope | **yes** | no | regression built on F_st |
| d_xy | diagnostic | **yes** | level; same denominator as π |
| genetic relatedness | diagnostic | **yes** | sum of centred products; not μ-invariant |

Invariant sites contribute zero to both numerator and denominator of F_st, so computing it over
variant sites only is correct. **The structure/migration side of the inference was never
compromised by the scale issue.**

Relatedness is a *level*: tskit applies `span_normalise=True`, making the simulated side per base
pair over the full 1e6 bp. `CalcGenRel.py` therefore normalizes by callable sites
(`NORMALISE_BY = "callable"`); `"segregating"` preserves the old behaviour for the tskit estimator
check only.

### 5.4 Cosmetic quirks that are not bugs [VERIFIED]

- **`ConvertBeagleToVCF.py:33` hardcodes `CHROM = 9`** for every file. Chromosome identity lives
  only in filenames. Harmless — every script keys off the filename consistently — but it means the
  VCF contents cannot tell you which assembly chromosome a file is.
- **`UserWarning: 'GT' FORMAT header not found`** from scikit-allel. `generate_VCF_header`
  (`ConvertBeagleToVCF.py:82-85`) omits the `##FORMAT=<ID=GT,...>` declaration, so allel falls
  back to its default GT spec (diploid), which matches the data. If allel had actually failed to
  parse GT, `chunk[0]["calldata/GT"]` would raise `KeyError`, not warn.
- **chr6 has the lowest SNP density** (6.26% vs 8.72% genome-wide). All chromosomes were processed
  identically, so there is no differential artifact to correct. Accepted as-is.

---

## 6. Known defects — read before changing the ABC

Ordered by how much they distort the inference.

### 6.1 `ancestral_Ne = 6700` is probably the wrong quantity [INFERRED — the biggest open question]

`pyslim.recapitate(ts, ..., ancestral_Ne=6700)`. The forward SLiM phase runs only 324 generations
while coalescence takes thousands, so most lineages coalesce in the *recapitated* ancestral phase
at this fixed Ne.

Recapitation's `ancestral_Ne` is the **long-term coalescent** effective size — the harmonic-mean Ne
over the thousands of generations in which lineages actually coalesce. 6700 looks much more like a
**contemporary or local** Ne, the kind produced by LD-based or temporal estimators. Those routinely
differ by orders of magnitude.

Working from the corrected π at the biological rate (μ ≈ 2.1e-9):

```
Ne = π/(4μ) = 0.0122 / (4 x 2.1e-9) ~ 1.46e6
```

against 6700 — a **~217× mismatch that the denominator fix does not remove.** For a widespread
agricultural pest, a long-term Ne near 10⁶ is unremarkable (Drosophila sits in that range).

How μ absorbed both errors at once:

| scenario | π target | required μ at Ne=6700 | × biological |
|---|---|---|---|
| both errors | 0.140 | 5.2e-6 | ~2400× |
| denominator fixed (now) | 0.0122 | 4.6e-7 | ~217× |
| both fixed (Ne ≈ 1.5e6) | 0.0122 | 2.1e-9 | 1× |

`ancestral_Ne` is a threaded parameter, **not** an ABC free parameter — it is confounded with μ in
π (they enter only as `4·Ne·μ`), so inferring it would build a second ridge. Exposed for
sensitivity analysis only. Scaling it with `population_modifier` was considered and **rejected**
(fabricates N-identifiability, conflates two demographic epochs).

**[OPEN] — needs the provenance of 6700.** It predates this project. See `TODO.md`.

### 6.2 Is N identifiable? — must be re-derived [OPEN]

The earlier conclusion was that population size is nearly unidentifiable from π: at
`ancestral_Ne = 6700` and μ = 5e-6, recapitation alone predicts `4·Ne·μ = 5.63e-5` against a
pipeline output of `6.38e-5`, i.e. **~7/8 of simulated diversity was set by a constant that
`population_modifier` does not touch.**

**That arithmetic is conditional on values now known to be wrong.** If ancestral Ne rises ~217×
and μ drops to biological, the forward/ancestral balance shifts and N may become identifiable.
**Re-derive, do not assume.** Use branch-mode diversity to diagnose. A POPMULT sweep
(`diagnostics/qdriver.py`) is the way to settle it; a previous attempt OOM'd at POPMULT=40000.

Regardless: π depends on N and μ only through **θ = 4Nμ** — they are confounded. If both stay
free, the posterior is a *ridge*, not a peak, and a peaked-looking N marginal is an artefact.
Report **θ=4Nμ** and **Nm**.

### 6.3 `recombination_rate` never reaches the forward simulation [VERIFIED]

`CPBSampleSimLinux.slim:12` hardcodes `initializeRecombinationRate(1e-8)`. The ABC parameter is
passed **only** to `pyslim.recapitate()`, so it cannot affect forward dynamics. It is currently
fixed (not inferred) at `DEFAULT_RECOMBINATION_RATE = 2.75e-6`.

Separately, recombination has **no signal at all** in π/d_xy/F_st — it shows up only in linkage
disequilibrium. Inferring it requires an LD summary statistic. Fix: pass it in as a `-d` constant
like `POPMULT`.

### 6.4 `simplificationRatio=INF` is the memory bug [VERIFIED, fix untested]

`CPBSampleSimLinux.slim:4` sets `initializeTreeSeq(simplificationRatio=INF)`, telling SLiM to
**never simplify during the forward run.** The edge table grows unbounded for all 324 generations.
Combined with `treeSeqRememberIndividuals(..., T)` at gens 308 and 316, this is almost certainly
the real cause of the OOM — not the mutation rate.

**[OPEN]** Removing it (letting SLiM auto-simplify) is untested but is the obvious first thing to
try. If it works, the "must inflate μ to keep the sim tractable" premise dissolves and large
POPMULT becomes affordable.

### 6.5 Resolved [VERIFIED]

- **pixy's `count_comparisons` overflowed int32** (fixed 2026-07-28). The field saturates to
  `INT32_MIN` once `comparisons_per_site × no_sites > 2^31`, i.e. at **≥14 diploid individuals**.
  Only `H53-2015` (19) ever crossed it, corrupting **244 of 2015's rows** — but just 18 showed a
  visible sign flip (π = −0.025); the rest, including two `Alsum25` d_xy pairs that saturated on a
  single chromosome, were merely **~18% high and looked entirely plausible**. Fixed by computing
  denominators analytically (§5.1). **Still live in two ways:** any future year with ≥14 sampled
  individuals re-triggers it (2023's max of 308 comparisons/site sits just under the 327.9
  threshold), and **any `empiricalStats` output produced before 2026-07-28 is suspect.**
- **Migration saturation.** The old code normalized each row of `exp(-d·modifier)` to sum to 1,
  making ~76% of each subpop's offspring immigrants every generation — effectively panmixia, which
  is why simulated F_st was tiny and d_xy ≈ π. Fixed by the `total_migration`/`scale` split (§2).
- **KMeans re-randomization.** `random_state=random.randint(0,1000)` gave a different cluster
  layout every run. Seed pinned at 42.
- **`csv.DictReader` ate subpop 0.** The π files have no header, so DictReader consumed the first
  value as a column name. Replaced with `_read_vector`/`_read_matrix`. (The remaining DictReader in
  `read_parameters_from_csv` is correct — that input CSV genuinely has a header.)
- **Missing header on `abc_results.csv`.** `csv_exists = Path(output_csv).exists()` skipped the
  header whenever the file existed — and CHTC's `run_code.sh` **pre-creates** it (so an evicted job
  fails with a real exit code instead of an errno-2 transfer hold), so every job emitted data rows
  with no column names. Now `needs_header` also treats a zero-byte file as needing one.
- **Coalescent rescaling cannot be applied as the model stands.** Standard rescaling (N→N/Q, μ→μQ,
  r→rQ, m→mQ) needs `m → m·Q`, but m was ≈0.76 — even Q=2 exceeds a probability; and the dominant
  term is fixed at `ancestral_Ne`, so the largest contribution to π is unaffected. Both blockers
  are parameter problems, not coalescent subtleties. Worth re-asking once §6.1/§6.4 are settled.

---

## 7. The ABC distance — design

In ABC there is no gradient-descent "loss." This is the **distance** `d(S_sim, S_obs)` used for
rejection/weighting; it only has to *rank* parameter draws sensibly.

**Fitted:** element-wise **log-π**, off-diagonal **F_st**, **IBD slope**.
**Diagnostic (computed, not fitted):** **d_xy**, **genetic relatedness**.

`calculate_losses` returns per-statistic, un-standardized, count-normalized distances (mean over
entries within a year, then mean over years):

```
pi_loss     = mean_years( mean_i | log pi_sim,i - log pi_obs,i | )   <- fitted, log-space
fst_loss    = mean_years( mean_pairs | Fst_sim - Fst_obs | )         <- fitted
ibd_loss    = mean_years( | slope_sim - slope_obs | )                <- fitted
dxy_loss    = mean_years( mean_pairs | dxy_sim - dxy_obs | )         <- diagnostic
genrel_loss = mean_years( mean_pairs | R_sim - R_obs | )             <- diagnostic
```

**There is deliberately no `total_loss` during the pass.** The combined standardized distance
`D = sqrt(Σ_j w_j (loss_j/σ_j)²)` with `σ_j = 1.4826·MAD` is built **offline** by
`Python_Code/abc_standardize.py`, using the run set as its own pilot batch, with equal weights to
start. This commits us to **rejection ABC**, which is what CHTC wants.

Rationale for the choices:

- **F_st, not raw d_xy.** d_xy ≈ π + differentiation, so its entries are nearly redundant with π
  and its differences are dominated by the diversity-level mismatch. Fitting d_xy would
  reintroduce the μ-degeneracy.
- **π element-wise, not mean/SD.** The specifier-matrix mapping makes simulated-subpop ↔ real-site
  a genuine within-year correspondence (§4), so element-wise is meaningful and dimensions always
  match.
- **π in log space; F_st and IBD not.** F_st is bounded near 0 and the IBD slope can be negative
  (log undefined). Log-π gives relative-error semantics and linearises the θ=4Nμ ridge.
- **IBD slope:** regress `F_st/(1 − F_st)` on **ln**(geographic distance) over all off-diagonal
  pairs (Rousset), take the OLS slope. One scalar regardless of k. Real-site coordinates from the
  specifier matrix are used for **both** the observed and simulated regressions.
- **Relatedness is not fitted** — it is linearly dependent by construction (centring makes rows sum
  to ~0) and measures the same signal as F_st. Kept as a **posterior-predictive check**: a
  statistic you didn't fit is far better validation than one you did.
- **Normalize out year entry counts** so 24- vs 17- vs 20-subpop years contribute comparably.

Verified behaviour [VERIFIED]: `calculate_losses(obs, obs)` → all five losses exactly 0.0.
Perturbed sim (π×1.1, F_st+0.02) → `pi_loss = 0.0953 = ln(1.1)` exactly, `fst_loss = 0.02`.
Real-data IBD slopes finite: 2015 −1.42e-3, 2019 +8.04e-4, 2023 +1.44e-4 (weak/mixed).

**Current priors** (`ABCAnalysisNoRedis.py`):

| parameter | prior | note |
|---|---|---|
| `m` (kernel decay) | `lognorm(s=1.5, scale=1e-4)` | reshapes *where* migrants come from |
| `total_migration` | `U(0.001, 0.301)` | **placeholder ceiling** — needs a biological bound |
| `pop` (POPMULT) | `U(2000, 12000)` | ≈ 6.7k–40k individuals |
| `numClusters` | {1,2,3} | **CSV records the raw draw; actual count is ×33** |
| `mutation_rate` | `lognorm(s=0.5, scale=5e-6)` | **calibrated to the broken π target — rebuild** |
| `recombination_rate` | fixed 2.75e-6 | not inferred (§6.3) |

---

## 8. Invariants — violate these and the sim/empirical comparison is meaningless

1. **Match the sampling design.** Subsample the tree sequence to the same subpops and per-subpop
   counts as the year in question. Pull them from the popfile, then `ts.simplify(samples=nodes,
   filter_populations=False)` — no re-simulation needed. **`filter_populations=False` is
   mandatory** (§2).
2. **Never slice a big matrix to get a small one.** `genetic_relatedness` is **centred on the
   populations present in the call**. A 3-pop matrix computed from scratch ≠ 3 rows/cols sliced
   from a 20-pop matrix — values differ and can flip sign. **Always recompute on the subset.**
   Biggest footgun in the project.
3. **Centring is per-year.** Absolute relatedness values are not comparable across years without
   re-centring on a common reference.
4. **Genome-wide = pool, don't average** (§5.2).
5. **Order every per-subpop matrix by the specifier matrix** (§4), never by `sorted(labels)`.
6. **Compare against a distribution, not a single run.** Coalescent and subsampling are both
   stochastic. Run replicates; build an envelope (e.g. 2.5–97.5%).
7. **Biallelic SNPs only.** tskit treats missing genotypes as ancestral; the empirical code
   computes frequencies over *called* genotypes only — a known, small source of divergence.
8. `ts.write_vcf` **rounds mutation positions to integers** — dense mutations on a short sequence
   can produce duplicate positions. Harmless for distance-binned LD; know about it.

---

## 9. Existing code

| File | What it does |
|---|---|
| `Python_Code/Main.py` | Pipeline entrypoint (§2). Threads `total_migration`, `ancestral_Ne`; `KMEANS_SEED = 42`. |
| `Python_Code/AnalyzeTreeSeq.py` | Recapitate → simplify → mutate → π, d_xy, F_st, relatedness (batched `indexes=`). |
| `Python_Code/ABCAnalysisNoRedis.py` | ABC driver, `calculate_losses`, IBD helpers (`get_site_geo_distances`, `ibd_slope`). CHTC entrypoint: `python ABCAnalysisNoRedis.py <job_id> [num_trials]` → `../out/abc_results.csv`. Seeds `np.random.seed(job_id)`. |
| `Python_Code/abc_standardize.py` | **Offline** post-processing: σ=1.4826·MAD per fitted stat → standardized `D` → ranked CSV + frozen σ JSON. Not run in the pass loop. |
| `Python_Code/GenerateSimulationParams.py` | `determine_migration_rates(distances, total_migration, scale, ...)`. |
| `Python_Code/GenerateClusterData.py` | KMeans clustering, distance matrix, genome→cluster assignment (`assign_genomes_to_clusters_idv_year` sets subpop→specifier-row mapping). |
| `SLiM_Code/CPBSampleSim{Linux,Win}.slim` | Forward sim. Neutral, `mutationRate(0)`, `simplificationRatio=INF` (§6.4, unfixed). |
| `diagnostics/qdriver.py` | Controlled sweep/rescaling harness: fixed KMeans seed, templated SLiM, scalable migration, reports branch-mode diversity. |
| `diagnostics/qpost.py` | Post-process an existing `.trees` → branch diversity, site π, F_st. Fast; no SLiM needed. |
| `ToUseOnBeagles/*` | Empirical-side pipeline (§5). Runs on the Beagle machine, not here. |

Both hand-rolled estimators (`CalcGenRel.py`, `CalculateLD,py`) were **verified bit-identical to
tskit** (`genetic_relatedness`, `ld_matrix(stat="r2")`, ~1e-16/1e-18). **Keep it that way** —
re-run that verification if you touch them. Note this is an *estimator* check on identical input;
it says nothing about which set of sites each side is computed over in production, which is
exactly where the §5.1 denominator mismatch lived.

`run_code.sh` is the CHTC wrapper and is **not in the repo**. It clones from `origin/main`, so
**any code or empirical-target change must be committed and pushed before a CHTC submission.**

---

## 10. Conventions

- Python; `numpy`, `scikit-allel`, `tskit`, `msprime`, `pyslim`.
- Scripts carry a `CONFIG` block at the top rather than argparse (`diagnostics/*` are the
  exception — they take flags, since they're meant to be swept).
- When implementing any new statistic, **verify it against tskit on a small msprime simulation
  first**, then apply it to real data. That pattern has caught real bugs three times.
- The pipeline **overwrites files in `data/` in place** (§2). Any experiment should run on a copy
  of the tree, not the repo, or redirect those paths.
- Prefer failing loudly over silent fallbacks for anything that changes units or scale — a default
  that reproduces a known-wrong scale is worse than a crash, because the output files carry no
  record of which scale they are on.
