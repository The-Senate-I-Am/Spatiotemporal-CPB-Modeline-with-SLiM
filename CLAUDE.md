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
4. `slim -d POPMULT=<pop> -d RECOMB=<r> SLiM_Code/CPBSampleSim{Win,Linux}.slim` →
   `out/simTreeSeq.trees`. Neither constant has a default inside the `.slim` files — an absent
   `-d` is a loud `undefined identifier` error, not a silent fallback to a different scale.
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

> **The same bug survived in the diagnostics harness until 2026-08-04** — `qdriver.py:154` and
> `qpost.py:52` both simplified with the default `filter_populations=True` and then queried
> `ts.samples(population=i, ...)` with original cluster-row indices. Both now pass
> `filter_populations=False`. Exposure was probably small at numClusters=33 (most demes are
> referenced), but **any §6.2-style sweep number that came from these two scripts rather than the
> full pipeline should be re-measured**, and this is a prerequisite for using them on the §6.1
> `--anc-ne` sweep. Grep for `.simplify(` before trusting any new script.

---

## 3. Environment [VERIFIED]

**`cpb-env` (from `environment3.yml`): SLiM 5.1, pyslim 1.1.1, tskit 1.0.2, msprime 1.4.1,
python 3.12.** Internally consistent SLiM-5 stack; the pipeline runs end to end. Invoke via
`conda run -n cpb-env python ...` from `Python_Code/` (paths are relative to that dir).

> An older note claimed "SLiM 4.3 tree sequences need pyslim 1.0.x; pyslim 1.1+ fails." That
> described the *old* env. Do **not** downgrade pyslim on this env.

### 3.1 Per-run cost [VERIFIED 2026-07-31, measured post-§6.3/§6.4, numClusters=33]

All on one 15.2 GB Windows box. Peak memory is the OS high-water mark (`peak_wset`), tracked
separately for the SLiM child and the Python process — **they do not overlap**, SLiM has exited
before recapitation starts.

| POPMULT | SLiM time | SLiM peak | `.trees` | analysis time | **analysis peak** | total |
|---|---|---|---|---|---|---|
| 500 | 4.0 s | 173 MB | 21.1 MB | ~274 s | 611 MB | 4.5 min |
| 5000 | 50.9 s | 1672 MB | 216.5 MB | 1764 s | **7632 MB** | **30.3 min** |
| 12000 | 169.6 s | 3938 MB | 519.8 MB | — **OOM** — | ≈20 GB (est.) | — |

**Scaling, measured, not assumed:**
- **SLiM memory and `.trees` are linear in POPMULT** (9.7× and 2.36× against 10× and 2.4×). This
  is what confirms §6.4's fix works — with `simplificationRatio=INF` the edge table grew with
  generations×individuals instead.
- **Analysis-phase memory is SUPERLINEAR**, exponent ≈1.10 (12.5× for 10×). A linear
  extrapolation *under*-predicts: it gave 6.1 GB at POPMULT=5000 against 7632 MB measured.
- **Analysis time is SUBlinear** (6.4× for 10×), projecting **~60 min/trial at POPMULT=12000**.

**Recapitation is still the bottleneck** — ~97% of wall time, and it is where the memory goes.

> **POPMULT=12000 OOMs on a 16 GB machine.** Killed inside `pyslim.recapitate`, before the
> `Simplifying tree sequence...` print. This is **not** a §6.4 regression: the forward phase
> completed fine at 3.9 GB. It is the memory floor §6.4 says simplification cannot touch — gens
> 308/316 permanently Remember *every* individual in *every* subpop, so recapitation faces ~240k
> sample nodes.

**CHTC sizing:** ~8 GB for POPMULT≈5000, ~20 GB for 12000, 30–60 min/trial. The prior ceiling is
12000, so **undersized `request_memory` would hold jobs mid-pass** — and rejection ABC pools
whatever survives, biasing the posterior toward draws small enough to finish. Size for the ceiling.

**Simplify-before-recapitate** was considered and **declined** (§9 `AnalyzeTreeSeq.py:126,146`
recapitates the full tree sequence and only then simplifies to the ~2.4k genomes actually
sampled). Two-thirds of that reasoning is now known to be wrong: it is *safe* with
`keep_input_roots=True` (pyslim documents exactly this), and the benefit is **not** local-only —
it is the difference between running and OOMing at POPMULT ≳ 9000, on CHTC nodes too. Left
unchanged deliberately (2026-07-31): nothing is broken, and CHTC can request more memory.
**Revisit if memory or throughput becomes binding.**

Total simulated N is **not** POPMULT. Subpop size is `Average Count × POPMULT / numSubpops`, so
`total N ≈ POPMULT × mean(Average Count) ≈ 3.33 × POPMULT`. The older OOM (POPMULT≈40000 on a
128 GB machine) predates the §6.4 fix; the forward phase is now linear and cheap, so if that
recurs it will be recapitation again, not the edge table.

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
- Subpop sizes range 2–19 diploid individuals. Small ones are very noisy — quantified in §7.0, and
  the reason `EXCLUDE_SMALL_SUBPOPS` drops n ≤ 3 from the fitted statistics. The large ones are
  not free either — at **≥14 individuals** pixy's comparison counter overflows (§6.5). Only
  `H53-2015` (19) has ever crossed it; 2023's largest is 11.
- A **typo-duplicated `Arlington` label** produces two near-identical subpopulations. [OPEN —
  decide deliberately whether to merge.] Note the small-subpop mask already drops `Arlington2015`
  (n=2) from the **fitted** statistics, so this currently bites only the diagnostics (§7.0).
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

### 6.1 `ancestral_Ne = 6700` — provenance found; the value fails its own source's internal check

**Source [VERIFIED 2026-08-04]: Cohen et al. 2022, *Evolutionary Applications* 15:1691–1705
(doi:10.1111/eva.13498), Figure 3a.** PDF is in the repo root. 6700 is `N_a`, the **dadi**-inferred
ancestral effective size of the common ancestor of the Hancock, WI and Long Island, NY pest
populations, at a split **325 generations (160 yr ± 1.6)** ago. The same figure gives
`N_WI = 15,000 (±163)` and `N_NY = 40,000 (±760)`.

**This one paper is the provenance of nearly every fixed constant in the project:**

| project constant | source in Cohen et al. |
|---|---|
| `ancestral_Ne = 6700` | Fig. 3a `N_a = 6,700 (±10)` |
| `DEFAULT_RECOMBINATION_RATE = 2.75e-6` | §3.2 `r_HAN = 2.75e-6` — the **Wisconsin** population, i.e. the right one for us |
| SLiM run length 324 generations (`CPBSampleSim*.slim:43,48,53`) | the 325-generation divergence |
| "biological μ ≈ 2.1e-9" | the *Chironomus riparius* midge rate Cohen used; there is no CPB-specific rate |
| `pop` prior `U(2000, 12000)` → total N 6.7k–40k | brackets `N_WI = 15,000` and `N_NY = 40,000` |

**The old guess in this section was half wrong.** 6700 is *not* a contemporary or LD-based
estimate. In dadi's `no_mig` model the ancestral population is **constant-size extending
infinitely into the past**, which is conceptually exactly what `recapitate(ancestral_Ne=)` wants.
**The role is right.** The value is not.

**The ~217× conflict is internal to Cohen et al., not a disagreement between their data and ours.
[VERIFIED — arithmetic]** Watterson's θ from *their own reported* 11.8M polymorphic sites over an
~870 Mb genome at n=28 diploids (a_56 = 4.594):

```
theta_W = 11.8e6 / (870e6 * 4.594)   = 2.95e-3 per site
Ne      = theta_W / (4 * 2.1e-9)     ~ 3.5e5      <- from Cohen's own SNP count
```

That is **52× their own reported `N_a` = 6700**, using their own mutation rate. Our π-derived
`Ne = 0.0122/(4·2.1e-9) ≈ 1.45e6` is only **4×** from *that* — same order of magnitude.
**Our π is not the outlier; 6700 is.**

**The paper says so itself, twice.** §4.1: sequencing was low-coverage (>5× average, ≥3× per
individual), so "possible heterozygous sites [are] mistaken as homozygous… reducing singletons and
causing a bias that results in **underestimating demography**… our results might lead to an
**underestimate in effective population size**." And their two methods disagree with each other —
"the dadi estimates… were ~4-fold larger than estimates from the Stairway plot," whose Figure 1
sits at 1k–20k. The authors treat this as a modest caveat; the arithmetic above says it is ~50×.

**Leading mechanistic candidate — and it is the mirror image of the §5.1 bug we just fixed on our
own side. [INFERRED]** `N_a` is derived, not measured: `N_a = θ_dadi / (4μL)` with
**L = 840 Mb, "all intergenic sequence data."** But the 2D-SFS was built from intergenic regions
"with **stringent quality thresholds for coverage and likelihood**." If θ was fit to an SFS drawn
from a heavily filtered subset while L was set to the full intergenic span, `N_a` is deflated by
exactly that fraction — the same denominator/numerator mismatch as §5.1, pointing the other way.
Reconciling with our π would need L ≈ 3.9 Mb, which is too small to be the whole story on its own,
so this is likely **compounded with** the low-coverage singleton loss the authors name. Confirming
it would need their supplement. Either way both named biases push the same direction: *up*.

How μ has been absorbing the error:

| scenario | π target | required μ at Ne=6700 | × biological |
|---|---|---|---|
| both errors | 0.140 | 5.2e-6 | ~2400× |
| denominator fixed (now) | 0.0122 | 4.6e-7 | ~217× |
| both fixed (Ne ≈ 1.5e6) | 0.0122 | 2.1e-9 | 1× |

`ancestral_Ne` is a threaded parameter, **not** an ABC free parameter — it is confounded with μ in
π (they enter only as `4·Ne·μ`), so inferring it would build a second ridge. Exposed for
sensitivity analysis only. Scaling it with `population_modifier` was considered and **rejected**
(fabricates N-identifiability, conflates two demographic epochs).

**Resolution [VERIFIED 2026-08-04 by the ridge sweep, §6.2.1 — this REVERSES the earlier
recommendation to set `ancestral_Ne ≈ 1.4e6`]:**

**Keep `ancestral_Ne = 6700`. Do not raise it.** Two measured facts force this:

1. **It would not change the inference.** π is invariant along the `4·Ne·μ = const` ridge in
   *both* its level (±2.6%) and its between-subpop relative spread (CV flat to <1%). So moving
   along the ridge is very nearly a no-op for the ABC.
2. **It is computationally impossible.** Recapitation cost scales as **Ne^2.34** (measured).
   Ne = 1.452e6 extrapolates to **~600 days per trial.**

**What must change is the language, not the constant.** μ = 5e-6 is **not a mutation rate** — it
is half of a calibration constant whose only meaningful content is the product `4·Ne_anc·μ`, set
to match observed π. Never report μ on its own, never present it as biological, and do not infer
it: report **θ = 4Nμ** (§6.2 already said this). The critique of Cohen's 6700 above stands as a
statement about CPB biology; it just does not license a code change, because the biologically
"correct" value cannot be simulated.

> **Calibration is POPMULT-dependent.** Setting `4·Ne·μ = 0.0122` does *not* yield π = 0.0122 —
> forward-phase coalescence pulls `branch_div` below `4·Ne_anc`, so the realised π is lower, and
> by a POPMULT-dependent factor (81% of target at POPMULT=500, ~50% at POPMULT=150). μ and POPMULT
> are therefore mildly coupled through the π level. Calibrate μ at the POPMULT you intend to run.
> **Done — see §6.1.1.**

### 6.1.1 μ recalibrated: 5e-6 → 4.646e-7 [VERIFIED 2026-08-11]

`diagnostics/mu_calibrate.py`, `numClusters=33`, `total_migration=0.05`, `ancestral_Ne=6700`,
seed 1. Raw records in `out/mu_calibration.jsonl`; `mu_calibrate_summary.py` tabulates them.

**Method.** Recapitate + simplify **once** (~97% of cost, and entirely μ-free), then sweep μ over
cheap mutation overlays on the ~2.4k-sample simplified tree. Branch-mode diversity `b_i` gives an
analytic first μ with no re-running; the loop then corrects the multiple-hit deficit. The
objective is the *actual* fitted loss, not mean-matching: since `log π_sim,i = log μ + log b_i`,
minimising `pi_loss` over `log μ` is a weighted-L1 problem whose exact minimiser is the **weighted
median** of `log π_obs,i − log b_i` with weights `1/(3·n_year)` — the same year-normalisation
`calculate_losses` uses, with the §7.0 mask applied to both sides.

| POPMULT | subpop N | `branch_div` | /ceiling | **μ_calib** | 4Ne·μ | `pi_loss` | `fst_loss` | sim F_st |
|---|---|---|---|---|---|---|---|---|
| 500 | 50 | 21647 | 0.789 | 5.564e-7 | 0.01491 | 0.0672 | 0.07145 | 0.0766 |
| 2000 | 202 | 25629 | 0.934 | 4.819e-7 | 0.01291 | 0.0290 | 0.01856 | 0.0197 |
| **5000** | 505 | 26847 | 0.978 | **4.646e-7** | 0.01245 | 0.0213 | **0.00830** | 0.0079 |

(observed F_st = 0.00645 in every row.) **`DEFAULT_MUTATION_RATE` is now the POPMULT=5000 value,
4.646e-7 — the old 5e-6 was 10.8× too large**, being calibrated against the pre-2026-07-28
per-SNP π target (§5.1).

**Three independent cross-checks passed**, which is what licenses trusting the harness: implied
`branch_div` at POPMULT=500 reproduces `ridge_sweep.jsonl`'s 21985; `fst_loss` at POPMULT=500
reproduces §6.2's 0.0719; and `fst_loss` at POPMULT=5000 reproduces §6.2's 0.00829 **exactly**.

**`site_pi = μ · branch_div` is not exact — it runs 1–2% low**, and the deficit grows with `μ·b`
(multiple hits at a site). Purely analytic calibration lands ~1.8% low, so the iteration matters.
Monte-Carlo precision of the result is **0.44–0.99%** (each iterate re-draws mutations, so the
loop oscillates rather than converging to a fixed point; `mu_calibrated` is the last iterate).

**`branch_div` saturates, which is why ONE fixed μ covers the whole prior.** Recapitation
coalesces any surviving pair at rate `1/(2·Ne_anc)`, so
`branch_div ≤ 2·(324 + 2·6700) = 27448` — a hard ceiling, giving μ a floor of ~4.45e-7. Fitting
the deficit `1 − b/ceiling ≈ 94.2·POPMULT^−0.973` (a **fit, not a derivation** — an `exp(−P/τ)`
form was tried and rejected for being unable to fit both ends) extrapolates to μ = 4.51e-7 at
POPMULT=8000 and 4.49e-7 at 12000. Holding μ at 4.646e-7 across the prior `U(2000, 12000)`
therefore drifts π by only **−3.2% to +2.1%** — and that drift is *signal*: it is how the π level
carries POPMULT information now that μ is effectively pinned.

**Consequence for §6.2 — the blocker is gone.** With μ calibrated per POPMULT, π and F_st no
longer pull in opposite directions; both improve monotonically together (π 0.067→0.029→0.021,
F_st 0.071→0.019→0.0083). The conflict was an artifact of the miscalibrated μ, not a feature of
the data.

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

**Partial answer [VERIFIED 2026-07-31]: N *is* identifiable — through F_st, not through π.**
A two-point POPMULT sweep at μ=5e-6, `total_migration=0.05`, numClusters=33:

| POPMULT | subpop size | Nm | sim F_st | **fst_loss** | sim π | **pi_loss** |
|---|---|---|---|---|---|---|
| 500 | ~50 | 2.5 | 0.0754–0.0778 | 0.0719 | 9.5e−2 | 2.043 |
| 5000 | ~505 | 25 | 0.0076–0.0082 | **0.00829** | 1.17e−1 | **2.262** |
| *observed* | — | 31–83 | 0.0032–0.0083 | — | 1.22e−2 | — |

F_st tracks `1/(1+4Nm)` almost exactly and `fst_loss` improves **8.7×**. So the structural side
carries real information about N. Do not conclude "N is unidentifiable" from the π argument alone.

**~~But the two fitted statistics currently pull in opposite directions.~~ RESOLVED 2026-08-11 by
§6.1.1.** The old reading was: the POPMULT that fits F_st drives π *further* from target (9.7× too
high at POPMULT=5000), so **no POPMULT satisfies both**. That was true only at the miscalibrated
μ = 5e-6. With μ recalibrated to 4.646e-7, π and F_st improve **together** with POPMULT
(π 0.067→0.029→0.021, F_st 0.071→0.019→0.0083 over POPMULT 500→2000→5000). The tradeoff was an
artifact of μ, not a feature of the data, exactly as this section suspected.

This section also **predicted the recalibration, and the arithmetic route won**: matching π at
POPMULT=5000 and Ne=6700 was estimated at μ ≈ **5.2e-7** by sweep extrapolation, against
**4.6e-7** from §6.1's π arithmetic. The measured value is **4.646e-7** — the arithmetic route was
right to within 1%, the extrapolation 12% high. Both remain ~220× the biological rate, which is
the §6.1 problem restated and is *not* resolved by this (nor can it be — §6.2.1).

### 6.2.1 The ridge sweep — π survives, but Ne_anc cannot be raised [VERIFIED 2026-08-04]

`diagnostics/ridge_sweep.py`, POPMULT=150, numClusters=33, seed 1, two points on
`4·Ne·μ = 0.0122`: **(Ne=6700, μ=4.55e-7)** and **(Ne=20000, μ=1.53e-7)**, a 2.985× step.

| quantity | Ne=6700 | Ne=20000 | ratio |
|---|---|---|---|
| `recap_s` | 183.8 | 2369.7 | **×12.89 → exponent 2.34** |
| `recap_peak_mb` | 292 | 291 | ×1.00 |
| site π (2015/19/23) | .00582/.00616/.00631 | .00570/.00600/.00615 | **−2.1/−2.6/−2.5%** |
| `branch_div` mean | 12918/13649/13992 | 37665/39635/40616 | ×2.92/2.90/2.90 |
| `branch_div` **sd** | 4010/4286/3816 | 11687/12508/11170 | ×2.92/2.92/2.93 |
| `branch_div` **CV** | .3104/.3140/.2727 | .3103/.3156/.2750 | **×1.000/1.005/1.008** |
| mean F_st | .2503/.2277/.2045 | .2518/.2291/.2063 | +0.6/+0.6/+0.9% |

**Test 1 — π is invariant along the ridge. PASSES.** π moves only −2.1 to −2.6% for a 3× change in
Ne_anc. The small residual drift is the forward-phase contribution shrinking in relative terms,
exactly as theory says it should.

**Test 2 — the between-subpop spread does NOT collapse. The prediction above was WRONG.** Mean and
SD of `branch_div` both scale ×2.9 with Ne_anc, so the **CV is flat to under 1%.** The mechanism:
the ancestral contribution to a subpop's coalescence time is weighted by the probability its pairs
did *not* already coalesce in the forward phase, and that probability varies by subpop. So the
ancestral phase **multiplies** the forward structure rather than **adding** a constant to it, and
relative structure is preserved no matter how deep the ancestral phase gets.

Consequences, all favourable:
- **π stays a fitted statistic.** It keeps its full relative information about POPMULT and
  migration. Do not demote it.
- **Log space is exactly right** (§7): `pi_loss` measures relative differences, which is precisely
  the quantity shown to be ridge-invariant.
- **The circularity worry is defused.** The element-wise variation is genuine independent signal
  and it does not shrink, so calibrating the *level* does not hollow out the statistic.
- **F_st is confirmed insensitive to `ancestral_Ne`** (<1%), as a forward-phase ratio should be.

**Test 3 — cost. This is the new blocker.** `recap_s` scales as **Ne^2.34** while memory stays
flat at ~291 MB. So the constraint is **wall time, not RAM** — the opposite of the §3.1 OOM
problem. Extrapolated from the Ne=6700 baseline at POPMULT=150:

| target Ne_anc | factor | projected recapitation |
|---|---|---|
| 2e5 | ×2,805 | ~143 h (6 days) |
| 1.452e6 (π-implied) | ×288,691 | **~14,700 h (614 days)** |

**Raising `ancestral_Ne` to the biologically-implied value is computationally out of reach**, by
about four orders of magnitude, and no amount of CHTC memory fixes a wall-time wall. Since §6.2.1
also shows raising it would barely move the inference, the correct move is to **keep 6700** — see
§6.1's resolution.

**Caveats, stated honestly.** Measured across 3× in Ne (6700→20000), not the full 217×; the
mechanism (mean and sd both ∝ Ne_anc) is clear and the extrapolation is principled, but the top of
the ridge is unverified. POPMULT=150 is far below the prior range, chosen because cost is
dominated by Ne_anc — dropping POPMULT 3.3× (500→150) cut recapitation only 19%, so POPMULT is
**not** a useful cost lever. The absolute CV is strongly POPMULT-dependent (0.27–0.31 at
POPMULT=150 vs 0.081–0.093 at POPMULT=500); what was shown invariant is its *insensitivity to
Ne_anc*, not its value.

### 6.3 `recombination_rate` now reaches the forward simulation [FIXED 2026-07-29]

Was: `CPBSampleSim{Linux,Win}.slim:12` hardcoded `initializeRecombinationRate(1e-8)`, so the ABC
parameter reached **only** `pyslim.recapitate()` and could not affect forward dynamics. Both files
now take `initializeRecombinationRate(RECOMB)` and `Main.py` passes `-d RECOMB=`. Verified against
SLiM 5.1: `-d RECOMB=2.75e-06` (the format Python's `!r` emits) parses; omitting it errors with
`undefined identifier RECOMB` and exit 1, which `check=True` on the subprocess turns into a raise.

**This is a 275× increase in forward recombination** (1e-8 → 2.75e-6), so expect many more edges
per generation and materially higher SLiM memory/runtime than any historical measurement. It is
the main reason §6.4 had to be fixed at the same time.

Still fixed, not inferred, at `DEFAULT_RECOMBINATION_RATE = 2.75e-6`: recombination has **no
signal at all** in π/d_xy/F_st — it shows up only in linkage disequilibrium, so inferring it needs
an LD summary statistic first.

### 6.4 `simplificationRatio=INF` removed [FIXED 2026-07-29, unrun]

Was: `CPBSampleSim{Linux,Win}.slim:4` set `initializeTreeSeq(simplificationRatio=INF)`, telling
SLiM to **never simplify during the forward run**, so the edge table grew unbounded for all 324
generations — almost certainly the real cause of the OOM, not the mutation rate. Now plain
`initializeTreeSeq(timeUnit="generations")`, i.e. SLiM's default ratio of 10.

**Safe — checked against the docs, not assumed.** Simplification is lossless for the genealogy of
retained samples; SLiM retains all living individuals plus everything permanently Remembered
(`treeSeqRememberIndividuals(..., T)` at gens 308/316 — `permanent=T` marks them as real samples),
and future generations descend only from living individuals, so nothing needed later is discarded.
Recombination is unaffected: breakpoints are recorded at reproduction, and simplification runs
afterwards on the already-written tables. The manual frames `simplificationRatio` purely as a
speed/memory tradeoff.

> The recapitation hazard is real but belongs to the **other** mechanism. Recapitation needs the
> input roots, and pyslim is explicit that a *Python-side* `ts.simplify()` before recapitating must
> pass `keep_input_roots=True` — that is why §3 declined simplify-before-recapitate. SLiM's own
> runtime simplification already uses `keep_input_roots`, which is why ordinary SLiM output (SLiM
> simplifies every ~20 ticks by default) is routinely recapitable. Do not conflate the two.

**[OPEN] Not yet run end-to-end.** Output will not be bit-identical (nodes are renumbered,
redundant edges merged); compare **branch-mode diversity under a fixed recapitation seed**, not
file hashes.

**Memory floor this cannot touch:** gens 308/316 permanently Remember *every individual in every
subpop*, pinning all of their ancestry. Downstream only 2–19 individuals per site are ever
sampled, so Remembering a bounded subset (say 50/subpop) would cut retained ancestry a lot. That
changes what is available to the sampler, so it is a design decision, not a free win.

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

**Fitted:** element-wise **log-π**, off-diagonal **F_st**.
**Diagnostic (computed, not fitted):** **IBD slope**, **d_xy**, **genetic relatedness**.

> **IBD was demoted from fitted on 2026-07-29** (§7.1). The switch that actually matters is
> `abc_standardize.py::FITTED_STATS`; `calculate_losses` still returns `ibd_loss`.

### 7.0 Small-subpop exclusion [VERIFIED]

`ABCAnalysisNoRedis.py` has `EXCLUDE_SMALL_SUBPOPS = True`, `MIN_SUBPOP_N = 4`. Subpops with
fewer than 4 diploid individuals are dropped from the **fitted** statistics via
`get_keep_mask(year)`, a boolean mask in specifier-matrix row order applied identically to the
observed and simulated sides.

Why: at n ≤ 3, **46.7% of 2015's pairs return a negative F_st** (vs 5.3% at 4≤n≤7 and 0% at n≥8) —
those entries scatter around a noise floor rather than measuring differentiation.

Only 2015 is affected. It drops exactly two sites — **`Arlington2015` (n=2) and `H67-2015` (n=2)**
— removing exactly the 45 noise-floor pairs. **Side effect worth knowing:** `Arlington2015` is the
typo duplicate (§4), so for the fitted statistics the duplicated-`Arlington` ambiguity is now moot
— only `Arlington-2015` (n=4) survives. The underlying data question is still open.

**Relatedness is deliberately NOT masked** — it is centred on the populations present when it was
computed, so slicing rows/cols is not the same as recomputing on the subset (invariant 2). Both
sides stay full-size. π, F_st and d_xy are all safe to slice.

`calculate_losses` returns per-statistic, un-standardized, count-normalized distances (mean over
entries within a year, then mean over years):

```
pi_loss     = mean_years( mean_i | log pi_sim,i - log pi_obs,i | )   <- fitted, log-space
fst_loss    = mean_years( mean_pairs | Fst_sim - Fst_obs | )         <- fitted
ibd_loss    = mean_years( | slope_sim - slope_obs | )                <- diagnostic (was fitted)
dxy_loss    = mean_years( mean_pairs | dxy_sim - dxy_obs | )         <- diagnostic
genrel_loss = mean_years( mean_pairs | R_sim - R_obs | )             <- diagnostic
```

### 7.1 Why IBD is not fitted [VERIFIED 2026-07-29]

**The observed IBD slope is indistinguishable from zero in all three years.** Mantel test, 9999
permutations of site labels, using the project's own `ibd_slope`/`get_site_geo_distances`:

| year | n | pairs | slope | Mantel r | **p (two-sided)** | \|slope\|/null_sd |
|---|---|---|---|---|---|---|
| 2015 | 24 | 550 | −1.42e−03 | −0.069 | **0.639** | 0.46 |
| 2019 | 17 | 272 | +5.03e−04 | +0.214 | **0.148** | 1.43 |
| 2023 | 20 | 380 | +5.53e−05 | +0.004 | **0.985** | 0.02 |

The sign flips across years — noise, not a weak real effect. **Identical conclusion on the old
per-SNP targets**, so this does not depend on the denominator fix.

A Mantel test is required here, not an OLS p-value: `ibd_slope` fits over all n(n−1) ordered
off-diagonal pairs (552 for 2015, from 24 sites), which are massively non-independent — every site
appears in 23 of them. An OLS p-value would treat correlated pairs as independent observations and
report significance almost regardless of signal. Permuting **site labels** is the exchangeable unit.

Consequences:
- **`scale` (dispersal-kernel decay) is unidentifiable from these data.** Fix it or report it as
  unidentified. Do not present it as inferred.
- Fitting a target that is noise adds a pure-noise term to the standardized `D`, costing
  acceptance efficiency and blurring the parameters that *are* identifiable.
- **`total_migration` is probably still identifiable, via the F_st level rather than IBD:** at
  mean F_st ≈ 0.003–0.008, **Nm ≈ 31–83**.

**Geographic scale is not the explanation** — sites span 1.7–160 km, median pairwise ~34 km, in all
three years. That is ample range to detect IBD in an insect. And **Rousset's slope assumes
drift–dispersal equilibrium**, which is questionable for a recent fast-spreading invader
(Whitlock & McCauley 1999), so a null result does not cleanly mean "no IBD" — it can equally mean
"not at equilibrium yet."

### 7.2 F_st carries real signal, but it is site-coherent, not distance-structured [VERIFIED]

F_st was checked for sample-size noise-domination before being left as the sole structural fitted
statistic. **It is not noise-dominated.** Spearman rho between `min(n_i,n_j)` and `|F_st|`, with
site-label permutation: 2015 −0.082 (p=0.63), 2019 +0.242 (p=0.23), 2023 +0.082 (p=0.71). No
association. (2019 has almost no leverage — sizes are 5–7.)

Instead, **each year's extremes converge on a single site**, which is what real differentiation
looks like and noise does not:

- **`Mortensen9-2015`** (n=5): mean F_st 0.0686, **5.7× the next-highest site.**
- **`H41-2023`** (n=7): mean F_st 0.0472, **3× the next-highest.**
- 2019: no isolate at all; max pairwise F_st 0.0092.

**These are almost certainly a within-site sampling artifact, not landscape structure. [INFERRED,
strong]** Both isolates are simultaneously the **lowest-π** and (near-)**highest within-site
relatedness** site in their year:

| year | corr(mean F_st, π) | corr(mean F_st, self-relatedness) | isolate |
|---|---|---|---|
| 2015 | **−0.721** | +0.339 | `Mortensen9-2015`: π rank 1 (lowest), selfRel rank 3 |
| 2019 | −0.380 | −0.089 | — |
| 2023 | **−0.915** | **+0.748** | `H41-2023`: π rank 1 (lowest), selfRel rank 1 |

Low within-site diversity + high within-site relatedness + high F_st against everything is the
signature of **a sample of close relatives** (e.g. beetles taken off one plant or one egg mass —
entirely plausible for CPB) or a recent founder/bottleneck event at that field. Either way it is a
*local* phenomenon the landscape migration model cannot and should not reproduce.

**Geography rules out the innocent explanation.** The isolates are geographically *ordinary*:
`Mortensen9-2015` ranks 19/24 on mean distance to other sites, `H41-2023` ranks 17/20. Meanwhile
the genuinely remote sites are undifferentiated — `Alsum59-2023` is the most remote (114.6 km) and
ranks 11/20 on F_st. corr(mean F_st, mean distance) = −0.065 / +0.251 / +0.014.

Two implications:
1. **The dispersal kernel may be misspecified.** The data says "one site is an isolate, the rest
   are near-panmictic"; an exponential distance-decay kernel with one global `total_migration`
   produces smooth structure and cannot make a single deme an isolate. Element-wise F_st fitting
   will be dominated by pairs the model structurally cannot match.
2. **π and F_st are not independent statistics here** (r = −0.72 / −0.92 in 2015/2023). Fitting
   both with equal weight after MAD-standardization partly double-counts one signal.
3. **The two isolates also dominate the observed π *spread*** [VERIFIED 2026-08-11]. Dropping the
   single site cuts the fitted between-site log-sd of π from **0.0422 → 0.0212** in 2015
   (`Mortensen9-2015`, 50% of the spread) and **0.0477 → 0.0155** in 2023 (`H41-2023`, 67%).
   2019, which has no isolate, sits at 0.0160 already. So the *genuine* between-site π spread is
   **0.014–0.021** in all three years, and roughly half of the raw spread is the artifact.

   This matters for reading §6.1.1: simulated π spread falls with POPMULT (log-sd 0.10–0.13 at
   POPMULT=500, 0.028 at 2000, 0.010 at 5000). Against the *raw* observed spread the simulation
   looks 4× too flat at POPMULT=5000; against the **cleaned** spread it is only ~1.5×, and the
   observed value is bracketed inside the prior (matching around POPMULT ≈ 3000–4000). The
   simulation is not failing to produce structure — it is failing to produce an artifact, which
   is correct behaviour.

   **Caveat, unresolved:** it is *not* established that simulated and observed π covary
   site-by-site. If they do not, then `pi_loss` partly rewards a flat simulation for being closer
   in L1 to a scattered target than a differently-scattered simulation would be — which would
   make π's apparent preference for large POPMULT partly spurious. Checking this needs the
   per-subpop π vectors, which `mu_calibrate.py` currently summarises rather than stores. **Do
   this before trusting a POPMULT posterior driven by π.**

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
- **IBD is retained as a diagnostic** for the same reason — with the slope no longer fitted, a
  simulated-vs-observed slope comparison becomes an honest posterior-predictive check.

Verified behaviour [VERIFIED]: `calculate_losses(obs, obs)` → all five losses exactly 0.0.
Perturbed sim (π×1.1, F_st+0.02) → `pi_loss = 0.0953 = ln(1.1)` exactly, `fst_loss = 0.02`.
Real-data IBD slopes finite: 2015 −1.42e-3, 2019 +8.04e-4, 2023 +1.44e-4 (weak/mixed).

**Current priors** (`ABCAnalysisNoRedis.py`):

| parameter | prior | note |
|---|---|---|
| `m` (kernel decay) | `lognorm(s=1.5, scale=1e-4)` | **unidentifiable — IBD is not fitted (§7.1)** |
| `total_migration` | `U(0.001, 0.301)` | **placeholder ceiling** — needs a biological bound |
| `pop` (POPMULT) | `U(2000, 12000)` | ≈ 6.7k–40k individuals |
| `numClusters` | {1,2,3} | **CSV records the raw draw; actual count is ×33** |
| `mutation_rate` | `lognorm(s=0.05, scale=4.646e-7)` | **recalibrated 2026-08-11 (§6.1.1).** `s` tightened 0.5→0.05: at 0.5 a single draw swung π ±65%, swamping both the observed between-site spread (0.014–0.021) and the ±3% POPMULT drift. Kept free as a nuisance dimension; **report θ=4Nμ, never μ** |
| `recombination_rate` | fixed 2.75e-6 | now reaches SLiM too (§6.3); still not inferred |

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
| `Python_Code/ABCAnalysisNoRedis.py` | ABC driver, `calculate_losses`, IBD helpers (`get_site_geo_distances`, `ibd_slope`), `get_keep_mask` + `EXCLUDE_SMALL_SUBPOPS`/`MIN_SUBPOP_N` (§7.0). CHTC entrypoint: `python ABCAnalysisNoRedis.py <job_id> [num_trials]` → `../out/abc_results.csv`. Seeds `np.random.seed(job_id)`. |
| `Python_Code/abc_standardize.py` | **Offline** post-processing: σ=1.4826·MAD per fitted stat → standardized `D` → ranked CSV + frozen σ JSON. Not run in the pass loop. **`FITTED_STATS` here is what actually decides what enters `D`.** |
| `Python_Code/GenerateSimulationParams.py` | `determine_migration_rates(distances, total_migration, scale, ...)`. |
| `Python_Code/GenerateClusterData.py` | KMeans clustering, distance matrix, genome→cluster assignment (`assign_genomes_to_clusters_idv_year` sets subpop→specifier-row mapping). |
| `SLiM_Code/CPBSampleSim{Linux,Win}.slim` | Forward sim. Neutral, `mutationRate(0)`. Takes `-d POPMULT` and `-d RECOMB` (§6.3), default simplification (§6.4). The two files are identical apart from path separators — **fix both or neither.** |
| `diagnostics/qdriver.py` | **BROKEN — do not use** (found 2026-08-04). Drifted from production three ways: `simplificationRatio=INF` in its SLiM template (the §6.4 bug), `--slim-rho` default `1e-8` (the §6.3 bug value), and a `determine_migration_rates(distances, modifier=...)` call whose signature no longer exists → `TypeError` on every run. Fix or delete before trusting anything it produced. |
| `diagnostics/ridge_sweep.py` | §6.2.1 harness. `--setup` builds a `.trees` via the production path; each subsequent call runs one `4·Ne·μ = const` ridge point and appends JSON to `out/ridge_*.jsonl` (one point per process, so a failure costs only that point). Reports branch-mode diversity mean/sd/**CV**, site π, F_st, plus wall time and peak RSS. |
| `diagnostics/mu_calibrate.py` | §6.1.1 harness. Recapitates **once** (μ-free), then sweeps μ over cheap mutation overlays to solve `pi_loss` exactly (weighted median in log space, §7.0 mask applied). One POPMULT per process; appends JSON to `out/mu_calibration.jsonl`. `--skip-slim` reuses the `.trees` on disk. Also reports `fst_loss` at the calibrated μ, so both fitted statistics land in one run. |
| `diagnostics/mu_calibrate_summary.py` | Tabulates `out/mu_calibration.jsonl`: μ vs POPMULT, `branch_div` against its hard ceiling, the Monte-Carlo spread of the μ iterates, per-year sim-vs-obs log spread, and the saturation extrapolation. |
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
