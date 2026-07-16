# CLAUDE.md

Context for Claude Code sessions on this project. Read this first.

Status markers used below:
- **[VERIFIED]** — checked by reading the code and/or running it this session. Trust it.
- **[OPEN]** — not yet established. Do not assume either way.
- **[DONE 2026-07-15]** — implemented/changed in the 2026-07-15 refactor session.

> **2026-07-15 refactor.** The ABC distance was reworked from raw L1 on {π, d_xy} to a per-
> statistic feature set {π (log), F_st, IBD slope} with F_st/IBD added, migration decoupled,
> `ancestral_Ne` un-hardcoded, the KMeans seed pinned, and offline standardization. Full detail
> and verification in **`ABC_REFACTOR_PLAN.md`** (design) and **`ABC_REFACTOR_CHANGELOG.md`**
> (what changed + tests). Sections below are annotated with what is now resolved.

---

## 1. The project

Colorado Potato Beetle (*Leptinotarsa decemlineata*) population genetics across the
Wisconsin landscape.

- **Empirical.** Sequenced beetle genomes, phased/imputed (Beagle) VCFs, 17 chromosomes.
  Three sampling years — **2015, 2019, 2023** — each with a *different* set of subpopulations
  ("sites") and different per-subpop sample sizes.
- **Simulated.** SLiM (forward) → pyslim recapitation → msprime mutation overlay → tskit tree
  sequence. Entrypoint: `Python_Code/Main.py::main()`. Nominal free parameters: migration,
  population size, number of clusters, mutation rate, recombination rate.

**Goal:** Approximate Bayesian Computation to infer simulator parameters from the empirical
data. The distance/loss lives in `Python_Code/ABCAnalysisNoRedis.py::calculate_losses`.

> **Read §5 before touching the ABC.** Several of the parameters currently in the prior do not
> actually reach or move the simulation. Tuning the distance function before fixing that is
> wasted effort.

---

## 2. How the pipeline actually works [VERIFIED]

`Main.py::main(num_clusters, migration_rates_modifier, population_modifier, total_migration=0.05,
mutation_rate, recombination_rate, ancestral_Ne=6700)` [signature updated 2026-07-15]:

1. `GenerateClusterData.cluster_coordinates(..., random_state=KMEANS_SEED)` — KMeans over field
   coordinates from `data/final_data_for_modeling.csv`. **Seed now pinned** (§5.6).
2. Writes `data/cluster_data.csv` and `data/cluster_distances.csv` — **overwriting them in place.**
3. `GenerateSimulationParams.determine_migration_rates(distances, total_migration, scale, ...)` →
   `data/migration_rates.csv`. **Migration decoupled** (§5.2).
4. `slim -d POPMULT=<pop> SLiM_Code/CPBSampleSim{Win,Linux}.slim` → `out/simTreeSeq.trees`.
5. `AnalyzeTreeSeq.analyze_tree_sequence(..., ancestral_Ne=6700)` — recapitate → simplify →
   overlay mutations → write, per year, `diversities`, `divergences` (d_xy), **`fst`, and
   `relatedness`** matrices under `data/Output_Data/`. Pairwise stats are computed in **single
   batched `indexes=` traversals** (fast; verified bit-identical to per-pair loops).

**Key architectural fact:** SLiM runs with `initializeMutationRate(0)`. It is a pure neutral
tree-sequence recorder. **All mutations are overlaid afterwards** by `msprime.sim_mutations()`,
*after* `simplify()` (`AnalyzeTreeSeq.py:118`).

Consequences:
- Mutation rate has **zero** effect on SLiM's memory or runtime.
- Site-level π is **exactly linear in μ**: `π = μ × (branch-mode diversity)`. Branch-mode
  diversity (`ts.diversity(mode="branch")`) is `2·E[T_pair]` in generations and is the
  mutation-free view of what the demography is doing. Use it when diagnosing.

---

## 3. Environment [VERIFIED 2026-07-15]

**Current working env is `cpb-env` (from `environment3.yml`): SLiM 5.1, pyslim 1.1.1,
tskit 1.0.2, msprime 1.4.1, python 3.12.** This is an internally consistent SLiM-5 stack and the
pipeline **runs end to end on it.** Run it via `conda run -n cpb-env python ...` (from
`Python_Code/`, since paths are relative to that dir).

> **Stale-warning correction.** An earlier version of this doc said "SLiM 4.3 tree sequences need
> pyslim 1.0.x; pyslim 1.1+ fails with `...top-level metadata`." That described the *old* SLiM 4.3
> env. The env has since been upgraded to **SLiM 5.1**, for which pyslim 1.1.1 is the correct
> match. Do **not** downgrade to pyslim 1.0.x on this env.

Per-run cost [VERIFIED 2026-07-15, POPMULT=2000, fsync-timed]: SLiM forward sim ~3 s; then
`analyze_tree_sequence` ~250 s, of which **recapitation is ~246 s (~98%)** — the batched π/d_xy/
F_st/relatedness stats are ~1 s total. Recapitation (coalescing ~42k lineages at r=2.75e-6 over
1e6 bp) is the throughput bottleneck, not the statistics. Fine for a parallel CHTC pass; only
slow for local iteration. Simplify-before-recapitate was considered and **declined** (unsafe
without `keep_input_roots=True`; benefit is local-only) — see `ABC_REFACTOR_CHANGELOG.md`.

Total simulated N is **not** POPMULT. Subpop size is
`Average Count × POPMULT / numSubpops`, so `total N ≈ POPMULT × mean(Average Count) ≈ 3.33 × POPMULT`.
Older OOM note (POPMULT≈40000 crashed a 128 GB machine) still stands as a caution; `simplificationRatio=INF`
(§5.5) remains unfixed, so keep POPMULT modest until it is.

---

## 4. Data layout

```
data/empiricalStats/averaged_{pi,dxy,fst,genRel}_{2015,2019,2023}.csv   # observed
data/Output_Data/{diversities,divergences,fst,relatedness}_{year}.csv   # simulated (fst,relatedness added 2026-07-15)
data/cluster_data.csv          # Cluster ID, Latitude, Longitude, Average Count, Genome Assignment {year}
data/cluster_distances.csv     # 66x66 pairwise geographic distance (metres), + header row & ID col
data/Genetic_Data/specifier_matrix_{year}.csv  # headerless; row = subpop index; col1=lat, col2=lon (REAL-site coords)
data/final_data_for_modeling.csv
diagnostics/qdriver.py         # controlled rescaling/sweep harness (see §6)
diagnostics/qpost.py           # post-process an existing .trees -> branch div, site pi, Fst
```

**Subpop ordering / IBD coordinates [VERIFIED 2026-07-15].** For year Y, subpop index `i`
corresponds to **row `i` of `specifier_matrix_Y.csv`**, whose columns 1/2 hold the real-site
lat/lon (see `GenerateClusterData.assign_genomes_to_clusters_idv_year`). Rows == the F_st/π/
relatedness matrix dimension (24/17/20). This is the ordering used for element-wise comparison
*within a year* (meaningful) and the real-site coordinate source for the IBD slope on **both**
observed and simulated sides. (Across years the matrices still cannot be aligned — different subpop
sets.)

Subpop counts per year: **2015 → 24, 2019 → 17, 2023 → 20.** All four empirical matrices and
`Genome Assignment {year}` agree on these. [VERIFIED]

**Geographic coordinates exist.** `cluster_data.csv` maps each year's subpop index (the
`Genome Assignment {year}` column, 1..k) to a Cluster ID with lat/lon, and pairwise distances
are in `cluster_distances.csv`. An older version of this file listed coordinates as an open
TODO; it is resolved. **The isolation-by-distance slope is buildable today.** [VERIFIED]

Notes that have already caused bugs:
- Subpop sizes range 2–19 diploid individuals. Small ones are very noisy.
- A **typo-duplicated `Arlington` label** produces two near-identical subpopulations. Decide
  deliberately whether to merge.
- Subpop sets differ between years → matrices have different dimensions and **cannot be
  aligned element-wise across years.**

---

## 5. Known defects — read before changing the ABC

These were all found by reading and running the code. They are ordered by how much they
distort the inference.

### 5.1 Simulated π is pinned by ancestral Ne, not by population size [VERIFIED]

> **[DONE 2026-07-15] `ancestral_Ne` un-hardcoded but deliberately kept fixed at 6700.** It is
> now a threaded parameter (`analyze_tree_sequence(..., ancestral_Ne=6700)` ← `Main.main`), **not**
> an ABC free parameter — it is a well-established empirical point estimate, and it is confounded
> with μ in π (they enter only as `4·Ne·μ`), so inferring it would just build a second ridge.
> Exposed for **sensitivity analysis only** (sweep it manually; μ fixed). Scaling it with
> `population_modifier` was considered and **rejected** (fabricates N-identifiability, conflates
> two demographic epochs). See `ABC_REFACTOR_PLAN.md` §2a and the discussion in the changelog.

Original analysis (still the reason it's fixed): `pyslim.recapitate(ts, ..., ancestral_Ne=6700)`.

The forward SLiM phase runs only 324 generations, while coalescence takes thousands. So most
lineages do not coalesce in the forward phase — they coalesce in the *recapitated* ancestral
phase, at the fixed Ne. The arithmetic:

```
predicted from recapitation alone:  4 · Ne_anc · μ = 4 · 6700 · 2.1e-9 = 5.63e-5
actually produced by the pipeline (POPMULT=10000):              π_mean = 6.38e-5
```

The ~13% excess is the forward phase's contribution. **Roughly 7/8 of simulated diversity is
set by a constant that `population_modifier` does not touch.**

Implication: population size is close to unidentifiable from π as the pipeline currently
stands, and "inflating μ to compensate for a small N" is not actually compensating — μ is
simply the *only* knob that moves π, which is exactly why inflating it appears to work.

**[OPEN]** A POPMULT sweep (2500 / 10000 / 40000, everything else held fixed, KMeans seed
pinned) was set up to confirm this by measuring branch-mode diversity across a 16× range in N.
It did not finish — the 40000 arm OOM'd and took the machine down. **This is the single most
valuable thing to finish.** Use `diagnostics/qdriver.py`; keep POPMULT ≤ ~20000 per arm, or
fix 5.5 first so larger N is affordable.

### 5.2 Migration was saturated — the population was near-panmictic [VERIFIED]

> **[DONE 2026-07-15] Migration decoupled.** `determine_migration_rates(distances,
> total_migration, scale, ...)` now separates the **total immigration fraction**
> (`total_migration`, a real bounded per-generation rate, new ABC prior `U(0.001, 0.301)`) from
> the **dispersal-kernel decay** (`scale`, the old `migration_rates_modifier` — it only reshapes
> *where* migrants come from). Kernel is over sources only (self excluded), normalized to 1, then
> scaled by `total_migration`, so each row's off-diagonals sum to exactly `total_migration`; SLiM
> fills the retention fraction as `1 − total_migration`. SLiM script unchanged. Verified: rows sum
> to `total_migration`. This is what makes migration identifiable and F_st/IBD move.

Original defect (the reason for the fix): the old code computed `exp(-d·modifier)` and then
**normalized each row to sum to 1.** SLiM applies the off-diagonals as immigration
probabilities (`CPBSampleSim{Win,Linux}.slim:34-39`).

Measured from the actual `migration_rates.csv` and confirmed with a SLiM probe:

```
mean total immigration per subpop per generation = 0.758
```

About **76% of each subpopulation's offspring are immigrants every generation.** That is
effectively panmixia — very little population structure can build, which is why simulated
F_st is tiny and d_xy ≈ π.

`migration_rates_modifier` is **not a migration rate.** It is a distance-decay scale (units of
1/distance) inside `exp(-d·m)`; larger m means *less* migration, and the relationship to total
immigration is strongly nonlinear because of the row normalization.

### 5.3 Coalescent rescaling cannot be applied to this model as written [VERIFIED]

Standard SLiM rescaling (N → N/Q, μ → μQ, r → rQ, m → mQ, generations → gens/Q) holds
θ = 4Nμ, ρ = 4Nr and 4Nm invariant. **It cannot be used here**, for two independent reasons:

1. **Migration has no headroom.** 4Nm invariance needs m → m·Q, but m is already ≈ 0.76.
   Even Q = 2 would require m > 1, which is not a probability. (5.2)
2. **The dominant term doesn't scale.** Most coalescence happens in recapitation at a fixed
   `ancestral_Ne=6700`. Unless that is also divided by Q, the largest contribution to π is
   simply unaffected by the rescaling. (5.1)

So the answer to "does rescaling hold?" is **no, not as the model currently stands** — and
crucially, the reason is not a subtlety of the coalescent, it is that two parameters are
hardcoded/saturated. Fix 5.1 and 5.2 and the question becomes worth re-asking.

### 5.4 `recombination_rate` never reaches the forward simulation [VERIFIED]

`CPBSampleSimLinux.slim:12` hardcodes `initializeRecombinationRate(1e-8)`. The
`recombination_rate` ABC parameter (prior centred on 2.75e-6) is passed **only** to
`pyslim.recapitate()`. It therefore cannot affect the forward dynamics.

Separately: recombination rate has **no signal at all** in π or d_xy — it only shows up in
linkage disequilibrium. Inferring it requires an LD summary statistic (`ld_decay.py` /
`ToUseOnBeagles/CalculateLD,py` already exist on the empirical side).

### 5.5 `simplificationRatio=INF` is the memory bug [VERIFIED, fix untested]

`CPBSampleSimLinux.slim:4` sets `initializeTreeSeq(simplificationRatio=INF)`, which tells SLiM
to **never simplify the tree sequence during the forward run.** The edge table grows unbounded
for all 324 generations. Combined with `treeSeqRememberIndividuals(..., T)` retaining every
individual at gens 308 and 316, this is almost certainly the real cause of the OOM — not the
mutation rate.

**[OPEN]** Removing `simplificationRatio=INF` (letting SLiM auto-simplify) has not been tested,
but is the obvious first thing to try. If it works, the entire premise of "must inflate μ to
keep the sim tractable" dissolves.

### 5.6 KMeans geography was re-randomized on every run [VERIFIED]

> **[DONE 2026-07-15] Seed pinned.** `Main.py` now defines `KMEANS_SEED = 42` and passes it as
> `cluster_coordinates(..., random_state=KMEANS_SEED)`. Cluster identity is stable across ABC
> iterations. (Files in `data/` are still overwritten in place each run — §9 caution stands.)

Original: `cluster_coordinates(..., random_state=random.randint(0, 1000))` produced a different
cluster layout every run, adding noise and making the subpop→lat/lon mapping non-reproducible.

### 5.7 `csv.DictReader` silently ate the first subpopulation [VERIFIED]

> **[DONE 2026-07-15] Fixed.** Both the simulated and observed π readers now use
> `_read_vector`/`_read_matrix` (plain `csv.reader`), which read **every** row, so subpop 0 is
> included on both sides. (The remaining `csv.DictReader` in `read_parameters_from_csv` is correct
> — the input-parameter CSV genuinely has a header.)

Original: the π files have no header row, so `csv.DictReader` consumed the first value as a column
name and silently dropped subpop 0 from the diversity distance in every year.

### 5.8 Distance function — replaced [DONE 2026-07-15]

The old `calculate_losses` (raw L1 `Σ|Δπ|·k + Σ_{i≠k}|Δd_xy|`, no standardization, no weights,
relatedness unused) has been **replaced**. `calculate_losses` now returns per-statistic,
un-standardized, count-normalized (mean over entries within a year, then mean over years)
distances:
- **fitted:** `pi_loss` (log-space), `fst_loss` (off-diagonal), `ibd_loss` (|IBD-slope diff|);
- **diagnostic (not fitted):** `dxy_loss`, `genrel_loss`.

There is deliberately **no `total_loss`** during the pass — the combined standardized distance
`D = sqrt(Σ_j w_j (loss_j/σ_j)²)` with `σ_j = 1.4826·MAD` is built **offline** by the new
`Python_Code/abc_standardize.py` (the run set is its own pilot batch; equal weights to start).
The `detailed_sim_results/run{n}/` copy now stores raw feature files (incl. F_st, relatedness) so
offline σ has raw values, not just losses. See §6 and `ABC_REFACTOR_PLAN.md` §4.

---

## 6. The ABC distance — design

> **[DONE 2026-07-15] What was actually built** (differs from the original recommendation below):
> - **π kept element-wise, not reduced to mean/SD.** The specifier-matrix mapping makes
>   simulated-subpop ↔ real-site a genuine within-year correspondence (§4), so element-wise is
>   meaningful; comparison is always within a year (dimensions match). π is compared in **log
>   space**.
> - **Fitted:** element-wise log-π, off-diagonal F_st, IBD slope. **Diagnostic (computed, not
>   fitted):** d_xy and genetic relatedness (relatedness kept as a posterior-predictive check).
> - **Standardization is offline** (`abc_standardize.py`), not a separate pilot batch — the pass
>   is its own pilot. Method is **rejection ABC** (no SMC yet; chosen for CHTC). Tajima's D / SFS
>   and LD were deferred (§5.4, `ABC_REFACTOR_PLAN.md` §5).
> The rationale below still applies (confounding, F_st-over-d_xy, IBD, MAD, noise floor).

In ABC there is no gradient-descent "loss." What is being designed is the **distance metric**
`d(S_sim, S_obs)` used for rejection/weighting. It only has to *rank* parameter draws sensibly.

### Do this first
Fix §5.1 / §5.2 / §5.5, and settle which parameters are actually identifiable. **No distance
metric can rescue a parameter the simulation doesn't respond to.** In particular:
- π depends on N and μ only through **θ = 4Nμ** — they are confounded. If both stay free, the
  posterior is a *ridge*, not a peak, and a peaked-looking marginal for N would be an artefact.
- r has zero signal without an LD statistic.

### Recommended feature vector (dimension-independent)
Years have different subpop sets, so raw matrix entries cannot be aligned across years. Reduce
each year to a fixed handful of scalars instead:

| | feature | targets |
|---|---|---|
| s₁ | mean π across subpops | θ = 4Nμ |
| s₂ | SD of π across subpops | N, cluster structure |
| s₃ | mean off-diagonal **F_st** (upper triangle) | migration, N |
| s₄ | SD of off-diagonal F_st | spatial heterogeneity |
| s₅ | **isolation-by-distance slope** | migration, directly |
| s₆ | *(phase 2)* LD decay half-distance | recombination, N |

15 features (5 × 3 years) instead of ~1000 raw matrix entries.

- **Use F_st, not raw d_xy.** d_xy ≈ π + differentiation, so its entries are nearly redundant
  with π and its differences are dominated by the diversity-level mismatch. `averaged_fst_*.csv`
  already exists and is currently unused; the sim side needs `ts.Fst`.
- **IBD slope:** regress `F_st/(1 − F_st)` on **ln**(geographic distance) over all subpop pairs
  (Rousset); take the OLS slope. Linear under 2-D isolation by distance, one scalar regardless
  of k. Use `cluster_data.csv` for coordinates. **Not yet built.**
- **Drop relatedness from the distance** — it is linearly dependent by construction (centring
  makes rows sum to ~0) and measures the same signal as F_st. Keep it as a *posterior-predictive
  check*: a statistic you didn't fit is far better validation than one you did.

### The distance

```
D(θ) = sqrt[ (1/3) · Σ_years Σ_j  w_j · ( (s_sim,j,y − s_obs,j,y) / σ_j )² ]
```

with `Σ_j w_j = 1`, and **σ_j = 1.4826 × MAD** of feature j across a pilot batch of ~200–500
prior-predictive simulations (MAD not SD — the 2-individual subpops throw outliers). Compute σ
once, save it, keep it fixed for the run.

Start with equal weights (w_j = 1/5); after standardization and feature reduction that is a
defensible baseline. Then let `pyabc`'s `AdaptivePNormDistance` re-learn w_j from running MAD
each SMC generation.

Average the feature vector over R ≈ 3–5 replicate simulations per draw. **Measure the noise
floor first:** simulate twice at the *same* parameters and compute D between the two runs. If
your acceptance threshold ε is below that floor, you are selecting on coalescent noise.

### Method
`pyabc` + SMC + `AdaptivePNormDistance` is the smallest step from where the code is now.
Keep `sbi` (SNPE) in view — with a 15-dim feature vector it is amortized, far more
simulation-efficient, and removes the weight-tuning question entirely. If you find yourself
spending real time on w_j, switch.

---

## 7. Invariants — violate these and the sim/empirical comparison is meaningless

1. **Match the sampling design.** Subsample the tree sequence to the same subpops and
   per-subpop counts as the year in question. Pull them from the popfile, then
   `ts.simplify(samples=nodes)` — no re-simulation needed.
2. **Never slice a big matrix to get a small one.** `genetic_relatedness` is **centred on the
   populations present in the call**. A 3-pop matrix computed from scratch ≠ 3 rows/cols sliced
   from a 20-pop matrix — values differ and can flip sign. **Always recompute on the subset.**
   Biggest footgun in the project.
3. **Centring is per-year.** Absolute relatedness values are not comparable across years without
   re-centring on a common reference.
4. **Genome-wide = pool, don't average.** These statistics are ratios (centred cross-products ÷
   segregating sites). Genome-wide = `Σ numerators / Σ segregating sites` across chromosomes. A
   plain mean of 17 per-chromosome matrices over-weights sparse chromosomes and won't match tskit.
5. **Compare against a distribution, not a single run.** Coalescent and subsampling are both
   stochastic. Run replicates; build an envelope (e.g. 2.5–97.5%).
6. **Biallelic SNPs only.** tskit treats missing genotypes as ancestral; the empirical code
   computes frequencies over *called* genotypes only — a known, small source of divergence.
7. `ts.write_vcf` **rounds mutation positions to integers** — dense mutations on a short sequence
   can produce duplicate positions. Harmless for distance-binned LD; know about it.

---

## 8. Existing code

| File | What it does |
|---|---|
| `Python_Code/Main.py` | Pipeline entrypoint. See §2. Threads `total_migration`, `ancestral_Ne`; `KMEANS_SEED` pinned. |
| `Python_Code/AnalyzeTreeSeq.py` | Recapitate → simplify → mutate → π, d_xy, **F_st, relatedness** (batched `indexes=`). `ancestral_Ne` now a parameter (§5.1). |
| `Python_Code/ABCAnalysisNoRedis.py` | ABC driver + rewritten `calculate_losses` (per-stat, log-π, F_st, IBD; §5.8) + IBD helpers (`get_site_geo_distances`, `ibd_slope`). |
| `Python_Code/abc_standardize.py` | **Offline** post-processing: σ=1.4826·MAD per fitted stat → standardized distance `D` → ranked CSV + frozen σ JSON. Not run in the pass loop (§5.8, `ABC_REFACTOR_PLAN.md` §4). |
| `Python_Code/GenerateSimulationParams.py` | `determine_migration_rates(distances, total_migration, scale, ...)` — decoupled (§5.2). |
| `Python_Code/GenerateClusterData.py` | KMeans clustering, distance matrix, genome→cluster assignment (`assign_genomes_to_clusters_idv_year` sets subpop→specifier-row mapping — §4). |
| `SLiM_Code/CPBSampleSim{Linux,Win}.slim` | Forward sim. Neutral, `mutationRate(0)`, `simplificationRatio=INF` (§5.5, still unfixed). |
| `diagnostics/qdriver.py` | Controlled sweep/rescaling harness: fixed KMeans seed, templated SLiM, scalable migration, reports branch-mode diversity. |
| `diagnostics/qpost.py` | Post-process an existing `.trees` → branch diversity, site π, F_st. Fast; no SLiM needed. |
| `ToUseOnBeagles/CalcGenRel.py` | Per-year empirical genetic-relatedness matrices from VCFs (reproduces tskit `genetic_relatedness` defaults; the sim side in AnalyzeTreeSeq matches it to ~1e-18). |
| `ld_decay.py` / `ToUseOnBeagles/CalculateLD,py` | Per-year, per-subpop LD decay: mean r² binned by physical distance (empirical LD; not yet in the distance). |
| (pixy) | Used for empirical π and d_xy. Handles the missing-vs-invariant-site denominator correctly. No relatedness statistic — hence the hand-rolled one. |

Both hand-rolled estimators were **verified bit-identical to tskit** (`genetic_relatedness`,
`ld_matrix(stat="r2")`, ~1e-16). **Keep it that way** — re-run that verification if you touch them.

---

## 9. Conventions

- Python; `numpy`, `scikit-allel`, `tskit`, `msprime`, `pyslim`.
- Scripts carry a `CONFIG` block at the top rather than argparse (`diagnostics/*` are the
  exception — they take flags, since they're meant to be swept).
- When implementing any new statistic, **verify it against tskit on a small msprime simulation
  first**, then apply it to real data. That pattern has caught real bugs twice.
- The pipeline **overwrites files in `data/` in place** (§2, §5.6). Any experiment should run on
  a copy of the tree, not the repo, or redirect those paths.

---

## 10. Immediate next steps

Done in the 2026-07-15 session (see `ABC_REFACTOR_CHANGELOG.md`): migration re-parameterized (§5.2),
`ancestral_Ne` threaded + kept fixed with justification (§5.1/§5.3), KMeans seed pinned (§5.6),
DictReader fixed (§5.7), feature vector + distance built (F_st + IBD + log-π; §5.8/§6), offline
standardizer written, pairwise stats batched. Env confirmed working (§3).

Remaining:
1. **One full integration run** through `run_sims_from_csv` (all 3 years → `abc_results.csv` with
   the new columns). Write to a **fresh** CSV — the legacy `Python_Code/abc_results.csv` has the old
   column layout. (~4 min; pieces already validated separately.)
2. **Run the big rejection pass** (CHTC), then `abc_standardize.py`, then inspect posteriors —
   report **θ=4Nμ** and **Nm**, and check whether the N marginal escapes the prior (§6, plan §4).
   **Measure the noise floor first** (replicate at identical params).
3. **Set the `total_migration` prior bounds** deliberately (currently placeholder `U(0.001,0.301)`)
   — a biological CPB immigration ceiling.
4. **[OPEN] POPMULT sweep** (§5.1) and **[OPEN] remove `simplificationRatio=INF`** (§5.5) — still
   the way to make N affordable/explorable at large values; not required for the first pass.
5. **Phase 2:** LD statistic for recombination (§5.4), Tajima's D as a posterior-predictive check
   (needs the genome-side computation), possibly SMC / `sbi`.
