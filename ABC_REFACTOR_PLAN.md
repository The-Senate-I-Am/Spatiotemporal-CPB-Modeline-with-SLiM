# ABC Refactor Plan — feature vector + loss redesign

Status: **proposed, nothing in this document is implemented yet** (except the migration
decoupling in §0, which is already done). This is the agreed design from discussion, written
down for review before any further code changes.

Goal for this pass: make the ABC distance measure *structure* (not just diversity level), so
that migration and N stop being masked by μ, and run **one big rejection-ABC pass** on the
prior. No SMC, no adaptive weights yet.

---

## 0. Already done (migration decoupling — for reference)

- `GenerateSimulationParams.determine_migration_rates(distances, total_migration, scale, ...)`
  now separates **total immigration fraction** (`total_migration`) from the **dispersal-kernel
  decay** (`scale`). Off-diagonals sum to exactly `total_migration`; SLiM fills the retention
  fraction as `1 − total_migration`. SLiM script unchanged.
- `Main.main(...)` takes `total_migration=0.05`, passes it through.
- `ABCAnalysisNoRedis.py` prior has `total_migration ~ U(0.001, 0.301)`, wired into `model()`.

Verified: each row's off-diagonals sum to `total_migration`, kernel still weights nearer
sources more.

---

## 1. Guiding decisions (the "why", settled in discussion)

1. **Fit these three features:** per-site **π** (log-transformed), pairwise **F_st**, and the
   **IBD slope**. These respond to the demography we're inferring.
2. **Compute but do NOT fit:** **d_xy** (μ-scaled → reintroduces the μ-degeneracy if fitted) and
   **genetic relatedness** (linearly dependent, redundant with F_st, per-year centred — §7 #2).
   Keep both as reported diagnostics / posterior-predictive checks.
3. **π is log-transformed; F_st and IBD are not** (F_st bounded near 0, IBD slope can be
   negative → log undefined). Log-π gives relative-error semantics and linearises the θ=4Nμ
   ridge.
4. **Keep π element-wise, not mean/SD.** The specifier-matrix mapping makes simulated-subpop ↔
   empirical-site a real correspondence, so element-wise is meaningful; within-year comparison
   means dimensions always match.
5. **Normalize out year entry counts:** aggregate each feature within a year by the **mean**
   over its entries (not the sum), so 24- vs 17- vs 20-subpop years contribute comparably. Then
   average across years (equal per-year weight).
6. **No total_loss during runs.** Store per-statistic losses; derive σ (=1.4826×MAD) and the
   combined standardized distance **offline, after the pass**, using the run set itself as the
   pilot batch (no separate pilot). This commits us to **rejection/offline ABC**, which is what
   we want for CHTC right now.
7. **N stays a free parameter but is reported via identifiable combinations** (θ=4Nμ, Nm), not
   as a standalone peak. `ancestral_Ne` is un-hardcoded (made explicit) but **fixed at 6700**,
   not randomized (point estimate only, no defensible prior width). μ remains free but is a
   nuisance diversity-scaler, not interpreted biologically.

---

## 2. Changes by file

### 2a. `Python_Code/AnalyzeTreeSeq.py`

**(i) Thread `ancestral_Ne` (un-hardcode, keep default 6700).**
- `analyze_tree_sequence(mutation_rate, recombination_rate)` →
  `analyze_tree_sequence(mutation_rate, recombination_rate, ancestral_Ne=6700)`.
- Line 101 `pyslim.recapitate(..., ancestral_Ne=6700)` → use the parameter.
- Add a comment: fixed empirical point estimate; swept only for sensitivity analysis, **not**
  inferred (confounded with μ via 4·Ne·μ). See CLAUDE.md §5.1.

**(ii) Add F_st output alongside π and d_xy.**
- In `calculate_diversity_and_divergence`, add a pairwise **F_st** matrix via `ts.Fst([pop_i,
  pop_j])` for each pair (same loop that already builds d_xy), write `fst_{year}.csv`.
- F_st is computed pairwise from scratch per pair → no centring footgun (that concern is
  relatedness-specific, §7 #2). Keep d_xy output as-is (now a diagnostic).
- New signature: add `output_fst_path`. Update the three call sites to pass
  `../data/Output_Data/fst_{year}.csv`.

**(iii) Add genetic-relatedness output (for the diagnostic / PP-check column).**
- Add a pairwise relatedness matrix via `ts.genetic_relatedness(...)` **recomputed on each
  year's subset** (never sliced from a bigger matrix — §7 #2), write `relatedness_{year}.csv`.
- If this proves fiddly, it can be deferred — it only feeds a diagnostic column, not the fitted
  loss. Flag, don't block.

**Invariant to respect:** F_st and relatedness must be computed on the same subsampled node set
already used for π/d_xy (the `pop_samples` for that year/time). Good — reuse `pop_samples`.

### 2b. `Python_Code/Main.py`

- Thread `ancestral_Ne` through: add `ancestral_Ne=6700` to `main(...)`, pass to
  `AnalyzeTreeSeq.analyze_tree_sequence(...)` (currently called with only mutation_rate,
  recombination_rate).
- **Pin the KMeans seed** (§5.6) so cluster identity — and therefore the subpop→coordinate
  mapping — is reproducible across the pass. Currently `random_state=random.randint(0,1000)`;
  replace with a fixed constant. **Confirmed for this pass.**

### 2c. `Python_Code/ABCAnalysisNoRedis.py`

**(i) Reading simulated + observed features — fix the DictReader bug (§5.7).**
- `model()` reads `diversities_{year}.csv` and `getObservedData()` reads `averaged_pi_{year}.csv`
  with `csv.DictReader`, but those files have **no header row** → the first subpop is silently
  eaten. Switch to `csv.reader` (or `np.genfromtxt`) so subpop 0 is included on both sides.
- Add reading of `fst_{year}.csv` (sim) and `averaged_fst_{year}.csv` (obs).
- Load **real-site geographic coordinates** for the IBD slope and build pairwise distances
  between real sampling sites, used for **both** the observed and simulated IBD regressions
  (resolved — §3.1). Source of the real-site coordinates (e.g. `final_data_for_modeling.csv`
  field coords aggregated per site, mapped through the specifier matrix / `Genome Assignment`)
  to be confirmed when wiring it up. Distances are raw metres (not centred) → no footgun.

**(ii) New feature computation.**
- Add an `ibd_slope(fst_matrix, geo_distances)` helper: regress `F_st/(1−F_st)` on
  `ln(distance)` over all off-diagonal pairs (OLS), return the slope. One scalar per year.
  **Verify against a hand computation / statsmodels on a small case first** (project convention).

**(iii) Rewrite `calculate_losses` → per-statistic, un-standardized, count-normalized, no total.**
Return a dict with (per run, already averaged over years with mean-over-entries):
- `pi_loss   = mean_years( mean_i | log π_sim,i − log π_obs,i | )`   ← **fitted**, log-space
- `fst_loss  = mean_years( mean_pairs | Fst_sim − Fst_obs | )`       ← **fitted**
- `ibd_loss  = mean_years( | slope_sim − slope_obs | )`              ← **fitted**
- `dxy_loss  = mean_years( mean_pairs | dxy_sim − dxy_obs | )`       ← diagnostic only
- `genrel_loss = mean_years( mean_pairs | R_sim − R_obs | )`         ← diagnostic only
- **No `total_loss`.**
These columns are for eyeballing/ranking-preview; the real standardized distance is built
offline (§4).

**(iv) Prior + parameter plumbing.**
- `sample_prior()` must include `total_migration` (currently missing).
- `read_parameters_from_csv`: add optional `total_migration` column; default to 0.05 if absent
  so the existing `sample_inputs.csv` (which has no such column) still runs.
- `model()` already reads `total_migration` — keep.

**(v) Output CSV — two artifacts.**
- **Main results CSV** (`abc_results.csv`): columns
  `iteration, m, total_migration, pop, numClusters, mutation_rate, recombination_rate,
   pi_loss, fst_loss, ibd_loss, dxy_loss, genrel_loss`.
  **No per-year columns, no total_loss.**
- **Detail artifact** (already exists as `detailed_sim_results/run{n}/`): extend the copy step to
  also copy `fst_{year}.csv` and `relatedness_{year}.csv`. This directory *is* the raw-feature
  store used for offline MAD/σ — it preserves raw sim feature values (not just absolute losses),
  which is exactly what post-hoc standardization needs.
- Store observed features once (they're constant) so the offline step has both sides.

### 2d. `Python_Code/GenerateSimulationParams.py`
- No further change (migration decoupling already done, §0).

### 2e. `SLiM_Code/CPBSampleSim{Linux,Win}.slim`
- **No change in this pass.** `simplificationRatio=INF` (memory bug, §5.5) is separable and only
  matters once we push N large; keep POPMULT modest for the first pass. Listed in §5 as
  explicitly deferred.

---

## 3. Design decisions

1. **IBD geographic-distance source — RESOLVED.** Use **real-site coordinates for both** the
   observed and simulated IBD slopes (each sim subpop represents a real site via the specifier
   matrix). Build one pairwise real-site distance matrix per year and use it on both sides.
   Exact coordinate source to confirm at wiring time (§2c-i). KMeans seed is pinned (§2b).

Remaining (non-blocking, decide offline):
2. **Per-stat σ vs per-stat-per-year σ.** Convenience columns collapse years, but the detail
   artifact keeps per-year raw features, so **either** is possible offline. Default: per-stat σ
   pooled across years (simpler). Revisit if a year is an outlier.
3. **Relatedness on the sim side** (§2a-iii): compute now as a diagnostic, or defer? Non-blocking
   — only feeds the `genrel_loss` diagnostic column.

---

## 4. Offline post-processing (after the pass — separate script, not in the run loop)

1. Read all runs' detail artifacts → assemble raw feature vectors per run (π, F_st, IBD; plus
   d_xy, relatedness for checks).
2. For each **fitted** feature, σ_j = 1.4826 × MAD of that feature across all runs (log-space for
   π). Freeze.
3. Standardized per-run distance:
   `D = sqrt( (1/3) [ mean_i(logπ resid/σ)² + mean_pairs(Fst resid/σ)² + (IBD resid/σ)² ] )`
   averaged over years (mean-over-entries within each block = count-normalized).
4. **Measure the noise floor:** simulate ≥2 replicates at identical params, compute D between
   them. Acceptance ε must sit above this. Average R≈3–5 replicates per draw if feasible.
5. Rank, take accepted set. Because everything was drawn from the prior, this rejection posterior
   is unbiased (up to ε). Report posteriors over **θ=4Nμ** and **Nm**, and check whether the N
   marginal escapes the prior. Use held-out relatedness / d_xy as posterior-predictive checks.

---

## 5. Explicitly NOT doing now

- **`simplificationRatio=INF` memory fix (§5.5)** — separable; only needed to push N large.
- **Adaptive ABC-SMC** — inconvenient on CHTC right now; manual top-X% iteration, if ever, needs
  importance weights to stay unbiased (single big rejection batch is unbiased with none).
- **Tajima's D / SFS statistics** — professor's idea; deferred (needs genome-side computation,
  and simulated SFS is ancestral-recapitation-dominated so likely weakly responsive here). Add
  later only if it passes the does-it-move-beyond-noise test; better as a PP-check for now.
- **Continuous-space SLiM model** — rejected for this use case.
- **Randomizing `ancestral_Ne`** — rejected (point estimate only; would fabricate uncertainty
  that dominates π).

---

## 6. Suggested implementation order

1. `AnalyzeTreeSeq.py`: thread `ancestral_Ne`; add F_st output (+ relatedness if easy).
2. `Main.py`: thread `ancestral_Ne`; pin KMeans seed.
3. `ABCAnalysisNoRedis.py`: fix DictReader; read F_st/dist; `ibd_slope` helper (verify vs
   reference); rewrite `calculate_losses`; prior/plumbing; new output columns + extended detail
   copy.
4. Smoke-test one run end to end; confirm all feature files written and columns populated.
5. Write the offline standardization/ranking script (§4).
6. Only then launch the big pass.
