# ABC Refactor — Changelog

Running log of the changes made while implementing `ABC_REFACTOR_PLAN.md`. Each entry: what
changed, why, and any verification. Newest section at the bottom.

Design reference: `ABC_REFACTOR_PLAN.md`. Defect references (e.g. §5.1) point at `CLAUDE.md`.

---

## Env note (read first) — CORRECTED

- The environment is **`cpb-env`** (defined by `environment3.yml`): **SLiM 5.1**, pyslim 1.1.1,
  tskit 1.0.2, msprime 1.4.1. This is an internally consistent SLiM-5 stack.
- CLAUDE.md §3's "needs pyslim 1.0.x, pyslim 1.1+ breaks recapitation" warning is **stale** — it
  described the old SLiM 4.3 env. On this upgraded SLiM-5 env, pyslim 1.1.1 is the correct match.
- **The end-to-end run is NOT blocked on the environment.** Verified: SLiM 5.1 runs
  `CPBSampleSimWin.slim` and writes `out/simTreeSeq.trees` in ~3 s; the pipeline then spends its
  time in `analyze_tree_sequence` (recapitation at r=2.75e-6 over 1e6 bp is expensive), consistent
  with CLAUDE.md's "~10 min end to end" note. Confirmed actively computing (process at ~300 CPU-s,
  responding) — slow, not hung, not OOM.
- **TODO (user's doc):** update CLAUDE.md §3 to reflect the SLiM-5 env.

---

## [done] Pre-existing (before this session): migration decoupling — plan §0

- `GenerateSimulationParams.determine_migration_rates` split into `total_migration` + `scale`.
- `Main.main` + ABC prior threaded with `total_migration`.
- (Documented here for completeness; see plan §0.)

---

## [done] Plan §2a — `Python_Code/AnalyzeTreeSeq.py`

### Changed
1. **`ancestral_Ne` un-hardcoded (default 6700).**
   - `analyze_tree_sequence(mutation_rate, recombination_rate)` →
     `analyze_tree_sequence(mutation_rate, recombination_rate, ancestral_Ne=6700)`.
   - `pyslim.recapitate(..., ancestral_Ne=6700)` (was a literal) now uses the parameter.
   - Docstring notes it's a fixed empirical point estimate exposed for **sensitivity analysis
     only**, not inferred (confounded with μ via π = 4·Ne·μ). Value unchanged → **no behavioural
     change** to existing runs.

2. **Added pairwise F_st output.**
   - `calculate_diversity_and_divergence(...)` now also computes a pairwise F_st matrix via
     `ts.Fst([pop_samples[i], pop_samples[j]])` (mirrors the existing d_xy loop; diagonal = 0),
     written to a new `output_fst_path`.
   - F_st computed independently per pair → no centring footgun (that's relatedness-only, §7 #2).

3. **Added pairwise genetic-relatedness output.**
   - Computes `ts.genetic_relatedness(pop_samples, indexes=all (i,j) pairs)` — **all of the
     year's sample sets passed in one call**, so centring is across that year's subpops (matches
     empirical per-year centring; never sliced from a bigger matrix — §7 #2/#3). Written to a new
     `output_relatedness_path`.

4. **Call sites updated (3×).** Each `calculate_diversity_and_divergence(...)` call now passes
   `../data/Output_Data/fst_{year}.csv` and `../data/Output_Data/relatedness_{year}.csv`.

### Not changed
- π and d_xy computation/outputs untouched (d_xy stays as a diagnostic).

### Verification
- `python -m py_compile AnalyzeTreeSeq.py` → OK.
- Small 3-deme msprime island-model check (`scratchpad/api_check.py`, run in `cpb-env`):
  - `ts.Fst` pairwise → symmetric, positive differentiation, diagonal 0. ✔
  - `ts.genetic_relatedness` all-pairs → symmetric, positive diagonal, **rows sum to ~0
    (centred)**. ✔
  - Simulated relatedness reproduces the empirical `CalcGenRel.py` estimator
    (centred cross-product ÷ segregating sites) to **max abs diff 1.7e-18**. ✔ (convention match)
- Not yet run end-to-end on a real SLiM tree (needs pyslim 1.0.x env — see Env note).

### Follow-ups / open
- The relatedness convention match was verified on tskit 1.0.2; the pinned pipeline env is
  tskit 0.6.4. CLAUDE.md §8 states the hand-rolled estimator was verified bit-identical to tskit
  `genetic_relatedness` (on the pinned env), so defaults should agree there too — worth a
  spot-check during the offline step.

---

## [done] Plan §2b — `Python_Code/Main.py`

### Changed
1. **Pinned the KMeans seed (§5.6).** Added module constant `KMEANS_SEED = 42`; the
   `cluster_coordinates(..., random_state=random.randint(0,1000))` call now uses `KMEANS_SEED`.
   Cluster identity is now stable across ABC iterations (the subpop→coordinate mapping the IBD
   slope depends on is reproducible). **Behavioural change:** clustering is now deterministic.
2. **Threaded `ancestral_Ne`.** `main(...)` gained `ancestral_Ne=6700`, passed through to
   `AnalyzeTreeSeq.analyze_tree_sequence(...)`. Default unchanged → no behavioural change.

### Notes
- `import random` is now unused in Main.py but left in place to minimise churn (harmless).

### Verification
- `python -m py_compile Main.py` → OK.

---

## [done] Plan §2c — `Python_Code/ABCAnalysisNoRedis.py`

### Changed
1. **Fixed the `csv.DictReader` bug (§5.7).** New `_read_vector` / `_read_matrix` helpers read
   **every** row, so subpop 0 is no longer silently dropped from π on either the simulated or
   observed side. (The remaining `csv.DictReader` at ~line 255 is the *input-parameter* reader,
   which correctly has a header — left as-is.)
2. **Added F_st + relatedness to feature loading.** `model()` and `getObservedData()` now load
   π (vector), d_xy, F_st, and relatedness (matrices) per year, from `fst_{year}.csv` /
   `relatedness_{year}.csv` (sim) and `averaged_fst_{year}.csv` / `averaged_genRel_{year}.csv` (obs).
3. **IBD slope implemented (real-site coordinates, plan 3.1).** `get_site_geo_distances(year)`
   builds a pairwise real-site distance matrix from `specifier_matrix_{year}.csv` cols 1/2
   (lat/lon) — **same subpop ordering as every stat matrix** (verified rows == Fst dims:
   24/17/20). `ibd_slope()` = OLS slope of `Fst/(1−Fst)` on `ln(distance)` over off-diagonal
   pairs, masking non-finite/`Fst≥1`/`dist≤0`. Same distances used for observed and simulated.
4. **Rewrote `calculate_losses`.** Returns per-statistic, un-standardized, **count-normalized**
   (mean over entries within a year, then mean over years) distances: `pi_loss` (log-space, fitted),
   `fst_loss` (fitted), `ibd_loss` (fitted), `dxy_loss` (diagnostic), `genrel_loss` (diagnostic).
   **No `total_loss`** — the standardized combined distance is built offline (plan §4).
5. **Prior + plumbing.** `sample_prior()` now includes `total_migration`; `read_parameters_from_csv`
   accepts an optional `total_migration` column (default 0.05, so legacy `sample_inputs.csv` runs).
6. **Output CSV columns.** Both run drivers now write
   `iteration, m, total_migration, pop, numClusters, mutation_rate, recombination_rate,
   pi_loss, fst_loss, ibd_loss, dxy_loss, genrel_loss` — no per-year columns, no total_loss.
   Per-run console line prints the five losses.
7. **Detail artifact extended.** The `detailed_sim_results/run{n}/` copy now includes
   `fst_{year}.csv` and `relatedness_{year}.csv` alongside diversities/divergences — this is the
   raw-feature store for offline standardization (plan 2c-v/4).

### Verification
- `python -m py_compile ABCAnalysisNoRedis.py` → OK; no leftover `total_loss`/`diversity_loss`/
  `divergence_loss` references.
- Functional test on **real empirical data** (`cpb-env`):
  - All matrices read at correct dims (24/17/20); π vector correct length. ✔
  - `calculate_losses(obs, obs)` → **all five losses exactly 0.0**. ✔
  - Real-data IBD slopes finite: 2015 −1.42e-3, 2019 +8.04e-4, 2023 +1.44e-4 (weak/mixed —
    expected for low-divergence pops). ✔
  - Perturbed sim (π×1.1, Fst+0.02): `pi_loss = 0.0953 = ln(1.1)` **exactly** (log-transform
    correct), `fst_loss = 0.02` (injected offset), `ibd_loss > 0`; dxy/genrel unchanged = 0. ✔
- **Not yet run end-to-end** on simulated output (needs a full SLiM run + pyslim 1.0.x env).

### Resolved design decision
- §3.2 σ granularity: default will be **per-statistic σ from the loss columns** in the offline
  step (simplest, uses `abc_results.csv` directly). The detail artifacts still enable a more
  principled per-entry / feature-spread σ later if wanted.

---

## [done] Plan §4/§6-step5 — new file `Python_Code/abc_standardize.py` (offline)

### Added
- Offline post-processing script (does **not** run simulations). Reads `../out/abc_results.csv`,
  computes robust `sigma_j = 1.4826*MAD` per fitted statistic (`pi_loss`, `fst_loss`, `ibd_loss`),
  combines into one standardized distance `D = sqrt( Σ_j w_j (loss_j/σ_j)^2 )` with equal weights,
  ranks runs, flags the top `ACCEPT_FRAC` (default 20%), and writes:
  - `../out/abc_results_ranked.csv` (results + `D` + `accepted`, sorted),
  - `../out/abc_sigmas.json` (frozen σ, weights, acceptance metadata).
- `dxy_loss` / `genrel_loss` carried through as diagnostics, deliberately **not** in `D`.
- CONFIG block at top (paths, fitted stats, weights, accept fraction) per repo convention (§9).
- Prints the plan §4 noise-floor reminder.

### Verification
- Ran on a synthetic 50-run results table (`cpb-env`): σ computed per stat, `D` strictly
  ascending after sort, exactly 10/50 flagged accepted, diagnostic columns preserved, σ/weights
  written to JSON. ✔
- Pending: validation against **real** pass output (format edge cases, NaN handling at scale).

---

## [done] Performance — batched pairwise stats in `AnalyzeTreeSeq.py`

### Changed
- Replaced the O(k²) per-pair loops for **divergence** and **Fst** with single `indexes=`
  traversals (`ts.divergence(pop_samples, indexes=pairs)`, `ts.Fst(...)`); diversity computed for
  all sets in one `ts.diversity(pop_samples)` call. This removes ~k² separate tree traversals per
  statistic per year (24² ≈ 576 → 1).
- Motivation: for a many-run ABC pass, per-run analysis time is the budget. The original code
  already looped divergence; the F_st added in §2a would have doubled that cost — batching removes
  it and speeds up the existing divergence too.

### Verification
- Batched vs looped on an 8-deme msprime sim: **bit-identical** divergence and Fst
  (`np.allclose` True), **24× faster** at k=8 (speedup grows with k²). ✔
- `python -m py_compile AnalyzeTreeSeq.py` → OK.
- End-to-end phase breakdown (POPMULT=2000, fsync-logged, `cpb-env`):
  - load 0s → **recapitate 245.8s** → simplify +2.3s → mutate +0.4s (717 sites)
    → **batched 2023 stats (pi+dxy+Fst+genrel) +1.1s** → DONE ~250s total.
  - **Recapitation is ~98% of analysis time; the batched stats are ~1s.** Confirms the stats
    optimization works and that the throughput bottleneck is recapitation, not the statistics.
  - Batched output validated on the real tree: `fst (20,20)` symmetric, diagonal 0; relatedness
    rows sum ~0 (centred). ✔
- **Optimization considered and DECLINED:** simplify-before-recapitate. Naive reordering is
  unsafe — default `simplify()` drops the first-generation root lineages `recapitate()` needs,
  silently biasing diversity downward; it would require `simplify(keep_input_roots=True)` plus an
  equivalence check. The benefit is local-only (CHTC parallelizes the pass, so ~4 min/run is a
  non-issue), so the recapitate → simplify order is **left unchanged** by user decision. The
  batched-stats win above is kept (pure, correctness-preserving). Recombination rate / sequence
  length remains a modeling-level lever, untouched.

---

## [done] Prior distributions — `ABCAnalysisNoRedis.py`

Per user decisions (2026-07-15), after pulling up the current priors:

- **`mutation_rate` recentered to ~5e-6** (`DEFAULT_MUTATION_RATE = 5e-6`; prior
  `lognorm(s=0.5, scale=5e-6)`, 2.5–97.5% ≈ [1.9e-6, 1.3e-5]). This is the value that matches
  empirical π ≈ 0.14 given fixed `ancestral_Ne=6700` (π = 4·Ne·μ). Documented as a **nuisance
  diversity-scaler, not the biological rate** (~2.1e-9). `Main.main` default μ also updated to
  5e-6 so direct calls don't silently produce π ~2500× too low.
- **`recombination_rate` fixed, not inferred.** Removed from `prior_distributions` and
  `sample_prior`; `read_parameters_from_csv` now treats the column as optional (defaults to
  `DEFAULT_RECOMBINATION_RATE = 2.75e-6`). Rationale: no signal in π/d_xy/F_st, only in LD
  (§5.4), so it was pure noise/dimensionality in the pass.
- **`pop` (POPMULT) capped.** Prior changed from unbounded `expon(loc=2000, scale=50000)` (which
  put most mass in OOM territory) to `uniform(2000, 12000)`. `POPMULT_MAX = 12000` ≈ 40k total
  individuals (total N ≈ 3.33·POPMULT) — CHTC-feasible.

Verified: prior keys now `{m, total_migration, pop, numClusters, mutation_rate}`; percentiles as
above; `sample_prior()` no longer emits `recombination_rate`; compiles.

**Still a placeholder:** `total_migration ~ U(0.001, 0.301)` upper bound (biological CPB
immigration ceiling — user judgment). `m` kernel-decay prior left as-is.

---

## [done] Full integration run — verified end-to-end

Ran one complete `run_sims_from_csv` iteration through the real driver (POPMULT=2000, μ=5e-6,
`cpb-env`), writing to a fresh `out/abc_results_integration_test.csv`. Confirms the whole chain:
SLiM → recapitate → simplify → mutate → all four stats → `model()` read → `calculate_losses` → CSV.
Output row:
```
iteration,m,total_migration,pop,numClusters,mutation_rate,recombination_rate,pi_loss,fst_loss,ibd_loss,dxy_loss,genrel_loss
0,0.0001,0.05,2000,1,5e-06,2.75e-06,0.2127,0.01676,0.00901,0.02655,0.000497
```
Validated:
- New columns present; **no `total_loss`, no per-year columns**. ✔
- **`recombination_rate` optional path works** — omitted from the input CSV, output shows the
  fixed default 2.75e-06. ✔
- All five losses finite and non-zero (pi/fst/ibd fitted; dxy/genrel diagnostic). ✔
- **μ=5e-6 makes `pi_loss` informative:** 0.21 (≈ e^0.21 ⇒ ~1.24× off in π), vs the ~7.8 it would
  be at μ≈2e-9 (fully saturated). Confirms the recentering brings π into range. ✔

## [done] CHTC per-job runner — `ABCAnalysisNoRedis.py` `__main__`

- **New entrypoint:** `python ABCAnalysisNoRedis.py <job_id> [num_trials]` (default 100 trials).
  Each job samples `num_trials` draws from the prior, runs the full pipeline per draw, and writes
  one row per trial to `../out/abc_results_job<job_id>.csv`. Combine per-job CSVs → `abc_standardize.py`.
- **Per-job reproducibility:** `np.random.seed(job_id)` seeds numpy's global RNG (which scipy
  `.rvs()` uses), so each job's draws are reproducible and distinct across job_ids. KMeans stays
  deterministic (fixed `Main.KMEANS_SEED`, independent of the global RNG).
- **Bug fixed:** `run_abc_simulation` referenced `parameters["recombination_rate"]` directly, but
  `sample_prior()` no longer emits that key (it's fixed) → would have `KeyError`'d on every
  prior-sampled trial. Both references (print + row) now use
  `.get("recombination_rate", DEFAULT_RECOMBINATION_RATE)`. (The CSV path was unaffected.)

### Verification (stubbed model, `cpb-env`)
- `sample_prior()` reproducible under a fixed seed; keys `{m, total_migration, pop, numClusters,
  mutation_rate}`. ✔
- `run_abc_simulation(3)` writes 3 rows, no `KeyError`; `recombination_rate` fixed 2.75e-6, `pop`
  in [2000, 12000], μ ~5e-6, all new loss columns present. ✔
- **Note:** the CSV records the raw `numClusters` draw {1,2,3}; the actual cluster count is ×33
  ({33,66,99}) (unchanged behavior — model multiplies internally). Recover with ×33 in analysis.
- **Runtime:** ~4 min/trial at POPMULT≈2000 (recapitation-dominated; longer at larger POPMULT),
  so 100 trials ≈ 7+ hours/job. Size jobs accordingly (fewer trials × more jobs is also fine).

## [done] CHTC submit-file alignment — `ABCAnalysisNoRedis.py`

Reviewing the user's `cpb-build.sub` / `run_code.sh` surfaced two code↔submit mismatches, now fixed:
- **Output filename:** `__main__` now writes the fixed `../out/abc_results.csv` (was
  `abc_results_job<id>.csv`). The submit file transfers `abc_results.csv` and remaps it
  per-process, so a fixed name is correct.
- **`detailed_sim_results`:** `run_abc_simulation` now creates `../out/detailed_sim_results/` and
  copies each trial's raw feature files into `run<N>/` (previously only `run_sims_from_csv` did).
  The submit file transfers this dir, so the prior-sampling path had to produce it.
Verified (stubbed): a run writes `abc_results.csv` + `detailed_sim_results/run1,run2/…`.

Operational items flagged to the user (not code): clone-URL vs push-remote mismatch
(`The-Senate-I-Am` vs `Sohan-All`), container must activate `cpb-env` + have `slim` on PATH,
`log/` dir must pre-exist, `run_code.sh` masks python's exit code, OSPool eviction risk on long
jobs. **These code changes are not yet committed — need commit + push to the cloned repo.**

## Status summary

- **Done + verified (code):** plan §2a, §2b, §2c, offline standardizer (§4), batched-stats
  perf fix, prior updates, a full end-to-end integration run, the CHTC per-job runner, and
  submit-file alignment fixes.
- **Environment: NOT a blocker.** SLiM 5.1 runs; the pipeline works end to end (~4 min/run,
  recapitation-dominated).
- **Not started (intentionally deferred):** SLiM `simplificationRatio=INF` fix (§5.5), Tajima's D,
  ABC-SMC — see plan §5.
