# TODO — updated 2026-08-12

Remaining work. Project context is in **`CLAUDE.md`**; section references (§) point there.

Split into **pipeline** work (code) and **Sohan** work (data, decisions, professor follow-ups).

---

## 0. State as of 2026-07-28

**The recalculation has been run.** It exposed an int32 overflow in pixy's `count_comparisons`
(CLAUDE.md §6.5), which has been fixed and the recalculation re-run.

| file | change |
|---|---|
| `ToUseOnBeagles/CallableSites.py` | **new** (07-27) — per-chromosome callable-site denominators (provisional: pixy `window_pos_2`) |
| `ToUseOnBeagles/AverageData.py` | rewritten (07-27) — pools across chromosomes, converts π/d_xy to per-site; **07-28: denominators now analytic, with a representability cross-check that raises** |
| `ToUseOnBeagles/CalcGenRel.py` | callable-site denominator; **row order fixed** to the specifier matrix; headerless output |
| `Python_Code/ABCAnalysisNoRedis.py` | `abc_results.csv` header written when the file is empty, not just absent |

Verified 07-28: a synthetic round-trip fixture recovers known π/d_xy to ~1e-9 under both int32
saturation and modular wrap, and raises on a non-overflow mismatch; the saturating-row count
predicted from sample sizes alone is **244, matching the real run exactly**; the full three-year
run completes without raising, which positively confirms `count_missing = 0` and that specifier
sample counts match pixy's popfile.

**The corrected targets are promoted and validated.** `data/empiricalStats/` now holds the
per-site values; `data/empiricalStats_old/` holds the old per-SNP ones (untracked). No code
changed — all four consumers (`ABCAnalysisNoRedis.py:226-229`, `ABCAnalysis.py:119,130`) point at
`data/empiricalStats/`, which is exactly why rename-in-place was the right promotion.

Validation passed 07-29, every check from the old list: 2015 mean π **0.0121171** as predicted;
old/new ratio uniform **11.448–11.523** across all 24 rows; `H53-2015` π **0.012371** (not
negative); `Alsum25×Refuge`/`×H15` d_xy **0.012565 / 0.012443**; **zero** negative π or d_xy in
any year; 2019/2023 unchanged; all four matrices match the specifier dimensions 24/17/20.

Two expected-but-unlisted differences, both accounted for: **F_st moved ≤3.7e-3** (the §5.2
SNP-weighted pooling change in `AverageData.py:274-285`, *not* the denominator — F_st is
denominator-invariant), and **relatedness rescaled by a per-year constant** (11.70/11.93/11.80,
identical to 4 s.f. across every entry — the `NORMALISE_BY` switch to callable sites).

### Done 2026-07-29

| item | where |
|---|---|
| `recombination_rate` now reaches SLiM via `-d RECOMB` | `CPBSampleSim{Linux,Win}.slim:12`, `Main.py` (§6.3) |
| `simplificationRatio=INF` removed | `CPBSampleSim{Linux,Win}.slim:4` (§6.4) |
| IBD demoted fitted → diagnostic | `abc_standardize.py::FITTED_STATS`, `calculate_losses` (§7.1) |
| Small-subpop exclusion | `ABCAnalysisNoRedis.py::get_keep_mask` (§7.0) |

### Immediate next steps

- [x] ~~**Run the pipeline end-to-end at low POPMULT.**~~ **Done 07-31.** Runs clean at POPMULT
      500 and 5000; **OOMs at 12000 on a 16 GB machine**, inside recapitation (not SLiM). Full
      timing/memory table and CHTC sizing now in CLAUDE.md §3.1. `-d RECOMB` confirmed threading
      through the real model; recapitation confirmed working on a runtime-simplified tree
      sequence, which is §6.4's safety argument actually exercised.
- [ ] **Commit and push** all of it together — CHTC clones from `origin/main`. Targets, SLiM
      files, `Main.py`, `ABCAnalysisNoRedis.py`, `abc_standardize.py`, `ToUseOnBeagles/AverageData.py`.
- [ ] **Decide whether to commit `data/empiricalStats_old/`.** Currently untracked. Recommend
      **not** committing: the old per-SNP values are already recoverable from git history (the
      previous commit of `data/empiricalStats/`), so committing duplicates history while leaving a
      live wrong-scale directory in the tree for someone to read by accident.
- [ ] **Delete `Python_Code/abc_results.csv`** (header-only, pre-refactor column layout).
      Precautionary — zero data rows. Still present as of 07-29.
- [ ] **Never pool ABC results across the 07-28/07-29 changes.** Rejection ABC pools trivially
      across jobs, which is exactly what makes incompatible batches easy to concatenate by
      accident. The fitted-statistic set itself changed, so old rows are not comparable.
- [ ] **Optional:** add the overflow fixture test to the repo. `ToUseOnBeagles/` has no tests, and
      this bug class is invisible in the output for any subpop under 14 individuals. The working
      version is in the session scratchpad (`test_averagedata.py`); it needs the `SRC` path made
      relative before it is worth committing.

---

## 1. Blocking the big pass

- [x] ~~**Find the provenance of `ancestral_Ne = 6700`**~~ — **done 08-04.** It is `N_a` from
      **Cohen et al. 2022, Evol. Appl. 15:1691–1705, Figure 3a** (PDF in repo root), the dadi
      ancestral size for the WI/NY split. The professor question is answered; see §6.1 for the
      full write-up and for the four *other* project constants that trace to the same paper
      (`2.75e-6`, the 324-generation run length, μ=2.1e-9, and the `pop` prior range).

- [x] ~~**Run the `--anc-ne` / `--mu` ridge sweep**~~ — **done 08-04**, `diagnostics/ridge_sweep.py`,
      results in `out/ridge_p150.jsonl`. Full table in §6.2.1. Three outcomes:
      **(1) π is ridge-invariant** (−2.1 to −2.6% over a 3× Ne step) — Test 1 passes.
      **(2) The between-subpop CV does NOT collapse** (flat to <1%; mean *and* sd both scale
      ×2.9 with Ne_anc) — **my prediction was wrong. π keeps its information and stays fitted.**
      **(3) Recapitation cost scales as Ne^2.34** at flat memory — the new blocker.

- [x] ~~**Reset `ancestral_Ne` to ~1.4e6 and μ to 2.1e-9**~~ — **REVERSED 08-04. Keep 6700.**
      Raising it would (a) barely change the inference, since π is ridge-invariant in both level
      and relative spread, and (b) cost **~600 days per trial** at Ne=1.452e6. The Cohen critique
      still stands as biology — 6700 fails its own paper's internal check by 52× — but it does not
      license a code change, because the defensible value cannot be simulated. **Wall time, not
      RAM, is the wall**, so CHTC does not rescue it either.

- [~] **Fix the *language* around μ** (§6.1). μ is not a mutation rate; it is half of a
      calibration constant whose only meaningful content is `4·Ne_anc·μ`.
      **(a) DECIDED 08-11 — μ stays a free nuisance parameter, but with `s` tightened 0.5 → 0.05.**
      (The originally-planned "drop it entirely" was considered and not taken.) At `s=0.5` a single
      draw swung π by ±65%, swamping both the observed between-site spread (0.014–0.021) and the
      ±3% POPMULT drift, so `pi_loss` would have ranked draws mostly on the μ draw. `s=0.05` is
      sized to the calibration's own uncertainty (~1% Monte-Carlo + ~3% POPMULT coupling).
      **~~Residual~~ RESOLVED 08-12 — `s` is now 0.02**, and the earlier residual was understated.
      Measured (§7.2.1): `calculate_losses` applies no level re-fit, so a μ draw shifts every
      simulated log π bodily into `pi_loss`. Median injected `pi_loss` is **0.0401 at `s=0.05`**
      against a POPMULT-driven signal range of **0.0459** — i.e. a typical μ draw moved the
      distance as far as the *entire* POPMULT prior, so `pi_loss` would have ranked draws mostly on
      μ. At `s=0.02` the injection is 0.0243 against a flat floor of 0.0209. `s=0.01` (0.0220) was
      rejected as overconfident — tighter than the ~3% POPMULT coupling the prior exists to absorb.
      **(b) done 08-11** — `ABCAnalysisNoRedis.py:12-16` rewritten with the calibration provenance.
      **(c) still open** — make `abc_standardize.py` report **θ = 4Nμ**, never μ alone.

- [x] ~~**Check whether simulated and observed π covary site-by-site**~~ — **done 08-12,
      `diagnostics/pi_covary.py`, write-up in §7.2.1. They do NOT.** Per-year Pearson
      −0.091/+0.538/−0.021 (p = 0.61/0.027/0.91), pooled +0.025 (p=0.86); 2019's nominal hit is
      leveraged (Spearman 0.194) and does not replicate — the sim's low-π demes are the same three
      cluster rows every year while the observed ranks shuffle. **The decisive number: a
      structureless simulation scores `pi_loss` = 0.02094**, and that floor depends only on the
      *observed* vectors, so it applies at every POPMULT already run — 500 is 3.21× worse than it,
      2000 is 1.38× worse, and 5000 sits **at** it (1.02×). Correctly-placed structure buys 2.0%.
      **So the suspicion was right: `pi_loss` is scoring flatness.** π's POPMULT information is
      one-sided and saturating — it excludes small POPMULT but cannot peak, because L1 never
      penalises a too-flat simulation. **Do not read a large-N π signal as corroborating F_st.**

- [ ] **NEW (§6.6) — simulated π is computed over whole demes (338–714 diploids) rather than the
      year's real per-site counts (4–19).** A 60–172× mismatch, and it reads as a violation of §8
      invariant 1. The π *level* is unaffected (pairwise diversity is unbiased at any n), so the
      μ calibration stands — but the *spread* comparison is not: the observed vector carries
      sampling noise the simulated one lacks entirely. Two consequences worth chasing before the
      pass: it means **"the sim is 1.5× too flat" (§7.2 impl. 3) is partly correct behaviour**, and
      it is a **live alternative explanation for the §7.2.1 null** (a real site-level signal could
      be buried in observed-side noise). Fix by subsampling the simulated demes to the real counts
      and **recomputing** — never by slicing (§8 invariant 2). Costs nothing extra: recapitation is
      already paid for and smaller sample sets make the statistics cheaper.
- [x] ~~**Re-calibrate μ at the POPMULT you actually intend to run.**~~ — **done 08-11**,
      `diagnostics/mu_calibrate.py`, results in `out/mu_calibration.jsonl`, write-up in §6.1.1.
      **`DEFAULT_MUTATION_RATE = 4.646e-7`** (was 5e-6, i.e. **10.8× too large**), calibrated at
      POPMULT=5000 / numClusters=33 / total_migration=0.05. Measured at three POPMULTs
      (500 / 2000 / 5000 → μ = 5.564e-7 / 4.819e-7 / 4.646e-7). `branch_div` saturates against a
      hard ceiling `2·(324 + 2·Ne_anc) = 27448`, so **one fixed μ covers the whole prior**: held
      at 4.646e-7, π drifts only −3.2% (POPMULT=2000) to +2.1% (12000). Three independent
      cross-checks against §6.2 and `ridge_sweep.jsonl` passed, one of them exact.

- [x] ~~**BLOCKING — π and F_st currently pull in opposite directions**~~ (§6.2) — **RESOLVED
      08-11 by the recalibration.** The conflict existed only at μ=5e-6. With μ calibrated, both
      fitted losses improve *together* with POPMULT: π 0.067→0.029→0.021 and F_st
      0.071→0.019→0.0083 over POPMULT 500→2000→5000. **This unblocks the pass** as far as §1 is
      concerned. F_st still wants slightly more than POPMULT=5000 (sim 0.0079 vs observed
      0.00645, implying POPMULT ≈ 6000), which is comfortably inside the prior.
- [~] **Re-derive whether N is identifiable** (§6.2) — **partially answered 07-31: yes, via F_st.**
      A 500→5000 POPMULT sweep moved simulated F_st from 0.076 onto the observed 0.008, improving
      `fst_loss` **8.7×**, tracking `1/(1+4Nm)`. The old "N is nearly unidentifiable" claim came
      from π alone and does not hold for the structural side. **Still open:** the full sweep with a
      corrected μ, and branch-mode diversity as the diagnostic. Keep POPMULT ≤ ~5000 locally
      (§3.1) or run it where there is more RAM.
- [ ] **Re-run the integration test** and confirm `pi_loss` is informative — it was a saturated
      ~7.8 at biological μ, and 0.21 under the broken scale. Do this before spending CHTC compute.
      Note π is now **one of only two** fitted statistics, so this matters more than it did.

---

## 2. Pipeline / code

- [ ] **Windows Smart App Control blocks scikit-learn on the dev box — `Main.py` cannot run
      locally** (found 08-12, §3). `VerifiedAndReputablePolicyState = 1`; the unsigned
      `sklearn/metrics/_dist_metrics.cp312-win_amd64.pyd` is refused at load. Only the KMeans step
      is affected — numpy/scipy/tskit/msprime/pyslim all load fine, and diagnostics reusing
      `data/cluster_data.csv` with `--skip-slim` are unaffected. **Sohan's call:** turning Smart App
      Control off is a one-way change via Windows Security → App & browser control. Not a CHTC
      problem. Mitigated in code by making `import Main` lazy inside `model()`
      (`ABCAnalysisNoRedis.py`), which also decouples the loss/statistics half from the simulation
      stack so diagnostics reuse the real `get_keep_mask`/readers.

- [ ] **`diagnostics/qdriver.py` is broken — fix or delete** (found 08-04). Three drifts from
      production, any one of which invalidates its output: `simplificationRatio=INF` in its SLiM
      template (§6.4 bug), `--slim-rho` default `1e-8` (§6.3 bug value), and a
      `determine_migration_rates(distances, modifier=...)` call whose signature no longer exists,
      which raises `TypeError` immediately. It has therefore not run since the migration refactor,
      so nothing recent depends on it — but **`qpost.py` shares its output-file convention**, so
      decide the two together. `ridge_sweep.py` was written to bypass both.
- [x] ~~**`filter_populations=False` missing in the diagnostics harness**~~ — **fixed 08-04**,
      `qdriver.py:154` and `qpost.py:52`. Same bug class as commit `c5963ae`: both simplified with
      the default `filter_populations=True`, then queried `ts.samples(population=i, ...)` with
      original cluster-row indices. Exposure was likely small at numClusters=33, but **re-measure
      any sweep number that came from these scripts** rather than the full pipeline, and note this
      was a prerequisite for the §6.1 `--anc-ne` sweep above.
- [x] ~~**`recombination_rate` never reaches SLiM**~~ (§6.3) — **done 07-29**, both .slim files.
- [x] ~~**Remove `simplificationRatio=INF`**~~ (§6.4) — **done 07-29**, both .slim files. Safety
      checked against the SLiM/pyslim docs, not assumed. **Not yet run.**
- [ ] **Consider Remembering a bounded subset at gens 308/316** (§6.4). Currently *every*
      individual in *every* subpop is permanently Remembered at both timepoints, pinning all their
      ancestry — a memory floor simplification cannot touch. Only 2–19 per site are ever sampled
      downstream, so ~50/subpop would be ample. Changes what the sampler can draw, so it is a
      design decision. Do this only if memory is still a problem after §6.4.
- [ ] **Pool F_st properly** (§5.2) — optional. pixy emits no variance components, so
      `AverageData.py` uses a SNP-weighted mean. Proper pooling needs `allel.weir_cockerham_fst`
      recomputed from the VCFs. Error is a few percent and F_st is scale-invariant, so this is the
      lowest-stakes item here.
- [ ] **Tajima's D as a posterior-predictive check** (not fitted). Attractive because it comes
      from the SFS and is **immune to the denominator problem entirely**. Sim: `ts.Tajimas_D()`.
      Empirical: per-subpop SFS via `allel.tajima_d` on the VCFs already loaded in `CalculateLD,py`.
      **Never MAF-filter first** — rare variants are the whole signal. Caveat: imputation bias hits
      rare variants hardest, biasing D upward.
- [ ] **LD statistic for recombination** — deferred, low priority. Blockers: recombination must
      reach SLiM first; 1e6 bp is too short for ~100 kb LD decay (few independent long-range
      pairs); Beagle imputation inflates empirical LD vs tskit's perfect phase (systematic, biases
      r downward); ρ=4N_er is another compound parameter (another ridge). Use tskit
      `ld_matrix(stat="r2")` directly, not via VCF (`write_vcf` rounds positions → duplicates on
      short sequences). Match MAF/bins/MAX_DIST/biallelic to the empirical side.
- [ ] **Optional cleanup:** add `##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">` to
      `ConvertBeagleToVCF.py::generate_VCF_header` to silence the scikit-allel warning (§5.4).
      Cosmetic; only matters if VCFs are ever regenerated.

---

## 3. Sohan — data, decisions, provenance

- [x] ~~**Ask the professor where `ancestral_Ne = 6700` came from.**~~ — **answered 08-04 from the
      Cohen et al. 2022 PDF** (§6.1). No longer a provenance question.

- [ ] **Still worth raising with the professor — but now it is a disagreement-with-a-published-
      number question, not a "where did this come from" question.** Cohen et al. is very likely
      his own lab's paper, so lead with the internal check, not with our π:

  > I traced our ancestral Ne = 6700 to Figure 3a of the 2022 Evolutionary Applications paper —
  > the dadi N_a for the WI/NY split. The trouble is it doesn't reconcile with our data by ~200×,
  > and I think the problem is upstream of us: if I take Watterson's θ from the SNP counts
  > reported in that paper itself — 11.8M polymorphic sites over ~870 Mb at n=28 — I get
  > Ne ≈ 3.5e5 at the same midge mutation rate, which is 52× the 6700 the dadi model reports.
  > Our corrected π puts us at ~1.45e6, only 4× from that. The paper does flag that low coverage
  > would bias the SFS downward, and that dadi and Stairway plot disagreed 4×. My guess is that
  > N_a = θ/(4μL) used L = 840 Mb (all intergenic sequence) while the SFS itself was built from a
  > much more stringently filtered subset — which would deflate N_a by exactly that ratio. Does
  > that sound right, and is the filtered site count recoverable? It decides whether we keep 6700
  > or recalibrate to ~1.4e6.

- [ ] **Ask whether `N_WI = 15,000` should anchor the `pop` prior more tightly** (§6.1). The
      current `U(2000, 12000)` gives total N ≈ 6.7k–40k, which brackets both Cohen estimates —
      but that bracketing looks incidental rather than deliberate, and the same low-coverage bias
      that hits `N_a` hits `N_WI` too.

- [ ] **Ask about the collection protocol for `Mortensen9-2015` and `H41-2023`** — the second
      highest-value question after Ne (§7.2):

  > Two of our sites look strongly differentiated from everything else — Mortensen9 in 2015 and
  > H41 in 2023, at roughly 3–6× the next-highest site. But they're also the *lowest*-diversity
  > and *most*-internally-related samples in their year, and they're not geographically remote
  > (the genuinely distant fields aren't differentiated at all). That pattern looks less like a
  > distinct population and more like the beetles in those samples being close relatives — say,
  > collected off a single plant or one egg mass. Do you know how those two fields were sampled,
  > or whether either was newly colonized that season?

- [ ] **Raise the dispersal-kernel specification** in the same conversation (§7.1, §7.2). Two
      independent results say connectivity here is not distance-limited: the IBD slope is null in
      all three years, and F_st structure is site-coherent rather than distance-organized. For a
      pest moved on farm equipment and seed potatoes that is unsurprising, but an exponential
      distance-decay kernel cannot produce "one isolate, rest panmictic." **This is a model
      question, not an ABC-tuning question**, and it may matter more than any prior.
- [ ] **Set the `total_migration` prior ceiling** deliberately — currently the placeholder
      `U(0.001, 0.301)` at `ABCAnalysisNoRedis.py:27`. Needs a biological CPB immigration ceiling
      from the dispersal literature or the professor. Anchor: observed mean F_st ≈ 0.003–0.008
      implies **Nm ≈ 31–83** (§7.1).
- [ ] **Decide on the duplicated `Arlington` label** (§4) — two near-identical subpopulations from
      a typo. Merge, or keep deliberately. **Lower urgency now:** the small-subpop mask already
      drops `Arlington2015` (n=2) from the fitted statistics, so it only affects diagnostics.
- [x] ~~**Check the empirical IBD slope is significantly ≠ 0**~~ — **done 07-29. It is not**, in any
      year (Mantel, 9999 perms: p = 0.639 / 0.148 / 0.985; sign flips across years). IBD demoted
      to diagnostic; **`scale` is unidentifiable and must not be reported as inferred** (§7.1).
- [ ] **Decide whether `scale` should stay a free parameter at all.** With IBD unfitted nothing
      constrains it, so sampling it just adds a nuisance dimension that widens the prior volume
      the pass has to cover. Fixing it would make the pass cheaper for free. Requires picking a
      defensible value — see the kernel-specification item above.
- [ ] **Optional:** get reference assembly chromosome lengths. Would give the other bound on
      callable sites (§5.1) and confirm that `chr{i}_cpb.vcf.gz` really is assembly chromosome
      `i` — currently unverifiable, since `ConvertBeagleToVCF.py` hardcodes `CHROM = 9` (§5.4).
      Not blocking; the bracket is ~10–20% against a ~2400× correction.

---

## 4. Running the pass (once §1 clears)

- [ ] **Measure the noise floor FIRST.** Run 2–3 replicates at identical parameters and compute
      the distance `D` between them. If the acceptance threshold ε lands below that floor, the ABC
      is selecting on coalescent/subsampling noise rather than signal.
- [ ] **Smoke test:** 5 jobs × 5 trials. Confirm each `abc_results.csv` has exactly 5 rows **and a
      header row** (the §6.5 fix), and that `slim` is on PATH and `cpb-env` activates inside the
      container. Clear any empty result CSVs from earlier failed runs so they aren't concatenated.
- [ ] **Size `request_memory` and `request_disk` from CLAUDE.md §3.1 before submitting.**
      ~8 GB at POPMULT≈5000, **~20 GB at the prior ceiling of 12000**. Undersized memory means
      jobs held or evicted *mid-pass*, and rejection ABC pools whatever survived — silently
      biasing the posterior toward the small-POPMULT draws that fit. Either request for the
      ceiling or lower the ceiling deliberately.
- [ ] **First real batch:** many short jobs. **Re-plan the trial count — it is no longer ~4
      min/trial.** Measured 07-31: **30 min/trial at POPMULT=5000**, projecting ~60 min at 12000
      (§3.1). At those rates 20 trials/job is 10–20 h, far too long for OSPool eviction; think
      **2–5 trials/job**. **Bump the `OFFSET`/seed per batch so job ids never repeat** — repeated
      ids re-draw identical parameter sets and bias the posterior.
- [ ] **After the pass:** run `abc_standardize.py` (offline σ=1.4826·MAD → standardized `D` →
      ranked CSV), then inspect posteriors. **Report θ=4Nμ and Nm**, not a standalone N peak, and
      check whether the N marginal escapes the prior. Use relatedness, d_xy, **the IBD slope**
      (newly available as a check now that it isn't fitted) and Tajima's D if added as
      posterior-predictive checks — statistics you didn't fit are the real validation.
      Remember `numClusters` in the CSV is the raw draw; multiply by 33.
      **Do not report a `scale` posterior** — nothing constrains it (§7.1).
- [ ] **Watch for π/F_st double-counting.** They are strongly correlated in the observed data
      (r = −0.72 in 2015, −0.92 in 2023; §7.2), so equal weights after MAD-standardization partly
      weight one signal twice. Now that the fitted set is only these two, that is the whole
      distance. Consider unequal `WEIGHTS` in `abc_standardize.py` once the pilot batch exists.
      **§7.2.1 sharpens this into a concrete risk.** `pi_loss` saturates at a floor of 0.0209 and
      is one-sided, so across a pilot batch drawn from `pop ~ U(2000, 12000)` — all of it at or
      near that floor — π's residual variation will be mostly the μ draw plus coalescent noise,
      not POPMULT. MAD-standardizing that small spread would then **inflate** a near-noise
      statistic to parity with F_st, which is the one carrying the peak. Inspect the per-statistic
      MADs before accepting equal weights, and expect π to warrant *less* than half.
- [ ] **Manual ABC-SMC iteration**, if wanted: take the top ~20% of trials, resample the prior
      around them, run another batch, repeat. Effectively SMC by hand. Note a single big rejection
      batch is unbiased with no importance weights; manual iteration needs them to stay that way.
